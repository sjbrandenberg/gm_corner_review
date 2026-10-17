"""Per-trace features and labels.

Two outputs per trace:
  * a flat dict of scalars (labels, metadata, automatic decisions, summaries)
  * a dict of fixed-length float32 curves (SNR, spectra, displacement-ratio
    curves, PSA) on common grids, for ML models.

Nothing here uses the analyst's choices to build predictors; analyst values
appear only in columns prefixed `y_`.
"""

import numpy as np

from . import gui_replica as gr

# Common grids ---------------------------------------------------------------
FREQ_GRID = np.logspace(np.log10(0.01), np.log10(50.0), 80)   # Hz, spectra/SNR
FCHP_GRID = np.logspace(np.log10(0.02), np.log10(3.0), 25)    # Hz, trial fchp
SNR_THRESHOLDS = (2.0, 3.0, 5.0, 10.0)
GUI_FCHP_CAP = gr.GUI_FCHP_KW["fchp_max"]


def _interp_loglog(x, y, grid):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.full(len(grid), np.nan, np.float32)
    out = np.interp(np.log(grid), np.log(x[ok]), np.log(y[ok]), left=np.nan, right=np.nan)
    return np.exp(out).astype(np.float32)


def snr_band_edges(freq, snr, threshold, f_lo=0.1, f_hi=10.0):
    """Walk outward from the SNR peak (within f_lo..f_hi) to where SNR < threshold."""
    freq = np.asarray(freq, float)
    snr = np.asarray(snr, float)
    ok = np.isfinite(snr) & (freq > 0)
    freq, snr = freq[ok], snr[ok]
    band = (freq >= f_lo) & (freq <= f_hi)
    if not band.any():
        return np.nan, np.nan
    ipk = np.flatnonzero(band)[np.argmax(snr[band])]
    if snr[ipk] < threshold:
        return np.nan, np.nan
    lo = ipk
    while lo > 0 and snr[lo - 1] >= threshold:
        lo -= 1
    hi = ipk
    while hi < len(snr) - 1 and snr[hi + 1] >= threshold:
        hi += 1
    return freq[lo], freq[hi]


NGAW2_T = np.array([
    0.01, 0.02, 0.022, 0.025, 0.029, 0.03, 0.032, 0.035, 0.036, 0.04, 0.042, 0.044, 0.045,
    0.046, 0.048, 0.05, 0.055, 0.06, 0.065, 0.067, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095,
    0.1, 0.11, 0.12, 0.13, 0.133, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.24, 0.25,
    0.26, 0.28, 0.29, 0.3, 0.32, 0.34, 0.35, 0.36, 0.38, 0.4, 0.42, 0.44, 0.45, 0.46, 0.48,
    0.5, 0.55, 0.6, 0.65, 0.667, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4,
    1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0, 3.2, 3.4, 3.5, 3.6, 3.8, 4.0,
    4.2, 4.4, 4.6, 4.8, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 20.0,
])


def psa_spectrum(acc, dt, T=NGAW2_T, D=0.05, chunk=16):
    """5%-damped pseudo-spectral acceleration via frequency-domain SDOF transfer
    function (same formulation as ucla_geotech_tools.response_spectrum, whose
    v3.0.3 get_ARS has a missing '*' bug). Zero-padded to avoid wrap-around."""
    import scipy.fft

    T = np.asarray(T, float)
    n = scipy.fft.next_fast_len(len(acc) + int(np.ceil(2 * T.max() / dt)))
    freq = np.fft.rfftfreq(n, dt)
    facc = np.fft.rfft(acc, n=n)
    out = np.empty(len(T))
    for i in range(0, len(T), chunk):
        Tc = T[i:i + chunk, None]
        h = (1.0 + 2.0j * D * freq * Tc) / (1.0 + 2.0j * D * freq * Tc - freq ** 2 * Tc ** 2)
        out[i:i + chunk] = np.max(np.abs(np.fft.irfft(facc * h, n=n)[:, :len(acc)]), axis=1)
    return out


def psa_shape_metrics(T, psa, pga):
    """Indicators of spiky short-period response spectra (high-frequency artifacts)."""
    T = np.asarray(T, float)
    lnS = np.log(np.maximum(np.asarray(psa, float), 1e-30))
    short = (T > 0) & (T <= 0.1)
    out = {"psa_rough_short": np.nan, "psa_max_bump_short": np.nan, "psa001_over_pga": np.nan}
    if short.sum() >= 4:
        lnT = np.log(T[short])
        s = lnS[short]
        d2 = np.diff(s, 2) / np.diff(lnT)[:-1] ** 2
        out["psa_rough_short"] = float(np.std(d2))
        trend = np.polyval(np.polyfit(lnT, s, 2), lnT)
        out["psa_max_bump_short"] = float(np.max(s - trend))
    i001 = np.argmin(np.abs(T - 0.01))
    if pga > 0:
        out["psa001_over_pga"] = float(np.exp(lnS[i001]) / pga)
    return out


def trace_features(ws, trace_id, compute_psa=True):
    """Return (scalars, curves) for one processed trace in workspace `ws`."""
    import obspy
    from obspy.geodetics import gps2dist_azimuth

    net, sta, loc, cha = trace_id.split(".")
    rec = {
        "event_id": ws.event_id, "label": ws.label, "trace_id": trace_id,
        "network": net, "station": sta, "location": loc, "channel": cha,
        "instrument": cha[:2], "band_code": cha[0], "inst_code": cha[1],
        "component": cha[2], "is_vertical": cha[2] == "Z",
        "source_file": str(ws.path.name),
    }
    curves = {}

    # --- event / station / distance
    ev = ws.event()
    rec.update(ev)
    netsta = ws.netsta(trace_id)
    inv = ws.inventory(netsta)
    try:
        coords = inv.get_coordinates(trace_id, obspy.UTCDateTime(ev["event_time"]))
        rec["sta_lat"], rec["sta_lon"] = coords["latitude"], coords["longitude"]
        rec["sta_elev_m"] = coords.get("elevation", np.nan)
        orient = inv.get_orientation(trace_id, obspy.UTCDateTime(ev["event_time"]))
        rec["channel_azimuth"], rec["channel_dip"] = orient["azimuth"], orient["dip"]
        repi = gps2dist_azimuth(ev["event_lat"], ev["event_lon"],
                                coords["latitude"], coords["longitude"])[0] / 1000.0
        rec["repi_km"] = repi
        rec["rhypo_km"] = float(np.hypot(repi, ev["event_depth_km"]))
    except Exception as exc:
        rec["coord_error"] = repr(exc)

    # --- analyst labels
    rv = ws.review(trace_id)
    rec["has_review"] = rv is not None
    rv = rv or {}
    rec["y_reviewed"] = rv.get("reviewed", 0)
    rec["y_accepted"] = rv.get("accepted", np.nan) if rv.get("reviewed", 0) == 1 else np.nan
    rec["colocated"] = rv.get("colocated", np.nan)
    rec["selected"] = rv.get("selected", np.nan)
    rec["review_username"] = rv.get("username")
    rec["review_timestamp"] = rv.get("timestamp")
    rec["y_fchp"] = rv.get("highpass", np.nan)
    y_fclp = rv.get("lowpass", np.nan)
    rec["y_fclp_raw"] = y_fclp

    # --- gmprocess automatic decisions
    tp = ws.processing_parameters(trace_id)
    cf = tp.get("corner_frequencies", {})
    rec["fchp_gmp"] = cf.get("highpass", np.nan)
    rec["fclp_gmp"] = cf.get("lowpass", np.nan)
    rec["corner_method_gmp"] = cf.get("type")
    fail = tp.get("failure") or {}
    rec["auto_failed"] = bool(fail)
    rec["auto_fail_module"] = fail.get("module")
    rec["auto_fail_reason"] = fail.get("reason")
    rec["stream_any_failure"] = ws.stream_parameters(trace_id).get("any_trace_failures")
    rec["clipping_prob"] = (tp.get("clipping_probability") or {}).get("probability", np.nan)
    rec["zero_crossing_rate"] = (tp.get("ZeroCrossingRate") or {}).get("crossing_rate", np.nan)
    rec["noise_duration"] = (tp.get("noise_spectrum") or {}).get("duration", np.nan)
    rec["signal_duration"] = (tp.get("signal_spectrum") or {}).get("duration", np.nan)
    fs = tp.get("fit_spectra") or {}
    for k in ("stress_drop", "f0", "magnitude", "kappa", "R2", "mean_squared_error"):
        rec[f"fit_{k}"] = fs.get(k, np.nan)
    rec["fit_success"] = fs.get("minimize_success")
    prov = ws.provenance(trace_id)
    rec["fchp_applied"], rec["fclp_applied"] = gr.applied_corners(prov)
    rec["ridder_changed_fchp"] = (
        np.isfinite(rec["fchp_applied"]) and np.isfinite(rec["fchp_gmp"])
        and abs(rec["fchp_applied"] - rec["fchp_gmp"]) > 1e-6 * rec["fchp_gmp"]
    )

    # --- cached SNR and smoothed spectra
    snr_f, snr = ws.cached(trace_id, "SnrFreq"), ws.cached(trace_id, "SnrSnr")
    if snr_f is not None and snr is not None:
        curves["snr"] = _interp_loglog(snr_f, snr, FREQ_GRID)
        for thr in SNR_THRESHOLDS:
            lo, hi = snr_band_edges(snr_f, snr, thr)
            rec[f"snr{int(thr)}_flo"], rec[f"snr{int(thr)}_fhi"] = lo, hi
        band = (snr_f >= 0.1) & (snr_f <= 10)
        rec["snr_max_0p1_10"] = float(np.nanmax(snr[band])) if band.any() else np.nan
        for f0 in (0.1, 0.3, 1.0, 3.0, 10.0):
            rec[f"snr_at_{f0:g}hz"] = float(np.interp(f0, snr_f, snr, left=np.nan, right=np.nan))
    for name, key in (("SmoothSignalSpectrum", "fas_signal"), ("SmoothNoiseSpectrum", "fas_noise")):
        fq, sp_ = ws.cached(trace_id, name + "Freq"), ws.cached(trace_id, name + "Spec")
        if fq is not None and sp_ is not None:
            curves[key] = _interp_loglog(fq, sp_, FREQ_GRID)

    # --- waveforms the analyst saw
    tr_p = ws.obspy_trace(trace_id, processed=True)
    tr_u = ws.obspy_trace(trace_id, processed=False)
    rec["sampling_rate"] = tr_p.stats.sampling_rate
    rec["nyquist"] = tr_p.stats.sampling_rate / 2.0
    rec["npts_processed"] = tr_p.stats.npts
    rec["pga_gmp_g"] = float(np.max(np.abs(tr_p.data)) / 981.0)

    # Lowpass sentinel: GUI stores max(freq)*100 when the lowpass box is unchecked
    rec["y_fclp_none"] = bool(np.isfinite(y_fclp) and y_fclp > rec["nyquist"] * 1.01)
    rec["y_fclp"] = np.nan if rec["y_fclp_none"] else y_fclp

    if tr_u is None:
        rec["gui_error"] = "no unprocessed waveform"
        return rec, curves

    acc_u, dt, nt = gr.gui_unprocessed_acc(tr_u, tr_p, inv, prov)
    rec["preevent_noise_s"] = nt
    rec["fchp_gui"] = gr.gui_fchp(acc_u, dt, nt)
    rec["gui_at_cap"] = rec["fchp_gui"] >= GUI_FCHP_CAP * (1 - 1e-6)
    rec["pga_unproc_g"] = float(np.max(np.abs(acc_u)) / 9.81)

    trial = np.concatenate([FCHP_GRID, [rec["fchp_gmp"], rec["fchp_gui"]]])
    trial = np.where(np.isfinite(trial) & (trial > 0), trial, FCHP_GRID[0])
    poly, pre = gr.displacement_ratio_curves(acc_u, dt, nt, trial)
    curves["poly_ratio"] = poly[:len(FCHP_GRID)].astype(np.float32)
    curves["pre_disp_ratio"] = pre[:len(FCHP_GRID)].astype(np.float32)
    rec["poly_ratio_at_gmp"], rec["poly_ratio_at_gui"] = poly[-2], poly[-1]
    rec["pre_ratio_at_gmp"], rec["pre_ratio_at_gui"] = pre[-2], pre[-1]

    # Which automatic value did the analyst adopt?
    if rec["y_reviewed"] == 1 and np.isfinite(rec["y_fchp"]):
        y = rec["y_fchp"]
        close = lambda a: np.isfinite(a) and abs(y - a) <= 1e-5 + 1e-4 * a
        rec["y_fchp_source"] = ("gmprocess" if close(rec["fchp_gmp"]) else
                                "gui_auto" if close(rec["fchp_gui"]) else "manual")

    # PSA of the record as first shown to the analyst (gmprocess corners, GUI filter)
    if compute_psa and np.isfinite(rec["fchp_gmp"]):
        fclp = rec["fclp_gmp"] if np.isfinite(rec["fclp_gmp"]) else 0
        acc_f = gr.gui_filter(acc_u, dt, rec["fchp_gmp"], fclp)
        psa = psa_spectrum(acc_f / 9.81, dt)
        curves["psa_gmp"] = psa.astype(np.float32)
        rec.update(psa_shape_metrics(NGAW2_T, psa, np.max(np.abs(acc_f)) / 9.81))

    return rec, curves
