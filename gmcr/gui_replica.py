"""Replica of the signal handling in the NGA-West3 review notebook
(ProcessorGUI-NGA-West3-Colocated-Review-v6.2), so that features describe the
waveforms the analyst actually looked at.

Verified against nc73819746.h5: `gui_fchp` reproduces the GUI's automatic
high-pass corner to 6 decimals for traces where the analyst adopted it.
"""

import warnings

import numpy as np
import scipy as sp
import ucla_geotech_tools.auto_fchp as af

# Settings hard-coded in the GUI's update_plots() call to af.get_fchp
GUI_FCHP_KW = dict(
    target=0.02, tol=0.001, filter_order=5, poly_order=6,
    fchp_min=0.001, fchp_max=1.2, maxiter=30, tukey_alpha=0.2,
    apply_disp_ratio=1, disp_ratio_target=0.05,
)
GUI_FILTER_ORDER = 5.0


def _parse_prefilt(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    txt = str(value).strip().strip("()[]")
    try:
        vals = tuple(float(v) for v in txt.split(","))
    except ValueError:
        return None
    return vals if len(vals) == 4 else None


def response_step(provenance):
    for label, attrs in provenance:
        if label == "Remove Response":
            return attrs
    return {}


def applied_corners(provenance):
    """Corner frequencies actually applied by gmprocess (from provenance)."""
    hp = lp = np.nan
    for label, attrs in provenance:
        if label == "Highpass Filter":
            hp = float(attrs.get("corner_frequency", np.nan))
        elif label == "Lowpass Filter":
            lp = float(attrs.get("corner_frequency", np.nan))
    return hp, lp


def gui_unprocessed_acc(tr_u, tr_proc, inventory, provenance):
    """Reproduce the GUI's 'acc_u' (m/s^2) from the raw trace.

    Returns (acc_u, dt, nt) where nt is the pre-event noise duration (s) kept
    ahead of the processed record's start time.
    """
    tr_u = tr_u.copy()
    rr = response_step(provenance)
    output = "ACC" if "cm/s^2" in str(rr.get("output_units", "")) else "VEL"
    water_level = rr.get("water_level")
    pre_filt = _parse_prefilt(rr.get("pre_filt_freqs"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tr_u.remove_response(inventory=inventory, output=output,
                                 water_level=float(water_level), pre_filt=pre_filt)
        except Exception:
            try:
                tr_u.remove_response(inventory=inventory, output=output)
            except Exception:
                tr_u.data = tr_u.data / 100.0

    nt = min(tr_proc.stats.starttime - tr_u.stats.starttime, 60)
    tr_u.trim(tr_proc.stats.starttime - nt, tr_proc.stats.endtime)
    tr_u.detrend(type="linear")
    window = sp.signal.windows.tukey(len(tr_u), alpha=0.05, sym=True)
    tr_u.data = window * (tr_u.data - np.average(tr_u.data, weights=window))
    if output == "ACC":
        acc_u = tr_u.data
    else:
        acc_u = tr_u.differentiate().data  # time-domain gradient, as in the GUI
    return np.asarray(acc_u, dtype=float), tr_u.stats.delta, float(nt)


def disp_ratio_time(nt):
    return nt - 2 if nt > 2 else nt


def gui_fchp(acc_u, dt, nt, **overrides):
    kw = dict(GUI_FCHP_KW, disp_ratio_time=disp_ratio_time(nt))
    kw.update(overrides)
    return float(af.get_fchp(dt=dt, acc=acc_u, **kw))


def _fdisp(acc, dt, tukey_alpha):
    """Same pre-processing as af.get_fchp."""
    if len(acc) % 2 != 0:
        acc = acc[:-1]
    window = sp.signal.windows.tukey(len(acc), alpha=tukey_alpha)
    acc = window * (acc - np.average(acc, weights=window))
    time = np.linspace(0, dt * len(acc), len(acc))
    freq = np.fft.rfftfreq(len(acc), dt)
    facc = np.fft.rfft(acc)
    fdisp = np.zeros(len(facc), dtype=complex)
    fdisp[freq > 0] = facc[freq > 0] / -((2 * np.pi * freq[freq > 0]) ** 2)
    return fdisp, time, freq


def displacement_ratio_curves(acc_u, dt, nt, fchp_values):
    """Polynomial-fit ratio and pre-event displacement ratio vs. fchp.

    These are the two criteria inside get_fchp (targets 0.02 and 0.05 in the
    GUI), evaluated on a grid so a model can re-learn the targets.
    """
    fdisp, time, freq = _fdisp(acc_u, dt, GUI_FCHP_KW["tukey_alpha"])
    t_pre = disp_ratio_time(nt)
    poly, pre = [], []
    for fc in np.atleast_1d(fchp_values):
        poly.append(af.get_residual1(fc, fdisp, time, GUI_FCHP_KW["poly_order"], 0.0,
                                     freq, GUI_FILTER_ORDER))
        pre.append(af.get_residual2(fc, fdisp, time, t_pre, 0.0, freq, GUI_FILTER_ORDER)
                   if np.any(time <= t_pre) else np.nan)
    return np.asarray(poly), np.asarray(pre)


def gui_filter(acc, dt, fchp, fclp, filter_order=GUI_FILTER_ORDER):
    """Acausal frequency-domain Butterworth used by the GUI's apply_filter."""
    facc = np.fft.rfft(acc)
    freq = np.fft.rfftfreq(len(acc), d=dt)
    pos = freq > 0
    facc[~pos] = 0
    facc[pos] /= np.sqrt(1 + (fchp / freq[pos]) ** (2 * filter_order))
    if fclp and fclp > 0 and np.isfinite(fclp):
        facc[pos] /= np.sqrt(1 + (freq[pos] / fclp) ** (2 * filter_order))
    out = np.fft.irfft(facc, n=len(acc))
    window = sp.signal.windows.tukey(len(out), alpha=0.05, sym=True)
    return window * (out - np.average(out, weights=window))
