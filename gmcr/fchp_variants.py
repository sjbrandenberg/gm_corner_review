"""Variants of the polynomial-fit criterion of ucla_geotech_tools.auto_fchp, for calibration.

The GUI (and get_fchp) fits a 6th-order polynomial to the high-pass filtered displacement of the whole
record and compares the amplitude of the fit with the amplitude of the displacement. With a fixed order,
the frequency band the criterion tests scales with the reciprocal of the record length, so records with
long pre- and post-event windows get lower corners. These curves evaluate the same ratio for other
polynomial orders and for fit windows tied to the event: [t5 - k*D5-95, t95 + k*D5-95], with t5 and t95
the 5% and 95% Arias intensity times of the unprocessed acceleration, clipped to the record.

Curves are named polyvar_p<order>_<window>, with values at features.FCHP_GRID. The displacement is
filtered as in the GUI (gui_replica._fdisp, Butterworth of order GUI_FILTER_ORDER), and
polyvar_p6_rec equals the GUI's poly_ratio curve.
"""

import numpy as np

from . import gui_replica as gr

ORDERS = (2, 4, 6, 8, 12)
WINDOWS = {"rec": None, "k0p5": 0.5, "k1": 1.0, "k2": 2.0}
NAMES = [f"polyvar_p{p}_{w}" for w in WINDOWS for p in ORDERS]


def arias_times(acc, dt, lo=0.05, hi=0.95):
    """Times at which the Arias intensity reaches fractions lo and hi of its final value"""
    ia = np.cumsum(np.asarray(acc, float) ** 2)
    if ia[-1] <= 0:
        return np.nan, np.nan
    ia /= ia[-1]
    return np.argmax(ia >= lo) * dt, np.argmax(ia >= hi) * dt


def poly_ratio_variants(acc_u, dt, fchp_values, orders=ORDERS, windows=WINDOWS):
    """Polynomial-fit amplitude ratio max|fit| / max|disp| of the filtered displacement over each window,
    for each order and trial corner frequency.

    Returns (curves, scalars): curves maps NAMES to arrays of len(fchp_values); scalars holds the Arias
    times and, for each event-tied window, the fraction of the requested window inside the record.
    """
    fdisp, time, freq = gr._fdisp(acc_u, dt, gr.GUI_FCHP_KW["tukey_alpha"])
    fchp_values = np.atleast_1d(np.asarray(fchp_values, float))
    pos = freq > 0
    filt = np.zeros((len(fchp_values), len(freq)))
    filt[:, pos] = 1.0 / np.sqrt(1.0 + (fchp_values[:, None] / freq[pos]) ** (2 * gr.GUI_FILTER_ORDER))
    disp = np.fft.irfft(fdisp * filt, n=len(time), axis=1)  # one row per trial fchp

    t5, t95 = arias_times(acc_u[: len(time)], dt)
    scalars = {"arias_t5": t5, "arias_t95": t95}
    curves = {}
    pmax = max(orders)
    for wname, k in windows.items():
        if k is None:
            i0, i1 = 0, len(time)
        else:
            D = t95 - t5
            start, end = t5 - k * D, t95 + k * D
            i0, i1 = max(int(np.floor(start / dt)), 0), min(int(np.ceil(end / dt)) + 1, len(time))
            scalars[f"polyvar_cover_{wname}"] = (i1 - i0) * dt / (end - start) if end > start else np.nan
        d = disp[:, i0:i1]
        t = time[i0:i1]
        if d.shape[1] <= pmax + 1:
            for p in orders:
                curves[f"polyvar_p{p}_{wname}"] = np.full(len(fchp_values), np.nan)
            continue
        # Least-squares polynomial fits of every order from one QR factorization: the column space of a
        # degree-p fit is spanned by the first p+1 columns of Q. Chebyshev columns keep it well conditioned.
        x = 2.0 * (t - t[0]) / (t[-1] - t[0]) - 1.0
        Q, _ = np.linalg.qr(np.polynomial.chebyshev.chebvander(x, pmax))
        coef = Q.T @ d.T  # (pmax+1, n_trial)
        amp = np.max(np.abs(d), axis=1)
        for p in orders:
            fit = Q[:, : p + 1] @ coef[: p + 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                curves[f"polyvar_p{p}_{wname}"] = np.max(np.abs(fit), axis=0) / amp
    return curves, scalars
