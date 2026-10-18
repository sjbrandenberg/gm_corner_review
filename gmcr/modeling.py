"""Shared pieces of the corner-frequency and accept/reject models: data assembly,
leakage-safe feature sets, event-grouped cross-validation, metrics, and the
simple (interpretable) log-linear model.

Conventions
  * Rows are traces from traces_joined.parquet (or traces_all.parquet), optionally
    joined with the fixed-grid curves of curves_all.parquet.
  * Predictors never use analyst choices (y_*), the flatfile corners (ff_*), or
    analyst_id; the accept/reject model also excludes st_* (stations appear in the
    database only if accepted for some event).
  * Cross-validation holds out whole events, so the three components and all
    stations of an event are never split between training and test data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

GUI_CAP = 1.2  # Hz, fchp_max of the review GUI's automatic corner

# ---------------------------------------------------------------------------- data


def load_traces(features_dir, curves=False):
    """Traces table (joined with database metadata if available), optionally with curves."""
    feat = Path(features_dir)
    path = feat / "traces_joined.parquet"
    if not path.exists():
        path = feat / "traces_all.parquet"
    df = pd.read_parquet(path)
    if curves:
        cur = pd.read_parquet(feat / "curves_all.parquet")
        df = df.merge(cur, on="trace_key", how="left")
    if "error" in df:
        df = df[df["error"].isna()]
    return df.reset_index(drop=True)


def _log(x):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return np.log(x.where(x > 0))


def _col(df, *names):
    """First existing column among names (as float), else NaN."""
    for n in names:
        if n in df:
            return pd.to_numeric(df[n], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index)


def base_features(df, include_site=True):
    """Scalar predictors available when a record is processed (no analyst input)."""
    X = pd.DataFrame(index=df.index)
    X["ln_fchp_gmp"] = _log(df["fchp_gmp"])
    X["ln_fchp_gui"] = _log(df["fchp_gui"])
    X["gui_at_cap"] = df["gui_at_cap"].astype(float)
    # 1 if gmprocess's ridder_fchp step (Ramos-Sepulveda polynomial check) raised the SNR
    # corner (type "snr_polyfit"), 0 if the SNR corner was kept ("snr"); analysts kept the
    # raised value less often in the test events (median ~0.57 vs ~0.80)
    X["method_polyfit"] = (df.get("corner_method_gmp", pd.Series("", index=df.index))
                           .astype(str).str.contains("polyfit")).astype(float)
    X["ln_fclp_gmp"] = _log(df["fclp_gmp"])
    X["magnitude"] = _col(df, "ev_magnitude", "magnitude")
    X["depth_km"] = _col(df, "ev_hypocenter_depth", "event_depth_km")
    X["ln_rrup"] = _log(_col(df, "d_rrup", "rhypo_km"))
    X["ln_rjb"] = np.log(_col(df, "d_rjb", "repi_km").clip(lower=0) + 1.0)
    X["rx"] = _col(df, "d_rx")
    X["ry0"] = _col(df, "d_ry0")
    X["ln_sampling_rate"] = _log(df["sampling_rate"])
    X["is_vertical"] = df["is_vertical"].astype(float)
    X["accelerometer"] = (df["inst_code"] == "N").astype(float)
    X["broadband"] = (df["band_code"] == "B").astype(float)
    for thr in (2, 3, 5, 10):
        X[f"ln_snr{thr}_flo"] = _log(_col(df, f"snr{thr}_flo"))
        X[f"ln_snr{thr}_fhi"] = _log(_col(df, f"snr{thr}_fhi"))
    for f0 in ("0.1", "0.3", "1", "3", "10"):
        X[f"ln_snr_at_{f0}hz"] = _log(_col(df, f"snr_at_{f0}hz"))
    X["ln_snr_max"] = _log(_col(df, "snr_max_0p1_10"))
    X["ln_stress_drop"] = _log(_col(df, "fit_stress_drop"))
    X["ln_f0"] = _log(_col(df, "fit_f0"))
    X["fit_magnitude"] = _col(df, "fit_magnitude")
    X["clipping_prob"] = _col(df, "clipping_prob")
    X["zero_crossing_rate"] = _col(df, "zero_crossing_rate")
    X["preevent_noise_s"] = _col(df, "preevent_noise_s")
    X["ln_signal_duration"] = _log(_col(df, "signal_duration"))
    X["ln_pga_gmp"] = _log(_col(df, "pga_gmp_g"))
    X["ln_pga_ratio"] = _log(_col(df, "pga_gmp_g")) - _log(_col(df, "pga_unproc_g"))
    for k in ("poly_ratio_at_gmp", "poly_ratio_at_gui", "pre_ratio_at_gmp", "pre_ratio_at_gui"):
        X[k] = _col(df, k)
    for k in ("psa_rough_short", "psa_max_bump_short", "psa001_over_pga"):
        X[k] = _col(df, k)
    if include_site:
        X["ln_vs30"] = _log(_col(df, "st_vs30"))
        X["ln_z1p0"] = _log(_col(df, "st_z1p0_preferred"))
    return X


def curve_features(df, prefixes=("snr_", "fas_signal_", "fas_noise_", "poly_ratio_",
                                 "pre_disp_ratio_", "psa_gmp_")):
    """Fixed-grid curves (log10 of positive spectra/SNR), for the ML model."""
    cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)
            and c.rsplit("_", 1)[-1].isdigit()]
    C = df[cols].astype(float)
    pos = [c for c in cols if not c.startswith(("poly_ratio_", "pre_disp_ratio_"))]
    C[pos] = np.log10(C[pos].where(C[pos] > 0))
    return C


def fchp_dataset(df):
    """Reviewed+accepted traces with an analyst high-pass corner."""
    m = (df["y_reviewed"] == 1) & (df["y_accepted"] == 1) & (df["y_fchp"] > 0) \
        & (df["fchp_gmp"] > 0) & (df["fchp_gui"] > 0)
    d = df[m].copy()
    d["y"] = np.log(d["y_fchp"])
    # analyst adopted the GUI value at its cap: their preference may be higher
    d["censored"] = (d["gui_at_cap"].astype(bool)
                     & np.isclose(d["y_fchp"], GUI_CAP, rtol=1e-4)).astype(int)
    return d


def fclp_dataset(df):
    """Reviewed+accepted traces with gmprocess and analyst low-pass corners."""
    m = (df["y_reviewed"] == 1) & (df["y_accepted"] == 1) & (df["fclp_gmp"] > 0)
    d = df[m].copy()
    none = d["y_fclp_none"].astype(bool)
    d["lowered"] = (~none & (d["y_fclp"] < d["fclp_gmp"] * (1 - 1e-3))).astype(int)
    d["y"] = np.log(np.where(none, d["fclp_gmp"], d["y_fclp"]))
    return d


def accept_dataset(df):
    """Traces the analysts reviewed (all passed the automatic checks)."""
    d = df[df["y_reviewed"] == 1].copy()
    d["y"] = d["y_accepted"].astype(int)
    return d


# ---------------------------------------------------------------------- validation


def event_folds(groups, n_splits=5, seed=0):
    """Fold index per row with whole events (groups) per fold, balanced by row count."""
    groups = np.asarray(groups)
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]  # big events first
    load = np.zeros(n_splits)
    fold_of_group = np.empty(len(uniq), dtype=int)
    for g in order:
        k = int(np.argmin(load))
        fold_of_group[g] = k
        load[k] += counts[g]
    return fold_of_group[inv]


def regression_metrics(y, p):
    e = np.asarray(p) - np.asarray(y)
    ok = np.isfinite(e)
    e = e[ok]
    return {
        "n": int(ok.sum()),
        "rmse_ln": float(np.sqrt(np.mean(e ** 2))),
        "bias_ln": float(np.mean(e)),
        "mae_ln": float(np.mean(np.abs(e))),
        "within_20pct": float(np.mean(np.abs(e) <= np.log(1.2))),
        "within_factor_1.5": float(np.mean(np.abs(e) <= np.log(1.5))),
    }


def classification_metrics(y, p, thresholds=(0.5,)):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = {"n": int(len(y)), "positive_rate": float(y.mean()),
           "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))}
    out["auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        out[f"t{t:g}"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                          "accuracy": (tp + tn) / max(len(y), 1)}
    return out


def bootstrap_rmse_difference(y, pa, pb, groups, n_boot=1000, seed=0):
    """RMSE(a) - RMSE(b) (ln units) and its 95% interval, resampling whole events."""
    y, pa, pb, groups = map(np.asarray, (y, pa, pb, groups))
    ok = np.isfinite(pa) & np.isfinite(pb) & np.isfinite(y)
    y, pa, pb, groups = y[ok], pa[ok], pb[ok], groups[ok]
    uniq, inv = np.unique(groups, return_inverse=True)
    sa = np.bincount(inv, (pa - y) ** 2, len(uniq))
    sb = np.bincount(inv, (pb - y) ** 2, len(uniq))
    n = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, len(uniq), len(uniq))
        diffs[i] = np.sqrt(sa[k].sum() / n[k].sum()) - np.sqrt(sb[k].sum() / n[k].sum())
    point = np.sqrt(sa.sum() / n.sum()) - np.sqrt(sb.sum() / n.sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(point), "ci95": [float(lo), float(hi)],
            "relative_reduction": float(point / np.sqrt(sa.sum() / n.sum()))}


# -------------------------------------------------------------- simple model


class LogLinearModel:
    """ln y = X b + e, e ~ N(0, s^2), with optional right censoring (Tobit) of rows
    flagged `censored` (the observed value is a lower bound). Missing predictors are
    filled with training medians; predictors are standardized internally, and the
    coefficients are reported on the original scale."""

    def __init__(self, features, censored=True):
        self.features = list(features)
        self.use_censoring = censored

    def _design(self, X):
        Z = X[self.features].astype(float).fillna(self.fill_)
        Z = (Z - self.mu_) / self.sd_
        return np.column_stack([np.ones(len(Z)), Z.to_numpy()])

    def fit(self, X, y, censored=None):
        from scipy import optimize, stats

        X = X[self.features].astype(float)
        self.fill_ = X.median()
        Xf = X.fillna(self.fill_)
        self.mu_ = Xf.mean()
        self.sd_ = Xf.std().replace(0, 1).fillna(1)
        A = self._design(X)
        y = np.asarray(y, float)
        b0, *_ = np.linalg.lstsq(A, y, rcond=None)
        s0 = np.std(y - A @ b0) or 0.1
        c = np.zeros(len(y), bool) if censored is None or not self.use_censoring \
            else np.asarray(censored, bool)
        if c.any():
            def nll(theta):
                b, ls = theta[:-1], theta[-1]
                s = np.exp(ls)
                r = (y - A @ b) / s
                ll = np.where(c, stats.norm.logsf(r), stats.norm.logpdf(r) - ls)
                return -ll.sum()

            res = optimize.minimize(nll, np.r_[b0, np.log(s0)], method="L-BFGS-B")
            b, s = res.x[:-1], float(np.exp(res.x[-1]))
        else:
            b, s = b0, float(np.sqrt(np.mean((y - A @ b0) ** 2)))
        self.coef_, self.sigma_ = b, s
        return self

    def predict(self, X):
        return self._design(X) @ self.coef_

    def coefficients(self):
        """Coefficients per unit of each original predictor (and the intercept)."""
        slopes = self.coef_[1:] / self.sd_.to_numpy()
        intercept = self.coef_[0] - np.sum(slopes * self.mu_.to_numpy())
        out = {"intercept": float(intercept)}
        out.update({f: float(v) for f, v in zip(self.features, slopes)})
        out["sigma_ln"] = self.sigma_
        return out


# ----------------------------------------------------------------------- report


def write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
