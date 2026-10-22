"""Calibrate the auto_fchp algorithm (ucla_geotech_tools.auto_fchp) on the analysts' high-pass corners.

    python scripts/calibrate_auto_fchp.py FEATURES_DIR --out models/auto_fchp
        [--variants-dir VARIANTS_FEATURES_DIR] [--compare-oof models/fchp/oof_predictions.parquet]
        [--jobs N]

Needs only the feature tables (traces_joined.parquet or traces_all.parquet, and curves_all.parquet), no
waveforms: about 6 minutes on one core with the existing curves, and about 6 minutes per curve family
per core with the --fchp-variants curves (20 families).

auto_fchp picks the lowest corner at which the polynomial-fit ratio of the filtered displacement,
max|fit| / max|disp|, falls to a target (objective 1), then raises it, if needed, until the ratio of the
pre-event displacement amplitude to the record's falls to a second target (objective 2), between
fchp_min and fchp_max. The review GUI used targets 0.02 and 0.05 and fchp_max = 1.2 Hz with a 6th-order
polynomial over the whole record (gmcr.gui_replica).

Both criteria are stored as curves at the 25 trial corners of features.FCHP_GRID (0.02 to 3 Hz):
poly_ratio_* and pre_disp_ratio_* in curves_all.parquet. auto_fchp is evaluated from the curves by
linear interpolation in ln(f), with the same logic as auto_fchp.solve_objective (lowest corner at which
an objective is met, fchp_max if it is not met there), so the calibration needs no waveforms. Corners
below the grid (0.02 Hz) are returned as 0.02 Hz. Extracting with `gmcr.extract --fchp-variants` adds
curves for other polynomial orders and fit windows tied to the event (gmcr.fchp_variants), which are
searched as well: pass the folder with --variants-dir if they were extracted separately.

Models (out-of-fold predictions, event-grouped CV as in train_corner_model.py; in each fold the
parameters are chosen to minimize the RMSE of ln(fchp) on the other folds):
  baseline_gmprocess  ln fchp_gmp
  baseline_gui        ln fchp_gui (the GUI's automatic value)
  gui_grid            auto_fchp with the GUI's settings, evaluated from the curves (checks the grid)
  auto                auto_fchp with calibrated targets, fchp_max, and curve family (order and window)
  auto_recal          a + b ln(auto), with the auto parameters chosen jointly with a and b
  gmp_max             max(fchp_gmp, auto): auto_fchp can only raise the gmprocess corner, the way
                      analysts adjusted it

Outputs in --out: report.json, report.md, oof_predictions.parquet (same layout as
train_corner_model.py, so scripts/residuals.py-style comparisons apply), and calibration_grid.parquet
(the full-data RMSE of every parameter combination for each model).
Positive residuals mean the analyst chose a higher corner than predicted.
"""

import argparse
import itertools
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmcr import features as ft  # noqa: E402
from gmcr import modeling as gm  # noqa: E402

GUI = {"family": "poly_ratio", "t_poly": 0.02, "t_pre": 0.05, "cap": gm.GUI_CAP}
MODELS = ("auto", "auto_recal", "gmp_max")
VARIANT_CURVE = re.compile(r"(polyvar_p\d+_[a-z0-9]+)_\d{3}$")  # gmcr.fchp_variants curves


# ----------------------------------------------------------------------- auto_fchp on a grid


def _interp_at(R, lng, f):
    """Values of curves R (rows) on the ln-frequency grid lng at frequencies f (one per row)."""
    lf = np.clip(np.log(f), lng[0], lng[-1])
    j = np.clip(np.searchsorted(lng, lf, side="right") - 1, 0, len(lng) - 2)
    w = (lf - lng[j]) / (lng[j + 1] - lng[j])
    rows = np.arange(len(R))
    return R[rows, j] * (1.0 - w) + R[rows, j + 1] * w


def solve_grid(R, lng, target, low, high):
    """Lowest frequency in [low, high] at which the curve falls to target, per row, as
    ucla_geotech_tools.auto_fchp.solve_objective does: low if the target is met there, high if it is
    not met at high, otherwise the first crossing above low (linear interpolation in ln f)."""
    n, m = R.shape
    Rl, Rh = _interp_at(R, lng, low), _interp_at(R, lng, high)
    F = np.empty((n, m + 2))
    V = np.empty((n, m + 2))
    F[:, 0], F[:, -1], F[:, 1:-1] = np.log(low), np.log(high), lng
    V[:, 0], V[:, -1], V[:, 1:-1] = Rl, Rh, R
    inside = (lng > np.log(low)[:, None]) & (lng < np.log(high)[:, None])
    V[:, 1:-1] = np.where(inside, V[:, 1:-1], np.nan)
    with np.errstate(invalid="ignore"):
        below = V <= target
    k = np.argmax(below, axis=1)
    idx = np.where(np.isfinite(V), np.arange(m + 2), -1)
    prev = np.maximum.accumulate(idx, axis=1)[np.arange(n), np.maximum(k - 1, 0)]
    prev = np.maximum(prev, 0)
    rows = np.arange(n)
    f0, f1, v0, v1 = F[rows, prev], F[rows, k], V[rows, prev], V[rows, k]
    with np.errstate(invalid="ignore", divide="ignore"):
        lnf = np.where(v1 != v0, f0 + (target - v0) * (f1 - f0) / (v1 - v0), f1)
    out = np.exp(np.clip(lnf, np.log(low), np.log(high)))
    with np.errstate(invalid="ignore"):
        out = np.where(Rl <= target, low, np.where(Rh >= target, high, out))
    return np.where(np.isfinite(Rl) & np.isfinite(Rh), out, np.nan)


def auto_on_grid(P, Q, lng, t_poly, t_pre, cap):
    """auto_fchp from the polynomial-ratio curves P and pre-event ratio curves Q (objective 2 is skipped
    when t_pre is None or the record has no pre-event curve)."""
    n = len(P)
    low = np.full(n, np.exp(lng[0]))
    high = np.full(n, float(cap))
    f1 = solve_grid(P, lng, t_poly, low, high)
    if t_pre is None:
        return f1
    has_pre = np.isfinite(Q).all(axis=1)
    f2 = solve_grid(np.where(has_pre[:, None], Q, 0.0), lng, t_pre, np.where(np.isfinite(f1), f1, low), high)
    return np.where(has_pre, f2, f1)


# ------------------------------------------------------------------------------ calibration


def _event_split(r, events):
    d = pd.DataFrame({"r": r, "e": events}).dropna()
    em = d.groupby("e")["r"].transform("mean")
    return {"tau_between_event": float(d.groupby("e")["r"].mean().std(ddof=1)),
            "phi_within_event": float((d["r"] - em).std(ddof=1))}


def _summary(y, p, events):
    r = np.asarray(y) - np.asarray(p)
    out = gm.regression_metrics(y, p)
    ok = np.isfinite(r)
    out.update({"residual_mean": float(np.mean(r[ok])), "residual_sd": float(np.std(r[ok], ddof=1))})
    out.update(_event_split(r, events))
    return out


def _recal_fit(S):
    """Least squares a, b of y = a + b x and the SSE, from sums S = (n, Sx, Sy, Sxx, Sxy, Syy)."""
    n, Sx, Sy, Sxx, Sxy, Syy = S
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (Sxy - Sx * Sy / n) / (Sxx - Sx**2 / n)
        a = (Sy - b * Sx) / n
        sse = Syy - 2 * a * Sy - 2 * b * Sxy + n * a**2 + 2 * a * b * Sx + b**2 * Sxx
    return a, b, sse


def _calibrate_family(fam, P, Q, y, ln_gmp, folds, grids, lng):
    """Statistics of every parameter combination of one curve family: per fold SSE (auto, gmp_max) and
    the sums for the recalibration."""
    nf = folds.max() + 1
    rows = []
    for t_poly, t_pre, cap in itertools.product(*grids):
        lnp = np.log(auto_on_grid(P, Q, lng, t_poly, t_pre, cap))
        rec = {"family": fam, "t_poly": t_poly, "t_pre": t_pre, "cap": cap}
        for name, pred in (("auto", lnp), ("gmp_max", np.maximum(ln_gmp, lnp))):
            rec[name] = np.bincount(folds, (y - pred) ** 2, nf)
        rec["recal"] = np.vstack([np.bincount(folds, w, nf) for w in
                                  (np.ones_like(y), lnp, y, lnp**2, lnp * y, y**2)])
        rows.append(rec)
    return rows


def calibrate(df, families, grid_t_poly, grid_t_pre, grid_cap, folds, lng, jobs=1):
    """Statistics of every parameter combination of every family, with families searched in parallel
    processes when jobs > 1."""
    y = df["y"].to_numpy()
    ln_gmp = np.log(df["fchp_gmp"].to_numpy(float))
    Q = df[[f"pre_disp_ratio_{i:03d}" for i in range(len(lng))]].to_numpy(float)
    grids = (grid_t_poly, grid_t_pre, grid_cap)

    def curves(fam):
        return df[[f"{fam}_{i:03d}" for i in range(len(lng))]].to_numpy(float)

    if jobs <= 1:
        return [r for fam in families for r in _calibrate_family(fam, curves(fam), Q, y, ln_gmp, folds, grids, lng)]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_calibrate_family, fam, curves(fam), Q, y, ln_gmp, folds, grids, lng)
                   for fam in families]
        return [r for f in futures for r in f.result()]


def select(rows, model, fold=None):
    """Parameter combination minimizing the SSE on all folds except `fold` (all folds if None)."""
    best, best_sse, best_ab = None, np.inf, None
    for rec in rows:
        if model == "auto_recal":
            S = rec["recal"].sum(axis=1) - (rec["recal"][:, fold] if fold is not None else 0)
            a, b, sse = _recal_fit(S)
        else:
            sse = rec[model].sum() - (rec[model][fold] if fold is not None else 0)
            a = b = None
        if np.isfinite(sse) and sse < best_sse:
            best, best_sse, best_ab = rec, sse, (a, b)
    params = {k: best[k] for k in ("family", "t_poly", "t_pre", "cap")}
    if model == "auto_recal":
        params.update(a=float(best_ab[0]), b=float(best_ab[1]))
    return params, best_sse


def predict(df, params, model, lng):
    Q = df[[f"pre_disp_ratio_{i:03d}" for i in range(len(lng))]].to_numpy(float)
    P = df[[f"{params['family']}_{i:03d}" for i in range(len(lng))]].to_numpy(float)
    lnp = np.log(auto_on_grid(P, Q, lng, params["t_poly"], params["t_pre"], params["cap"]))
    if model == "gmp_max":
        return np.maximum(np.log(df["fchp_gmp"].to_numpy(float)), lnp)
    if model == "auto_recal":
        return params["a"] + params["b"] * lnp
    return lnp


# -------------------------------------------------------------------------------------- main


def _fmt(v):
    return "off" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--variants-dir", default=None,
                   help="features folder whose curves_all.parquet has polyvar_* curves (gmcr.extract "
                        "--fchp-variants), if not in FEATURES_DIR")
    p.add_argument("--compare-oof", default=None,
                   help="oof_predictions.parquet of train_corner_model.py --target fchp, to compare with "
                        "the trained models on the same traces")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-poly", default="0.003:0.3:25", help="min:max:n, log-spaced polynomial-ratio targets")
    p.add_argument("--t-pre", default="0.01:0.5:12",
                   help="min:max:n, log-spaced pre-event ratio targets (objective 2 off is always tried)")
    p.add_argument("--cap", default="0.5,0.8,1.2,2,3", help="fchp_max values (Hz), up to 3")
    p.add_argument("--families", default="",
                   help="comma-separated curve families to search (default: poly_ratio and all polyvar_*)")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=1, help="processes for searching curve families in parallel")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def logspace(spec):
        lo, hi, n = spec.split(":")
        return np.logspace(np.log10(float(lo)), np.log10(float(hi)), int(n)).round(6).tolist()

    grid_t_poly = logspace(args.t_poly)
    grid_t_pre = [None] + logspace(args.t_pre)
    grid_cap = [float(c) for c in args.cap.split(",")]
    lng = np.log(ft.FCHP_GRID)
    if max(grid_cap) > ft.FCHP_GRID[-1] + 1e-9:
        sys.exit(f"--cap: the curves end at {ft.FCHP_GRID[-1]:g} Hz")

    df = gm.load_traces(args.features_dir, curves=True)
    if args.variants_dir:
        var = pd.read_parquet(Path(args.variants_dir) / "curves_all.parquet")
        var = var[["trace_key"] + [c for c in var.columns if VARIANT_CURVE.match(c)]]
        df = df.drop(columns=[c for c in df.columns if VARIANT_CURVE.match(c)]).merge(var, on="trace_key", how="left")
    df = gm.fchp_dataset(df).reset_index(drop=True)
    available = sorted({m.group(1) for m in map(VARIANT_CURVE.match, df.columns) if m})
    families = [f for f in args.families.split(",") if f] or ["poly_ratio"] + available
    need = [f"{fam}_{i:03d}" for fam in families + ["pre_disp_ratio"] for i in range(len(lng))]
    missing = sorted({c.rsplit("_", 1)[0] for c in need if c not in df})
    if missing:
        sys.exit(f"missing curves: {missing}")
    # Score every model on the same traces: those with complete polynomial-ratio curves
    poly_cols = [c for c in need if not c.startswith("pre_disp_ratio")]
    complete = np.isfinite(df[poly_cols].to_numpy(float)).all(axis=1)
    n_all = len(df)
    df = df[complete].reset_index(drop=True)
    events = df["event_id"].to_numpy()
    if df["event_id"].nunique() < args.folds:
        sys.exit(f"{df['event_id'].nunique()} events: need at least --folds ({args.folds})")
    folds = gm.event_folds(events, args.folds, args.seed)
    y = df["y"].to_numpy()
    print(f"{len(df)} traces ({n_all - len(df)} without complete curves dropped), {df['event_id'].nunique()} "
          f"events; families {families}; {len(families) * len(grid_t_poly) * len(grid_t_pre) * len(grid_cap)} "
          f"parameter combinations", flush=True)

    rows = calibrate(df, families, grid_t_poly, grid_t_pre, grid_cap, folds, lng, args.jobs)
    print(f"searched in {time.time() - t0:.0f} s", flush=True)

    oof = pd.DataFrame({"trace_key": df["trace_key"], "event_id": events, "y": y,
                        "censored": df["censored"].to_numpy(),
                        "baseline_gmprocess": np.log(df["fchp_gmp"].to_numpy(float)),
                        "baseline_gui": np.log(df["fchp_gui"].to_numpy(float)),
                        "gui_grid": predict(df, GUI, "auto", lng)})
    report = {"n_traces": int(len(df)), "n_dropped_incomplete_curves": int(n_all - len(df)),
              "n_events": int(df["event_id"].nunique()), "folds": args.folds, "families": families,
              "grid": {"t_poly": grid_t_poly, "t_pre": grid_t_pre, "cap": grid_cap},
              "fraction_analyst_below_grid": float(np.mean(np.exp(y) < ft.FCHP_GRID[0])),
              "selected": {}, "metrics": {}, "comparisons": {}}
    for model in MODELS:
        oof[model] = np.nan
        per_fold = []
        for k in range(args.folds):
            params, _ = select(rows, model, fold=k)
            te = folds == k
            oof.loc[te, model] = predict(df[te], params, model, lng)
            per_fold.append(params)
        full, sse = select(rows, model)
        report["selected"][model] = {"all_data": full, "all_data_rmse_ln": float(np.sqrt(sse / len(df))),
                                     "per_fold": per_fold}
    # Fraction of auto corners at the bottom of the grid (true value may be lower)
    p_auto = predict(df, report["selected"]["auto"]["all_data"], "auto", lng)
    report["fraction_auto_at_grid_floor"] = float(np.mean(np.isclose(p_auto, lng[0])))
    gui_ok = (df["fchp_gui"] > ft.FCHP_GRID[0] * 1.001).to_numpy()
    report["gui_grid_vs_gui"] = {"median_abs_ln_ratio": float(np.median(np.abs(oof["gui_grid"] - oof["baseline_gui"])[gui_ok])),
                                 "p95_abs_ln_ratio": float(np.quantile(np.abs(oof["gui_grid"] - oof["baseline_gui"])[gui_ok], 0.95))}

    columns = ["baseline_gmprocess", "baseline_gui", "gui_grid", *MODELS]
    if args.compare_oof:
        ref = pd.read_parquet(args.compare_oof)
        trained = [c for c in ("simple", "lightgbm", "lightgbm_2stage_soft") if c in ref]
        oof = oof.merge(ref[["trace_key"] + trained].rename(columns={c: f"trained_{c}" for c in trained}),
                        on="trace_key", how="left")
        columns += [f"trained_{c}" for c in trained]
    for m in columns:
        report["metrics"][m] = _summary(oof["y"], oof[m], oof["event_id"])
    for a, b in [("auto", "baseline_gmprocess"), ("gmp_max", "baseline_gmprocess"), ("auto_recal", "baseline_gmprocess")] + \
            [(m, f"trained_{c}") for m in MODELS for c in ("lightgbm", "simple") if f"trained_{c}" in oof]:
        report["comparisons"][f"{a}_minus_{b}"] = gm.bootstrap_rmse_difference(
            oof["y"], oof[a], oof[b], oof["event_id"], n_boot=args.boot)

    oof.to_parquet(out / "oof_predictions.parquet", index=False)
    pd.DataFrame([{**{k: r[k] for k in ("family", "t_poly", "t_pre", "cap")},
                   "rmse_auto": float(np.sqrt(r["auto"].sum() / len(df))),
                   "rmse_gmp_max": float(np.sqrt(r["gmp_max"].sum() / len(df))),
                   "rmse_auto_recal": float(np.sqrt(_recal_fit(r["recal"].sum(axis=1))[2] / len(df)))}
                  for r in rows]).to_parquet(out / "calibration_grid.parquet", index=False)
    report["seconds"] = round(time.time() - t0, 1)
    gm.write_json(report, out / "report.json")

    lines = [f"# auto_fchp calibration ({len(df)} traces, {report['n_events']} events, {args.folds}-fold "
             f"event-grouped CV)", "",
             "Residual = ln(analyst / predicted). RMSE, tau, and phi in ln units.", "",
             "| model | RMSE | mean resid | sd resid | tau | phi | within 1.5x |", "|---|---|---|---|---|---|---|"]
    for m in columns:
        r = report["metrics"][m]
        lines.append(f"| {m} | {r['rmse_ln']:.3f} | {r['residual_mean']:+.3f} | {r['residual_sd']:.3f} | "
                     f"{r['tau_between_event']:.3f} | {r['phi_within_event']:.3f} | {r['within_factor_1.5']:.2f} |")
    lines += ["", "Selected parameters (all data):", ""]
    for m in MODELS:
        s = report["selected"][m]["all_data"]
        extra = f", a = {s['a']:.3f}, b = {s['b']:.3f}" if m == "auto_recal" else ""
        lines.append(f"* {m}: {s['family']}, target {_fmt(s['t_poly'])}, pre-event target {_fmt(s['t_pre'])}, "
                     f"fchp_max {_fmt(s['cap'])} Hz{extra}")
    lines += ["", "RMSE differences (95% event-bootstrap interval):", ""]
    for k, v in report["comparisons"].items():
        lines.append(f"* {k}: {v['difference']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    lines += ["", f"gui_grid vs the GUI's value: median |ln ratio| {report['gui_grid_vs_gui']['median_abs_ln_ratio']:.3f}, "
              f"95th percentile {report['gui_grid_vs_gui']['p95_abs_ln_ratio']:.3f}.",
              f"Analyst corners below the 0.02 Hz grid: {report['fraction_analyst_below_grid']:.1%}; "
              f"calibrated auto corners at the grid floor: {report['fraction_auto_at_grid_floor']:.1%}."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out} in {report['seconds']} s")


if __name__ == "__main__":
    main()
