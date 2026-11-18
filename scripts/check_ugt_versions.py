"""Would ucla_geotech_tools 3.1.0 bias the models trained on 3.0.3 features?

The review GUI's automatic high-pass corner (fchp_gui, from ucla_geotech_tools.auto_fchp.get_fchp)
is a predictor of every model. The training features were extracted with ucla_geotech_tools 3.0.3,
whose get_fchp differs from 3.1.0's in one case (commit fad6738 of ucla_geotech_tools):

    objective 2 (pre-event displacement ratio <= 0.05) is already met at objective 1's corner f1,
    but not at fchp_max (1.2 Hz), because the ratio rises again at high corners.

3.0.3 then ran the root finder anyway and returned the corner f_old > f1 where the ratio crosses
0.05 on its way up; 3.1.0 returns f1. (The same can happen to objective 1 at fchp_min = 0.001 Hz;
that is counted, not corrected, since the curves start at 0.02 Hz.) The other changes (time axis,
polynomial fit in normalized time) move fchp_gui by less than the root finder's tolerance.

The workspaces are not needed. For each trace, the training tables hold fchp_gui and the two
criteria at fchp_gui (poly_ratio_at_gui, pre_ratio_at_gui), plus both criteria on a 25-point grid
of trial corners (curves_all.parquet). A trace took the 3.0.3-only branch if

    objective 2 moved the corner: pre_ratio_at_gui ~ 0.05 and poly_ratio_at_gui < 0.02, and
    the pre-event ratio at 1.2 Hz (interpolated on the grid) is above 0.05.

For those traces the 3.1.0 values are f1 (where the polynomial-fit ratio on the grid falls to
0.02), poly_ratio_at_gui = 0.02, and pre_ratio_at_gui = the pre-event ratio at f1. Traces whose
interpolated ratio at 1.2 Hz is within --margin of 0.05 are reported as ambiguous and left alone.

The models are then applied twice, to the features as extracted and with the 3.1.0 values, and
the change in their predictions is the bias new records would see.

    python scripts/check_ugt_versions.py FEATURES_DIR --out ugt_check [--models accept fchp fclp]

FEATURES_DIR holds traces_joined.parquet (or traces_all.parquet), curves_all.parquet and
grids.json, e.g. $SCRATCH/gmcr_run/features. Writes OUT/summary.md, OUT/traces.csv (one row per
trace: class, fchp_gui as extracted and under 3.1.0) and OUT/predictions_<model>.csv.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The review GUI's get_fchp settings (gmcr.gui_replica.GUI_FCHP_KW)
TARGET_POLY = 0.02
TARGET_PRE = 0.05
FCHP_MAX = 1.2


def log_interp(f, grid, values):
    """Values on the grid, interpolated in ln(f), at f (NaN outside the grid or where missing)."""
    out = np.full(len(f), np.nan)
    lg = np.log(grid)
    for i, (fi, row) in enumerate(zip(f, values)):
        ok = np.isfinite(row)
        if fi > 0 and np.isfinite(fi) and ok.sum() >= 2 and lg[ok][0] <= np.log(fi) <= lg[ok][-1]:
            out[i] = np.interp(np.log(fi), lg[ok], row[ok])
    return out


def first_crossing_below(grid, values, target):
    """Lowest grid corner at which values fall to target, interpolated in ln(f) and ln(value)
    between the bracketing points. -inf if already at or below target at the grid's first point,
    NaN if never."""
    out = np.full(len(values), np.nan)
    lg = np.log(grid)
    for i, row in enumerate(values):
        ok = np.isfinite(row) & (row > 0)
        if not ok.any():
            continue
        g, v = lg[ok], row[ok]
        below = np.nonzero(v <= target)[0]
        if not len(below):
            continue
        k = below[0]
        if k == 0:
            out[i] = -np.inf
            continue
        x0, x1, y0, y1 = g[k - 1], g[k], np.log(v[k - 1]), np.log(v[k])
        out[i] = np.exp(x0 + (np.log(target) - y0) * (x1 - x0) / (y1 - y0))
    return out


def classify(df, curves, grid, margin, pre_tol, poly_tol):
    poly = curves[[f"poly_ratio_{k:03d}" for k in range(len(grid))]].to_numpy(float)
    pre = curves[[f"pre_disp_ratio_{k:03d}" for k in range(len(grid))]].to_numpy(float)
    f_old = df["fchp_gui"].to_numpy(float)
    pre_at_cap = log_interp(np.full(len(df), FCHP_MAX), grid, pre)
    f1 = first_crossing_below(grid, poly, TARGET_POLY)
    pre_at_f1 = log_interp(np.where(f1 > 0, f1, np.nan), grid, pre)

    at_cap = df["gui_at_cap"].astype(bool).to_numpy()
    obj2_moved = ((np.abs(df["pre_ratio_at_gui"].to_numpy(float) - TARGET_PRE) <= pre_tol * TARGET_PRE)
                  & (df["poly_ratio_at_gui"].to_numpy(float) < TARGET_POLY * (1 - poly_tol)))
    obj1_at_target = np.abs(df["poly_ratio_at_gui"].to_numpy(float) - TARGET_POLY) <= poly_tol * TARGET_POLY
    poly_at_cap = log_interp(np.full(len(df), FCHP_MAX), grid, poly)

    cls = np.full(len(df), "objective 1", dtype=object)
    cls[obj2_moved] = "objective 2 raised it (both versions)"
    cls[~np.isfinite(f_old)] = "no fchp_gui"
    cls[at_cap] = "at the 1.2 Hz cap"
    rising = obj2_moved & ~at_cap & (pre_at_cap > TARGET_PRE)
    cls[rising & (pre_at_cap > TARGET_PRE * (1 + margin))] = "3.0.3 only: objective 2 rising root"
    cls[rising & (pre_at_cap <= TARGET_PRE * (1 + margin))] = "ambiguous (ratio at 1.2 Hz near 0.05)"
    # objective 1 met at fchp_min but not at 1.2 Hz: 3.0.3 root-found, 3.1.0 returns 0.001 Hz
    obj1_rising = (~at_cap & ~obj2_moved & obj1_at_target & (f1 == -np.inf)
                   & (poly_at_cap > TARGET_POLY))
    cls[obj1_rising] = "3.0.3 only: objective 1 rising root (not corrected)"

    out = pd.DataFrame({
        "trace_key": df["trace_key"], "class": cls, "fchp_gui_303": f_old,
        "f1_from_curves": np.where(f1 > 0, f1, np.nan), "pre_ratio_at_1p2hz": pre_at_cap,
        "pre_ratio_at_f1": pre_at_f1,
    })
    fix = (out["class"] == "3.0.3 only: objective 2 rising root") & (f1 > 0) & (pre_at_f1 <= TARGET_PRE)
    out["corrected"] = fix
    out["fchp_gui_310"] = np.where(fix, f1, f_old)
    return out


def corrected_traces(df, cls):
    new = df.copy()
    fix = cls["corrected"].to_numpy()
    new.loc[fix, "fchp_gui"] = cls.loc[fix, "fchp_gui_310"].to_numpy()
    new.loc[fix, "poly_ratio_at_gui"] = TARGET_POLY
    new.loc[fix, "pre_ratio_at_gui"] = cls.loc[fix, "pre_ratio_at_f1"].to_numpy()
    return new


def predict(model, feat_dir, out_csv):
    script = Path(__file__).resolve().parent / "predict.py"
    cmd = [sys.executable, str(script), model, "--features", str(feat_dir), "--out", str(out_csv)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"predict.py {model} failed:\n{r.stdout}\n{r.stderr}")
    return pd.read_csv(out_csv)


def describe(x):
    x = pd.Series(x).dropna()
    if not len(x):
        return "none"
    p = np.percentile(x, [5, 50, 95])
    return f"mean {x.mean():+.4f}, median {p[1]:+.4f}, 5-95% [{p[0]:+.4f}, {p[2]:+.4f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features", help="folder with traces_joined/traces_all, curves_all, grids.json")
    p.add_argument("--out", required=True)
    p.add_argument("--models", nargs="+", default=["accept", "fchp", "fclp"],
                   help="bundled model names or model folders (default: accept fchp fclp)")
    p.add_argument("--margin", type=float, default=0.10,
                   help="relative margin around 0.05 within which the interpolated pre-event ratio at "
                        "1.2 Hz is called ambiguous (default 0.10)")
    p.add_argument("--pre-tol", type=float, default=0.02,
                   help="relative closeness of pre_ratio_at_gui to 0.05 that means objective 2 set the "
                        "corner (default 0.02)")
    p.add_argument("--poly-tol", type=float, default=0.02,
                   help="relative closeness of poly_ratio_at_gui to 0.02 that means objective 1 set the "
                        "corner (default 0.02)")
    args = p.parse_args()

    feat, out = Path(args.features), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    traces_name = "traces_joined.parquet" if (feat / "traces_joined.parquet").exists() else "traces_all.parquet"
    df = pd.read_parquet(feat / traces_name)
    if "error" in df:
        df = df[df["error"].isna()]
    df = df.reset_index(drop=True)
    curves = pd.read_parquet(feat / "curves_all.parquet")
    curves = df[["trace_key"]].merge(curves, on="trace_key", how="left")
    grid = np.asarray(json.loads((feat / "grids.json").read_text())["poly_ratio"], float)

    cls = classify(df, curves, grid, args.margin, args.pre_tol, args.poly_tol)
    cls["reviewed"] = df.get("y_reviewed", pd.Series(np.nan, index=df.index)).fillna(0).astype(bool)
    cls["auto_failed"] = df.get("auto_failed", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    for c in ("y_fchp", "y_fchp_source", "y_accepted"):
        if c in df:
            cls[c] = df[c]
    cls.to_csv(out / "traces.csv", index=False)

    # Both feature sets, so predict.py builds predictors exactly as it does for new records
    dirs = {"303": out / "features_303", "310": out / "features_310"}
    for tag, d in dirs.items():
        d.mkdir(exist_ok=True)
        (df if tag == "303" else corrected_traces(df, cls)).to_parquet(d / traces_name, index=False)
        for name in ("curves_all.parquet", "grids.json"):
            if not (d / name).exists():
                shutil.copy(feat / name, d / name)

    lines = ["# fchp_gui: ucla_geotech_tools 3.0.3 (training) vs 3.1.0", "",
             f"Features: `{feat / traces_name}`, {len(df)} traces, "
             f"{df['trace_key'].str.split('|').str[0].nunique()} events.", "",
             "| class | traces | reviewed | reviewed & passed gmprocess |", "|---|---|---|---|"]
    for name, g in cls.groupby("class"):
        lines.append(f"| {name} | {len(g)} | {int(g['reviewed'].sum())} | "
                     f"{int((g['reviewed'] & ~g['auto_failed']).sum())} |")
    fixed = cls[cls["corrected"]]
    lines += ["", f"Corrected to their 3.1.0 value: {len(fixed)} traces "
              f"({100 * len(fixed) / max(len(df), 1):.2f}%; reviewed and passed: "
              f"{int((fixed['reviewed'] & ~fixed['auto_failed']).sum())}).",
              f"ln(fchp_gui 3.1.0 / 3.0.3) on them: {describe(np.log(fixed['fchp_gui_310'] / fixed['fchp_gui_303']))}."]
    if "y_fchp" in fixed and len(fixed):
        rev = fixed[fixed["reviewed"] & fixed["y_fchp"].notna()]
        if len(rev):
            lines += [f"Analysts' fchp on the reviewed ones: ln(y / fchp_gui 3.0.3) "
                      f"{describe(np.log(rev['y_fchp'] / rev['fchp_gui_303']))}; ln(y / fchp_gui 3.1.0) "
                      f"{describe(np.log(rev['y_fchp'] / rev['fchp_gui_310']))}."]
            if "y_fchp_source" in rev:
                lines.append(f"Their fchp source: {rev['y_fchp_source'].value_counts().to_dict()}.")

    lines += ["", "## Predictions with the 3.1.0 values minus with the 3.0.3 values", "",
              "| model | column | all traces | corrected traces |", "|---|---|---|---|"]
    keys = set(fixed["trace_key"])
    for model in args.models:
        tag = Path(model).name
        a = predict(model, dirs["303"], out / f"predictions_{tag}_303.csv").set_index("trace_key")
        b = predict(model, dirs["310"], out / f"predictions_{tag}_310.csv").set_index("trace_key")
        for col in ("p_accept", "pred_fchp_hz", "pred_fclp_hz"):
            if col not in a:
                continue
            d = (b[col] - a[col]) if col == "p_accept" else np.log(b[col] / a[col])
            label = col if col == "p_accept" else f"ln {col}"
            lines.append(f"| {tag} | {label} | {describe(d)} | {describe(d[d.index.isin(keys)])} |")
            pd.DataFrame({"303": a[col], "310": b[col], "diff": d}).to_csv(out / f"predictions_{tag}.csv")

    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
