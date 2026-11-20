"""How much does the gmprocess version alone change a model's predictions? Apply the same models to the
same reviewed traces twice, with their features from two extractions (e.g. the gmprocess-1.3
workspaces the analysts reviewed, and the same events reprocessed with gmprocess 2.9), and score both
against the analysts' decisions.

    python scripts/compare_generations.py --old FEATURES_13 --new FEATURES_29 --out OUT_DIR
                                          [--models gmprocess-1.3] [--flag-share 80]

FEATURES_13: the merged 1.3 extraction (traces_joined.parquet, from join_metadata.py, or
traces_all.parquet). FEATURES_29: the training folder scripts/transfer_labels.py wrote (its rows keep
the 1.3 trace_key, so they pair with the 1.3 rows); run join_metadata.py on it first if the 1.3 side is
joined, so both sides have the same distance and site predictors. --models: a bundled model generation
(default gmprocess-1.3: accept, fchp, fclp) or folders, each applied with its recommended model
exactly as scripts/predict.py applies it.

Only traces in both tables are scored (pairs), and for each model only the pairs that are in its
training dataset on both sides (gmcr.modeling accept_dataset, fchp_dataset, fclp_dataset), so the two
sides are scored on the same records. Labels are the analysts' on both sides; where an fclp label is
"no low-pass" it equals each side's own gmprocess corner, as in training.

The models were trained on the 1.3 side's labels, so its scores are in-sample and better than the
models' cross-validated ones; what matters is the difference between the sides, which is the cost of
the gmprocess change for models that aren't retrained. Also reported: gmprocess's own corners as the
prediction (each side's), and, for the accept model, the share of rejections flagged and of accepted
records lost at the threshold that flagged --flag-share percent of rejections in cross-validation.
Everything is broken down by sensor (accelerometer, broadband, other) and magnitude.

Writes OUT_DIR/summary.md, metrics.csv (one row per model, side, and group), and predictions.parquet.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from gmcr import modeling as gm  # noqa: E402
from gmcr import portable  # noqa: E402
import predict as pr  # noqa: E402

MAG_BINS = [(-np.inf, 4.0, "M<4"), (4.0, 5.0, "M4-5"), (5.0, 6.0, "M5-6"), (6.0, np.inf, "M>=6")]
DATASETS = {"accept": gm.accept_dataset, "fchp": gm.fchp_dataset, "fclp": gm.fclp_dataset}


def sensor(df):
    inst = df["inst_code"].astype(str)
    band = df["band_code"].astype(str)
    out = pd.Series("other", index=df.index)
    out[inst == "N"] = "accelerometer"
    out[inst.isin(["H", "L"]) & band.isin(["H", "B"])] = "broadband"
    return out


def mag_bin(m):
    m = pd.to_numeric(m, errors="coerce")
    out = pd.Series("M unknown", index=m.index)
    for lo, hi, name in MAG_BINS:
        out[(m >= lo) & (m < hi)] = name
    return out


def load(path):
    joined = (Path(path) / "traces_joined.parquet").exists()
    df = gm.load_traces(path, curves=False)
    return df.drop_duplicates("trace_key").set_index("trace_key", drop=False), joined


def models_from(arg):
    if Path(arg[0]).is_dir():
        return [Path(a) for a in arg]
    gen = portable.bundled_dirs(arg[0])
    if not gen:
        sys.exit(f"no bundled models of generation {arg[0]}")
    return [gen[k] for k in ("accept", "fchp", "fclp") if k in gen]


def predict_one(kind, report, bundle, df):
    """(prediction, label) Series on df's index: ln corner and ln label, or p_accept and 0/1."""
    name = pr.default_model(kind, report)
    X, _ = pr.predictors(df, bundle, include_site=(kind != "accept"))
    if kind == "accept":
        return pd.Series(pr.predict_accept(bundle, X, name), index=df.index), name
    fn = pr.predict_fchp if kind == "fchp" else pr.predict_fclp
    ln, _ = fn(bundle["models"], X, name)
    return pd.Series(ln, index=df.index), name


def metrics(kind, y, p):
    if kind == "accept":
        m = gm.classification_metrics(y, p)
        return {"n": m["n"], "auc": m["auc"], "log_loss": m["log_loss"], "brier": m["brier"],
                "mean_p_accepted": float(np.mean(p[y == 1])) if (y == 1).any() else np.nan,
                "mean_p_rejected": float(np.mean(p[y == 0])) if (y == 0).any() else np.nan}
    m = gm.regression_metrics(y, p)
    return {k: m[k] for k in ("n", "rmse_ln", "bias_ln", "within_20pct")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="features of the reviewed workspaces (e.g. gmprocess 1.3)")
    ap.add_argument("--new", required=True, help="transfer_labels.py output for the reprocessed traces")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["gmprocess-1.3"],
                    help="a bundled generation, or model folders (default gmprocess-1.3)")
    ap.add_argument("--flag-share", type=int, choices=[50, 80, 90], default=80)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    old, old_joined = load(a.old)
    new, new_joined = load(a.new)
    if old_joined != new_joined:
        sys.exit("one side is joined with database metadata (traces_joined.parquet) and the other isn't: "
                 "run scripts/join_metadata.py on the other (copy its db/ folder first)")
    pairs = old.index.intersection(new.index)
    old, new = old.loc[pairs], new.loc[pairs]
    print(f"{len(pairs):,} traces in both tables ({old['event_id'].nunique()} events)", flush=True)
    groups = pd.DataFrame({"sensor": sensor(new), "magnitude": mag_bin(new["magnitude"])}, index=pairs)

    rows, preds, notes = [], [], []
    for mdir in models_from(a.models):
        kind, report, bundle = pr.model_kind(mdir)
        if kind not in DATASETS:
            continue
        d_old, d_new = DATASETS[kind](old.reset_index(drop=True)), DATASETS[kind](new.reset_index(drop=True))
        keys = pd.Index(d_old["trace_key"]).intersection(pd.Index(d_new["trace_key"]))
        d_old = d_old.set_index("trace_key", drop=False).loc[keys]
        d_new = d_new.set_index("trace_key", drop=False).loc[keys]
        sides = {}
        for side, d in (("old", d_old), ("new", d_new)):
            p, name = predict_one(kind, report, bundle, d)
            sides[side] = (d, p)
            preds.append(pd.DataFrame({"model": kind, "side": side, "trace_key": keys, "y": d["y"].to_numpy(),
                                       "pred": p.to_numpy()}))
        tag = f"{portable.model_generation(mdir)}/{kind} ({name})"
        for side, (d, p) in sides.items():
            y = d["y"].astype(float)
            for gname, gcol in [("all", None), ("sensor", "sensor"), ("magnitude", "magnitude")]:
                labels = ["all"] if gcol is None else sorted(groups.loc[keys, gcol].unique())
                for lab in labels:
                    sel = np.ones(len(keys), bool) if gcol is None else (groups.loc[keys, gcol] == lab).to_numpy()
                    if gcol is not None and sel.sum() < 20:  # small groups say little
                        continue
                    r = {"model": tag, "side": side, "group": gname, "value": lab,
                         **metrics(kind, y.to_numpy()[sel], p.to_numpy()[sel])}
                    if kind != "accept":  # gmprocess's own corner as the prediction
                        base = np.log(pd.to_numeric(d[f"{kind}_gmp"], errors="coerce")).to_numpy()[sel]
                        r["gmprocess_rmse_ln"] = gm.regression_metrics(y.to_numpy()[sel], base)["rmse_ln"]
                        r["gmprocess_bias_ln"] = gm.regression_metrics(y.to_numpy()[sel], base)["bias_ln"]
                    rows.append(r)
            if kind == "accept":
                op = report.get("operating_points", {}).get(name, {}).get(f"flag_{a.flag_share}pct_of_rejections")
                yy = y.to_numpy()
                if op and (yy == 0).any() and (yy == 1).any():
                    t = op["p_reject_threshold"]
                    flag = (1 - p.to_numpy()) >= t
                    notes.append(f"{tag}, {side}: P(reject) >= {t:.3f} flags "
                                 f"{100 * flag[yy == 0].mean():.1f}% of rejections and "
                                 f"{100 * flag[yy == 1].mean():.1f}% of accepted records "
                                 f"(cross-validation: {a.flag_share}% and "
                                 f"{100 * op['accepted_records_lost']:.1f}%)")
        print(f"{tag}: {len(keys):,} records scored on both sides", flush=True)

    met = pd.DataFrame(rows)
    met.to_csv(out / "metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)

    lines = ["# The same models on two gmprocess versions' features", "",
             f"Old: `{a.old}`; new: `{a.new}`; {len(pairs):,} paired traces from "
             f"{old['event_id'].nunique()} events. Models: {', '.join(map(str, a.models))}. The old side is "
             "in-sample for models trained on it; the difference between the sides is the point.", ""]
    for tag, g in met[met.group == "all"].groupby("model", sort=False):
        lines += [f"## {tag}", ""]
        cols = [c for c in ("n", "auc", "log_loss", "mean_p_accepted", "mean_p_rejected", "rmse_ln", "bias_ln",
                            "within_20pct", "gmprocess_rmse_ln", "gmprocess_bias_ln") if c in g and g[c].notna().any()]
        lines += ["| side | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for r in g.itertuples():
            lines.append(f"| {r.side} | " + " | ".join(
                f"{getattr(r, c):.4f}" if isinstance(getattr(r, c), float) else str(getattr(r, c)) for c in cols) + " |")
        key = "auc" if "auc" in g and g["auc"].notna().any() else "rmse_ln"
        sub = met[(met.model == tag) & (met.group != "all")].pivot_table(
            index=["group", "value"], columns="side", values=key)
        if not sub.empty:
            lines += ["", f"{key} by group:", "", "| group | value | old | new |", "|---|---|---|---|"]
            lines += [f"| {gi} | {vi} | {r.get('old', np.nan):.4f} | {r.get('new', np.nan):.4f} |"
                      for (gi, vi), r in sub.iterrows()]
        lines.append("")
    if notes:
        lines += ["## Flagging rejections", ""] + [f"- {n}" for n in notes] + [""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
