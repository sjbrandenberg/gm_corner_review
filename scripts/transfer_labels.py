"""Training tables for a new model generation: the analysts' decisions on the gmprocess-1.3 workspaces
joined to the features of the same traces reprocessed with the generation's gmprocess (gmcr.reprocess).

    python scripts/transfer_labels.py --labels FEATURES_13 --features FEATURES_29 --out OUT_DIR
                                      [--kept {absolute,relative}] [--keep-auto-failed]

FEATURES_13 and FEATURES_29 are merged extraction folders (scripts/merge_outputs.py): traces_all.parquet
and curves_all.parquet. Rows are matched on (event_id, trace_id); a trace reviewed in more than one 1.3
file keeps one row per review, as in the 1.3 training tables. OUT_DIR gets traces_all.parquet,
curves_all.parquet, and grids.json in the layout of an extraction folder, so the usual steps follow:
copy the 1.3 run's db/ folder (or run fetch_flatfile.py), then tacc/train.slurm with FEATURES_DIR=OUT_DIR.
The rows keep the new generation's `generation` column, so the training reports and exported models
are of that generation.

Columns: every feature comes from the reprocessed trace; the labels (y_*, review_*, has_review,
colocated, selected) come from the 1.3 review; the 1.3 automatic values are kept for comparison as
v13_fchp_gmp, v13_fclp_gmp, v13_fchp_gui, v13_auto_failed, v13_trace_key. trace_key is the 1.3 key
(unique per review); trace_key_new is the reprocessed trace's.

Labels are the analysts' corners in Hz, as they chose them ("absolute"). Where the analyst kept
gmprocess 1.3's corner, they might also have kept the new gmprocess corner, which differs (2.9's fclp
is ~30% higher); --kept relative replaces such labels with the new automatic corner instead (fchp:
y_fchp == v13_fchp_gmp; fclp: y_fclp == v13_fclp_gmp), to compare the two readings. An unchecked
low-pass box in the GUI (y_fclp_none: no low-pass needed) carries over as is: the training scripts
read it as "not lowered" from whatever the automatic corner is.

Traces the new gmprocess rejects automatically are dropped (the analysts never saw records 1.3
rejected, and new records rejected automatically never reach the models); --keep-auto-failed keeps
them. summary.json counts the matches, what was dropped and why, and the shifts of the automatic
corners between the versions.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmcr import tables  # noqa: E402

LABELS = ["y_reviewed", "y_accepted", "y_fchp", "y_fclp", "y_fclp_raw", "y_fclp_none", "y_fchp_source",
          "has_review", "review_username", "review_timestamp", "colocated", "selected"]
V13 = {"fchp_gmp": "v13_fchp_gmp", "fclp_gmp": "v13_fclp_gmp", "fchp_gui": "v13_fchp_gui",
       "auto_failed": "v13_auto_failed", "trace_key": "v13_trace_key",
       "gmprocess_version": "v13_gmprocess_version"}


def _norm_id(s):
    # gmprocess 2.x writes an empty location code as "--" in some places
    return s.astype(str).str.replace(".--.", "..", regex=False)


def _close(a, b):
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    return (a - b).abs() <= 1e-5 + 1e-4 * b.abs()


def _ln_ratio(a, b):
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    r = np.log(a.where(a > 0) / b.where(b > 0)).dropna()
    if r.empty:
        return None
    return {"n": int(len(r)), "median": round(float(r.median()), 3),
            "p05": round(float(r.quantile(0.05)), 3), "p95": round(float(r.quantile(0.95)), 3)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="merged 1.3 extraction folder (reviewed traces)")
    p.add_argument("--features", required=True, help="merged extraction folder of the reprocessed traces")
    p.add_argument("--out", required=True)
    p.add_argument("--kept", choices=["absolute", "relative"], default="absolute",
                   help="labels where the analyst kept gmprocess 1.3's corner (see above)")
    p.add_argument("--keep-auto-failed", action="store_true",
                   help="keep traces the new gmprocess rejects automatically")
    a = p.parse_args()
    lab_dir, feat_dir, out = Path(a.labels), Path(a.features), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    old = pd.read_parquet(lab_dir / "traces_all.parquet")
    new = pd.read_parquet(feat_dir / "traces_all.parquet")
    if "error" in new:
        new = new[new["error"].isna()]
    gens = sorted(new.get("generation", pd.Series(dtype=str)).dropna().unique())
    if len(gens) != 1 or gens[0] == "gmprocess-1.3":
        sys.exit(f"--features must hold one new generation's extraction (found {gens or 'none'})")
    old = old[old["y_reviewed"] == 1].copy()
    for df in (old, new):
        df["_tid"] = _norm_id(df["trace_id"])
    if new.duplicated(["event_id", "_tid"]).any():
        n = int(new.duplicated(["event_id", "_tid"]).sum())
        print(f"{n} reprocessed traces appear more than once (event in several batches); keeping the first")
        new = new.drop_duplicates(["event_id", "_tid"])

    lab = old[[c for c in LABELS if c in old] + [c for c in V13 if c in old] + ["event_id", "_tid"]]
    lab = lab.rename(columns=V13)
    feats = new.drop(columns=[c for c in LABELS if c in new]).rename(columns={"trace_key": "trace_key_new"})
    m = lab.merge(feats, on=["event_id", "_tid"], how="inner")
    summary = {"generation": gens[0], "reviewed_1p3": len(old),
               "reviewed_events_1p3": int(old["event_id"].nunique()),
               "reprocessed_traces": len(new), "matched": len(m)}
    unmatched = old.merge(new[["event_id", "_tid"]], on=["event_id", "_tid"], how="left", indicator=True)
    unmatched = unmatched[unmatched["_merge"] == "left_only"]
    summary["unmatched_reviews"] = len(unmatched)
    summary["unmatched_events_not_reprocessed"] = int(
        (~unmatched["event_id"].isin(new["event_id"])).sum())
    unmatched[["event_id", "trace_id", "trace_key", "y_accepted"]].to_csv(
        out / "unmatched_reviews.csv", index=False)

    failed = m["auto_failed"].astype("boolean").fillna(False).astype(bool)
    summary["auto_failed_new"] = {
        "traces": int(failed.sum()),
        "of_which_accepted_by_analyst": int((failed & (m["y_accepted"] == 1)).sum()),
        "reasons": m.loc[failed, "auto_fail_module"].value_counts().head(10).to_dict()}
    if not a.keep_auto_failed:
        m = m[~failed]

    acc = m["y_accepted"] == 1
    kept_hp = acc & _close(m["y_fchp"], m["v13_fchp_gmp"])
    kept_lp = acc & ~m["y_fclp_none"].astype("boolean").fillna(False).astype(bool) \
        & _close(m["y_fclp"], m["v13_fclp_gmp"])
    summary["analyst_kept_1p3_corner"] = {"fchp": int(kept_hp.sum()), "fclp": int(kept_lp.sum()),
                                          "accepted": int(acc.sum())}
    if a.kept == "relative":
        m.loc[kept_hp, "y_fchp"] = m.loc[kept_hp, "fchp_gmp"]
        m.loc[kept_lp, "y_fclp"] = m.loc[kept_lp, "fclp_gmp"]
        m.loc[kept_hp, "y_fchp_source"] = "gmprocess"
    summary["kept"] = a.kept
    summary["shift_new_vs_1p3_ln"] = {
        "fchp_gmp": _ln_ratio(m["fchp_gmp"], m["v13_fchp_gmp"]),
        "fclp_gmp": _ln_ratio(m["fclp_gmp"], m["v13_fclp_gmp"]),
        "fchp_gui": _ln_ratio(m["fchp_gui"], m["v13_fchp_gui"])}
    summary["training_rows"] = len(m)
    summary["training_events"] = int(m["event_id"].nunique())
    summary["accepted"] = int(acc.loc[m.index].sum())

    m["trace_key"] = m["v13_trace_key"]
    m = m.drop(columns=["_tid"])
    curves = pd.read_parquet(feat_dir / "curves_all.parquet").rename(columns={"trace_key": "trace_key_new"})
    cur = m[["trace_key", "trace_key_new"]].merge(curves, on="trace_key_new", how="left") \
        .drop(columns=["trace_key_new"])
    tables.parquet_safe(m).to_parquet(out / "traces_all.parquet", index=False)
    cur.to_parquet(out / "curves_all.parquet", index=False)
    shutil.copy2(feat_dir / "grids.json", out / "grids.json")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
