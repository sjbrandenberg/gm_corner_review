"""Merge per-file extraction outputs into single tables and print a QA summary.

    python scripts/merge_outputs.py FEATURES_DIR

Writes FEATURES_DIR/traces_all.parquet, curves_all.parquet, grids.json,
errors.csv and summary.json.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmcr import features as ft  # noqa: E402
from gmcr import fchp_variants as fv  # noqa: E402


def main(feat_dir):
    feat = Path(feat_dir)
    tr_files = sorted((feat / "traces").glob("*.parquet"))
    if not tr_files:
        sys.exit(f"no outputs in {feat / 'traces'}")
    traces = pd.concat((pd.read_parquet(f) for f in tr_files), ignore_index=True)
    # A column can be typed differently in different files (e.g. bool in one, text or
    # empty in another); make every non-numeric column text so parquet accepts it.
    for c in traces.columns:
        if traces[c].dtype == object:
            traces[c] = traces[c].map(lambda v: None if v is None or (isinstance(v, float) and v != v)
                                      else str(v))
    curves = pd.concat((pd.read_parquet(feat / "curves" / f.name) for f in tr_files
                        if (feat / "curves" / f.name).exists()), ignore_index=True)
    traces.to_parquet(feat / "traces_all.parquet", index=False)
    curves.to_parquet(feat / "curves_all.parquet", index=False)
    (feat / "grids.json").write_text(json.dumps({
        "snr": ft.FREQ_GRID.tolist(), "fas_signal": ft.FREQ_GRID.tolist(),
        "fas_noise": ft.FREQ_GRID.tolist(), "poly_ratio": ft.FCHP_GRID.tolist(),
        "pre_disp_ratio": ft.FCHP_GRID.tolist(), "psa_gmp": ft.NGAW2_T.tolist(),
        **{name: ft.FCHP_GRID.tolist() for name in fv.NAMES
           if f"{name}_000" in curves.columns},
    }, indent=1))

    logs = [json.loads(p.read_text()) for p in (feat / "logs").glob("*.json")]
    err_rows = [{"file": lg["file"], "trace_id": t, "error": e.strip().splitlines()[-1]}
                for lg in logs for t, e in lg.get("errors", {}).items()]
    err_rows += [{"file": lg["file"], "trace_id": None, "error": lg["file_error"].strip().splitlines()[-1]}
                 for lg in logs if "file_error" in lg]
    pd.DataFrame(err_rows, columns=["file", "trace_id", "error"]).to_csv(feat / "errors.csv", index=False)

    rev = traces[traces.get("y_reviewed", 0) == 1]
    summary = {
        "files": len(tr_files),
        "file_errors": sum("file_error" in lg for lg in logs),
        "traces": len(traces),
        "trace_errors": int(traces["error"].notna().sum()) if "error" in traces else 0,
        "reviewed": len(rev),
        "reviewed_accepted": int((rev["y_accepted"] == 1).sum()),
        "reviewed_rejected": int((rev["y_accepted"] == 0).sum()),
        "fchp_source": rev["y_fchp_source"].value_counts().to_dict() if "y_fchp_source" in rev else {},
        "gui_at_cap_frac": float(rev["gui_at_cap"].astype(float).mean()) if "gui_at_cap" in rev else None,
        "fclp_none_frac": float(rev["y_fclp_none"].astype(float).mean()) if "y_fclp_none" in rev else None,
        "gmprocess_versions": pd.Series([lg.get("gmprocess_version") for lg in logs]).value_counts().to_dict(),
        "review_usernames": rev["review_username"].value_counts().head(20).to_dict() if "review_username" in rev else {},
        "extract_seconds_total": round(sum(lg.get("seconds", 0) for lg in logs), 1),
    }
    # the same trace from more than one workspace file (events are split across files, and
    # some events exist in several folders): report them so one copy can be chosen
    if {"event_id", "trace_id", "source_file"} <= set(traces.columns):
        stem = traces["trace_key"].str.split("|").str[-1]
        dup = traces.assign(stem=stem).groupby(["event_id", "trace_id"])["stem"].transform("nunique") > 1
        d = traces.loc[dup, ["event_id", "trace_id", "trace_key", "y_reviewed", "y_accepted", "y_fchp"]
                       if "y_fchp" in traces else ["event_id", "trace_id", "trace_key"]]
        d.sort_values(["event_id", "trace_id"]).to_csv(feat / "duplicate_traces.csv", index=False)
        summary["duplicate_traces"] = {
            "traces_in_more_than_one_file": int(dup.sum()),
            "distinct_traces_affected": int(d[["event_id", "trace_id"]].drop_duplicates().shape[0]),
            "events_affected": int(d["event_id"].nunique()),
        }
    staged = [lg for lg in logs if "stage_bytes" in lg]
    if staged:
        gb = sum(lg["stage_bytes"] for lg in staged) / 1e9
        sec = sum(lg["stage_seconds"] for lg in staged)
        summary["staging"] = {"files": len(staged), "gb_copied": round(gb, 2),
                              "seconds": round(sec, 1),
                              "mb_per_s_per_task": round(1e3 * gb / max(sec, 1e-9), 1),
                              "share_of_task_time": round(sec / max(summary["extract_seconds_total"], 1e-9), 3)}
    (feat / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
