"""Task list for reprocessing one prepared batch (gmcr.reprocess run), and a report of all batches.

    python tacc/make_reprocess_tasks.py RUN_DIR BATCH [--keep] [--no-delete-source]
    python tacc/make_reprocess_tasks.py RUN_DIR --report

The batch's event folders are RUN_DIR/events/<batch>/<event id>/ (written by `gmcr.reprocess prepare`).
One task per event, largest first (by reviewed instruments), so long events don't straggle; events
whose features already exist are left out. Each task also lists every trace of the event's 1.3 files
with its outcome into RUN_DIR/records13 (for scripts/record_status.py; --no-records skips it), so
events with nothing to reprocess still get a task. Writes RUN_DIR/tasks_<batch>.txt.

--report summarizes every event's reprocess.json: status, traces, time per stage, raw traces rounded
or left out, and the errors (RUN_DIR/reprocess_report.csv).
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def tasks(run, batch, keep, delete_source, records=True):
    events = run / "events" / batch
    done = {f.stem for f in (run / "features" / "traces").glob("*.parquet")} \
        if (run / "features" / "traces").exists() else set()
    todo, no_sis = [], []
    for d in sorted(p for p in events.iterdir() if p.is_dir()):
        src = d / "sources.json"
        if not src.exists():
            continue
        info = json.loads(src.read_text())
        if f"{d.name}__workspace_{batch}" in done:
            continue
        if not info.get("sources"):
            status = d / "reprocess.json"
            if not records or (status.exists() and
                               json.loads(status.read_text()).get("status") == "nothing to run"):
                continue
        elif not (d / "sis" / "manifest.json").exists():
            no_sis.append(d.name)  # prepare hasn't fetched its SIS metadata yet (or it failed)
            continue
        todo.append((len(info["sources"]), d))
    todo.sort(key=lambda t: -t[0])
    flags = (f" --records {run / 'records13'}" * records + " --keep" * keep
             + " --delete-source" * delete_source)
    lines = [f"python -m gmcr.reprocess run {d} --features {run / 'features'}{flags}" for _, d in todo]
    out = run / f"tasks_{batch}.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"{batch}: {len(todo)} events to run ({sum(n for n, _ in todo)} instruments), "
          f"{len(done)} files already extracted -> {out}")
    if no_sis:
        print(f"  left out until prepare has their SIS metadata (rerun it): {len(no_sis)} events, "
              f"e.g. {', '.join(no_sis[:5])}")


def report(run):
    rows = []
    for f in sorted((run / "events").glob("*/*/reprocess.json")):
        st = json.loads(f.read_text())
        exp = st.get("export") or {}
        rows.append({"batch": f.parent.parent.name, "event_id": st.get("event_id"),
                     "status": st.get("status"), "n_traces": st.get("n_traces"), "n_ok": st.get("n_ok"),
                     "rounded": exp.get("rounded"), "physical_units": exp.get("physical_units"),
                     **{k: st.get(k) for k in ("export_seconds", "assemble_seconds", "process_seconds",
                                               "extract_seconds", "seconds")},
                     "error": (st.get("error") or "").strip().splitlines()[-1:] or None})
    if not rows:
        print("no reprocess.json yet")
        return
    df = pd.DataFrame(rows)
    df.to_csv(run / "reprocess_report.csv", index=False)
    print(df.groupby("status").agg(events=("event_id", "count"), traces=("n_ok", "sum")).to_string())
    print(f"raw traces rounded to counts: {int(df['rounded'].fillna(0).sum())}; left out (physical "
          f"units): {int(df['physical_units'].fillna(0).sum())}; core-hours: "
          f"{df['seconds'].fillna(0).sum() / 3600:.1f} -> {run / 'reprocess_report.csv'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("batch", nargs="?")
    p.add_argument("--keep", action="store_true", help="keep the MiniSEED files and 2.9 workspaces")
    p.add_argument("--no-delete-source", action="store_true",
                   help="keep the staged 1.3 files after their events finish")
    p.add_argument("--no-records", action="store_true",
                   help="don't list the 1.3 records with their outcomes (records13/)")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    run = Path(a.run_dir).resolve()
    if a.report:
        report(run)
    elif a.batch:
        tasks(run, a.batch, a.keep, not a.no_delete_source, not a.no_records)
    else:
        p.error("give a batch, or --report")


if __name__ == "__main__":
    main()
