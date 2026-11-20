"""Prepare a rerun of some reprocessed events (gmcr.reprocess): clear their outputs and write a Globus
batch file that stages their 1.3 workspace files again (the first run deleted them).

    python tacc/redo_events.py RUN_DIR --name NAME (--events ID [ID ...] | --unmatched)
                               [--globus-dir $SCRATCH/gmcr_run/globus] [--dry-run]

--unmatched picks the events with reviewed instruments that were prepared but are missing from the
reprocessed features (RUN_DIR/train/unmatched_reviews.csv from scripts/transfer_labels.py), e.g. the
non-free-field stations gmprocess's own assemble dropped before gmcr kept them.

For each event (found in RUN_DIR/events/<batch>/<event>): deletes its features
(features/{traces,curves}/<event>__workspace_<batch>.parquet, features/logs/...json) and
reprocess.json, so make_reprocess_tasks.py lists it again; its prepared folder (event.json, StationXML,
STREC, SIS snapshot) is kept. Writes GLOBUS_DIR/NAME.txt: the lines of the batch files (bNNN.txt) whose
destination is one of the event's files (sources.json). The record list (records13) is not touched.
Then:

    bash tacc/stage_batch.sh NAME $SCRATCH/gmcr_run/globus
    for b in <batches printed>; do python tacc/make_reprocess_tasks.py RUN_DIR $b; done
    cat RUN_DIR/tasks_<batch>.txt ... > RUN_DIR/tasks_NAME.txt     # one job for all of them
    GMCR_PACK_SRC=$WORK/gmp29_env.tar.gz bash tacc/submit.sh reprocess RUN_DIR=... TASKS=RUN_DIR/tasks_NAME.txt -- ...
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path


def instrument(trace_id):
    net, sta, loc, cha = trace_id.split(".")
    return f"{net}.{sta}.{loc}.{cha[:2]}"


def unmatched_events(run):
    import pandas as pd

    u = pd.read_csv(run / "train" / "unmatched_reviews.csv", dtype=str, keep_default_na=False)
    out = set()
    for ev, tid in zip(u["event_id"], u["trace_id"]):
        d = glob.glob(str(run / "events" / "b*" / ev / "sources.json"))
        if d and instrument(tid) in json.loads(Path(d[0]).read_text()).get("sources", {}):
            out.add(ev)
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--name", required=True, help="name of the Globus batch file to write (e.g. redo_nonfree)")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--events", nargs="+")
    which.add_argument("--unmatched", action="store_true")
    p.add_argument("--globus-dir", default=os.path.expandvars("$SCRATCH/gmcr_run/globus"))
    p.add_argument("--dry-run", action="store_true", help="list what would be done, change nothing")
    a = p.parse_args()
    run, globus = Path(a.run_dir).resolve(), Path(a.globus_dir)

    events = unmatched_events(run) if a.unmatched else sorted(set(a.events))
    lines_by_dest = {}
    for bf in sorted(globus.glob("b[0-9][0-9][0-9].txt")):
        for ln in bf.read_text().splitlines():
            parts = re.findall(r'"[^"]*"|\S+', ln)
            if len(parts) >= 2:
                lines_by_dest[parts[-1].strip('"')] = ln
    keep, batches, missing = [], {}, []
    for ev in events:
        dirs = sorted(glob.glob(str(run / "events" / "b*" / ev)))
        if not dirs:
            missing.append(ev)
            continue
        d = Path(dirs[0])
        batch = d.parent.name
        batches[batch] = batches.get(batch, 0) + 1
        for f in json.loads((d / "sources.json").read_text())["files"]:
            if f in lines_by_dest:
                keep.append(lines_by_dest[f])
            else:
                missing.append(f"{ev}: no batch-file line for {f}")
        stem = f"{ev}__workspace_{batch}"
        doomed = [run / "features" / sub / f"{stem}.parquet" for sub in ("traces", "curves")]
        doomed += [run / "features" / "logs" / f"{stem}.json", d / "reprocess.json"]
        for f in doomed:
            if f.exists() and not a.dry_run:
                f.unlink()
    out = globus / f"{a.name}.txt"
    if not a.dry_run:
        out.write_text("\n".join(keep) + ("\n" if keep else ""))
    print(f"{len(events)} events ({', '.join(f'{b}: {n}' for b, n in sorted(batches.items()))}), "
          f"{len(keep)} files to stage" + (" (dry run: nothing changed)" if a.dry_run else f" -> {out}"))
    for m in missing:
        print("  not found:", m)
    if not a.dry_run and keep:
        print(f"next: bash tacc/stage_batch.sh {a.name} {globus}; then make_reprocess_tasks.py for "
              f"{' '.join(sorted(batches))} and submit (see this script's docstring)")


if __name__ == "__main__":
    main()
