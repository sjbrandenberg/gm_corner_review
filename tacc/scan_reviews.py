"""Choose which workspace files to transfer, from their review records.

Run on a Stampede3 login node after tacc/globus_batches.py (reads its events.csv):

    python tacc/scan_reviews.py $SCRATCH/gmcr_run/globus [--workers 8]

Event folders hold many kinds of .h5 files (workspace.h5, <id>.h5, per-analyst copies
like workspace-Shako.h5, Copy1/Copy2, Frontera runs, p_arrival_shift experiments,
-old/-corrupt/-bad files), and many events were never reviewed. Names don't say which
files matter, so this opens every .h5 file and reads only its review records: for each
reviewed trace, the accept/reject decision and the analyst's corner frequencies.

Per event (all locations together):
    unreadable   file can't be opened or read                       -> not transferred
    unreviewed   no reviewed traces (automatic gmprocess output)    -> not transferred
    redundant    its reviewed traces, decisions, and corners are all
                 contained in another file of the event              -> not transferred
    selected     everything else                                     -> transferred
Files with different reviews of the same traces (e.g. two analysts) are all selected;
duplicate traces are reported after extraction (summary.json, duplicate_traces.csv).

Outputs (same folder):
    files_reviews.csv   one row per file: status, reviewed/accepted traces, last review,
                        users, gmprocess version, bytes, redundant_with, error
    select.csv          location, comcatid, file of the selected files

Then build the batches from the selection:

    python tacc/globus_batches.py ... --select $SCRATCH/gmcr_run/globus/select.csv
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np


def _attrs(obj):
    out = {}
    for k, v in obj.attrs.items():
        out[k] = v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v)
    return out


def _corner(g, which):
    key = f"corner_frequencies/{which}"
    if key not in g:
        return None
    return round(float(np.asarray(g[key][()]).ravel()[0]), 4)


def scan_file(path):
    """Review signature and metadata of one workspace file."""
    out = {"path": path, "bytes": os.path.getsize(path), "reviews": {}, "accepted": 0,
           "last_review": "", "users": "", "version": "", "processed": 0, "error": ""}
    users = set()
    try:
        with h5py.File(path, "r") as f:
            if "AuxiliaryData/gmprocess_version/version" in f:
                out["version"] = bytes(
                    f["AuxiliaryData/gmprocess_version/version"][()].tobytes()).decode(errors="replace")
            for sta in f.get("Waveforms", {}).values():
                out["processed"] += sum(1 for k in sta if k != "StationXML" and not k.endswith("unprocessed"))
            rv = f.get("AuxiliaryData/review")
            if rv is not None:
                for sta in rv.values():
                    for key, g in sta.items():
                        if "accepted" not in g:
                            continue
                        a = _attrs(g["accepted"])
                        if a.get("reviewed", 0) != 1:
                            continue
                        acc = int(a.get("accepted", 0))
                        out["reviews"][key.split("_")[0]] = (acc, _corner(g, "highpass"), _corner(g, "lowpass"))
                        out["accepted"] += acc
                        out["last_review"] = max(out["last_review"], str(a.get("timestamp", "")))
                        if a.get("username"):
                            users.add(str(a["username"]))
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)[:200]
    out["users"] = ";".join(sorted(users))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="the --out folder of globus_batches.py")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    out = Path(a.out)
    if not (out / "events.csv").exists():
        sys.exit(f"no {out / 'events.csv'}: run tacc/globus_batches.py (current version) first")
    events = list(csv.DictReader(open(out / "events.csv", newline="")))
    jobs = []
    for e in events:
        for fn in sorted(os.listdir(e["path"])):
            if fn.endswith(".h5"):
                jobs.append((e["location"], e["comcatid"], fn, os.path.join(e["path"], fn)))
    print(f"{len(events)} event folders, {len(jobs)} .h5 files to scan with {a.workers} workers", flush=True)

    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(scan_file, [j[3] for j in jobs], chunksize=2), 1):
            results[r["path"]] = r
            if k % 100 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} files ({time.time() - t0:.0f} s)", flush=True)

    by_event = defaultdict(list)
    for loc, cid, fn, path in jobs:
        by_event[cid].append((loc, fn, path))
    status, redundant_with = {}, {}
    for cid, files in by_event.items():
        cands = []
        for loc, fn, path in files:
            r = results[path]
            if r["error"]:
                status[path] = "unreadable"
            elif not r["reviews"]:
                status[path] = "unreviewed"
            else:
                cands.append((loc, fn, path, set(r["reviews"].items())))
        # most reviews first; then latest review; then largest file
        # most reviews first; then latest review; then, among equals, the standard file name
        # (workspace.h5, <id>.h5) over copies/old/check files, and the top-level location
        def rank(c):
            loc, fn, path, sig = c
            low = fn.lower()
            name = 2 if fn in ("workspace.h5", f"{cid}.h5") else \
                0 if any(w in low for w in ("copy", "old", "bad", "check", "corrupt", "unreviewed")) else 1
            return (len(sig), results[path]["last_review"], name, -loc.count("-"), results[path]["bytes"])

        cands.sort(key=rank, reverse=True)
        kept = []
        for loc, fn, path, sig in cands:
            sup = next((k for k in kept if sig <= k[3]), None)
            if sup is not None:
                status[path] = "redundant"
                redundant_with[path] = f"{sup[0]}/{cid}/{sup[1]}"
            else:
                status[path] = "selected"
                kept.append((loc, fn, path, sig))

    rows = []
    for loc, cid, fn, path in jobs:
        r = results[path]
        rows.append({"location": loc, "comcatid": cid, "file": fn, "status": status[path],
                     "reviewed_traces": len(r["reviews"]), "accepted_traces": r["accepted"],
                     "processed_traces": r["processed"], "last_review": r["last_review"],
                     "users": r["users"], "gmprocess_version": r["version"], "bytes": r["bytes"],
                     "redundant_with": redundant_with.get(path, ""), "error": r["error"]})
    with open(out / "files_reviews.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "select.csv", "w", newline="") as fs:
        w = csv.writer(fs)
        w.writerow(["location", "comcatid", "file"])
        w.writerows((r["location"], r["comcatid"], r["file"]) for r in rows if r["status"] == "selected")

    st = Counter(r["status"] for r in rows)
    tb = defaultdict(int)
    for r in rows:
        tb[r["status"]] += int(r["bytes"])
    sel = [r for r in rows if r["status"] == "selected"]
    print(f"\n{len(rows)} files in {len(by_event)} events")
    for s in ("selected", "redundant", "unreviewed", "unreadable"):
        print(f"  {s:11s} {st[s]:6d} files  {tb[s] / 1e12:6.2f} TB")
    print(f"selected: {len({r['comcatid'] for r in sel})} events, "
          f"{sum(r['reviewed_traces'] for r in sel)} reviewed traces "
          f"({sum(r['accepted_traces'] for r in sel)} accepted)")
    multi = Counter(r["comcatid"] for r in sel)
    print(f"events with more than one selected file: {sum(1 for v in multi.values() if v > 1)}")
    names = Counter(r["file"].replace(r["comcatid"], "<id>") for r in sel)
    print("selected file names:", ", ".join(f"{n} ({c})" for n, c in names.most_common(12)))
    print(f"details: {out / 'files_reviews.csv'}   selection: {out / 'select.csv'}")


if __name__ == "__main__":
    main()
