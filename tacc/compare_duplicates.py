"""Compare the copies of events that exist in more than one location, before staging.

Run on a Stampede3 login node after tacc/globus_batches.py (it reads events.csv and
duplicates.csv from the same --out folder); it opens each copy's .h5 files on /corral,
reading only the small review records (a few minutes for ~1,500 copies):

    python tacc/compare_duplicates.py $SCRATCH/gmcr_run/globus

For every copy of a duplicated event: workspace files, processed traces, reviewed and
accepted traces, latest review time, review user names, gmprocess version(s), and size.
Each event is then classified by comparing the sets of reviewed traces of its copies:

    identical       every copy has the same reviewed traces (and the same decisions)
    nested          one copy's reviewed traces include all the others'
    unreviewed      only one copy has any reviews (the others are automatic runs)
    different       the copies have different reviewed traces (none contains the others)
    none_reviewed   no copy has reviews

Outputs (in the same folder):
    duplicates_compared.csv   one row per copy, with the class and whether it is kept
    skip.csv                  copies to leave out (location, comcatid): for identical,
                              nested, and unreviewed events, every copy except the one with
                              the most reviewed traces (ties: latest review, then largest);
                              "different" events keep all copies (duplicate traces are
                              reported after extraction, see summary.json)

Then rebuild the batches without the skipped copies:

    python tacc/globus_batches.py ... --skip $SCRATCH/gmcr_run/globus/skip.csv
"""

import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py


def _attrs(obj):
    out = {}
    for k, v in obj.attrs.items():
        out[k] = v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v)
    return out


def review_summary(path):
    """Reviewed traces {review key: accepted} and metadata of one workspace file."""
    info = {"reviewed": {}, "processed": 0, "last_review": "", "users": set(), "version": ""}
    with h5py.File(path, "r") as f:
        if "AuxiliaryData/gmprocess_version/version" in f:
            info["version"] = bytes(f["AuxiliaryData/gmprocess_version/version"][()].tobytes()).decode(
                errors="replace")
        for sta in f.get("Waveforms", {}).values():
            info["processed"] += sum(1 for k in sta if k != "StationXML" and not k.endswith("unprocessed"))
        rv = f.get("AuxiliaryData/review")
        if rv is None:
            return info
        for sta in rv.values():
            for key, g in sta.items():
                if "accepted" not in g:
                    continue
                a = _attrs(g["accepted"])
                if a.get("reviewed", 0) != 1:
                    continue
                # key = <NET.STA.LOC.CHA>_<eventid>_<label>: compare by trace id only
                trace = key.split("_")[0]
                info["reviewed"][trace] = int(a.get("accepted", 0))
                info["last_review"] = max(info["last_review"], str(a.get("timestamp", "")))
                if a.get("username"):
                    info["users"].add(str(a["username"]))
    return info


def main():
    out = Path(sys.argv[1])
    events = list(csv.DictReader(open(out / "events.csv", newline="")))
    dup_ids = {r["comcatid"] for r in csv.DictReader(open(out / "duplicates.csv", newline=""))}
    copies = defaultdict(list)
    for e in events:
        if e["comcatid"] in dup_ids:
            copies[e["comcatid"]].append(e)

    rows, skip, classes = [], [], Counter()
    for k, (cid, cps) in enumerate(sorted(copies.items()), 1):
        summ = []
        for c in cps:
            agg = {"reviewed": {}, "processed": 0, "last_review": "", "users": set(),
                   "versions": set(), "files": 0, "error": ""}
            for fn in sorted(os.listdir(c["path"])):
                if not fn.endswith(".h5"):
                    continue
                agg["files"] += 1
                try:
                    s = review_summary(os.path.join(c["path"], fn))
                except Exception as exc:  # noqa: BLE001
                    agg["error"] = f"{fn}: {exc!r}"[:200]
                    continue
                agg["reviewed"].update(s["reviewed"])
                agg["processed"] += s["processed"]
                agg["last_review"] = max(agg["last_review"], s["last_review"])
                agg["users"] |= s["users"]
                if s["version"]:
                    agg["versions"].add(s["version"])
            summ.append((c, agg))

        sets = [set(a["reviewed"].items()) for _, a in summ]
        n_rev = [len(s) for s in sets]
        if max(n_rev) == 0:
            cls = "none_reviewed"
        elif sum(n > 0 for n in n_rev) == 1:
            cls = "unreviewed"
        elif all(s == sets[0] for s in sets):
            cls = "identical"
        elif any(all(o <= s for o in sets) for s in sets):
            cls = "nested"
        else:
            cls = "different"
        classes[cls] += 1
        best = max(range(len(summ)), key=lambda i: (n_rev[i], summ[i][1]["last_review"],
                                                     int(summ[i][0]["bytes"])))
        for i, (c, a) in enumerate(summ):
            keep = cls in ("different",) or i == best
            if not keep:
                skip.append((c["location"], cid))
            rows.append({
                "comcatid": cid, "location": c["location"], "class": cls, "keep": int(keep),
                "files": a["files"], "processed_traces": a["processed"],
                "reviewed_traces": len(a["reviewed"]),
                "accepted_traces": sum(a["reviewed"].values()),
                "last_review": a["last_review"], "users": ";".join(sorted(a["users"])),
                "gmprocess_versions": ";".join(sorted(a["versions"])), "bytes": c["bytes"],
                "error": a["error"], "path": c["path"],
            })
        if k % 50 == 0:
            print(f"  {k}/{len(copies)} events compared", flush=True)

    with open(out / "duplicates_compared.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "skip.csv", "w", newline="") as fs:
        w = csv.writer(fs)
        w.writerow(["location", "comcatid"])
        w.writerows(skip)

    skipped_tb = sum(int(r["bytes"]) for r in rows if not r["keep"]) / 1e12
    print(f"{len(copies)} duplicated events, {len(rows)} copies")
    for cls, n in classes.most_common():
        print(f"  {cls:14s} {n}")
    by_loc = Counter((r["location"], r["keep"]) for r in rows)
    for loc in sorted({r["location"] for r in rows}):
        print(f"  {loc:20s} kept {by_loc[(loc, 1)]:5d}  skipped {by_loc[(loc, 0)]:5d}")
    errors = [r for r in rows if r["error"]]
    if errors:
        print(f"  {len(errors)} copies had unreadable files (see the error column)")
    print(f"skip.csv: {len(skip)} copies, {skipped_tb:.2f} TB not to transfer")
    print(f"details: {out / 'duplicates_compared.csv'}")


if __name__ == "__main__":
    main()
