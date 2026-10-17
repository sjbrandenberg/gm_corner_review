"""Split the DesignSafe project's workspace files into Globus transfer batches.

Run on a Stampede3 *login* node (login nodes see /corral; compute nodes don't):

    P=/corral/projects/NHERI/projects/<project id>
    python tacc/globus_batches.py --root $P/data --root $P/data2 --root $P/data_new \
        --dest-root $SCRATCH/gm_stage --batch-gb 400 --out $SCRATCH/gmcr_run/globus

Each --root holds event directories (<comcatid>/) with gmprocess workspace .h5 files.
Only the .h5 files directly inside each event directory are transferred, one line per
file, so raw/, plots, and reports in the event folders stay behind. Events are packed,
whole and in sorted order, into batches of about --batch-gb.

Destination: <dest-root>/<batch>/<root label>__<comcatid>/<file>.h5, with the root label
the root's folder name (e.g. data2__nc73819746). The label keeps an event that appears
under several roots apart; such events are listed in duplicates.csv so you can decide
which copy to keep (features are extracted from every copy, with the label in the file
name; the event id itself is read from inside each file).

Outputs in --out:
    b001.txt ...    Globus CLI batch files, one quoted "SRC DST" pair per file
    manifest.csv    root, comcatid, batch, files, bytes
    duplicates.csv  comcatids found under more than one root (if any)
    commands.sh     the `globus transfer` command of each batch

--globus-root ROOT=PATH maps a root to its path in the DesignSafe Globus collection if it
differs from the /corral path (for the "TACC Corral3" collection it doesn't).
Destination paths must be absolute on the Stampede3 collection ($SCRATCH expanded).
"""

import argparse
import csv
import os
import shlex
from collections import defaultdict
from pathlib import Path


def event_files(root):
    """(comcatid, [(file name, bytes), ...]) for each event directory with .h5 files."""
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        files = sorted((f.name, f.stat().st_size) for f in os.scandir(entry.path)
                       if f.is_file() and f.name.endswith(".h5"))
        if files:
            yield entry.name, files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", action="append", required=True,
                   help="folder of event directories on /corral (repeat for several)")
    p.add_argument("--globus-root", action="append", default=[],
                   help="ROOT=PATH if the Globus collection shows ROOT under another path")
    p.add_argument("--dest-root", required=True)
    p.add_argument("--batch-gb", type=float, default=400.0)
    p.add_argument("--out", required=True)
    p.add_argument("--src-collection", default="$DESIGNSAFE_COLLECTION")
    p.add_argument("--dst-collection", default="$STAMPEDE3_COLLECTION")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dest_root = os.path.abspath(os.path.expandvars(a.dest_root))
    gmap = dict(g.split("=", 1) for g in a.globus_root)
    roots = [r.rstrip("/") for r in a.root]
    labels = [os.path.basename(r) for r in roots]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"root folder names must differ (got {labels})")

    events = []  # (label, root, comcatid, files)
    seen = defaultdict(list)
    for root, label in zip(roots, labels):
        n0 = len(events)
        for name, files in event_files(root):
            events.append((label, root, name, files))
            seen[name].append(label)
        print(f"{root}: {len(events) - n0} event directories")

    limit = a.batch_gb * 1e9
    batches, current, size = [], [], 0
    for ev in events:
        nbytes = sum(b for _, b in ev[3])
        if current and size + nbytes > limit:
            batches.append(current)
            current, size = [], 0
        current.append(ev)
        size += nbytes
    if current:
        batches.append(current)

    cmds = ["#!/bin/bash", "# one Globus transfer per batch; tacc/run_batch.sh runs these for you", ""]
    with open(out / "manifest.csv", "w", newline="") as fm:
        w = csv.writer(fm)
        w.writerow(["root", "comcatid", "batch", "files", "bytes"])
        for k, batch in enumerate(batches, 1):
            tag = f"b{k:03d}"
            lines = []
            for label, root, name, files in batch:
                w.writerow([label, name, tag, len(files), sum(b for _, b in files)])
                src_root = gmap.get(root, root)
                for fname, _ in files:
                    src = f"{src_root}/{name}/{fname}"
                    dst = f"{dest_root}/{tag}/{label}__{name}/{fname}"
                    lines.append(f"{shlex.quote(src)} {shlex.quote(dst)}")
            (out / f"{tag}.txt").write_text("\n".join(lines) + "\n")
            cmds.append(f"globus transfer {a.src_collection} {a.dst_collection} "
                        f"--batch {out / (tag + '.txt')} --label gmcr-{tag} --verify-checksum")
    (out / "commands.sh").write_text("\n".join(cmds) + "\n")

    dups = {k: v for k, v in seen.items() if len(v) > 1}
    if dups:
        with open(out / "duplicates.csv", "w", newline="") as fd:
            w = csv.writer(fd)
            w.writerow(["comcatid", "roots"])
            for k, v in sorted(dups.items()):
                w.writerow([k, ";".join(v)])

    total = sum(b for *_, files in events for _, b in files)
    n_files = sum(len(files) for *_, files in events)
    print(f"{len(events)} event directories, {n_files} .h5 files, {total / 1e12:.2f} TB, "
          f"{len(batches)} batches of <= {a.batch_gb:g} GB")
    for k, batch in enumerate(batches, 1):
        gb = sum(b for *_, files in batch for _, b in files) / 1e9
        print(f"  b{k:03d}: {len(batch):6d} events {gb:8.1f} GB")
    if dups:
        print(f"WARNING: {len(dups)} comcatids appear under more than one root: see {out / 'duplicates.csv'}")
    print(f"batch files, manifest, and commands in {out}")


if __name__ == "__main__":
    main()
