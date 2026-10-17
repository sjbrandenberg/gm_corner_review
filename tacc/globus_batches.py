"""Split the DesignSafe project into Globus transfer batches of whole event folders.

Run on a Stampede3 *login* node (login nodes see /corral; compute nodes don't):

    python tacc/globus_batches.py \
        --corral-root /corral/projects/NHERI/projects/<project-uuid>/<folder with event dirs> \
        --globus-root /<same folder as shown in the DesignSafe Globus collection> \
        --dest-root   $SCRATCH/gm_stage \
        --batch-gb 400 --out $SCRATCH/gmcr_run/globus

For every event directory (<comcatid>/ containing .h5 files) it measures the .h5 size
and packs directories, in sorted order, into batches of about --batch-gb. It writes

    <out>/b001.txt ...   Globus CLI batch files: "--recursive SRC DST" per event folder
                         (paths quoted, so folders with spaces work),
                         SRC under --globus-root, DST = <dest-root>/b001/<comcatid>
    <out>/manifest.csv   comcatid, batch, files, bytes
    <out>/commands.sh    the `globus transfer` command for each batch

--globus-root is needed because the DesignSafe Globus collection shows the project
under its own path, not /corral/...; open the project in the Globus web app once to
read it off. Destination paths must be absolute paths on the Stampede3 collection
($SCRATCH or $WORK expanded, e.g. /scratch/01234/username/gm_stage).
"""

import argparse
import csv
import os
import shlex
from pathlib import Path


def event_dirs(root):
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        files, size = 0, 0
        for f in os.scandir(entry.path):
            if f.is_file() and f.name.endswith(".h5"):
                files += 1
                size += f.stat().st_size
        if files:
            yield entry.name, files, size


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corral-root", required=True)
    p.add_argument("--globus-root", required=True)
    p.add_argument("--dest-root", required=True)
    p.add_argument("--batch-gb", type=float, default=400.0)
    p.add_argument("--out", required=True)
    p.add_argument("--src-collection", default="$DESIGNSAFE_COLLECTION",
                   help="Globus collection id of DesignSafe (or leave the shell variable)")
    p.add_argument("--dst-collection", default="$STAMPEDE3_COLLECTION",
                   help="Globus collection id of Stampede3 (or leave the shell variable)")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dest_root = os.path.abspath(os.path.expandvars(a.dest_root))
    limit = a.batch_gb * 1e9
    batches, current, size = [], [], 0
    for name, files, nbytes in event_dirs(a.corral_root):
        if current and size + nbytes > limit:
            batches.append(current)
            current, size = [], 0
        current.append((name, files, nbytes))
        size += nbytes
    if current:
        batches.append(current)

    with open(out / "manifest.csv", "w", newline="") as fm:
        w = csv.writer(fm)
        w.writerow(["comcatid", "batch", "files", "bytes"])
        cmds = ["#!/bin/bash", "# one Globus transfer per batch; run them one or two at a time",
                f"# export DESIGNSAFE_COLLECTION=<id>  STAMPEDE3_COLLECTION=<id>", ""]
        for k, batch in enumerate(batches, 1):
            tag = f"b{k:03d}"
            lines = []
            for name, files, nbytes in batch:
                w.writerow([name, tag, files, nbytes])
                src = f"{a.globus_root.rstrip('/')}/{name}"
                dst = f"{dest_root}/{tag}/{name}"
                # quoted: project folders can contain spaces (e.g. "Will be Public")
                lines.append(f"--recursive {shlex.quote(src)} {shlex.quote(dst)}")
            (out / f"{tag}.txt").write_text("\n".join(lines) + "\n")
            cmds.append(f"globus transfer {a.src_collection} {a.dst_collection} "
                        f"--batch {out / (tag + '.txt')} --label gmcr-{tag} --verify-checksum")
        (out / "commands.sh").write_text("\n".join(cmds) + "\n")

    total = sum(b for batch in batches for *_, b in batch)
    n_ev = sum(len(b) for b in batches)
    print(f"{n_ev} event folders, {total / 1e12:.2f} TB, {len(batches)} batches of <= {a.batch_gb:g} GB")
    for k, batch in enumerate(batches, 1):
        print(f"  b{k:03d}: {len(batch):6d} events {sum(x[2] for x in batch) / 1e9:8.1f} GB")
    print(f"batch files, manifest, and commands in {out}")


if __name__ == "__main__":
    main()
