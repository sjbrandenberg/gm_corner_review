"""Inventory of staged workspace files: counts, sizes, and a sample read check.

    python tacc/inventory.py $SCRATCH/gm_data [--sample 20]

Expects <root>/<comcatid>/*.h5. Reports event directories, files, total size,
empty directories, files outside the expected layout, and opens a random sample of
files to count their reviewed traces (catches truncated transfers early).
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--sample", type=int, default=20)
    a = p.parse_args()
    root = Path(a.root)
    files = sorted(root.glob("*/*.h5"))
    dirs = [d for d in root.iterdir() if d.is_dir()]
    empty = [d.name for d in dirs if not any(d.glob("*.h5"))]
    stray = [str(f.relative_to(root)) for f in root.rglob("*.h5") if f.parent.parent != root]
    size = sum(f.stat().st_size for f in files)
    print(f"event directories: {len(dirs)}  files: {len(files)}  total: {size / 1e9:.1f} GB")
    print(f"directories without .h5: {len(empty)} {empty[:10]}")
    print(f".h5 files outside <root>/<comcatid>/: {len(stray)} {stray[:10]}")

    from gmcr.asdf_reader import Workspace

    bad, reviewed = [], 0
    for f in random.Random(0).sample(files, min(a.sample, len(files))):
        try:
            with Workspace(f) as ws:
                for tid in ws.processed_trace_ids():
                    rv = ws.review(tid)
                    reviewed += bool(rv and rv.get("reviewed", 0) == 1)
        except Exception as exc:  # noqa: BLE001
            bad.append((f.name, repr(exc)[:120]))
    n = min(a.sample, len(files))
    print(f"sample of {n} files: {len(bad)} unreadable, {reviewed} reviewed traces "
          f"(~{reviewed / max(n, 1) * len(files):.0f} extrapolated)")
    for name, err in bad:
        print("  UNREADABLE", name, err)


if __name__ == "__main__":
    main()
