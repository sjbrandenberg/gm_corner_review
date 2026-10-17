"""Copy a gmprocess workspace, keeping only what gmcr needs (about 40% of the
original size). Useful before transferring files to TACC.

Dropped: full-resolution Signal/Event/Noise spectrum caches and the cached
pre-event noise trace (the smoothed spectra and SNR curves are kept).

    python scripts/slim_workspace.py SRC_ROOT DST_ROOT [--pattern '*/*.h5'] [--workers 8]

Directory structure under SRC_ROOT (e.g. <comcatid>/<file>.h5) is preserved.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmcr.slim import slim  # noqa: E402


def _job(args):
    src, dst = args
    if dst.exists():
        return 0, 0
    try:
        return slim(src, dst)
    except Exception as exc:
        print(f"FAILED {src}: {exc}", flush=True)
        return 0, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src_root")
    p.add_argument("dst_root")
    p.add_argument("--pattern", default="*/*.h5")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    src_root, dst_root = Path(a.src_root), Path(a.dst_root)
    jobs = [(f, dst_root / f.relative_to(src_root)) for f in sorted(src_root.glob(a.pattern))]
    tot_in = tot_out = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (b_in, b_out) in enumerate(ex.map(_job, jobs, chunksize=4), 1):
            tot_in += b_in
            tot_out += b_out
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} files, {tot_in / 1e9:.1f} GB -> {tot_out / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
