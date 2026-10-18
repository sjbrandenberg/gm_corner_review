"""Build a pylauncher task file for feature extraction.

Expects one directory per ComCat ID under DATA_ROOT, each holding one (or more)
workspace .h5 files:  $SCRATCH/gm_data/<comcatid>/*.h5

Files are grouped into chunks (each chunk = one task = one Python start-up) so
Python import overhead is amortised over many small files.

    python tacc/make_tasks.py $SCRATCH/gm_data $SCRATCH/gmcr_run --files-per-task 10
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root")
    p.add_argument("run_dir", help="output directory (tasks, chunk lists, results)")
    p.add_argument("--files-per-task", type=int, default=10)
    p.add_argument("--pattern", default="*/*.h5")
    p.add_argument("--extra-args", default="",
                   help="passed to gmcr.extract, e.g. --extra-args='--reviewed-only --delete-source'")
    p.add_argument("--limit", type=int, default=0, help="only first N files (testing)")
    p.add_argument("--smallest-first", action="store_true",
                   help="order files smallest first (quick tests with --limit); the default, "
                        "largest first, lets a full run finish evenly")
    p.add_argument("--batch", default="", help="name of a staged batch: writes tasks_<batch>.txt "
                   "and chunks/<batch>/ so several batches can share one run directory")
    a = p.parse_args()

    run = Path(a.run_dir).resolve()
    out = run / "features"
    chunks = run / "chunks" / a.batch if a.batch else run / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)

    # Largest files first so the long tasks don't straggle at the end
    files = sorted(Path(a.data_root).resolve().glob(a.pattern),
                   key=lambda f: f.stat().st_size, reverse=not a.smallest_first)
    if a.limit:
        files = files[: a.limit]
    done = {f.stem for f in (out / "traces").glob("*.parquet")} if (out / "traces").exists() else set()
    todo = [f for f in files if f"{f.parent.name}__{f.stem}" not in done]

    lines = []
    n = a.files_per_task
    for i in range(0, len(todo), n):
        lst = chunks / f"chunk_{i // n:06d}.txt"
        lst.write_text("\n".join(str(f) for f in todo[i:i + n]) + "\n")
        lines.append(f"python -m gmcr.extract --list {lst} --out {out} {a.extra_args}".strip())
    tasks = run / (f"tasks_{a.batch}.txt" if a.batch else "tasks.txt")
    tasks.write_text("\n".join(lines) + "\n")
    print(f"{len(files)} files found, {len(done)} already done, "
          f"{len(todo)} to run in {len(lines)} tasks -> {tasks}")


if __name__ == "__main__":
    main()
