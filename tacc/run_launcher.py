"""Run every line of a task file as an independent single-core task across all
nodes of the SLURM allocation (TACC pylauncher).

Tasks on other nodes don't inherit this job's PATH, so a task's leading "python" is
replaced by the full path of the interpreter running this script (the unpacked env in
node-local /tmp exists at the same path on every node, as does the env in $WORK).
"""

import sys

import pylauncher

tasks = sys.argv[1]
fixed = f"{tasks}.resolved"
n = 0
with open(tasks) as fi, open(fixed, "w") as fo:
    for line in fi:
        if not line.strip():
            continue
        if line.startswith("python "):
            line = sys.executable + line[len("python"):]
        fo.write(line)
        n += 1
if n == 0:
    sys.exit(f"{tasks} has no tasks: make_tasks.py found no files to extract "
             "(not staged yet, wrong folder, or all already extracted)")
print(f"tasks: {fixed} (python = {sys.executable})", flush=True)
pylauncher.ClassicLauncher(fixed, debug="job", cores=1)
