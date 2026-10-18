"""Run every line of a task file as an independent single-core task across all
nodes of the SLURM allocation (TACC pylauncher).

Tasks on other nodes don't inherit this job's environment, so a task's leading
"python" is replaced by

    env PYTHONPATH=<repository> <full path of this interpreter>

The interpreter (the unpacked env in node-local /tmp, or the env in $WORK) exists at the
same path on every node, and PYTHONPATH makes tasks import gmcr from the repository, so
`git pull` is enough after code changes (no rebuild of the packed env).
"""

import shlex
import sys
from pathlib import Path

import pylauncher

repo = Path(__file__).resolve().parents[1]
tasks = sys.argv[1]
fixed = f"{tasks}.resolved"
prefix = f"env PYTHONPATH={shlex.quote(str(repo))} {shlex.quote(sys.executable)}"
n = 0
with open(tasks) as fi, open(fixed, "w") as fo:
    for line in fi:
        if not line.strip():
            continue
        if line.startswith("python "):
            line = prefix + line[len("python"):]
        fo.write(line)
        n += 1
if n == 0:
    sys.exit(f"{tasks} has no tasks: make_tasks.py found no files to extract "
             "(not staged yet, wrong folder, or all already extracted)")
print(f"tasks: {fixed} ({n} tasks; python = {sys.executable}; gmcr from {repo})", flush=True)
pylauncher.ClassicLauncher(fixed, debug="job", cores=1)
