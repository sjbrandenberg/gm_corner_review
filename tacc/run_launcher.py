"""Run every line of tasks.txt as an independent single-core task across all
nodes of the SLURM allocation (TACC pylauncher)."""

import sys

import pylauncher

tasks = sys.argv[1]
pylauncher.ClassicLauncher(tasks, debug="job", cores=1)
