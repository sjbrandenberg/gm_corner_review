#!/bin/bash
# Reprocess one batch of the training workspaces with gmprocess 2.9 (login node, inside tmux):
#
#   bash tacc/run_reprocess_batch.sh b001
#
# Steps: stage the batch with Globus (the same batch files as the 1.3 extraction, made by
# tacc/globus_batches.py with --select), run `gmcr.reprocess prepare` on it here (it needs the
# internet for the ANSS SIS metadata; it reads only StationXML and review records), build the task
# list, and submit tacc/reprocess.slurm, whose tasks export the raw data, run gmprocess 2.9, extract
# the features, and delete the staged files of each event that finished. See docs/RUNBOOK.md section 6.
#
# Settings (environment variables, with defaults):
#   DESIGNSAFE_COLLECTION, STAMPEDE3_COLLECTION   Globus collection ids (for the transfer)
#   GLOBUS_DIR=$SCRATCH/gmcr_run/globus  the batch files (b001.txt, ...)
#   STAGE_ROOT=$SCRATCH/gm_stage         where Globus puts the batches (their --dest-root)
#   RUN_DIR=$SCRATCH/gmcr29_run          event folders (events/<batch>/), features/, tasks, report
#   NODES=1                              nodes for the job
#   SKIP_TRANSFER=1                      if the batch is already staged
#   SKIP_SUBMIT=1                        prepare and write the task list only

set -euo pipefail
B=${1:?usage: run_reprocess_batch.sh <batch, e.g. b001>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
GLOBUS_DIR=${GLOBUS_DIR:-$SCRATCH/gmcr_run/globus}
STAGE_ROOT=${STAGE_ROOT:-$SCRATCH/gm_stage}
RUN_DIR=${RUN_DIR:-$SCRATCH/gmcr29_run}
NODES=${NODES:-1}
CONDA_ROOT=${CONDA_ROOT:-$WORK/miniforge3}
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if [ -z "${SKIP_TRANSFER:-}" ]; then
    conda activate gmcr   # has the globus CLI
    bash "$REPO/tacc/stage_batch.sh" "$B" "$GLOBUS_DIR"
    conda deactivate
fi
[ -d "$STAGE_ROOT/$B" ] || { echo "$STAGE_ROOT/$B doesn't exist: stage the batch first"; exit 1; }

conda activate gmp29
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}   # gmcr from this checkout
mkdir -p "$RUN_DIR/events"
echo "$(date) preparing $B (STREC, SIS metadata)"
python -m gmcr.reprocess prepare "$STAGE_ROOT/$B" "$RUN_DIR/events/$B" --sis-cache "$RUN_DIR/sis_cache"
python "$REPO/tacc/make_reprocess_tasks.py" "$RUN_DIR" "$B"

if [ -z "${SKIP_SUBMIT:-}" ]; then
    # submit.sh copies the code and the gmp29 env to $SCRATCH and runs the job from there (logs in
    # $SCRATCH/gmcr_logs/gmcr29_<batch>.<jobid>.out)
    GMCR_PACK_SRC=$WORK/gmp29_env.tar.gz bash "$REPO/tacc/submit.sh" reprocess \
        RUN_DIR="$RUN_DIR" TASKS="$RUN_DIR/tasks_$B.txt" -- -N "$NODES" -J "gmcr29_$B"
fi
