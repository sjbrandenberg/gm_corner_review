#!/bin/bash
# Stage one Globus batch and submit its extraction job (run on a Stampede3 login node).
#
#   bash tacc/run_batch.sh b001
#
# Steps: start the Globus transfer of the batch (tacc/globus_batches.py made the batch
# file), wait for it, check the staged files, build the task list, and submit
# extract.slurm. Each task copies its file to node-local /tmp, extracts it, and deletes
# both the copy and the staged file, so the batch folder empties as the job runs.
# Start the next batch while this one is extracting; run it inside tmux or screen
# (the wait can take hours).
#
# Settings (environment variables, with defaults):
#   DESIGNSAFE_COLLECTION, STAMPEDE3_COLLECTION   Globus collection ids (required)
#   STAGE_ROOT=$SCRATCH/gm_stage   where Globus puts the batches (same as --dest-root)
#   RUN_DIR=$SCRATCH/gmcr_run      outputs; globus/ batch files are in $RUN_DIR/globus
#   NODES=1                        nodes for the extraction job
#   SKIP_TRANSFER=1                if the batch is already staged
#   EXTRACT_ARGS='--reviewed-only --stage-dir /tmp/gmcr_stage --delete-source'
#                                  flags for gmcr.extract (e.g. add --no-psa --fchp-variants)

set -euo pipefail
B=${1:?usage: run_batch.sh <batch, e.g. b001>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
STAGE_ROOT=${STAGE_ROOT:-$SCRATCH/gm_stage}
RUN_DIR=${RUN_DIR:-$SCRATCH/gmcr_run}
NODES=${NODES:-1}
EXTRACT_ARGS=${EXTRACT_ARGS:---reviewed-only --stage-dir /tmp/gmcr_stage --delete-source}
BATCH_FILE=$RUN_DIR/globus/$B.txt
[ -f "$BATCH_FILE" ] || { echo "no $BATCH_FILE: run tacc/globus_batches.py first"; exit 1; }

source "${CONDA_ROOT:-$WORK/miniforge3}/etc/profile.d/conda.sh"
conda activate gmcr

if [ -z "${SKIP_TRANSFER:-}" ]; then
    : "${DESIGNSAFE_COLLECTION:?export DESIGNSAFE_COLLECTION}" "${STAMPEDE3_COLLECTION:?export STAMPEDE3_COLLECTION}"
    TASK=$(globus transfer "$DESIGNSAFE_COLLECTION" "$STAMPEDE3_COLLECTION" \
        --batch "$BATCH_FILE" --label "gmcr-$B" --verify-checksum \
        --jmespath task_id --format unix)
    echo "$(date) Globus task $TASK started for $B"
    echo "$TASK" > "$RUN_DIR/globus/$B.task"
    until globus task wait "$TASK" --timeout 3600 --polling-interval 60 >/dev/null 2>&1; do
        STATUS=$(globus task show "$TASK" --jmespath status --format unix)
        echo "$(date) $B: $STATUS"
        [ "$STATUS" = "FAILED" ] && { echo "transfer failed: globus task show $TASK"; exit 1; }
    done
    echo "$(date) $B transferred"
fi

python "$REPO/tacc/inventory.py" "$STAGE_ROOT/$B" --sample 20
python "$REPO/tacc/make_tasks.py" "$STAGE_ROOT/$B" "$RUN_DIR" --batch "$B" \
    "--extra-args=$EXTRACT_ARGS"
# submit.sh copies code and env to $SCRATCH and runs the job from there (logs in
# $SCRATCH/gmcr_logs/gmcr_<batch>.<jobid>.out)
bash "$REPO/tacc/submit.sh" extract RUN_DIR="$RUN_DIR" TASKS="$RUN_DIR/tasks_$B.txt" \
    -- -N "$NODES" -J "gmcr_$B"
