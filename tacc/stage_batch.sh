#!/bin/bash
# Stage one Globus batch from DesignSafe and wait for it (login node; used by run_batch.sh and
# run_reprocess_batch.sh). The globus CLI is in the gmcr env.
#
#   bash tacc/stage_batch.sh <batch> <folder of the batch files, e.g. $SCRATCH/gmcr_run/globus>
#
# Needs DESIGNSAFE_COLLECTION and STAMPEDE3_COLLECTION. Writes <folder>/<batch>.task (the task id).

set -euo pipefail
B=${1:?usage: stage_batch.sh <batch> <globus folder>}
GLOBUS_DIR=${2:?usage: stage_batch.sh <batch> <globus folder>}
BATCH_FILE=$GLOBUS_DIR/$B.txt
[ -f "$BATCH_FILE" ] || { echo "no $BATCH_FILE: run tacc/globus_batches.py first"; exit 1; }
: "${DESIGNSAFE_COLLECTION:?export DESIGNSAFE_COLLECTION}" "${STAMPEDE3_COLLECTION:?export STAMPEDE3_COLLECTION}"

TASK=$(globus transfer "$DESIGNSAFE_COLLECTION" "$STAMPEDE3_COLLECTION" \
    --batch "$BATCH_FILE" --label "gmcr-$B" --verify-checksum \
    --jmespath task_id --format unix)
echo "$(date) Globus task $TASK started for $B"
echo "$TASK" > "$GLOBUS_DIR/$B.task"
# Ctrl-C stops the waiting, not the transfer, which runs on Globus's side; say so and how to cancel it.
trap 'echo; echo "Stopped waiting. Globus transfer $TASK ($B) continues; to cancel it: globus task cancel $TASK"; exit 130' INT
until globus task wait "$TASK" --timeout 3600 --polling-interval 60 >/dev/null 2>&1; do
    STATUS=$(globus task show "$TASK" --jmespath status --format unix)
    echo "$(date) $B: $STATUS"
    [ "$STATUS" = "FAILED" ] && { echo "transfer failed: globus task show $TASK"; exit 1; }
done
echo "$(date) $B transferred"
