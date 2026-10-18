#!/bin/bash
# Submit a gmcr job so that it runs entirely from $SCRATCH (run on a login node).
#
#   bash tacc/submit.sh <extract|train|slim> [VAR=value ...] [-- sbatch options ...]
#
#   bash tacc/submit.sh train FEATURES_DIR=$SCRATCH/gmcr_run/features
#   bash tacc/submit.sh extract RUN_DIR=$SCRATCH/gmcr_test -- -N 1 -t 01:00:00
#
# TACC asks that compute nodes not use $WORK (Stockyard) except to stage data before or
# after a job (https://docs.tacc.utexas.edu/basics/conduct/). The code and the packed env
# live in $WORK; this script copies them to $SCRATCH on the login node first:
#   $WORK/gm_corner_review  -> $SCRATCH/gm_corner_review   (rsync; only changes are copied)
#   $WORK/gmcr_env.tar.gz   -> $SCRATCH/gmcr_env.tar.gz    (only if newer)
# and submits from $SCRATCH/gm_corner_review with the logs in $SCRATCH/gmcr_logs
# (<job name>.<job id>.out / .err) and GMCR_REPO / GMCR_PACK pointing at the copies.

set -euo pipefail
JOB=${1:?usage: submit.sh <extract|train|slim> [VAR=value ...] [-- sbatch options ...]}
shift
SRC=$(cd "$(dirname "$0")/.." && pwd)
DST=$SCRATCH/gm_corner_review
LOGS=$SCRATCH/gmcr_logs
PACK_SRC=${GMCR_PACK_SRC:-$WORK/gmcr_env.tar.gz}
PACK=$SCRATCH/gmcr_env.tar.gz

VARS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do VARS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
SBATCH_OPTS=("$@")

[ -f "$SRC/tacc/$JOB.slurm" ] || { echo "no tacc/$JOB.slurm"; exit 1; }
mkdir -p "$LOGS"
if [ "$SRC" != "$DST" ]; then
    rsync -a --delete --exclude .git "$SRC/" "$DST/"
fi
if [ -f "$PACK_SRC" ] && [ "$PACK_SRC" != "$PACK" ] && { [ ! -f "$PACK" ] || [ "$PACK_SRC" -nt "$PACK" ]; }; then
    echo "copying $PACK_SRC to $PACK"
    cp "$PACK_SRC" "$PACK.tmp" && mv "$PACK.tmp" "$PACK"
fi

EXPORT="ALL,GMCR_REPO=$DST"
[ -f "$PACK" ] && EXPORT+=",GMCR_PACK=$PACK"
for v in "${VARS[@]}"; do EXPORT+=",$v"; done

cd "$DST"
echo "submitting tacc/$JOB.slurm from $DST (logs in $LOGS)"
sbatch -o "$LOGS/%x.%j.out" -e "$LOGS/%x.%j.err" --export="$EXPORT" "${SBATCH_OPTS[@]}" "tacc/$JOB.slurm"
