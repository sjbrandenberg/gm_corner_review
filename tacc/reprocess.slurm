#!/bin/bash
#SBATCH -J gmcr_reprocess
#SBATCH -o gmcr_reprocess.%j.out
#SBATCH -e gmcr_reprocess.%j.err
#SBATCH -p skx                 # 48 cores / 192 GB per node
#SBATCH -N 1
#SBATCH --ntasks-per-node 24   # one event per task; gmprocess holds an event's streams in memory
#SBATCH -t 08:00:00
#SBATCH -A BCS22014

# Reprocess prepared events with gmprocess 2.9 and extract their features (gmcr.reprocess run).
# Submit with tacc/run_reprocess_batch.sh, or by hand (runs everything from $SCRATCH):
#   python tacc/make_reprocess_tasks.py $SCRATCH/gmcr29_run b001
#   GMCR_PACK_SRC=$WORK/gmp29_env.tar.gz bash tacc/submit.sh reprocess \
#       RUN_DIR=$SCRATCH/gmcr29_run TASKS=$SCRATCH/gmcr29_run/tasks_b001.txt
# Resubmitting is safe: events whose features exist are skipped (rerun make_reprocess_tasks.py first).

set -euo pipefail
export PYTHONUNBUFFERED=1
RUN_DIR=${RUN_DIR:?set RUN_DIR}
TASKS=${TASKS:?set TASKS}
REPO=${GMCR_REPO:-$SCRATCH/gm_corner_review}
[ -f "$REPO/tacc/run_launcher.py" ] || REPO=$SLURM_SUBMIT_DIR
[ -f "$REPO/tacc/run_launcher.py" ] || { echo "can't find gm_corner_review (submit with tacc/submit.sh)" >&2; exit 1; }
PACK=${GMCR_PACK:-$SCRATCH/gmp29_env.tar.gz}
[ -f "$PACK" ] || { echo "no $PACK: run tacc/setup_gmp29_env.sh, and submit with GMCR_PACK_SRC=\$WORK/gmp29_env.tar.gz" >&2; exit 1; }
LOCAL_ENV=/tmp/gmp29_env_$SLURM_JOB_ID

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# unpack the env once per node onto local disk
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    bash -c "rm -rf $LOCAL_ENV && mkdir -p $LOCAL_ENV && tar -xzf $PACK -C $LOCAL_ENV && $LOCAL_ENV/bin/conda-unpack"
export PATH=$LOCAL_ENV/bin:$PATH
echo "python: $(command -v python)   repo: $REPO"
python -c "import gmprocess, gmcr; from importlib.metadata import version; print('gmprocess', version('gmprocess'))"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

module load pylauncher
cd "$RUN_DIR"
python "$REPO/tacc/run_launcher.py" "$TASKS"

python "$REPO/scripts/merge_outputs.py" "$RUN_DIR/features"
if [ -d "$RUN_DIR/records13/traces" ]; then
    python "$REPO/scripts/merge_outputs.py" "$RUN_DIR/records13" > /dev/null   # record list (RUNBOOK 6.4)
fi
python "$REPO/tacc/make_reprocess_tasks.py" "$RUN_DIR" --report
