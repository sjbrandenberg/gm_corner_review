#!/bin/bash
#SBATCH -J gmcr_extract
#SBATCH -o gmcr_extract.%j.out
#SBATCH -e gmcr_extract.%j.err
#SBATCH -p skx                 # 48 cores / 192 GB per node; icx or spr also work
#SBATCH -N 2
#SBATCH --ntasks-per-node 48
#SBATCH -t 04:00:00
#SBATCH -A BCS22014
##SBATCH --mail-user=you@ucla.edu
##SBATCH --mail-type=end

# Submit with tacc/submit.sh (runs everything from $SCRATCH, as TACC asks):
#   python tacc/make_tasks.py $SCRATCH/gm_stage/b001 $SCRATCH/gmcr_run --batch b001 ...
#   bash tacc/submit.sh extract RUN_DIR=$SCRATCH/gmcr_run TASKS=$SCRATCH/gmcr_run/tasks_b001.txt
# Resubmitting is safe: re-run make_tasks.py first and finished files are skipped.

set -euo pipefail
export PYTHONUNBUFFERED=1   # write log lines as they happen
RUN_DIR=${RUN_DIR:?set RUN_DIR}
TASKS=${TASKS:-$RUN_DIR/tasks.txt}   # tasks_<batch>.txt for staged batches
# code and packed env: the $SCRATCH copies made by submit.sh (compute nodes should not
# use $WORK); GMCR_REPO / GMCR_PACK override
REPO=${GMCR_REPO:-$SCRATCH/gm_corner_review}
[ -f "$REPO/tacc/run_launcher.py" ] || REPO=$SLURM_SUBMIT_DIR
[ -f "$REPO/tacc/run_launcher.py" ] || { echo "can't find gm_corner_review (submit with tacc/submit.sh)" >&2; exit 1; }
echo "repo: $REPO"
CONDA_ROOT=${CONDA_ROOT:-$WORK/miniforge3}
PACK=${GMCR_PACK:-$SCRATCH/gmcr_env.tar.gz}
[ -f "$PACK" ] || PACK=$WORK/gmcr_env.tar.gz
LOCAL_ENV=/tmp/gmcr_env_$SLURM_JOB_ID

# One BLAS/FFT thread per task; otherwise 48 tasks x N threads oversubscribe the node
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

USE_LOCAL=0
if [ -f "$PACK" ]; then
    # Unpack the env exactly once per node onto local disk. The explicit node/task counts
    # matter: without them srun inherits the job's 48 tasks per node and 48 tars collide.
    # If unpacking fails, fall back to the env in $WORK instead of stopping the job.
    if srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
            bash -c "rm -rf $LOCAL_ENV && mkdir -p $LOCAL_ENV && tar -xzf $PACK -C $LOCAL_ENV && $LOCAL_ENV/bin/conda-unpack"; then
        USE_LOCAL=1
    else
        echo "unpacking $PACK failed; using the env in \$WORK (last resort; TACC discourages it)" >&2
    fi
fi
if [ "$USE_LOCAL" = 1 ]; then
    export PATH=$LOCAL_ENV/bin:$PATH
else
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate gmcr
fi
echo "python: $(command -v python)"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}   # gmcr from the repository: git pull is enough after changes

module load pylauncher
cd "$RUN_DIR"
python "$REPO/tacc/run_launcher.py" "$TASKS"

python "$REPO/scripts/merge_outputs.py" "$RUN_DIR/features"
