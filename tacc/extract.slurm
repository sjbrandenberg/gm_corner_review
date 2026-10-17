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

# Usage (from the repository root):
#   python tacc/make_tasks.py $SCRATCH/gm_data $SCRATCH/gmcr_run
#   sbatch --export=ALL,RUN_DIR=$SCRATCH/gmcr_run tacc/extract.slurm
# Resubmitting is safe: re-run make_tasks.py first and finished files are skipped.
# Staged batches (see docs/RUNBOOK.md):
#   sbatch --export=ALL,RUN_DIR=$SCRATCH/gmcr_run,TASKS=$SCRATCH/gmcr_run/tasks_b001.txt tacc/extract.slurm

set -euo pipefail
RUN_DIR=${RUN_DIR:?set RUN_DIR}
TASKS=${TASKS:-$RUN_DIR/tasks.txt}   # tasks_<batch>.txt for staged batches
REPO=${SLURM_SUBMIT_DIR}
CONDA_ROOT=${CONDA_ROOT:-$WORK/miniforge3}
PACK=${GMCR_PACK:-$WORK/gmcr_env.tar.gz}
LOCAL_ENV=/tmp/gmcr_env_$SLURM_JOB_ID

# One BLAS/FFT thread per task; otherwise 48 tasks x N threads oversubscribe the node
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

if [ -f "$PACK" ]; then
    # Unpack the env once per node onto local disk
    srun --ntasks-per-node=1 bash -c "mkdir -p $LOCAL_ENV && tar -xzf $PACK -C $LOCAL_ENV && $LOCAL_ENV/bin/conda-unpack"
    export PATH=$LOCAL_ENV/bin:$PATH
else
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate gmcr
fi

module load pylauncher
cd "$RUN_DIR"
python "$REPO/tacc/run_launcher.py" "$TASKS"

python "$REPO/scripts/merge_outputs.py" "$RUN_DIR/features"
