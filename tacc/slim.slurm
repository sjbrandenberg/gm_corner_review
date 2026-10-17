#!/bin/bash
#SBATCH -J gmcr_slim
#SBATCH -o gmcr_slim.%j.out
#SBATCH -e gmcr_slim.%j.err
#SBATCH -p skx
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -t 08:00:00
#SBATCH -A BCS22014

# Copy the workspaces from RAW_DIR to DATA_DIR keeping only what gmcr reads (~36% of the
# size). Resumable: files already in DATA_DIR are skipped.
#   sbatch --export=ALL,RAW_DIR=$SCRATCH/gm_raw,DATA_DIR=$SCRATCH/gm_data tacc/slim.slurm

set -euo pipefail
RAW_DIR=${RAW_DIR:?set RAW_DIR}
DATA_DIR=${DATA_DIR:?set DATA_DIR}
CONDA_ROOT=${CONDA_ROOT:-$WORK/miniforge3}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate gmcr
export OMP_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE

python "$SLURM_SUBMIT_DIR/scripts/slim_workspace.py" "$RAW_DIR" "$DATA_DIR" --workers 32
python "$SLURM_SUBMIT_DIR/tacc/inventory.py" "$DATA_DIR"
