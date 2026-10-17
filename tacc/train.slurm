#!/bin/bash
#SBATCH -J gmcr_train
#SBATCH -o gmcr_train.%j.out
#SBATCH -e gmcr_train.%j.err
#SBATCH -p skx
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -t 08:00:00
#SBATCH -A BCS22014

# Train and compare the models on one node (LightGBM uses all cores).
# Needs FEATURES_DIR/traces_joined.parquet: run scripts/fetch_flatfile.py on a login
# node first (compute nodes have no internet); this job runs join_metadata.py.
#   sbatch --export=ALL,FEATURES_DIR=$SCRATCH/gmcr_run/features tacc/train.slurm

set -euo pipefail
FEATURES_DIR=${FEATURES_DIR:?set FEATURES_DIR}
MODELS_DIR=${MODELS_DIR:-$FEATURES_DIR/../models}
REPO=${SLURM_SUBMIT_DIR}
CONDA_ROOT=${CONDA_ROOT:-$WORK/miniforge3}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate gmcr
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-48}

if [ -d "$FEATURES_DIR/db" ]; then
    python "$REPO/scripts/join_metadata.py" "$FEATURES_DIR"
else
    echo "no $FEATURES_DIR/db: skipping join_metadata (models use HDF5 features only)"
fi

python "$REPO/scripts/train_corner_model.py" "$FEATURES_DIR" --target fchp --out "$MODELS_DIR/fchp"
python "$REPO/scripts/train_corner_model.py" "$FEATURES_DIR" --target fchp --curves --out "$MODELS_DIR/fchp_curves"
python "$REPO/scripts/train_corner_model.py" "$FEATURES_DIR" --target fclp --out "$MODELS_DIR/fclp"
python "$REPO/scripts/train_accept_model.py" "$FEATURES_DIR" --out "$MODELS_DIR/accept"
python "$REPO/scripts/train_accept_model.py" "$FEATURES_DIR" --curves --out "$MODELS_DIR/accept_curves"
