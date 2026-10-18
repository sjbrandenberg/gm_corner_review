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
#   bash tacc/submit.sh train FEATURES_DIR=$SCRATCH/gmcr_run/features

set -euo pipefail
export PYTHONUNBUFFERED=1   # write log lines as they happen
FEATURES_DIR=${FEATURES_DIR:?set FEATURES_DIR}
MODELS_DIR=${MODELS_DIR:-$FEATURES_DIR/../models}
# code and packed env: the $SCRATCH copies made by submit.sh; GMCR_REPO / GMCR_PACK override
REPO=${GMCR_REPO:-$SCRATCH/gm_corner_review}
[ -f "$REPO/scripts/train_corner_model.py" ] || REPO=$SLURM_SUBMIT_DIR
CONDA_ROOT=${CONDA_ROOT:-$WORK/miniforge3}
PACK=${GMCR_PACK:-$SCRATCH/gmcr_env.tar.gz}
[ -f "$PACK" ] || PACK=$WORK/gmcr_env.tar.gz
LOCAL=/tmp/gmcr_train_$SLURM_JOB_ID

# Copy the code and the packed env to node-local disk once (retrying) and run from there
# (they are on $SCRATCH; $WORK gave I/O errors on compute nodes, and TACC discourages it).
retry() { for i in 1 2 3 4 5; do "$@" && return 0; echo "attempt $i failed: $*" >&2; sleep 30; done; return 1; }
mkdir -p "$LOCAL"
retry cp -r "$REPO/gmcr" "$REPO/scripts" "$LOCAL/"
REPO=$LOCAL
if [ -f "$PACK" ] && retry tar -xzf "$PACK" -C "$LOCAL" && "$LOCAL/bin/conda-unpack"; then
    export PATH=$LOCAL/bin:$PATH
else
    echo "no usable $PACK; activating the env in $WORK" >&2
    retry source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate gmcr
fi
export PYTHONPATH=$REPO
echo "python: $(command -v python)   code: $REPO"
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
