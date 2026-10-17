#!/bin/bash
# One-time setup of the gmcr Python environment on Stampede3.
# Run from the repository root, preferably inside an idev session:
#     idev -p skx -N 1 -t 01:00:00
#     bash tacc/setup_env.sh
#
# Installs Miniforge under $WORK (not $HOME: small quota; not $SCRATCH: purged),
# creates the "gmcr" env, installs this package, and packs a copy of the env
# that jobs can unpack to node-local /tmp (avoids hammering $WORK at startup).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
CONDA_ROOT=${CONDA_ROOT:-$WORK/miniforge3}
PACK=${GMCR_PACK:-$WORK/gmcr_env.tar.gz}

if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
    echo "Installing Miniforge to $CONDA_ROOT"
    curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_ROOT"
    rm /tmp/miniforge.sh
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if conda env list | grep -qE '^gmcr\s'; then
    conda env update -n gmcr -f "$REPO/environment.yml" --prune
else
    conda env create -f "$REPO/environment.yml"
fi
conda activate gmcr
# non-editable install so the packed env is self-contained
pip install --no-deps "$REPO"

python -c "import gmcr, obspy, h5py, ucla_geotech_tools.auto_fchp; print('gmcr env OK')"

echo "Packing env to $PACK"
rm -f "$PACK"
conda-pack -p "$CONDA_PREFIX" -o "$PACK" --ignore-editable-packages  # conda-pack lives in the gmcr env, not base
echo "Done. Re-run this script after changing gmcr code so the packed env is updated."
