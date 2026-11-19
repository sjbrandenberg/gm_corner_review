#!/bin/bash
# One-time setup of the gmprocess 2.9 environment ("gmp29") for reprocessing the training events
# (gmcr.reprocess, docs/RUNBOOK.md section 6). Run from the repository root ON A LOGIN NODE: it needs the
# internet (pip, STREC's data); the build itself takes a few minutes.
#     bash tacc/setup_gmp29_env.sh
#
# Uses the Miniforge of tacc/setup_env.sh ($WORK/miniforge3; run that first). Creates the conda env
# "gmp29" (Python 3.12, which gmprocess 2.9's wheels need, with its numpy < 2.4 pin) with gmprocess,
# gmcr, and pylauncher's paramiko; packs it to $WORK/gmp29_env.tar.gz for jobs to unpack on node-local
# disk; and downloads STREC's slab models and GCMT catalog to $WORK/strec_data (the `prepare` step
# classifies each event's tectonic region with them, as gmprocess's download does; STREC keeps its
# settings in ~/.strec/config.ini). Rerun after changing gmprocess's version; gmcr code changes need
# only a git pull (jobs import gmcr from the repository).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
CONDA_ROOT=${CONDA_ROOT:-$WORK/miniforge3}
PACK=${GMP29_PACK:-$WORK/gmp29_env.tar.gz}
STREC_DATA=${STREC_DATA:-$WORK/strec_data}
GMPROCESS=${GMPROCESS:-gmprocess==2.9.0}

[ -x "$CONDA_ROOT/bin/conda" ] || { echo "no Miniforge in $CONDA_ROOT: run tacc/setup_env.sh first"; exit 1; }
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | grep -qE '^gmp29\s'; then
    conda create -y -n gmp29 -c conda-forge python=3.12 pip conda-pack paramiko
fi
conda activate gmp29
pip install --upgrade "$GMPROCESS" lightgbm pyarrow
# non-editable, so the packed env is self-contained; jobs still import gmcr from the repository
pip install --upgrade "$REPO"

python - <<'EOF'
from importlib.metadata import version
from gmprocess.utils.config import get_config
from gmcr import generations
conf = generations.config_dir("gmprocess-2.9")
c = get_config(conf)
print(f"gmprocess {version('gmprocess')}, gmcr {version('gmcr')}, numpy {version('numpy')}, "
      f"obspy {version('obspy')}; config {conf}: {len(c['processing'])} processing steps")
EOF

if [ -z "$(ls -A "$STREC_DATA" 2>/dev/null)" ] || [ -n "${REFRESH_STREC:-}" ]; then
    mkdir -p "$STREC_DATA"
    echo "Downloading STREC's data to $STREC_DATA (a few minutes)"
    strec_cfg update --datafolder "$STREC_DATA" --slab --gcmt
fi
python -c "
from strec.subtype import SubductionSelector
r = SubductionSelector().getSubductionType(37.9147, -122.3043, 4.6, 3.6)
print('STREC answers:', r['TectonicRegion'], r['FocalMechanism'])"

echo "Packing env to $PACK"
rm -f "$PACK"
conda-pack -n gmp29 -o "$PACK" --ignore-editable-packages
echo "Done."
