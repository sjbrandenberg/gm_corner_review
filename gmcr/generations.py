"""Model generations: which gmprocess version a set of trained models goes with.

gmcr's predictors describe gmprocess's own decisions (its corner frequencies, SNR, failure checks), so
models trained on workspaces from one gmprocess version are biased on another: gmprocess 2.9's SNR
corners differ from 1.3's (on nc73819746, the low-pass corner is about 30% higher for the same raw
data). Each generation therefore names

    gmprocess   the gmprocess version (major, minor) whose workspaces it was trained on and applies to
    auto_fchp   the ucla_geotech_tools version of auto_fchp that computes fchp_gui and the
                displacement-ratio curves (gmcr.auto_fchp_303, gmcr.auto_fchp_310)
    config      the gmprocess config folder (gmrecords --confdir) the training workspaces were
                processed with, for generations gmcr reprocessed itself (gmcr/conf/<generation>/);
                new records must be processed with it for the models to apply

Bundled models live in gmcr/trained/<generation>/<name>/, and their model.json records the generation.
gmcr.extract picks the generation from the workspace's gmprocess version, and scripts/predict.py
refuses models of another generation than the records.

Labels: analysts reviewed records processed with gmprocess 1.3, and fchp_gui in those reviews came from
auto_fchp 3.0.3. A later generation is trained on the same events reprocessed with its gmprocess
version, with the analysts' decisions joined from the gmprocess-1.3 extraction by trace id.
"""

import re
from pathlib import Path

CONF = Path(__file__).resolve().parent / "conf"

GENERATIONS = {
    "gmprocess-1.3": {
        "gmprocess": (1, 3),
        "auto_fchp": "3.0.3",
        "note": "NGA-West3 review workspaces (gmprocess 1.3 at 045580a1, review GUI v6.2)",
    },
    "gmprocess-2.9": {
        "gmprocess": (2, 9),
        "auto_fchp": "3.1.0",
        "config": "gmprocess-2.9",
        "note": "training events reprocessed with gmprocess 2.9 and gmcr/conf/gmprocess-2.9 "
                "(gmcr.reprocess)",
    },
}

LEGACY = "gmprocess-1.3"  # the generation of models and extractions from before generations existed


def parse_version(version):
    """(major, minor) of a gmprocess version string such as '1.3.1.dev31+g045580a1' or '2.9.0'."""
    m = re.match(r"\s*v?(\d+)\.(\d+)", str(version or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def for_gmprocess(version):
    """The generation for a gmprocess version string, or None if no generation covers it."""
    mm = parse_version(version)
    for name, g in GENERATIONS.items():
        if g["gmprocess"] == mm:
            return name
    return None


def get(name):
    if name not in GENERATIONS:
        raise ValueError(f"unknown model generation {name!r}; known: {', '.join(GENERATIONS)}")
    return GENERATIONS[name]


def auto_fchp_version(name):
    return get(name)["auto_fchp"]


def config_dir(name):
    """The gmprocess config folder (gmrecords --confdir) of a generation; None for gmprocess-1.3,
    whose workspaces were processed outside gmcr (DesignSafe/gmprocess/conf/user.yml)."""
    sub = get(name).get("config")
    return CONF / sub if sub else None


def describe(name):
    g = get(name)
    return f"{name} (gmprocess {g['gmprocess'][0]}.{g['gmprocess'][1]}, auto_fchp {g['auto_fchp']})"
