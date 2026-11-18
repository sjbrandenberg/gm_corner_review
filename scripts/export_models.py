"""Export trained model folders into the package, in a version-independent format.

    python scripts/export_models.py MODEL_DIR [MODEL_DIR ...] [--name NAME ...] [--only-recommended]
                                    [--generation GENERATION]

    # e.g. on Stampede3, after training:
    python scripts/export_models.py $SCRATCH/gmcr_run/models/fchp $SCRATCH/gmcr_run/models/accept

Each MODEL_DIR is an output folder of train_corner_model.py, train_accept_model.py, or
calibrate_auto_fchp.py. It is written to gmcr/trained/<generation>/<name>/ (name = the folder
name unless --name is given) as model.json + LightGBM text files + report + provenance
(gmcr.portable), which load with any version of scikit-learn / LightGBM / numpy, unlike
models.joblib. The generation (gmcr.generations: which gmprocess version the models go with) is
the one in the training report (training runs record the generation of their features), or
--generation; training runs from before generations existed are gmprocess-1.3.
Commit gmcr/trained/ to ship the models: `pip install` then includes them, and
`predict.py <name> ...` finds them (predict.py --list-models).

Every export is checked: the exported models must give the same predictions as models.joblib
(to 1e-6) on the sample event of examples/predict (or --check-features DIR). The export
stops with an error if they don't.

--only-recommended keeps only the models the report recommends (smaller; --model in
predict.py then can't pick the others). LightGBM models can be several MB each.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from gmcr import generations  # noqa: E402
from gmcr import modeling as gm  # noqa: E402
from gmcr import portable  # noqa: E402

SAMPLE = REPO / "examples" / "predict" / "sample_features"


def _outputs(model, X):
    """Every prediction a stored model makes, as a list of arrays."""
    if isinstance(model, tuple):  # fclp simple
        clf, reg = model
        out = []
        cols = getattr(clf, "feature_names_in_", None)
        if cols is not None:
            out.append(clf.predict_proba(X[list(cols)])[:, 1])
        if reg is not None:
            out.append(reg.predict(X))
        return out
    cls = type(model).__name__
    if cls == "LightGBM":  # exported LightGBM
        return [model.predict_proba(X)[:, 1] if model.binary else model.predict(X)]
    if cls == "LGBMClassifier":
        return [model.predict_proba(X)[:, 1]]
    if cls in ("LGBMRegressor", "LogLinear", "LogLinearModel"):
        return [model.predict(X)]
    return [model.predict_proba(X[list(model.feature_names_in_)])[:, 1]]  # logistic


def check(src, dst, features_dir):
    """Largest difference between the joblib and exported models' predictions."""
    import joblib
    import predict  # scripts/predict.py

    if not (Path(src) / "models.joblib").exists():
        return 0.0, 0  # auto_fchp calibration: settings only, copied as is
    original = joblib.load(Path(src) / "models.joblib")
    kind, _, exported = portable.load_model_dir(dst)
    df = gm.load_traces(features_dir, curves=True)
    X, _ = predict.predictors(df, original, include_site=(kind != "accept"))
    worst, n = 0.0, 0
    for key, m in exported["models"].items():
        a, b = _outputs(original["models"][key], X), _outputs(m, X)
        if len(a) != len(b):
            raise AssertionError(f"{key}: {len(a)} outputs from models.joblib, {len(b)} exported")
        for u, v in zip(a, b):
            u, v = np.asarray(u, float), np.asarray(v, float)
            if not np.array_equal(np.isnan(u), np.isnan(v)):
                raise AssertionError(f"{key}: missing predictions differ")
            d = np.nanmax(np.abs(u - v)) if np.isfinite(u).any() else 0.0
            worst, n = max(worst, d), n + 1
    return worst, n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dirs", nargs="+")
    p.add_argument("--name", nargs="+", default=None, help="names in gmcr/trained/ (default: folder names)")
    p.add_argument("--into", default=str(portable.TRAINED),
                   help="destination; models go into <into>/<generation>/<name> (default: gmcr/trained)")
    p.add_argument("--generation", default=None, choices=list(generations.GENERATIONS),
                   help="model generation (default: the training report's)")
    p.add_argument("--only-recommended", action="store_true")
    p.add_argument("--note", default="", help="free text stored in provenance.json")
    p.add_argument("--check-features", default=str(SAMPLE),
                   help="features folder for the check (default: the example's sample event)")
    a = p.parse_args()
    names = a.name or [Path(d).resolve().name for d in a.model_dirs]
    if len(names) != len(a.model_dirs):
        sys.exit("--name: give one name per model folder")
    for src, name in zip(a.model_dirs, names):
        gen = a.generation or portable.model_generation(src)
        dst = Path(a.into) / gen / name
        meta = portable.export_model_dir(src, dst, a.only_recommended, a.note, gen)
        worst, n = check(src, dst, a.check_features)
        if worst > 1e-6:
            sys.exit(f"{name}: exported models differ from models.joblib by {worst:.3g}; not exported correctly")
        size = sum(f.stat().st_size for f in dst.iterdir()) / 1e6
        models = ", ".join(meta.get("models", {})) or "settings only"
        print(f"{gen}/{name:14s} {meta['kind']:9s} -> {dst}  ({size:.2f} MB; {models}; "
              f"{n} outputs checked, max difference {worst:.1e})")
    print("\nCommit gmcr/trained/ to ship these models; list them with: python scripts/predict.py --list-models")


if __name__ == "__main__":
    main()
