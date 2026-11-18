"""Step-by-step example of scripts/predict.py, with checks that explain what's wrong.

Run from anywhere:

    python examples/predict/run_example.py                          # bundled models, sample event
    python examples/predict/run_example.py --model fchp accept      # chosen bundled models
    python examples/predict/run_example.py --workspace FILE.h5 [FILE2.h5 ...]
    python examples/predict/run_example.py --model-dir PATH         # a model folder
    python examples/predict/run_example.py --demo                   # the demo models

Steps
  1. Python and packages: are the packages predict.py needs installed and importable?
  2. Models: by default every model bundled with the package (gmcr/trained/, listed by
     `python scripts/predict.py --list-models`); --model picks some of them by name. If no
     models are bundled yet (or with --demo), demo models trained on synthetic labels are
     built the first time, in a few seconds, by make_demo_models.py. --model-dir takes folders
     exported with scripts/export_models.py or training output folders (report.json +
     models.joblib). Each model is checked and loaded here, so problems show up with an
     explanation.
  3. Input: the sample event shipped with this example (sample_features/, already extracted,
     no .h5 file needed), or the workspace files given with --workspace.
  4. predict.py is run with exactly the command printed (bundled models by name), so you can
     copy it and run it yourself. The predictions go to output/ next to this script.
  5. The first rows of each predictions file are printed.
"""

import argparse
import importlib
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PREDICT = REPO / "scripts" / "predict.py"

PACKAGES = [  # (import name, pip/conda name)
    ("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
    ("sklearn", "scikit-learn"), ("joblib", "joblib"), ("h5py", "h5py"), ("obspy", "obspy"),
    ("pyproj", "pyproj"), ("matplotlib", "matplotlib"),
]
OPTIONAL = [("lightgbm", "lightgbm")]  # only for models whose recommendation is LightGBM


def step(n, text):
    print(f"\n=== {n}. {text}", flush=True)


def fail(msg):
    print(f"\nSTOPPED: {msg}")
    sys.exit(1)


def check_packages():
    print(f"Python {platform.python_version()} at {sys.executable}")
    if sys.version_info < (3, 10):
        fail("Python 3.10 or newer is needed (the project's conda env uses 3.11).")
    missing = []
    for mod, pkg in PACKAGES + OPTIONAL:
        try:
            m = importlib.import_module(mod)
            print(f"  ok       {mod:30s} {getattr(m, '__version__', '')}")
        except ImportError as e:
            if (mod, pkg) in PACKAGES:
                print(f"  MISSING  {mod:30s} ({e})")
                missing.append(pkg)
            else:
                print(f"  -        {mod:30s} not installed (optional: needed only for LightGBM models)")
        except Exception as e:  # broken installs, e.g. a pandas/pyarrow version clash
            print(f"  BROKEN   {mod:30s} {type(e).__name__}: {e}")
            missing.append(pkg)
    if missing:
        fail("install or repair: " + " ".join(missing) + "\n"
             "  Easiest: the project's environment (conda env create -f environment.yml, then\n"
             "  conda activate gmcr). 'BROKEN' usually means two installations of the same\n"
             "  package are mixed (e.g. one in your user folder): try setting PYTHONNOUSERSITE=1,\n"
             "  or use a fresh conda/venv environment.")
    sys.path.insert(0, str(REPO))
    try:
        import gmcr  # noqa: F401
    except Exception as e:
        fail(f"can't import gmcr from {REPO}: {e}")
    print(f"  ok       gmcr (from {REPO})")


def _need_lightgbm(rec):
    if rec and rec.startswith("lightgbm"):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            fail("the default model is LightGBM: pip install lightgbm (or conda install lightgbm)")


def check_portable(path):
    """An exported model folder (bundled in gmcr/trained/, or from export_models.py)."""
    from gmcr import portable

    meta = json.loads((path / "model.json").read_text())
    report = json.loads((path / "report.json").read_text())
    prov = json.loads((path / "provenance.json").read_text()) if (path / "provenance.json").exists() else {}
    if meta["kind"] == "auto_fchp":
        rmse = {m: report["metrics"][m]["rmse_ln"] for m in report["selected"] if m in report.get("metrics", {})}
        rec = min(rmse, key=rmse.get)
        print(f"  auto_fchp calibration; default variant: {rec}; settings {report['selected'][rec]['all_data']}")
    else:
        rec = meta.get("recommendation")
        print(f"  kind: {meta['kind']}; default model: {rec}; {len(meta['features'])} predictors"
              + ("; trained with curves" if meta.get("curves") else "")
              + (f"; without {', '.join(meta['dropped'])}" if meta.get("dropped") else ""))
        _need_lightgbm(rec)
        try:
            portable.load_model_dir(path)
        except Exception as e:
            fail(f"the exported models could not be read ({type(e).__name__}: {e})")
    if report.get("n_traces"):
        print(f"  trained on {report['n_traces']} traces from {report.get('n_events', '?')} events; "
              f"exported {prov.get('exported_utc', '?')[:10]} (code {str(prov.get('gmcr_commit'))[:7]})")
    if "DEMO" in prov.get("note", ""):
        print("  (DEMO model: trained on synthetic labels, its predictions mean nothing)")


def check_model_dir(path):
    path = Path(path)
    if not path.is_dir():
        fail(f"{path} is not a folder. Give a model folder: one exported with export_models.py, or "
             "the output folder of a training run (e.g. $SCRATCH/gmcr_run/models/fchp), copied in full.")
    files = sorted(p.name for p in path.iterdir())
    print(f"{path}: {', '.join(files) or '(empty)'}")
    if "report.json" not in files:
        fail("no report.json: copy the whole model folder, not single files.")
    if "model.json" in files:
        check_portable(path)
        return
    report = json.loads((path / "report.json").read_text())
    if "selected" in report and "models.joblib" not in files:
        best = min((m for m in report["selected"] if m in report.get("metrics", {})),
                   key=lambda m: report["metrics"][m]["rmse_ln"])
        print(f"  auto_fchp calibration; best variant: {best}; settings "
              f"{report['selected'][best]['all_data']}")
        return
    if "models.joblib" not in files:
        fail("no models.joblib: copy the whole model folder.")
    import joblib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            bundle = joblib.load(path / "models.joblib")
        except ModuleNotFoundError as e:
            fail(f"loading models.joblib needs a package that isn't installed: {e}")
        except Exception as e:
            fail(f"models.joblib could not be loaded ({type(e).__name__}: {e}). It was probably saved "
                 "with other versions of scikit-learn / LightGBM / numpy than you have: use the\n"
                 "  project's environment (environment.yml), with the versions of the env that trained it.")
    versions = [w for w in caught if "version" in str(w.message).lower()]
    for w in versions[:3]:
        print(f"  WARNING: {w.message}")
    if versions:
        print("  The models were saved with other package versions. Predictions usually still work,\n"
              "  but if they look wrong, use the same versions as the training env on Stampede3.")
    kind = bundle.get("target") or ("accept" if "simple_features" in bundle else "?")
    rec = report.get("recommendation")
    print(f"  kind: {kind}; recommended model: {rec}; {len(bundle['features'])} predictors"
          + ("; trained with curves" if bundle.get("curves") else ""))
    print("  (training output: a pickle tied to package versions; export_models.py makes a portable copy)")
    _need_lightgbm(rec)


def choose_models(a):
    """[(label, argument for predict.py, folder to check)]."""
    from gmcr import portable

    if a.model_dir:
        return [(Path(m).name, str(Path(m)), Path(m)) for m in a.model_dir]
    bundled = portable.bundled_dirs()
    # Bundled models are <generation>/<name>. A name alone makes predict.py use the model of the
    # records' generation (gmprocess version); the folder checked here is the first one with it.
    names = a.model or list(dict.fromkeys(k.split("/", 1)[1] for k in bundled))
    if names and not a.demo:
        if not a.model:
            print(f"models bundled with the package ({portable.TRAINED}): {', '.join(bundled)}")
        chosen = []
        for m in names:
            match = [k for k in bundled if k == m or k.split("/", 1)[1] == m]
            if not match:
                fail(f"not bundled: {m}. Bundled models: {', '.join(bundled) or 'none'}")
            chosen.append((m.replace("/", "_"), m, bundled[match[0]]))
        return chosen
    if not a.demo:
        print("No models are bundled with this copy of the package yet (gmcr/trained/ is empty).\n"
              "They are added by exporting the trained models (docs/RUNBOOK.md, 'Bundling the trained\n"
              "models with the package'); `git pull` to get them once they are committed.\n"
              "Using DEMO models for now.")
    demo = HERE / "demo_models"
    if not all((demo / n / "model.json").exists() for n in ("fchp", "accept")):
        print("building the demo models (first run only)...", flush=True)
        r = subprocess.run([sys.executable, str(HERE / "make_demo_models.py")])
        if r.returncode:
            fail("building the demo models failed (see above)")
    return [(f"demo_{n}", str(demo / n), demo / n) for n in ("fchp", "accept")]


def run_predict(model_arg, source_args, out):
    cmd = [sys.executable, str(PREDICT), str(model_arg), *source_args, "--out", str(out)]
    print("command:\n  " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("\n".join("  | " + ln for ln in r.stdout.strip().splitlines()[-12:]))
    if r.returncode:
        print("\n".join("  ! " + ln for ln in r.stderr.strip().splitlines()[-25:]))
        fail("predict.py failed; the error is just above (lines starting with '!').")


def show(out):
    import pandas as pd

    df = pd.read_parquet(out) if out.suffix == ".parquet" else pd.read_csv(out)
    pred = [c for c in ("pred_fchp_hz", "pred_fclp_hz", "p_accept", "p_keep_gmprocess", "p_lowered",
                        "flag_reject", "ratio_to_gmprocess") if c in df]
    cols = [c for c in ("trace_id", "auto_failed", "fchp_gmp", "fchp_gui") if c in df] + pred
    print(f"{out.name}: {len(df)} traces; columns: {', '.join(df.columns)}")
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.precision", 3):
        print(df[cols].head(12).to_string(index=False))
    if "pred_fchp_hz" in df and df["pred_fchp_hz"].isna().any():
        n = int(df["pred_fchp_hz"].isna().sum())
        print(f"  {n} traces have no prediction: gmprocess rejected them (auto_failed) and gave no corner.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = p.add_mutually_exclusive_group()
    which.add_argument("--model", nargs="+", default=None,
                       help="bundled model name(s) (default: all bundled models)")
    which.add_argument("--model-dir", nargs="+", default=None,
                       help="model folder(s): exported with export_models.py, or training output")
    which.add_argument("--demo", action="store_true", help="use the demo models")
    p.add_argument("--workspace", nargs="+", default=None,
                   help="gmprocess workspace .h5 file(s) to predict for (default: the sample event)")
    a = p.parse_args()

    step(1, "Python and packages")
    check_packages()

    step(2, "Models")
    models = choose_models(a)
    for _, _, folder in models:
        check_model_dir(folder)

    step(3, "Input records")
    if a.workspace:
        missing = [w for w in a.workspace if not Path(w).exists()]
        if missing:
            fail(f"not found: {missing}")
        source = ["--workspaces", *a.workspace]
        print(f"{len(a.workspace)} workspace file(s); predict.py extracts their features first "
              "(about half a second per trace)")
    else:
        source = ["--features", str(HERE / "sample_features")]
        print(f"sample event nc73819746 (63 traces), already extracted: {HERE / 'sample_features'}")

    outputs = []
    for label, arg, _ in models:
        step(4, f"predict.py with {label}")
        out = HERE / "output" / f"predictions_{label}.csv"
        out.parent.mkdir(exist_ok=True)
        run_predict(arg, source, out)
        outputs.append(out)

    step(5, "Results")
    for out in outputs:
        show(out)
        print()
    print("Done. To predict for your own records, copy a command from step 4 and change the input\n"
          "(--workspaces FILES or --features FOLDER) and, if you like, the model (a bundled name from\n"
          "`python scripts/predict.py --list-models`, or a model folder).")


if __name__ == "__main__":
    main()
