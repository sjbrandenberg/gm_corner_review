"""Step-by-step example of scripts/predict.py, with checks that explain what's wrong.

Run from anywhere:

    python examples/predict/run_example.py                         # demo models, sample event
    python examples/predict/run_example.py --model-dir PATH        # a real model folder
    python examples/predict/run_example.py --model-dir PATH --workspace FILE.h5 [FILE2.h5 ...]

Steps
  1. Python and packages: are the packages predict.py needs installed and importable?
  2. Model folder(s): with no --model-dir, the demo models (built the first time, in a few
     seconds, by make_demo_models.py); otherwise the folder you give, which must hold
     report.json and models.joblib as written by the training scripts. The model is loaded
     once here, so version problems show up with an explanation.
  3. Input: the sample event shipped with this example (sample_features/, already extracted,
     no .h5 file needed), or the workspace files given with --workspace.
  4. predict.py is run with exactly the command printed, so you can copy it and run it
     yourself. The predictions go to output/ next to this script.
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
    ("ucla_geotech_tools.auto_fchp", "ucla_geotech_tools==3.0.3"),
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


def check_model_dir(path):
    path = Path(path)
    if not path.is_dir():
        fail(f"{path} is not a folder. Give the output folder of a training run, e.g. "
             "$SCRATCH/gmcr_run/models/fchp, copied in full.")
    files = sorted(p.name for p in path.iterdir())
    print(f"{path}: {', '.join(files) or '(empty)'}")
    if "report.json" not in files:
        fail("no report.json: copy the whole model folder, not only models.joblib.")
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
    if rec and rec.startswith("lightgbm"):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            fail("the recommended model is LightGBM: pip install lightgbm (or conda install lightgbm)")
    note = path.parent / "README.txt"  # demo_models/README.txt
    if note.exists() and "DEMO" in note.read_text():
        print("  (DEMO model: trained on synthetic labels, its predictions mean nothing)")


def run_predict(model_dir, source_args, out):
    cmd = [sys.executable, str(PREDICT), str(model_dir), *source_args, "--out", str(out)]
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
    p.add_argument("--model-dir", nargs="+", default=None,
                   help="real model folder(s) (default: the demo models)")
    p.add_argument("--workspace", nargs="+", default=None,
                   help="gmprocess workspace .h5 file(s) to predict for (default: the sample event)")
    a = p.parse_args()

    step(1, "Python and packages")
    check_packages()

    step(2, "Model folder(s)")
    if a.model_dir:
        model_dirs = [Path(m) for m in a.model_dir]
    else:
        demo = HERE / "demo_models"
        if not (demo / "fchp" / "models.joblib").exists() or not (demo / "accept" / "models.joblib").exists():
            print("building the demo models (first run only)...", flush=True)
            r = subprocess.run([sys.executable, str(HERE / "make_demo_models.py")])
            if r.returncode:
                fail("building the demo models failed (see above)")
        model_dirs = [demo / "fchp", demo / "accept"]
    for m in model_dirs:
        check_model_dir(m)

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
    for m in model_dirs:
        step(4, f"predict.py with {m.name}")
        out = HERE / "output" / f"predictions_{m.name}.csv"
        out.parent.mkdir(exist_ok=True)
        run_predict(m, source, out)
        outputs.append(out)

    step(5, "Results")
    for out in outputs:
        show(out)
        print()
    print("Done. To predict for your own records, copy a command from step 4 and change the model\n"
          "folder and the input (--workspaces FILES or --features FOLDER).")


if __name__ == "__main__":
    main()
