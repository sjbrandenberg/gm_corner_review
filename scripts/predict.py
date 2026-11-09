"""Apply a trained model to new records.

    # from a folder of extracted features (gmcr.extract + merge_outputs [+ join_metadata])
    python scripts/predict.py MODEL_DIR --features FEATURES_DIR --out predictions.csv

    # straight from workspace files (features are extracted to a temporary folder first)
    python scripts/predict.py MODEL_DIR --workspaces EVENT_DIR/*.h5 --out predictions.csv

MODEL_DIR is an output folder of one of the training scripts:

  train_corner_model.py --target fchp   high-pass corner (Hz)
  train_corner_model.py --target fclp   low-pass corner (Hz)
  train_accept_model.py                 probability that an analyst would accept the record
  calibrate_auto_fchp.py                high-pass corner (Hz) from re-tuned auto_fchp settings

By default the model the folder's report.json recommends is applied ("recommendation" for
the trained models; for an auto_fchp calibration, the calibrated variant with the lowest
cross-validated RMSE). --model picks another one, e.g. --model simple. The models are the
ones refit on all the training data (models.joblib), or the all-data settings of the
calibration.

Predictors are built exactly as in training (gmcr.modeling.base_features, plus the curves
for models trained with --curves, without the predictors the run dropped). Predictors that
can't be computed for the new records are left missing: the simple models fill them with
their training medians and LightGBM handles them natively, but predictions are better with
the same inputs as in training. In particular, the training tables had finite-fault
distances and site data from join_metadata.py; without them, distances fall back to the
hypocentral and epicentral distances from the workspace, and site terms are missing.

Output (CSV, or Parquet if --out ends in .parquet), one row per trace: trace_key, event_id,
trace_id, the automatic values (fchp_gmp, fchp_gui, fclp_gmp), missing_predictors (how many
predictors were unavailable for the trace), and

  fchp, fclp models   pred_fchp_hz / pred_fclp_hz, ratio_to_gmprocess, and p_keep_gmprocess
                      (two-stage fchp models) or p_lowered (fclp)
  accept models       p_accept, and flag_reject if --flag-share is given (the threshold that
                      flagged that share of the analysts' rejections in cross-validation)

Traces the automatic checks rejected (auto_failed) are included; filter on that column if
only the records gmprocess passed are of interest, as in training. Most of them have no
gmprocess corners, so the corner models (which start from fchp_gmp / fclp_gmp) give NaN.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmcr import modeling as gm  # noqa: E402

ID_COLUMNS = ["trace_key", "event_id", "trace_id", "network", "station", "channel",
              "auto_failed", "fchp_gmp", "fchp_gui", "fclp_gmp"]


# ------------------------------------------------------------------------------ input


def features_from_workspaces(paths, fchp_variants=False):
    """Extract features from workspace files into a temporary folder; return the
    traces table merged with the curves (as gmcr.modeling.load_traces does)."""
    from gmcr.extract import process_file

    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.h5")) if p.is_dir() else [p]
    if not files:
        sys.exit("no .h5 files given")
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            status = process_file(f, tmp, fchp_variants=fchp_variants)
            print(f"{status:8s} {f}", flush=True)
        tr = sorted(Path(tmp, "traces").glob("*.parquet"))
        if not tr:
            sys.exit("no traces extracted (see the errors above)")
        df = pd.concat((pd.read_parquet(f) for f in tr), ignore_index=True)
        cur = pd.concat((pd.read_parquet(Path(tmp, "curves", f.name)) for f in tr
                         if Path(tmp, "curves", f.name).exists()), ignore_index=True)
    df = df.merge(cur, on="trace_key", how="left")
    if "error" in df:
        bad = df["error"].notna()
        if bad.any():
            print(f"{bad.sum()} traces could not be extracted and are left out", flush=True)
        df = df[~bad]
    return df.reset_index(drop=True)


def predictors(df, bundle, include_site):
    """The predictor table of a trained model, columns in training order."""
    X = gm.base_features(df, include_site=include_site)
    if bundle.get("curves"):
        X = pd.concat([X, gm.curve_features(df)], axis=1)
    missing = [c for c in bundle["features"] if c not in X]
    X = X.reindex(columns=bundle["features"])
    return X, missing


# ------------------------------------------------------------------------- models


def _proba(model, X):
    return model.predict_proba(X)[:, 1]


def _pipeline_columns(pipe):
    """Input columns of a fitted sklearn pipeline (None if it was never fitted)."""
    cols = getattr(pipe, "feature_names_in_", None)
    return list(cols) if cols is not None else []


def predict_fchp(models, X, name):
    """ln(fchp) and P(analyst keeps the gmprocess value) (NaN for direct models)."""
    ln_gmp = X["ln_fchp_gmp"].to_numpy() if "ln_fchp_gmp" in X else None
    p_keep = np.full(len(X), np.nan)
    family = "simple" if name.startswith("simple") else "lightgbm"
    if family == "simple":
        direct = models["simple"]
        feats = direct.features
        if name == "simple":
            return direct.predict(X), p_keep
        p_keep = _proba(models["simple_keep"], X[feats])
        stage2 = models["simple_adjusted"].predict(X)
    else:
        if name == "lightgbm":
            return models["lightgbm"].predict(X), p_keep
        p_keep = _proba(models["lightgbm_keep"], X)
        stage2 = models["lightgbm_adjusted"].predict(X)
    if ln_gmp is None:
        sys.exit(f"{name} needs ln_fchp_gmp, which the model was trained without")
    if name.endswith("_soft"):
        return p_keep * ln_gmp + (1 - p_keep) * stage2, p_keep
    return np.where(p_keep >= 0.5, ln_gmp, stage2), p_keep


def predict_fclp(models, X, name):
    """ln(fclp) and P(analyst lowers the gmprocess value)."""
    base = X["ln_fclp_gmp"].to_numpy()
    if name == "simple":
        clf, reg = models["simple"]
        cols = _pipeline_columns(clf)
        p = _proba(clf, X[cols]) if cols else np.zeros(len(X))
        shift = reg.predict(X) if reg is not None else np.zeros(len(X))
    else:
        p = _proba(models["lightgbm_classifier"], X)
        r = models.get("lightgbm_shift")
        shift = r.predict(X) if r is not None else np.zeros(len(X))
    return base + p * np.minimum(shift, 0.0), p


def predict_accept(bundle, X, name):
    models = bundle["models"]
    if name == "simple":
        return _proba(models["simple"], X[bundle["simple_features"]])
    return _proba(models[name], X)


def predict_auto(df, report, name):
    """ln(fchp) from calibrated auto_fchp settings, evaluated from the stored curves."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import calibrate_auto_fchp as caf
    from gmcr import features as ft

    params = report["selected"][name]["all_data"]
    lng = np.log(ft.FCHP_GRID)
    need = [f"{fam}_{i:03d}" for fam in (params["family"], "pre_disp_ratio") for i in range(len(lng))]
    if any(c not in df for c in need):
        hint = " (extract with --fchp-variants, or --workspaces with --fchp-variants)" \
            if params["family"].startswith("polyvar") else ""
        sys.exit(f"the {params['family']} curves are missing{hint}")
    return caf.predict(df, params, name, lng), params


# --------------------------------------------------------------------------- main


def model_kind(model_dir):
    model_dir = Path(model_dir)
    report = json.loads((model_dir / "report.json").read_text())
    if "selected" in report and not (model_dir / "models.joblib").exists():
        return "auto_fchp", report, None
    import joblib

    bundle = joblib.load(model_dir / "models.joblib")
    kind = bundle.get("target") or ("accept" if "simple_features" in bundle else None)
    if kind is None:
        sys.exit(f"{model_dir}: not an output of train_corner_model.py or train_accept_model.py")
    return kind, report, bundle


def default_model(kind, report):
    if kind == "auto_fchp":
        rmse = {m: report["metrics"][m]["rmse_ln"] for m in report["selected"] if m in report["metrics"]}
        return min(rmse, key=rmse.get)
    return report["recommendation"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", help="output folder of a training or calibration run")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", help="folder with traces_joined.parquet or traces_all.parquet "
                                        "and curves_all.parquet")
    src.add_argument("--workspaces", nargs="+", help="workspace .h5 files or folders of them")
    p.add_argument("--out", required=True, help="predictions file (.csv or .parquet)")
    p.add_argument("--model", default=None, help="model to apply (default: the report's recommendation)")
    p.add_argument("--flag-share", type=int, choices=[50, 80, 90], default=None,
                   help="accept models: flag records with the threshold that flagged this percentage "
                        "of the analysts' rejections in cross-validation")
    p.add_argument("--fchp-variants", action="store_true",
                   help="with --workspaces: also extract the polyvar_* curves (needed by auto_fchp "
                        "calibrations that selected one of them)")
    args = p.parse_args()

    kind, report, bundle = model_kind(args.model_dir)
    name = args.model or default_model(kind, report)
    print(f"model: {kind} / {name}"
          + ("" if args.model else " (recommended)"), flush=True)

    if args.features:
        df = gm.load_traces(args.features, curves=True)
    else:
        df = features_from_workspaces(args.workspaces, args.fchp_variants)
    out = df[[c for c in ID_COLUMNS if c in df]].copy()
    out.insert(0, "model", f"{kind}/{name}")

    if kind == "auto_fchp":
        ln, params = predict_auto(df, report, name)
        out["pred_fchp_hz"] = np.exp(ln)
        print(f"settings: {params}")
    else:
        X, missing = predictors(df, bundle, include_site=(kind != "accept"))
        absent = [c for c in bundle["features"] if X[c].isna().all()]
        if absent:
            print(f"{len(absent)} predictors unavailable for these records (left missing): "
                  f"{', '.join(absent[:15])}{' ...' if len(absent) > 15 else ''}", flush=True)
        out["missing_predictors"] = X.isna().sum(axis=1).to_numpy()
        models = bundle["models"]
        if kind == "fchp":
            if name not in ("simple", "simple_2stage_soft", "simple_2stage_hard",
                            "lightgbm", "lightgbm_2stage_soft", "lightgbm_2stage_hard"):
                sys.exit(f"unknown fchp model {name}")
            ln, p_keep = predict_fchp(models, X, name)
            out["pred_fchp_hz"] = np.exp(ln)
            if np.isfinite(p_keep).any():
                out["p_keep_gmprocess"] = p_keep
        elif kind == "fclp":
            if name not in ("simple", "lightgbm"):
                sys.exit(f"unknown fclp model {name}")
            ln, p_low = predict_fclp(models, X, name)
            out["pred_fclp_hz"] = np.exp(ln)
            out["p_lowered"] = p_low
        else:
            if name not in ("simple", "lightgbm"):
                sys.exit(f"unknown accept model {name}")
            out["p_accept"] = predict_accept(bundle, X, name)
            if args.flag_share:
                op = report.get("operating_points", {}).get(name, {}).get(
                    f"flag_{args.flag_share}pct_of_rejections")
                if not op:
                    sys.exit(f"report.json has no {args.flag_share}% operating point for {name}")
                out["flag_reject"] = (1 - out["p_accept"]) >= op["p_reject_threshold"]
                print(f"flag_reject: P(reject) >= {op['p_reject_threshold']:.3f} (flagged "
                      f"{args.flag_share}% of rejections and {100 * op['accepted_records_lost']:.1f}% "
                      f"of accepted records in cross-validation)")
    for col, base in (("pred_fchp_hz", "fchp_gmp"), ("pred_fclp_hz", "fclp_gmp")):
        if col in out and base in out:
            out["ratio_to_gmprocess"] = out[col] / pd.to_numeric(out[base], errors="coerce")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    (out.to_parquet if path.suffix == ".parquet" else out.to_csv)(path, index=False)
    print(f"{len(out)} traces -> {path}")


if __name__ == "__main__":
    main()
