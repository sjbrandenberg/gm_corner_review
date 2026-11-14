"""Version-independent storage of trained models (no pickles).

models.joblib (written by the training scripts) is a Python pickle: it may not load with
other versions of scikit-learn, LightGBM, or numpy. For distribution, export_model_dir()
writes the same models as

    model.json        kind, predictors, and every simple model as plain numbers:
                      log-linear models (training medians, standardization, coefficients) and
                      logistic models (imputer medians, scaler mean/scale, coefficients)
    lightgbm_*.txt    LightGBM models in LightGBM's own text format (best iteration only),
                      which any LightGBM version reads
    report.json/.md   the training run's report (recommendation, metrics, coefficients)
    provenance.json   where and when the models were exported, code version, package versions

An auto_fchp calibration folder has no fitted models: its report.json (with the selected
settings) is exported as is.

load_model_dir() reads such a folder back into the same structure as models.joblib (a dict
with "models", "features", ...), with small classes that have the predict / predict_proba
methods predict.py uses, so the rest of predict.py works unchanged.

The bundled models of the package are in gmcr/trained/<name>/ (bundled_dirs()).
"""

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT_VERSION = 1
TRAINED = Path(__file__).resolve().parent / "trained"


# ------------------------------------------------------------------ portable models


class LogLinear:
    """gmcr.modeling.LogLinearModel without pickling: same predict()."""

    def __init__(self, spec):
        self.features = list(spec["features"])
        self.fill_ = pd.Series(spec["fill"], index=self.features, dtype=float)
        self.mu_ = pd.Series(spec["mean"], index=self.features, dtype=float)
        self.sd_ = pd.Series(spec["sd"], index=self.features, dtype=float)
        self.coef_ = np.asarray(spec["coef"], dtype=float)
        self.sigma_ = spec.get("sigma_ln")

    def predict(self, X):
        Z = X[self.features].astype(float).fillna(self.fill_)
        Z = (Z - self.mu_) / self.sd_
        return np.column_stack([np.ones(len(Z)), Z.to_numpy()]) @ self.coef_


class Logistic:
    """sklearn Pipeline(SimpleImputer(median), StandardScaler, LogisticRegression) without
    pickling: same predict_proba() and feature_names_in_."""

    def __init__(self, spec):
        self.feature_names_in_ = np.asarray(spec["inputs"], dtype=object)
        self.used = list(spec["used"])
        self.median = np.asarray(spec["median"], dtype=float)
        self.mean = np.asarray(spec["mean"], dtype=float)
        self.scale = np.asarray(spec["scale"], dtype=float)
        self.coef = np.asarray(spec["coef"], dtype=float)
        self.intercept = float(spec["intercept"])

    def predict_proba(self, X):
        Z = X[self.used].to_numpy(dtype=float)
        Z = np.where(np.isnan(Z), self.median, Z)
        z = ((Z - self.mean) / self.scale) @ self.coef + self.intercept
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])


class Unfitted:
    """A classifier that was never fitted (one class only in training)."""

    feature_names_in_ = None


class LightGBM:
    """A LightGBM model read from its text format: predict() for regression,
    predict_proba() for binary classification, as the sklearn wrappers."""

    def __init__(self, path, objective):
        import lightgbm as lgb

        self.booster = lgb.Booster(model_file=str(path))
        self.features = self.booster.feature_name()
        self.binary = objective == "binary"

    def _raw(self, X):
        return self.booster.predict(X[self.features].to_numpy(dtype=float))

    def predict(self, X):
        return self._raw(X)

    def predict_proba(self, X):
        p = self._raw(X)
        return np.column_stack([1 - p, p])


# ------------------------------------------------------------------------- export


def _loglinear_spec(m):
    return {"type": "loglinear", "features": list(m.features), "fill": m.fill_.tolist(),
            "mean": m.mu_.tolist(), "sd": m.sd_.tolist(), "coef": np.asarray(m.coef_).tolist(),
            "sigma_ln": float(m.sigma_)}


def _logistic_spec(pipe):
    if pipe is None or not hasattr(pipe[-1], "coef_"):
        return {"type": "unfitted"}
    imp, sc, lr = pipe[0], pipe[1], pipe[-1]
    inputs = [str(c) for c in pipe.feature_names_in_]
    keep = ~np.isnan(imp.statistics_)  # SimpleImputer drops columns that were all missing
    return {"type": "logistic", "inputs": inputs,
            "used": [c for c, k in zip(inputs, keep) if k],
            "median": imp.statistics_[keep].tolist(), "mean": sc.mean_.tolist(),
            "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0])}


def _lightgbm_spec(model, out, key):
    fname = f"lightgbm_{key}.txt"
    best = getattr(model, "best_iteration_", None) or None
    model.booster_.save_model(str(out / fname), num_iteration=best)
    return {"type": "lightgbm", "file": fname, "objective": model.objective_
            if isinstance(model.objective_, str) else str(model.get_params()["objective"])}


def _spec(key, model, out):
    cls = type(model).__name__
    if cls == "LogLinearModel":
        return _loglinear_spec(model)
    if cls == "Pipeline" or model is None:
        return _logistic_spec(model)
    if cls in ("LGBMRegressor", "LGBMClassifier"):
        return _lightgbm_spec(model, out, key)
    if isinstance(model, tuple):  # fclp simple: (classifier pipeline, log-linear shift or None)
        clf, reg = model
        return {"type": "fclp_simple", "classifier": _logistic_spec(clf),
                "shift": _loglinear_spec(reg) if reg is not None else None}
    raise TypeError(f"{key}: don't know how to export a {cls}")


# models each recommendation needs (to export only those with only_recommended=True)
NEEDS = {
    ("fchp", "simple"): ["simple"],
    ("fchp", "simple_2stage_soft"): ["simple", "simple_keep", "simple_adjusted"],
    ("fchp", "simple_2stage_hard"): ["simple", "simple_keep", "simple_adjusted"],
    ("fchp", "lightgbm"): ["lightgbm"],
    ("fchp", "lightgbm_2stage_soft"): ["lightgbm_keep", "lightgbm_adjusted"],
    ("fchp", "lightgbm_2stage_hard"): ["lightgbm_keep", "lightgbm_adjusted"],
    ("fclp", "simple"): ["simple"],
    ("fclp", "lightgbm"): ["lightgbm_classifier", "lightgbm_shift"],
    ("accept", "simple"): ["simple"],
    ("accept", "lightgbm"): ["lightgbm"],
}


def _git_commit(repo):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _versions():
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return out


def export_model_dir(src, dst, only_recommended=False, note=""):
    """Export a training/calibration output folder `src` to a portable folder `dst`.
    Returns the model.json content."""
    src, dst = Path(src), Path(dst)
    report = json.loads((src / "report.json").read_text())
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in ("report.json", "report.md"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    if "selected" in report and not (src / "models.joblib").exists():
        meta = {"format_version": FORMAT_VERSION, "kind": "auto_fchp",
                "settings": {m: report["selected"][m]["all_data"] for m in report["selected"]}}
    else:
        import joblib

        bundle = joblib.load(src / "models.joblib")
        kind = bundle.get("target") or ("accept" if "simple_features" in bundle else None)
        if kind is None:
            raise ValueError(f"{src}: not an output of the training scripts")
        keys = list(bundle["models"])
        rec = report.get("recommendation")
        if only_recommended:
            keys = [k for k in NEEDS.get((kind, rec), keys) if k in bundle["models"]]
        meta = {"format_version": FORMAT_VERSION, "kind": kind, "recommendation": rec,
                "features": list(bundle["features"]), "curves": bool(bundle.get("curves")),
                "dropped": bundle.get("dropped", report.get("dropped", [])),
                "simple_features": bundle.get("simple_features"),
                "models": {k: _spec(k, bundle["models"][k], dst) for k in keys}}
    (dst / "model.json").write_text(json.dumps(meta, indent=1))
    repo = Path(__file__).resolve().parents[1]
    prov = {"exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": str(src.resolve()), "gmcr_commit": _git_commit(repo),
            "packages_at_export": _versions(), "note": note,
            "training": {k: report.get(k) for k in ("n_traces", "n_events", "folds", "recommendation",
                                                     "dropped", "curves", "target") if k in report}}
    (dst / "provenance.json").write_text(json.dumps(prov, indent=1))
    return meta


# --------------------------------------------------------------------------- load


def is_portable(model_dir):
    return (Path(model_dir) / "model.json").exists()


def load_model_dir(model_dir):
    """(kind, report, bundle) of a portable folder; bundle has the layout of models.joblib
    (None for auto_fchp calibrations)."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "model.json").read_text())
    if meta.get("format_version", 1) > FORMAT_VERSION:
        raise ValueError(f"{model_dir}: model format {meta['format_version']} is newer than this gmcr; update it")
    report = json.loads((model_dir / "report.json").read_text())
    if meta["kind"] == "auto_fchp":
        return "auto_fchp", report, None
    models = {}
    for key, spec in meta["models"].items():
        t = spec["type"]
        if t == "loglinear":
            models[key] = LogLinear(spec)
        elif t == "logistic":
            models[key] = Logistic(spec)
        elif t == "unfitted":
            models[key] = Unfitted()
        elif t == "lightgbm":
            models[key] = LightGBM(model_dir / spec["file"], spec["objective"])
        elif t == "fclp_simple":
            clf = Logistic(spec["classifier"]) if spec["classifier"]["type"] == "logistic" else Unfitted()
            models[key] = (clf, LogLinear(spec["shift"]) if spec["shift"] else None)
        else:
            raise ValueError(f"{model_dir}: unknown model type {t}")
    bundle = {"models": models, "features": meta["features"], "target": meta["kind"]
              if meta["kind"] != "accept" else None, "curves": meta["curves"],
              "dropped": meta.get("dropped", [])}
    if meta.get("simple_features"):
        bundle["simple_features"] = meta["simple_features"]
    return meta["kind"], report, bundle


def bundled_dirs():
    """{name: folder} of the models bundled with the package (gmcr/trained/<name>/)."""
    if not TRAINED.exists():
        return {}
    return {p.name: p for p in sorted(TRAINED.iterdir()) if (p / "model.json").exists()}
