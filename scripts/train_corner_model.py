"""Train and compare corner-frequency models against the analysts' choices.

    python scripts/train_corner_model.py FEATURES_DIR --target fchp --out models/fchp
    python scripts/train_corner_model.py FEATURES_DIR --target fclp --out models/fclp

FEATURES_DIR holds traces_joined.parquet (or traces_all.parquet) and
curves_all.parquet. All models are scored by out-of-fold predictions from
cross-validation that holds out whole events, on ln(frequency).

fchp (high-pass) models
  baseline_gmprocess  ln fchp_gmp (current gmprocess + Ramos-Sepulveda value)
  baseline_gui        ln fchp_gui (the review GUI's automatic value)
  simple              log-linear model (gmcr.modeling.LogLinearModel) on a short list
                      of interpretable predictors; analyst picks at the GUI's 1.2 Hz
                      cap are treated as right-censored (Tobit)
  lightgbm            gradient boosting on all scalar predictors (+ curves with --curves)

fclp (low-pass) models: analysts mostly keep the gmprocess value, so each model
predicts the probability p that the analyst lowered it and ln(fclp / fclp_gmp) when
lowered; the prediction is fclp_gmp * exp(p * shift).

The ML model is recommended over the simple one only if it lowers the RMSE by at
least --min-gain (default 10%) and the 95% interval of the event-bootstrap
difference excludes zero.

Outputs in --out: report.json, report.md, oof_predictions.parquet, the fitted
models (models.joblib, refit on all data), and figures (*.png).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmcr import modeling as gm  # noqa: E402

SIMPLE_FCHP = ["ln_fchp_gmp", "ln_fchp_gui", "gui_at_cap", "method_polyfit", "magnitude", "ln_rrup",
               "accelerometer", "is_vertical", "ln_snr3_flo", "pre_ratio_at_gmp"]
SIMPLE_FCLP = ["ln_fclp_gmp", "magnitude", "ln_rrup", "accelerometer", "is_vertical",
               "ln_snr3_fhi", "psa_rough_short", "psa_max_bump_short"]

LGBM_REG = dict(objective="regression", learning_rate=0.03, num_leaves=31,
                min_child_samples=20, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, n_estimators=3000, verbose=-1)
LGBM_CLF = dict(LGBM_REG, objective="binary")


def _lgbm_fit(params, X, y, groups, seed):
    """LightGBM with early stopping on 15% of the training events."""
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    ev = np.unique(groups)
    val_ev = rng.choice(ev, max(1, int(0.15 * len(ev))), replace=False)
    val = np.isin(groups, val_ev)
    if val.all() or not val.any():
        val = rng.random(len(y)) < 0.15
    cls = lgb.LGBMClassifier if params["objective"] == "binary" else lgb.LGBMRegressor
    model = cls(**params, random_state=seed)
    model.fit(X[~val], y[~val], eval_set=[(X[val], y[val])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def _predict_lgbm(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def run_fchp(df, X, folds, args):
    y = df["y"].to_numpy()
    cens = df["censored"].to_numpy()
    groups = df["event_id"].to_numpy()
    oof = pd.DataFrame({"trace_key": df["trace_key"], "event_id": groups, "y": y,
                        "censored": cens,
                        "baseline_gmprocess": X["ln_fchp_gmp"].to_numpy(),
                        "baseline_gui": X["ln_fchp_gui"].to_numpy(),
                        "simple": np.nan, "lightgbm": np.nan})
    simple_feats = [f for f in SIMPLE_FCHP if X[f].notna().any()]
    for k in range(args.folds):
        tr, te = folds != k, folds == k
        if not te.any():
            continue
        m = gm.LogLinearModel(simple_feats).fit(X[tr], y[tr], cens[tr])
        oof.loc[te, "simple"] = m.predict(X[te])
        if not args.no_ml:
            g = _lgbm_fit(LGBM_REG, X[tr], y[tr], groups[tr], args.seed + k)
            oof.loc[te, "lightgbm"] = _predict_lgbm(g, X[te])
    final = {"simple": gm.LogLinearModel(simple_feats).fit(X, y, cens)}
    if not args.no_ml:
        final["lightgbm"] = _lgbm_fit(LGBM_REG, X, y, groups, args.seed)
    return oof, final


def run_fclp(df, X, folds, args):
    groups = df["event_id"].to_numpy()
    base = X["ln_fclp_gmp"].to_numpy()
    lowered = df["lowered"].to_numpy()
    shift = df["y"].to_numpy() - base  # <= 0 when lowered, 0 otherwise
    oof = pd.DataFrame({"trace_key": df["trace_key"], "event_id": groups,
                        "y": df["y"].to_numpy(), "lowered": lowered,
                        "baseline_gmprocess": base, "simple": np.nan, "lightgbm": np.nan,
                        "p_simple": np.nan, "p_lightgbm": np.nan})
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    feats = [f for f in SIMPLE_FCLP if X[f].notna().any()]

    def simple_fit(Xa, la, sa):
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(Xa[feats], la) if 0 < la.sum() < len(la) else None
        reg = gm.LogLinearModel(feats, censored=False).fit(Xa[la == 1], sa[la == 1]) \
            if la.sum() >= len(feats) + 2 else None
        return clf, reg

    def simple_predict(model, Xb, la_train):
        clf, reg = model
        p = clf.predict_proba(Xb[feats])[:, 1] if hasattr(clf[-1], "coef_") else \
            np.full(len(Xb), la_train.mean())
        s = reg.predict(Xb) if reg is not None else np.zeros(len(Xb))
        return p, np.minimum(s, 0.0)

    for k in range(args.folds):
        tr, te = folds != k, folds == k
        if not te.any():
            continue
        model = simple_fit(X[tr], lowered[tr], shift[tr])
        p, s = simple_predict(model, X[te], lowered[tr])
        oof.loc[te, "p_simple"] = p
        oof.loc[te, "simple"] = base[te] + p * s
        if not args.no_ml and 0 < lowered[tr].sum() < tr.sum():
            c = _lgbm_fit(LGBM_CLF, X[tr], lowered[tr], groups[tr], args.seed + k)
            low = tr & (lowered == 1)
            r = _lgbm_fit(LGBM_REG, X[low], shift[low], groups[low], args.seed + k) \
                if low.sum() >= 50 else None
            p = _predict_lgbm(c, X[te])
            s = np.minimum(r.predict(X[te]), 0.0) if r is not None else np.zeros(te.sum())
            oof.loc[te, "p_lightgbm"] = p
            oof.loc[te, "lightgbm"] = base[te] + p * s
    final = {"simple": simple_fit(X, lowered, shift)}
    if not args.no_ml and 0 < lowered.sum() < len(lowered):
        final["lightgbm_classifier"] = _lgbm_fit(LGBM_CLF, X, lowered, groups, args.seed)
        low = lowered == 1
        if low.sum() >= 50:
            final["lightgbm_shift"] = _lgbm_fit(LGBM_REG, X[low], shift[low], groups[low], args.seed)
    return oof, final


def figures(oof, models, out, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lim = np.exp(np.nanpercentile(oof["y"], [0.5, 99.5]))
    for ax, m in zip(axes, models):
        ax.loglog(np.exp(oof["y"]), np.exp(oof[m]), ".", ms=2, alpha=0.3)
        ax.plot(lim, lim, "k-", lw=0.8)
        ax.set_title(m)
        ax.set_xlabel(f"analyst {target} (Hz)")
    axes[0].set_ylabel(f"predicted {target} (Hz)")
    fig.tight_layout()
    fig.savefig(out / "predicted_vs_analyst.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features_dir")
    p.add_argument("--target", choices=["fchp", "fclp"], default="fchp")
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--curves", action="store_true", help="add the fixed-grid curves to the ML model")
    p.add_argument("--no-ml", action="store_true", help="skip LightGBM")
    p.add_argument("--min-gain", type=float, default=0.10)
    p.add_argument("--boot", type=int, default=1000)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = gm.load_traces(args.features_dir, curves=args.curves)
    df = gm.fchp_dataset(df) if args.target == "fchp" else gm.fclp_dataset(df)
    df = df.reset_index(drop=True)
    X = gm.base_features(df, include_site=True)
    if args.curves:
        X = pd.concat([X, gm.curve_features(df)], axis=1)
    n_events = df["event_id"].nunique()
    if n_events < args.folds:
        sys.exit(f"{n_events} events: need at least --folds ({args.folds}) for event-grouped CV")
    folds = gm.event_folds(df["event_id"], args.folds, args.seed)

    oof, final = (run_fchp if args.target == "fchp" else run_fclp)(df, X, folds, args)
    groups = oof["event_id"].to_numpy()
    models = [c for c in ("baseline_gmprocess", "baseline_gui", "simple", "lightgbm")
              if c in oof and oof[c].notna().any()]
    report = {"target": args.target, "n_traces": int(len(df)), "n_events": int(n_events),
              "folds": args.folds, "curves": args.curves, "metrics": {}, "comparisons": {}}
    for m in models:
        report["metrics"][m] = gm.regression_metrics(oof["y"], oof[m])
    if args.target == "fchp":
        report["censored_fraction"] = float(oof["censored"].mean())
        unc = oof["censored"] == 0
        report["metrics_uncensored"] = {m: gm.regression_metrics(oof.loc[unc, "y"], oof.loc[unc, m])
                                        for m in models}
        if "analyst_id" in df:
            res = oof["simple"] - oof["y"]
            report["simple_residual_by_analyst"] = (
                res.groupby(df["analyst_id"]).agg(["mean", "std", "size"]).round(4).to_dict("index"))
    else:
        report["lowered_fraction"] = float(oof["lowered"].mean())
        for m in ("simple", "lightgbm"):
            if oof[f"p_{m}"].notna().any():
                report["metrics"][f"lowered_classifier_{m}"] = gm.classification_metrics(
                    oof["lowered"], oof[f"p_{m}"])
    base = "baseline_gmprocess"
    for m in models:
        if m != base:
            report["comparisons"][f"{base}_vs_{m}"] = gm.bootstrap_rmse_difference(
                oof["y"], oof[base], oof[m], groups, args.boot, args.seed)
    if "lightgbm" in models:
        cmp = gm.bootstrap_rmse_difference(oof["y"], oof["simple"], oof["lightgbm"], groups,
                                           args.boot, args.seed)
        report["comparisons"]["simple_vs_lightgbm"] = cmp
        report["recommendation"] = ("lightgbm" if cmp["relative_reduction"] >= args.min_gain
                                    and cmp["ci95"][0] > 0 else "simple")
    else:
        report["recommendation"] = "simple"
    if "simple" in final and hasattr(final["simple"], "coefficients"):
        report["simple_coefficients"] = final["simple"].coefficients()
    elif "simple" in final and final["simple"][1] is not None:
        report["simple_shift_coefficients"] = final["simple"][1].coefficients()
    if "lightgbm" in final:
        imp = pd.Series(final["lightgbm"].feature_importances_, index=X.columns)
        report["lightgbm_top_features"] = imp.sort_values(ascending=False).head(25).to_dict()
    report["seconds"] = round(time.time() - t0, 1)

    import joblib

    joblib.dump({"models": final, "features": list(X.columns), "target": args.target,
                 "curves": args.curves}, out / "models.joblib")
    oof.to_parquet(out / "oof_predictions.parquet", index=False)
    gm.write_json(report, out / "report.json")
    figures(oof, models, out, args.target)
    lines = [f"# {args.target} model comparison", "",
             f"{len(df)} traces, {n_events} events, {args.folds}-fold event-grouped CV", "",
             "| model | RMSE ln | bias ln | within 20% | within x1.5 |", "|---|---|---|---|---|"]
    for m in models:
        r = report["metrics"][m]
        lines.append(f"| {m} | {r['rmse_ln']:.3f} | {r['bias_ln']:+.3f} | {r['within_20pct']:.2f} | "
                     f"{r['within_factor_1.5']:.2f} |")
    lines += ["", "Comparisons (RMSE difference, ln units, event bootstrap 95% CI):", ""]
    for k, v in report["comparisons"].items():
        lines.append(f"- {k}: {v['difference']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], "
                     f"relative {100 * v['relative_reduction']:.1f}%")
    lines += ["", f"Recommendation: **{report['recommendation']}** (ML needs >= "
                  f"{100 * args.min_gain:.0f}% lower RMSE with a CI excluding 0)"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
