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

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated")  # LightGBM >= 4.6

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmcr import modeling as gm  # noqa: E402

SIMPLE_FCHP = ["ln_fchp_gmp", "ln_fchp_gui", "gui_at_cap", "method_polyfit", "magnitude", "ln_rrup",
               "accelerometer", "is_vertical", "ln_snr2_flo", "pre_ratio_at_gmp"]
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


def _logistic():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=5000))


def run_fchp(df, X, folds, args):
    """Direct models of ln(fchp) and keep-or-adjust (two-stage) models.

    Two-stage: stage 1 predicts p = P(analyst kept the gmprocess value); stage 2 predicts
    ln(fchp) of the traces the analyst adjusted (fit on those traces only). Two ways to
    combine them: "soft" = p ln(fchp_gmp) + (1 - p) stage2, and "hard" = fchp_gmp if
    p >= 0.5, else stage2.
    """
    y = df["y"].to_numpy()
    cens = df["censored"].to_numpy()
    groups = df["event_id"].to_numpy()
    base = X["ln_fchp_gmp"].to_numpy()
    keep = np.isclose(df["y_fchp"].astype(float), df["fchp_gmp"].astype(float), rtol=1e-4).astype(int)
    cols = ["simple", "lightgbm", "simple_2stage_soft", "simple_2stage_hard",
            "lightgbm_2stage_soft", "lightgbm_2stage_hard", "p_keep_simple", "p_keep_lightgbm"]
    oof = pd.DataFrame({"trace_key": df["trace_key"], "event_id": groups, "y": y,
                        "censored": cens, "kept_gmprocess": keep,
                        "baseline_gmprocess": base,
                        "baseline_gui": X["ln_fchp_gui"].to_numpy() if "ln_fchp_gui" in X else np.nan,
                        **{c: np.nan for c in cols}})
    simple_feats = [f for f in SIMPLE_FCHP if f in X and X[f].notna().any()]

    def combine(p, stage2, rows):
        return p * base[rows] + (1 - p) * stage2, np.where(p >= 0.5, base[rows], stage2)

    for k in range(args.folds):
        tr, te = folds != k, folds == k
        if not te.any():
            continue
        adj = tr & (keep == 0)
        m = gm.LogLinearModel(simple_feats).fit(X[tr], y[tr], cens[tr])
        oof.loc[te, "simple"] = m.predict(X[te])
        if 0 < keep[tr].sum() < tr.sum():
            c = _logistic().fit(X.loc[tr, simple_feats], keep[tr])
            r = gm.LogLinearModel(simple_feats).fit(X[adj], y[adj], cens[adj])
            p = c.predict_proba(X.loc[te, simple_feats])[:, 1]
            soft, hard = combine(p, r.predict(X[te]), te)
            oof.loc[te, ["p_keep_simple", "simple_2stage_soft", "simple_2stage_hard"]] = np.c_[p, soft, hard]
        if not args.no_ml:
            g = _lgbm_fit(LGBM_REG, X[tr], y[tr], groups[tr], args.seed + k)
            oof.loc[te, "lightgbm"] = _predict_lgbm(g, X[te])
            if 0 < keep[tr].sum() < tr.sum():
                c = _lgbm_fit(LGBM_CLF, X[tr], keep[tr], groups[tr], args.seed + k)
                r = _lgbm_fit(LGBM_REG, X[adj], y[adj], groups[adj], args.seed + k)
                p = _predict_lgbm(c, X[te])
                soft, hard = combine(p, _predict_lgbm(r, X[te]), te)
                oof.loc[te, ["p_keep_lightgbm", "lightgbm_2stage_soft", "lightgbm_2stage_hard"]] = \
                    np.c_[p, soft, hard]
    adj = keep == 0
    both = 0 < keep.sum() < len(keep)
    final = {"simple": gm.LogLinearModel(simple_feats).fit(X, y, cens)}
    if both:
        final["simple_keep"] = _logistic().fit(X[simple_feats], keep)
        final["simple_adjusted"] = gm.LogLinearModel(simple_feats).fit(X[adj], y[adj], cens[adj])
    if not args.no_ml:
        final["lightgbm"] = _lgbm_fit(LGBM_REG, X, y, groups, args.seed)
        if both:
            final["lightgbm_keep"] = _lgbm_fit(LGBM_CLF, X, keep, groups, args.seed)
            final["lightgbm_adjusted"] = _lgbm_fit(LGBM_REG, X[adj], y[adj], groups[adj], args.seed)
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

    feats = [f for f in SIMPLE_FCLP if f in X and X[f].notna().any()]

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


def _importance(model, columns, n=25):
    """Split counts and share of total gain of a fitted LightGBM model, top n by gain."""
    gain = pd.Series(model.booster_.feature_importance("gain"), index=columns)
    split = pd.Series(model.booster_.feature_importance("split"), index=columns)
    top = gain.sort_values(ascending=False).head(n).index
    return {"gain_share": (gain[top] / gain.sum()).round(4).to_dict(),
            "split": split[top].astype(int).to_dict()}


def _by_analyst(df, oof, models, y="y"):
    """RMSE and bias (ln) of each model per analyst (flatfile user_id of the event)."""
    if "analyst_id" not in df or df["analyst_id"].isna().all():
        return {}
    out = {}
    a = pd.to_numeric(df["analyst_id"], errors="coerce").fillna(-1).to_numpy()
    for aid in pd.unique(a):
        sel = a == aid
        row = {"n_traces": int(sel.sum()), "n_events": int(oof.loc[sel, "event_id"].nunique())}
        for m in models:
            e = (oof.loc[sel, m] - oof.loc[sel, y]).dropna()
            row[m] = round(float(np.sqrt(np.mean(e ** 2))), 4) if len(e) else None
            row[m + "_bias"] = round(float(e.mean()), 4) if len(e) else None
        out[str(int(aid)) if aid != -1 else "unknown"] = row
    return out


SIMPLE_FAMILY = ("simple", "simple_2stage_soft", "simple_2stage_hard")
ML_FAMILY = ("lightgbm", "lightgbm_2stage_soft", "lightgbm_2stage_hard")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features_dir")
    p.add_argument("--target", choices=["fchp", "fclp"], default="fchp")
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--curves", action="store_true", help="add the fixed-grid curves to the ML model")
    p.add_argument("--no-ml", action="store_true", help="skip LightGBM")
    p.add_argument("--drop", default="",
                   help="comma-separated predictors to leave out, e.g. magnitude,depth_km "
                        "(event-level features, which can stand in for the analyst)")
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
    dropped = [c for c in args.drop.split(",") if c]
    unknown = [c for c in dropped if c not in X]
    if unknown:
        sys.exit(f"--drop: unknown predictors {unknown}")
    X = X.drop(columns=dropped)
    n_events = df["event_id"].nunique()
    if n_events < args.folds:
        sys.exit(f"{n_events} events: need at least --folds ({args.folds}) for event-grouped CV")
    folds = gm.event_folds(df["event_id"], args.folds, args.seed)

    oof, final = (run_fchp if args.target == "fchp" else run_fclp)(df, X, folds, args)
    groups = oof["event_id"].to_numpy()
    candidates = ("baseline_gmprocess", "baseline_gui") + SIMPLE_FAMILY + ML_FAMILY
    models = [c for c in candidates if c in oof and oof[c].notna().any()]
    report = {"target": args.target, "n_traces": int(len(df)), "n_events": int(n_events),
              "folds": args.folds, "curves": args.curves, "dropped": dropped,
              "metrics": {}, "comparisons": {}}
    for m in models:
        report["metrics"][m] = gm.regression_metrics(oof["y"], oof[m])
    if args.target == "fchp":
        report["censored_fraction"] = float(oof["censored"].mean())
        report["kept_gmprocess_fraction"] = float(oof["kept_gmprocess"].mean())
        unc = oof["censored"] == 0
        report["metrics_uncensored"] = {m: gm.regression_metrics(oof.loc[unc, "y"], oof.loc[unc, m])
                                        for m in models}
        for m in ("simple", "lightgbm"):
            if oof[f"p_keep_{m}"].notna().any():
                report["metrics"][f"keep_classifier_{m}"] = gm.classification_metrics(
                    oof["kept_gmprocess"], oof[f"p_keep_{m}"])
    else:
        report["lowered_fraction"] = float(oof["lowered"].mean())
        for m in ("simple", "lightgbm"):
            if oof[f"p_{m}"].notna().any():
                report["metrics"][f"lowered_classifier_{m}"] = gm.classification_metrics(
                    oof["lowered"], oof[f"p_{m}"])
    report["by_analyst"] = _by_analyst(df, oof, [m for m in models if m != "baseline_gui"])

    base = "baseline_gmprocess"
    for m in models:
        if m != base:
            report["comparisons"][f"{base}_vs_{m}"] = gm.bootstrap_rmse_difference(
                oof["y"], oof[base], oof[m], groups, args.boot, args.seed)
    rmse = {m: report["metrics"][m]["rmse_ln"] for m in models}
    best_simple = min((m for m in SIMPLE_FAMILY if m in rmse), key=rmse.get)
    best_ml = min((m for m in ML_FAMILY if m in rmse), key=rmse.get, default=None)
    report["best_simple"], report["best_ml"] = best_simple, best_ml
    if best_ml:
        cmp = gm.bootstrap_rmse_difference(oof["y"], oof[best_simple], oof[best_ml], groups,
                                           args.boot, args.seed)
        report["comparisons"][f"{best_simple}_vs_{best_ml}"] = cmp
        report["recommendation"] = (best_ml if cmp["relative_reduction"] >= args.min_gain
                                    and cmp["ci95"][0] > 0 else best_simple)
    else:
        report["recommendation"] = best_simple
    if "simple" in final and hasattr(final["simple"], "coefficients"):
        report["simple_coefficients"] = final["simple"].coefficients()
    elif "simple" in final and final["simple"][1] is not None:
        report["simple_shift_coefficients"] = final["simple"][1].coefficients()
    for name in ("lightgbm", "lightgbm_keep", "lightgbm_adjusted", "lightgbm_classifier", "lightgbm_shift"):
        if name in final:
            report[f"importance_{name}"] = _importance(final[name], X.columns)
    if "lightgbm" in final:  # split counts, as in earlier reports
        imp = pd.Series(final["lightgbm"].feature_importances_, index=X.columns)
        report["lightgbm_top_features"] = imp.sort_values(ascending=False).head(25).to_dict()
    report["seconds"] = round(time.time() - t0, 1)

    import joblib

    joblib.dump({"models": final, "features": list(X.columns), "target": args.target,
                 "curves": args.curves, "dropped": dropped}, out / "models.joblib")
    oof.to_parquet(out / "oof_predictions.parquet", index=False)
    gm.write_json(report, out / "report.json")
    figures(oof, [m for m in models if m in ("baseline_gmprocess", best_simple, best_ml)], out, args.target)

    lines = [f"# {args.target} model comparison", "",
             f"{len(df)} traces, {n_events} events, {args.folds}-fold event-grouped CV"
             + (f"; without {', '.join(dropped)}" if dropped else ""), ""]
    if args.target == "fchp":
        lines += [f"Analyst kept the gmprocess value for {100 * report['kept_gmprocess_fraction']:.1f}% "
                  "of these traces.", ""]
    lines += ["| model | RMSE ln | bias ln | within 20% | within x1.5 |", "|---|---|---|---|---|"]
    for m in models:
        r = report["metrics"][m]
        lines.append(f"| {m} | {r['rmse_ln']:.3f} | {r['bias_ln']:+.3f} | {r['within_20pct']:.2f} | "
                     f"{r['within_factor_1.5']:.2f} |")
    for m in ("simple", "lightgbm"):
        k = report["metrics"].get(f"keep_classifier_{m}")
        if k:
            lines += ["", f"Keep-gmprocess classifier ({m}): AUC {k['auc']:.3f}, Brier {k['brier']:.4f}"]
    lines += ["", "Comparisons (RMSE difference, ln units, event bootstrap 95% CI):", ""]
    for k, v in report["comparisons"].items():
        lines.append(f"- {k}: {v['difference']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], "
                     f"relative {100 * v['relative_reduction']:.1f}%")
    if report["by_analyst"]:
        show = [m for m in ("baseline_gmprocess", best_simple, best_ml) if m]
        lines += ["", "RMSE (ln) by analyst:", "",
                  "| analyst | events | traces | " + " | ".join(show) + " |",
                  "|---|---|---|" + "---|" * len(show)]
        for aid, r in sorted(report["by_analyst"].items(), key=lambda kv: -kv[1]["n_traces"]):
            lines.append(f"| {aid} | {r['n_events']} | {r['n_traces']} | "
                         + " | ".join(f"{r[m]:.3f}" if r.get(m) is not None else "" for m in show) + " |")
    imp_key = f"importance_{best_ml.split('_2stage')[0]}" if best_ml else None
    if imp_key in report:
        lines += ["", "Top LightGBM predictors (share of total gain):", ""]
        lines += [f"- {k}: {100 * v:.1f}%" for k, v in list(report[imp_key]["gain_share"].items())[:12]]
    lines += ["", f"Best simple: {best_simple}; best ML: {best_ml}.",
              f"Recommendation: **{report['recommendation']}** (ML needs >= "
              f"{100 * args.min_gain:.0f}% lower RMSE than the best simple model, with a CI excluding 0)"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
