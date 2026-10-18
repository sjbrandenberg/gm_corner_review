"""Train and compare accept/reject models against the analysts' decisions.

    python scripts/train_accept_model.py FEATURES_DIR --out models/accept

Rows are the traces analysts reviewed. Every one of them passed the automatic
checks (analysts never saw auto-rejected records), so these models can only learn
which *passed* records analysts reject, i.e. how to tighten the criteria.
Predictors exclude st_* and ff_* (present in the database only for accepted
records) and analyst_id.

Models (out-of-fold, event-grouped cross-validation)
  baseline   constant acceptance rate (what the current pipeline implies)
  simple     L2 logistic regression on a short list of interpretable predictors
  lightgbm   gradient boosting on all safe scalar predictors (+ curves with --curves)

For each model the report gives AUC, Brier score, log loss, and the operating
points that would flag 50/80/90% of the analysts' rejections, with the share of
accepted records that would be lost at each. The ML model is recommended only if
its log loss is at least --min-gain lower than the simple model's and the event
bootstrap interval of the difference excludes zero.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmcr import modeling as gm  # noqa: E402

SIMPLE_ACCEPT = ["ln_snr_max", "ln_snr2_flo", "ln_snr2_fhi", "ln_fchp_gmp", "ln_fchp_gui",
                 "gui_at_cap", "pre_ratio_at_gmp", "poly_ratio_at_gmp", "psa_rough_short",
                 "psa_max_bump_short", "clipping_prob", "magnitude", "ln_rrup",
                 "accelerometer", "is_vertical"]

LGBM_CLF = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=20,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                n_estimators=3000, verbose=-1)


def _logloss_rows(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap_logloss_difference(y, pa, pb, groups, n_boot, seed):
    la, lb = _logloss_rows(y, pa), _logloss_rows(y, pb)
    uniq, inv = np.unique(groups, return_inverse=True)
    sa, sb = np.bincount(inv, la), np.bincount(inv, lb)
    n = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, len(uniq), len(uniq))
        d[i] = (sa[k].sum() - sb[k].sum()) / n[k].sum()
    point = (sa.sum() - sb.sum()) / n.sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"difference": float(point), "ci95": [float(lo), float(hi)],
            "relative_reduction": float(point / (sa.sum() / n.sum()))}


def operating_points(y, p_accept, recalls=(0.5, 0.8, 0.9)):
    """Thresholds on P(reject) = 1 - p_accept that flag a share of the rejections."""
    rej = 1 - np.asarray(y)
    q = 1 - np.asarray(p_accept)
    out = {}
    if rej.sum() == 0:
        return out
    for r in recalls:
        t = np.quantile(q[rej == 1], 1 - r)
        flag = q >= t
        out[f"flag_{int(r * 100)}pct_of_rejections"] = {
            "p_reject_threshold": float(t),
            "accepted_records_lost": float(flag[rej == 0].mean()) if (rej == 0).any() else float("nan"),
            "flagged_share": float(flag.mean()),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--curves", action="store_true")
    p.add_argument("--no-ml", action="store_true")
    p.add_argument("--drop", default="", help="comma-separated predictors to leave out")
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--boot", type=int, default=1000)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = gm.accept_dataset(gm.load_traces(args.features_dir, curves=args.curves)).reset_index(drop=True)
    X = gm.base_features(df, include_site=False)
    if args.curves:
        X = pd.concat([X, gm.curve_features(df)], axis=1)
    dropped = [c for c in args.drop.split(",") if c]
    if [c for c in dropped if c not in X]:
        sys.exit(f"--drop: unknown predictors {[c for c in dropped if c not in X]}")
    X = X.drop(columns=dropped)
    y = df["y"].to_numpy()
    groups = df["event_id"].to_numpy()
    n_events = len(np.unique(groups))
    if n_events < args.folds:
        sys.exit(f"{n_events} events: need at least --folds ({args.folds}) for event-grouped CV")
    if y.min() == y.max():
        sys.exit("all reviewed records have the same decision; nothing to learn")
    folds = gm.event_folds(groups, args.folds, args.seed)
    feats = [f for f in SIMPLE_ACCEPT if f in X and X[f].notna().any()]

    def simple():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=5000))

    oof = pd.DataFrame({"trace_key": df["trace_key"], "event_id": groups, "y": y,
                        "baseline": np.nan, "simple": np.nan, "lightgbm": np.nan})
    if not args.no_ml:
        from train_corner_model import _lgbm_fit  # noqa: E402  (same directory)
    for k in range(args.folds):
        tr, te = folds != k, folds == k
        if not te.any():
            continue
        oof.loc[te, "baseline"] = y[tr].mean()
        if 0 < y[tr].sum() < tr.sum():
            m = simple().fit(X.loc[tr, feats], y[tr])
            oof.loc[te, "simple"] = m.predict_proba(X.loc[te, feats])[:, 1]
            if not args.no_ml:
                g = _lgbm_fit(LGBM_CLF, X[tr], y[tr], groups[tr], args.seed + k)
                oof.loc[te, "lightgbm"] = g.predict_proba(X[te])[:, 1]
        else:
            oof.loc[te, "simple"] = y[tr].mean()

    models = [m for m in ("baseline", "simple", "lightgbm") if oof[m].notna().any()]
    report = {"n_traces": int(len(df)), "n_events": int(n_events), "folds": args.folds, "dropped": dropped,
              "accept_rate": float(y.mean()), "metrics": {}, "operating_points": {}, "comparisons": {}}
    for m in models:
        report["metrics"][m] = gm.classification_metrics(y, oof[m])
        report["operating_points"][m] = operating_points(y, oof[m].to_numpy())
    for a, b in [("baseline", "simple"), ("simple", "lightgbm")]:
        if a in models and b in models:
            report["comparisons"][f"{a}_vs_{b}"] = bootstrap_logloss_difference(
                y, oof[a].to_numpy(), oof[b].to_numpy(), groups, args.boot, args.seed)
    cmp = report["comparisons"].get("simple_vs_lightgbm")
    report["recommendation"] = ("lightgbm" if cmp and cmp["relative_reduction"] >= args.min_gain
                                and cmp["ci95"][0] > 0 else "simple")

    final = {"simple": simple().fit(X[feats], y)}
    lr = final["simple"][-1]
    report["simple_coefficients_standardized"] = dict(zip(feats, map(float, lr.coef_[0])))
    if not args.no_ml:
        final["lightgbm"] = _lgbm_fit(LGBM_CLF, X, y, groups, args.seed)
        imp = pd.Series(final["lightgbm"].feature_importances_, index=X.columns)
        report["lightgbm_top_features"] = imp.sort_values(ascending=False).head(25).to_dict()
        from train_corner_model import _importance  # noqa: E402
        report["importance_lightgbm"] = _importance(final["lightgbm"], X.columns)
    if "analyst_id" in df and df["analyst_id"].notna().any():
        from sklearn.metrics import roc_auc_score
        by = {}
        a = pd.to_numeric(df["analyst_id"], errors="coerce").fillna(-1).to_numpy()
        for aid in pd.unique(a):
            sel = a == aid
            row = {"n_traces": int(sel.sum()), "n_events": int(len(np.unique(groups[sel]))),
                   "accept_rate": round(float(y[sel].mean()), 4)}
            for m in models:
                pm = np.clip(oof.loc[sel, m].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
                row[f"{m}_log_loss"] = round(float(np.mean(_logloss_rows(y[sel], pm))), 4)
                row[f"{m}_auc"] = (round(float(roc_auc_score(y[sel], pm)), 4)
                                   if 0 < y[sel].sum() < sel.sum() else None)
            by[str(int(aid)) if aid != -1 else "unknown"] = row
        report["by_analyst"] = by
    report["seconds"] = round(time.time() - t0, 1)

    import joblib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    joblib.dump({"models": final, "simple_features": feats, "features": list(X.columns),
                 "curves": args.curves}, out / "models.joblib")
    oof.to_parquet(out / "oof_predictions.parquet", index=False)
    gm.write_json(report, out / "report.json")

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    bins = np.linspace(0, 1, 11)
    for m in models:
        idx = np.clip(np.digitize(oof[m], bins) - 1, 0, 9)
        obs = pd.Series(y).groupby(idx).mean()
        pred = oof[m].groupby(idx).mean()
        ax.plot(pred, obs, "o-", label=m)
    ax.plot([0, 1], [0, 1], "k-", lw=0.8)
    ax.set_xlabel("predicted P(accept)")
    ax.set_ylabel("observed accept rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=150)
    plt.close(fig)

    lines = ["# accept/reject model comparison", "",
             f"{len(df)} reviewed traces, {n_events} events, accept rate {y.mean():.3f}"
             + (f"; without {', '.join(dropped)}" if dropped else ""), "",
             "| model | AUC | Brier | log loss |", "|---|---|---|---|"]
    for m in models:
        r = report["metrics"][m]
        lines.append(f"| {m} | {r['auc']:.3f} | {r['brier']:.4f} | {r['log_loss']:.4f} |")
    lines += ["", "Accepted records lost when flagging a share of the analysts' rejections:", ""]
    for m in models:
        for k, v in report["operating_points"][m].items():
            lines.append(f"- {m}, {k}: {100 * v['accepted_records_lost']:.1f}% of accepted records flagged")
    lines += ["", "Comparisons (log loss difference, event bootstrap 95% CI):", ""]
    for k, v in report["comparisons"].items():
        lines.append(f"- {k}: {v['difference']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    if report.get("by_analyst"):
        show = [m for m in ("simple", "lightgbm") if m in models]
        lines += ["", "By analyst (AUC / log loss):", "",
                  "| analyst | events | traces | accept rate | " + " | ".join(show) + " |",
                  "|---|---|---|---|" + "---|" * len(show)]
        for aid, r in sorted(report["by_analyst"].items(), key=lambda kv: -kv[1]["n_traces"]):
            cells = [(f"{r[m + '_auc']:.3f}" if r[m + "_auc"] is not None else "-") + f" / {r[m + '_log_loss']:.3f}"
                     for m in show]
            lines.append(f"| {aid} | {r['n_events']} | {r['n_traces']} | {r['accept_rate']:.3f} | "
                         + " | ".join(cells) + " |")
    if "importance_lightgbm" in report:
        lines += ["", "Top LightGBM predictors (share of total gain):", ""]
        lines += [f"- {k}: {100 * v:.1f}%" for k, v in list(report["importance_lightgbm"]["gain_share"].items())[:12]]
    lines += ["", f"Recommendation: **{report['recommendation']}**"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
