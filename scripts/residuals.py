"""Residuals of the corner-frequency models: ln(analyst corner) - ln(predicted corner).

    python scripts/residuals.py $WORK/gmcr_results/models $WORK/gmcr_results/traces_joined.parquet

For every corner-frequency run in the models folder (fchp, fchp_curves, fclp, ... : folders
with oof_predictions.parquet) and every model in it, uses the out-of-fold predictions
(event-grouped CV) and writes to <models>/residuals/:

  residuals.json   mean, standard deviation and quantiles of the residuals; the split into
                   between-event (sd of the event means, tau) and within-event (phi) parts;
                   a histogram on fixed bins; and the mean and sd of the residuals in decile
                   bins of the predicted corner, the gmprocess corner, magnitude, ln Rrup
                   and ln Vs30
  <run>.png        histogram and residual-versus-predictor plots, simple vs LightGBM

Positive residuals mean the analyst chose a higher corner than predicted. For fchp picks
censored at the GUI cap (1.2 Hz) the residual is a lower bound; they are counted separately.
Takes a few seconds on one core.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("baseline_gmprocess", "baseline_gui", "simple", "simple_2stage_soft", "simple_2stage_hard",
          "lightgbm", "lightgbm_2stage_soft", "lightgbm_2stage_hard")
HIST_EDGES = np.round(np.arange(-3.0, 3.0001, 0.1), 2)


def _stats(r):
    r = r[np.isfinite(r)]
    if not len(r):
        return None
    q = np.quantile(r, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(r)), "mean": round(float(r.mean()), 4), "sd": round(float(r.std(ddof=1)), 4),
            "q05": round(q[0], 4), "q25": round(q[1], 4), "median": round(q[2], 4),
            "q75": round(q[3], 4), "q95": round(q[4], 4)}


def _event_split(r, events):
    d = pd.DataFrame({"r": r, "e": events}).dropna()
    em = d.groupby("e")["r"].transform("mean")
    return {"tau_between_event": round(float(d.groupby("e")["r"].mean().std(ddof=1)), 4),
            "phi_within_event": round(float((d["r"] - em).std(ddof=1)), 4),
            "n_events": int(d["e"].nunique())}


def _binned(x, r, nbins=10):
    d = pd.DataFrame({"x": x, "r": r}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 50 or d["x"].nunique() < 3:
        return None
    d["b"] = pd.qcut(d["x"], nbins, duplicates="drop")
    g = d.groupby("b", observed=True)
    return {"x_median": g["x"].median().round(4).tolist(), "mean": g["r"].mean().round(4).tolist(),
            "sd": g["r"].std(ddof=1).round(4).tolist(), "n": g["r"].size().astype(int).tolist()}


def _covariates(traces):
    t = traces.drop_duplicates("trace_key").set_index("trace_key")

    def col(*names):
        for n in names:
            if n in t:
                return pd.to_numeric(t[n], errors="coerce")
        return pd.Series(np.nan, index=t.index)
    pos = lambda s: np.log(s.where(s > 0))
    return pd.DataFrame({"ln_fchp_gmp": pos(col("fchp_gmp")), "ln_fclp_gmp": pos(col("fclp_gmp")),
                         "magnitude": col("ev_magnitude", "magnitude"),
                         "ln_rrup": pos(col("d_rrup", "rhypo_km")), "ln_vs30": pos(col("st_vs30"))})


def run(oof, cov, target):
    d = oof.merge(cov, left_on="trace_key", right_index=True, how="left")
    unc = (d["censored"] == 0).to_numpy() if "censored" in d else np.ones(len(d), bool)
    gmp = "ln_fchp_gmp" if target == "fchp" else "ln_fclp_gmp"
    out = {"target": target, "n_traces": int(len(d)), "n_censored": int((~unc).sum()), "models": {}}
    for m in MODELS:
        if m not in d or d[m].isna().all():
            continue
        r = (d["y"] - d[m]).to_numpy(float)
        res = {"all": _stats(r), "uncensored": _stats(r[unc]),
               **_event_split(r[unc], d["event_id"].to_numpy()[unc]),
               "hist_counts": np.histogram(np.clip(r[unc & np.isfinite(r)], -2.999, 2.999),
                                           HIST_EDGES)[0].astype(int).tolist(),
               "vs": {}}
        for name, x in (("predicted", d[m]), ("gmprocess", d[gmp]), ("magnitude", d["magnitude"]),
                        ("ln_rrup", d["ln_rrup"]), ("ln_vs30", d["ln_vs30"])):
            b = _binned(x.to_numpy(float)[unc], r[unc])
            if b:
                res["vs"][name] = b
        out["models"][m] = res
    out["hist_edges"] = HIST_EDGES.tolist()
    return out


def figure(name, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pick = [m for m in ("simple_2stage_soft" if res["target"] == "fchp" else "simple",
                        "lightgbm_2stage_soft" if res["target"] == "fchp" else "lightgbm")
            if m in res["models"]]
    panels = ["hist", "predicted", "magnitude", "ln_rrup", "ln_vs30"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6))
    edges = np.array(res["hist_edges"])
    for m, c in zip(pick, ("tab:blue", "tab:orange")):
        mm = res["models"][m]
        s = mm["uncensored"]
        axes[0].stairs(mm["hist_counts"], edges, color=c,
                       label=f"{m}: mean {s['mean']:+.3f}, sd {s['sd']:.3f}")
        for ax, p in zip(axes[1:], panels[1:]):
            b = mm["vs"].get(p)
            if b:
                ax.errorbar(b["x_median"], b["mean"], yerr=b["sd"], color=c, marker="o", ms=4, capsize=2)
    axes[0].set_xlabel("residual, ln(analyst / predicted)")
    axes[0].legend(fontsize=7)
    for ax, p in zip(axes[1:], panels[1:]):
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(p)
        ax.set_ylabel("residual (bin mean ± sd)")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(models_dir, traces_path):
    root = Path(models_dir)
    cov = _covariates(pd.read_parquet(traces_path))
    out_dir = root / "residuals"
    out_dir.mkdir(exist_ok=True)
    allres = {}
    for f in sorted(root.glob("*/oof_predictions.parquet")):
        rep = f.parent / "report.json"
        target = json.loads(rep.read_text()).get("target") if rep.exists() else None
        if target not in ("fchp", "fclp"):
            continue
        res = run(pd.read_parquet(f), cov, target)
        allres[f.parent.name] = res
        figure(f.parent.name, res, out_dir / f"{f.parent.name}.png")
        for m, v in res["models"].items():
            s = v["uncensored"]
            print(f"{f.parent.name:22s} {m:22s} n={s['n']:7d} mean={s['mean']:+.4f} sd={s['sd']:.4f} "
                  f"tau={v['tau_between_event']:.4f} phi={v['phi_within_event']:.4f}")
    (out_dir / "residuals.json").write_text(json.dumps(allres))
    print(f"wrote {out_dir}/residuals.json and one PNG per run")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
