"""One-page summary of all model comparisons in a models folder.

    python scripts/summarize_models.py $SCRATCH/gmcr_run/models

Reads each <folder>/report.json written by train_corner_model.py / train_accept_model.py
and prints a markdown summary: per corner-frequency run the RMSE (ln) and share within
20% of the gmprocess baseline and of each model; per accept/reject run the AUC and log
loss; and, for every *_noevent run, the change from its counterpart with the event-level
predictors.
"""

import json
import sys
from pathlib import Path


def main(models_dir):
    root = Path(models_dir)
    reports = {}
    for f in sorted(root.glob("*/report.json")):
        reports[f.parent.name] = json.loads(f.read_text())
    corner = {k: r for k, r in reports.items() if "target" in r}
    accept = {k: r for k, r in reports.items() if "accept_rate" in r}

    out = ["# Model comparison summary", ""]
    if corner:
        out += ["## Corner frequencies (RMSE of ln f, share within 20%)", "",
                "| run | traces | gmprocess | best simple | best ML | recommendation |",
                "|---|---|---|---|---|---|"]
        for k, r in corner.items():
            m = r["metrics"]

            def cell(name):
                if not name or name not in m:
                    return "-"
                return f"{name}: {m[name]['rmse_ln']:.3f} / {m[name]['within_20pct']:.2f}"
            bs = r.get("best_simple", "simple")
            bm = r.get("best_ml", "lightgbm" if "lightgbm" in m else None)
            out.append(f"| {k} | {r['n_traces']} | {cell('baseline_gmprocess')} | {cell(bs)} | "
                       f"{cell(bm)} | {r['recommendation']} |")
        out.append("")
    if accept:
        out += ["## Accept/reject (AUC / log loss)", "",
                "| run | traces | simple | lightgbm | recommendation |", "|---|---|---|---|---|"]
        for k, r in accept.items():
            m = r["metrics"]
            c = lambda n: f"{m[n]['auc']:.3f} / {m[n]['log_loss']:.4f}" if n in m else "-"
            out.append(f"| {k} | {r['n_traces']} | {c('simple')} | {c('lightgbm')} | {r['recommendation']} |")
        out.append("")

    pairs = [(k, k.replace("_noevent", "")) for k in reports if k.endswith("_noevent")
             and k.replace("_noevent", "") in reports]
    if pairs:
        out += ["## Without the event-level predictors (magnitude, depth)", ""]
        for k, base in pairs:
            a, b = reports[base], reports[k]
            if "target" in a:
                name = b.get("best_ml") or b.get("best_simple")
                ref = a.get("best_ml") or a.get("best_simple")
                ra, rb = a["metrics"][ref]["rmse_ln"], b["metrics"][name]["rmse_ln"]
                out.append(f"- {base}: best RMSE {ra:.3f} ({ref}) -> {rb:.3f} ({name}) without them "
                           f"({100 * (rb - ra) / ra:+.1f}%)")
            else:
                la, lb = a["metrics"]["lightgbm"]["log_loss"], b["metrics"]["lightgbm"]["log_loss"]
                aa, ab = a["metrics"]["lightgbm"]["auc"], b["metrics"]["lightgbm"]["auc"]
                out.append(f"- {base}: LightGBM AUC {aa:.3f} -> {ab:.3f}, log loss {la:.4f} -> {lb:.4f} "
                           f"({100 * (lb - la) / la:+.1f}%)")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
