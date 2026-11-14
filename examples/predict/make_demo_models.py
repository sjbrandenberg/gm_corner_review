"""Build small DEMO models so the predict.py example runs without the real models.

    python examples/predict/make_demo_models.py

The demo models are trained on SYNTHETIC analyst labels: the sample event
(sample_features/, nc73819746) copied 60 times with random changes to the predictors and
made-up analyst decisions. They only show how predict.py works. Their predictions mean
nothing; use the model folders from the real training runs on Stampede3 for real work.

They are built on your machine (a few seconds, simple models only), so that the repository
holds no fake models next to the real ones in gmcr/trained/.

Writes demo_models/fchp and demo_models/accept next to this script, exported in the same
version-independent format as the models bundled with the package (gmcr.portable), so the
example uses them exactly as it uses bundled models.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def synthetic_training_set(src, dst, n_events=60, seed=0):
    tr = pd.read_parquet(src / "traces_all.parquet")
    cu = pd.read_parquet(src / "curves_all.parquet").set_index("trace_key").loc[tr["trace_key"]].reset_index()
    rng = np.random.default_rng(seed)
    T, C = [], []
    for e in range(n_events):
        t, c = tr.copy(), cu.copy()
        t["event_id"] = f"demo{e:03d}"
        t["trace_key"] = t["trace_key"] + f"#demo{e}"
        c["trace_key"] = t["trace_key"].to_numpy()
        t["magnitude"] = rng.uniform(3, 7.5)
        for k in ("fchp_gmp", "fchp_gui", "snr2_flo", "snr3_flo", "repi_km", "rhypo_km",
                  "pre_ratio_at_gmp", "psa_rough_short"):
            if k in t:
                t[k] = t[k] * np.exp(rng.normal(0, 0.3, len(t)))
        t["fchp_gui"] = np.minimum(t["fchp_gui"], 1.2)
        t["gui_at_cap"] = t["fchp_gui"] >= 1.2 - 1e-9
        # made-up analyst decisions
        t["y_reviewed"] = np.where(t["instrument"] == "HH", 1, 0)
        guess = np.exp(0.4 * np.log(t["fchp_gmp"]) + 0.6 * np.log(t["fchp_gui"])
                       + rng.normal(0, 0.1, len(t)))
        kept = rng.random(len(t)) < 0.3
        t["y_fchp"] = np.where(kept, t["fchp_gmp"], np.round(guess, 2))
        z = 1 + 2 * np.log(t["snr3_flo"].fillna(1) + 1) - t["psa_rough_short"].fillna(5) / 5
        accepted = rng.random(len(t)) < 1 / (1 + np.exp(-z))
        t["y_accepted"] = np.where(t["y_reviewed"] == 1, accepted.astype(float), np.nan)
        t["y_fclp"], t["y_fclp_none"] = t["fclp_gmp"], False
        T.append(t)
        C.append(c)
    pd.concat(T).to_parquet(dst / "traces_all.parquet", index=False)
    pd.concat(C).to_parquet(dst / "curves_all.parquet", index=False)


DEMO_NOTE = "DEMO model trained on synthetic labels (make_demo_models.py): its predictions mean nothing"


def main():
    sys.path.insert(0, str(REPO))
    from gmcr import portable

    out = HERE / "demo_models"
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        data.mkdir()
        synthetic_training_set(HERE / "sample_features", data)
        for script, extra, name in (("train_corner_model.py", ["--target", "fchp"], "fchp"),
                                    ("train_accept_model.py", [], "accept")):
            trained = Path(tmp) / name
            cmd = [sys.executable, str(REPO / "scripts" / script), str(data), *extra,
                   "--no-ml", "--boot", "100", "--out", str(trained)]
            print("building demo model:", name, flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                print(r.stdout[-2000:], r.stderr[-3000:])
                sys.exit(f"building the {name} demo model failed (see above)")
            portable.export_model_dir(trained, out / name, note=DEMO_NOTE)
    (out / "README.txt").write_text(
        "DEMO models trained on synthetic labels (make_demo_models.py). Their predictions\n"
        "mean nothing; they only show how predict.py works.\n")
    print(f"demo models in {out}")


if __name__ == "__main__":
    main()
