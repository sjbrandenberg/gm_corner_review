# Example: predicting corner frequencies and accept/reject with `predict.py`

This folder shows how to apply the trained models to gmprocess records with
[`scripts/predict.py`](../../scripts/predict.py). The models are **bundled with the package**
(in `gmcr/trained/`), so no files from Stampede3 are needed: get the repository, set up Python,
and run the example. It ships one extracted event to predict for.

To see which models are bundled:

```bash
python scripts/predict.py --list-models
```

If none are bundled yet (they are added by exporting the trained models; see "Bundling the
trained models with the package" in [docs/RUNBOOK.md](../../docs/RUNBOOK.md)), the example uses
**demo models** built on your machine, so you can still check your setup.

> **The demo models are trained on synthetic analyst labels. Their predictions mean nothing.**
> The example says so whenever it uses them.

## 1. Set up Python (once)

You need Python 3.10 or newer and this repository:

```bash
git clone https://github.com/sjbrandenberg/gm_corner_review.git
cd gm_corner_review
```

(If you already have it: `git pull`, which also brings newly bundled models.)

Then **either** create the project's conda environment (recommended; same packages as on
Stampede3):

```bash
conda env create -f environment.yml
conda activate gmcr
```

**or** install into an environment of your own with pip:

```bash
pip install -e ".[ml]"      # [ml] adds LightGBM, needed only for LightGBM models
```

## 2. Run the example

From the repository folder:

```bash
python examples/predict/run_example.py
```

It goes through five steps and stops with an explanation at the first problem:

1. checks Python and every package `predict.py` needs;
2. checks every bundled model (or builds the demo models, the first time, if none are bundled);
3. uses the sample event that ships with the example (`sample_features/`: event
   nc73819746, 63 traces, already extracted, so no `.h5` file is needed);
4. runs `predict.py` for each model, printing the **exact command**, so you can copy it;
5. prints the first rows of the predictions, which are saved in `examples/predict/output/`.

If this works, your setup is fine. Other ways to run it:

```bash
python examples/predict/run_example.py --model fchp accept              # only these bundled models
python examples/predict/run_example.py --workspace path/to/<event>.h5   # your own records
python examples/predict/run_example.py --model-dir path/to/model_folder # a model folder (below)
python examples/predict/run_example.py --demo                           # the demo models
```

## 3. Which models

The bundled models have the names of the training runs:

| name | predicts |
|---|---|
| `fchp`, `fchp_curves`, ... | high-pass corner frequency (Hz) |
| `fclp`, ... | low-pass corner frequency (Hz) |
| `accept`, ... | probability that an analyst would accept the record |
| an `auto_fchp` calibration | high-pass corner from re-tuned `auto_fchp` settings |

Names ending in `_noevent` were trained without magnitude and depth. Each model's folder in
`gmcr/trained/` has its training report (`report.md`: performance, recommendation,
coefficients) and `provenance.json` (when and from what it was exported).

A model can also be given as a folder (`--model-dir`, or as the first argument of
`predict.py`): one written by `scripts/export_models.py`, or a training output folder from
Stampede3 (`report.json` + `models.joblib`, copied whole). Training output folders hold Python
pickles that may not load with other package versions; exported folders and bundled models
don't have that problem.

## 4. Run `predict.py` yourself

Once the example works, call `predict.py` directly:

```bash
# from gmprocess workspace files (features are extracted first, ~0.5 s per trace)
python scripts/predict.py fchp --workspaces path/to/event1.h5 path/to/event2.h5 --out fchp.csv

# from a folder of already-extracted features (gmcr.extract + merge_outputs.py)
python scripts/predict.py fchp --features path/to/features --out fchp.csv

# accept/reject, flagging records at the threshold that caught 80% of the analysts' rejections
python scripts/predict.py accept --workspaces path/to/event.h5 --flag-share 80 --out accept.csv
```

By default `predict.py` applies the model each training run recommends (its `report.json`);
`--model NAME` picks another variant of that run (the names are in its `report.md`).

## Reading the output

One row per trace (component), with `trace_id` (network.station.location.channel) and:

| column | meaning |
|---|---|
| `pred_fchp_hz` / `pred_fclp_hz` | predicted corner frequency (Hz) |
| `ratio_to_gmprocess` | predicted corner divided by gmprocess's corner |
| `p_keep_gmprocess` | two-stage fchp models: probability an analyst would keep gmprocess's corner |
| `p_lowered` | fclp models: probability an analyst would lower gmprocess's low-pass corner |
| `p_accept`, `flag_reject` | accept models: probability of acceptance; flag from `--flag-share` |
| `fchp_gmp`, `fchp_gui`, `fclp_gmp` | the automatic corners, for comparison |
| `auto_failed` | gmprocess's automatic checks rejected the trace |
| `missing_predictors` | how many model inputs could not be computed for the trace |

Traces with `auto_failed = True` usually have no gmprocess corner and therefore no corner
prediction (empty cells). That is expected; the models were trained on records gmprocess passed.

`predict.py` also prints which inputs were unavailable. Reading `.h5` files directly, the
finite-fault distances (`rx`, `ry0`) and site data (`ln_vs30`, `ln_z1p0`) are missing because
they come from the NGA-West3 database (`scripts/fetch_flatfile.py` + `scripts/join_metadata.py`).
The models still run, but predictions are closest to the training conditions when those steps
are done first and `--features` is used.

## Troubleshooting

| message | fix |
|---|---|
| `MISSING ...` in step 1 | install the listed packages, or use the conda environment (step 1 above) |
| `BROKEN ...` in step 1, or `module 'pyarrow' has no attribute '__version__'` | two installations of a package are mixed (often one in your user folder): set `PYTHONNOUSERSITE=1`, or use a fresh environment |
| `no report.json` / `no models.joblib` (with `--model-dir`) | copy the whole model folder, not single files |
| `models.joblib could not be loaded` or a version warning (training output folders only) | use the bundled or exported models instead, or the conda environment from `environment.yml` (the versions used on Stampede3) |
| `No models are bundled ... yet` | `git pull` to get them once they are committed; until then the example uses demo models |
| `not bundled: NAME` | use a name from `python scripts/predict.py --list-models` |
| `the default model is LightGBM` | `pip install lightgbm` |
| `the polyvar_... curves are missing` | an `auto_fchp` calibration chose a variant curve family: add `--fchp-variants` (with `--workspaces`) |
| `no traces extracted` | the `.h5` file is not a gmprocess workspace, or is corrupt; the lines above it give the error |

## Files

| file | |
|---|---|
| `run_example.py` | the step-by-step example with checks |
| `make_demo_models.py` | builds the demo models (called by `run_example.py`) |
| `sample_features/` | extracted features of event nc73819746 (input for the example) |
| `demo_models/`, `output/` | created when you run the example (not in git) |
