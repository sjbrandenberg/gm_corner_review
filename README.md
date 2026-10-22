# gm_corner_review

Learn high-/low-pass corner frequencies and accept/reject decisions from
human-reviewed gmprocess workspaces (NGA-West3 review GUI v6.2).

1. Extract one row of features + labels per trace, in parallel on Stampede3.
2. Attach database metadata (nga-gm.org) and computed distances.
3. Train and compare corner-frequency and accept/reject models.

**To run it end to end (staging with Globus, Stampede3 jobs, training), follow
[docs/RUNBOOK.md](docs/RUNBOOK.md).**

## Layout

```
gmcr/asdf_reader.py    read-only h5py access to workspace files (no gmprocess needed)
gmcr/gui_replica.py    the review GUI's signal handling + its automatic fchp (verified exact)
gmcr/features.py       per-trace scalars and fixed-grid curves
gmcr/fchp_variants.py  polynomial-fit ratio curves for other orders and event-tied fit windows (--fchp-variants)
gmcr/extract.py        CLI: workspace .h5 -> parquet (resumable; optional copy to node-local
                       /tmp and deletion of the staged source after extraction)
gmcr/slim.py           copy only what gmcr reads from a workspace
gmcr/flatfile_api.py   nga-gm.org API client (reference tables, flatfile)
gmcr/distances.py      Rrup/Rjb (site-centred projection) and GC2 Rx/Ry0 from finite faults
gmcr/gc2.py            GC2 coordinates (Spudich & Chiou 2015)
scripts/slim_workspace.py  optional: strip unused caches before transfer (~36% of size)
scripts/merge_outputs.py   merge per-file parquet, QA summary
scripts/fetch_flatfile.py  download database metadata for the extracted events
scripts/join_metadata.py   attach event/site/flatfile metadata and computed distances
tacc/setup_env.sh      Miniforge + conda env in $WORK, packed for node-local use
tacc/make_tasks.py     pylauncher task list (chunks of files)
tacc/submit.sh         submit any job so it runs from $SCRATCH (use this, not sbatch)
tacc/extract.slurm     multi-node feature extraction
tacc/globus_batches.py split the DesignSafe project into Globus transfer batches (login node)
tacc/run_batch.sh      stage one batch with Globus, check it, submit its extraction
tacc/slim.slurm        slim staged files on one node (not needed with staged batches)
tacc/inventory.py      count/size/read-check staged files
tacc/train.slurm       join metadata + train all models on one node
gmcr/modeling.py       datasets, leakage-safe features, event-grouped CV, metrics, log-linear/Tobit model
scripts/train_corner_model.py   fchp / fclp: baselines vs simple vs LightGBM
scripts/train_accept_model.py   accept/reject: constant vs logistic vs LightGBM
scripts/calibrate_auto_fchp.py  calibrate auto_fchp's targets, fchp_max, order, and fit window on the analysts' fchp
```

## Outputs

`traces_all.parquet`, one row per processed trace, key `trace_key = "<eventid>|<NET.STA.LOC.CHA>|<file stem>"`.

| group | columns |
|---|---|
| labels | `y_reviewed`, `y_accepted` (NaN unless reviewed), `y_fchp`, `y_fclp` (NaN when `y_fclp_none`, i.e. GUI lowpass box unchecked), `y_fclp_raw`, `y_fchp_source` (gmprocess / gui_auto / manual), `review_username`, `review_timestamp` |
| automatic | `fchp_gmp`, `fclp_gmp`, `fchp_applied`, `fchp_gui`, `gui_at_cap` (GUI fchp hit its 1.2 Hz cap), `auto_failed`, `auto_fail_module/reason`, `colocated`, `selected` |
| metadata | magnitude, depth, `repi_km`, `rhypo_km`, instrument/component codes, sampling rate |
| signal quality | SNR band edges at 2/3/5/10, SNR at fixed frequencies, `fit_*` (Brune fit), clipping prob., zero crossings |
| displacement | `poly_ratio_at_gmp/gui`, `pre_ratio_at_gmp/gui` (the two get_fchp criteria) |
| HF artifacts | `psa_rough_short`, `psa_max_bump_short`, `psa001_over_pga` |

`curves_all.parquet` (float32, grids in `grids.json`): `snr_*`, `fas_signal_*`,
`fas_noise_*` (80 log-spaced freqs, 0.01–50 Hz), `poly_ratio_*`, `pre_disp_ratio_*`
(25 trial fchp, 0.02–3 Hz), `psa_gmp_*` (NGA-West2 periods).

No predictor uses the analyst's choices; only `y_*` columns do.

## Local test

```bash
pip install -e .            # or: conda env create -f environment.yml
python -m gmcr.extract ~/Downloads/nc73819746.h5 --out test_out
python scripts/merge_outputs.py test_out
```

## Stampede3

```bash
# once (in an idev session)
bash tacc/setup_env.sh

# optional, before/after transfer: strip unused caches
python scripts/slim_workspace.py /path/raw $SCRATCH/gm_data --workers 16

# test on a handful of files first
python tacc/make_tasks.py $SCRATCH/gm_data $SCRATCH/gmcr_test --limit 20 --files-per-task 2
sbatch -N 1 -t 00:30:00 --export=ALL,RUN_DIR=$SCRATCH/gmcr_test tacc/extract.slurm

# full run (edit -A allocation in extract.slurm)
python tacc/make_tasks.py $SCRATCH/gm_data $SCRATCH/gmcr_run
sbatch --export=ALL,RUN_DIR=$SCRATCH/gmcr_run tacc/extract.slurm
```

Rerunning `make_tasks.py` + `sbatch` resumes; finished files are skipped.
Expect ~0.4 s/trace/core (laptop measurement).

## Flatfile metadata (nga-gm.org API)

Run where there is internet (laptop or TACC login node), after merging:

```bash
python scripts/fetch_flatfile.py $SCRATCH/gmcr_run/features --token-file ~/token.txt   # or --login
python scripts/join_metadata.py  $SCRATCH/gmcr_run/features
```

`traces_joined.parquet` adds:

| prefix | source | coverage | accept/reject model? |
|---|---|---|---|
| `ev_*` | events + finiteFaults tables | all events | yes |
| `d_*` | Rrup, Rjb, GC2 Rx/Ry0/U computed from the finite fault and StationXML coords | all traces | yes |
| `analyst_id` | event's flatfile `user_id` (events were split among analysts) | events with an accepted motion | yes |
| `st_*` | stations + sites tables | stations accepted for some event | **no** (missing hints at rejection) |
| `ff_*` | flatfile row | accepted motions only | **no** |

`ff_hpass_fc` / `ff_lpass_fc` are the analyst's final corners, used only to
cross-check the HDF5 review labels.

Distances are computed in a site-centred azimuthal-equidistant frame on a
6371 km sphere (the convention that reproduces the database), so Rrup >= Rjb.
On 1,987 flatfile stations (Ridgecrest, South Napa, nc73819746) Rrup/Rjb match
to a median of 15 m (max 0.17 km). Rx/Ry0 use GC2 per NGA sub-rupture
(`seg_sub_rupture_number`), each site taking the sub-rupture with the smallest
Rrup; over all 21 multi-segment faults plus two single-segment events the median
errors are 0.27 km (Rx), 0.15 km (U), and 0.12 km (Ry0).

## Calibrating auto_fchp

`scripts/calibrate_auto_fchp.py` asks how close the auto_fchp algorithm itself can get to the analysts'
fchp once its parameters are calibrated, scored like the models above (out-of-fold, event-grouped CV,
RMSE of ln fchp, tau/phi). It evaluates auto_fchp from the stored `poly_ratio_*` and `pre_disp_ratio_*`
curves (the GUI's two criteria on the 25 trial corners), so it needs only the feature tables, no
waveforms: about 6 minutes on one core.

```bash
python scripts/calibrate_auto_fchp.py $FEATURES_DIR --out models/auto_fchp     --compare-oof models/fchp/oof_predictions.parquet
```

It compares `auto` (calibrated targets and fchp_max), `auto_recal` (a + b ln auto), and `gmp_max`
(max of the gmprocess corner and auto, i.e. auto_fchp only raises the gmprocess value) with the
gmprocess and GUI baselines and, with `--compare-oof`, the trained simple and LightGBM models on the
same traces (event-bootstrap RMSE differences). `gui_grid` reproduces the GUI's settings from the curves,
to check the grid interpolation (median 0.8% from the GUI's value on nc73819746).

Other polynomial orders and fit windows tied to the event ([t5 - k D5-95, t95 + k D5-95]) need the
waveforms: extract with `--fchp-variants` (adds 20 curve families, about 0.2 s/trace), e.g. with
`--extra-args='--reviewed-only --no-psa --fchp-variants'` in `tacc/make_tasks.py`, merge, and pass the
folder with `--variants-dir` (use `--jobs` to search the families in parallel).

## Notes

* `ucla_geotech_tools==3.0.3`: `auto_fchp` matches the GUI; `response_spectrum.get_ARS`
  has a bug (`Fmotion(1.0 + ...)` missing `*`; `zeropad` ignored), so gmcr computes PSA itself.
* Analysts never reviewed auto-rejected records, so accept/reject labels can only
  tighten the existing criteria.
