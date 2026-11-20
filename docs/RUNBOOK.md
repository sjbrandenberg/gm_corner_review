# Runbook: staging, extraction, and model training on Stampede3

Allocation: **BCS22014** (already set in `tacc/*.slurm`).

**Jobs run entirely from `$SCRATCH`.** TACC asks that compute nodes not use `$WORK`
(Stockyard) except to stage data before or after a job
([Good Conduct Guide](https://docs.tacc.utexas.edu/basics/conduct/); the
[Stampede3 guide](https://docs.tacc.utexas.edu/hpc/stampede3/): "Always run your jobs out
of $SCRATCH"). Jobs submitted from `$WORK` failed here on 2026-10-18 with `Input/output
error` or were "CANCELLED by 0" after 2:02 without logs. So the code and the conda env are
*kept* in `$WORK`, and every job is submitted with **`bash tacc/submit.sh <job> ...`**, which
on the login node copies the code and the packed env to `$SCRATCH` (only what changed) and
submits from there, with logs in `$SCRATCH/gmcr_logs/<job name>.<job id>.out` and `.err`.
Don't call `sbatch` on the `tacc/*.slurm` files directly.

Stampede3 compute nodes cannot read Corral, so the HDF5 files are copied from the
DesignSafe project to Stampede3 **in batches** with Globus. Each extraction task copies its
file to the node's local `/tmp`, extracts it, then deletes the `/tmp` copy *and* the staged
file. A batch folder empties as its job runs, so only one or two batches (~400–800 GB) are
on disk at any time instead of 4 TB.

| stage | where | script |
|---|---|---|
| 0. one-time setup (env, Globus CLI, batch lists) | login node / `idev` | `tacc/setup_env.sh`, `tacc/globus_batches.py` |
| 1. test on 20 files | login + 1 compute node | `tacc/run_batch.sh` pieces |
| 2. stage + extract, batch after batch | login node (tmux) + 1–2 compute nodes per batch | `tacc/run_batch.sh` |
| 3a. fetch database metadata | login node (needs internet) | `scripts/fetch_flatfile.py` |
| 3b. join + train models | 1 compute node | `tacc/train.slurm` |
| 4. keep the results | | `cp`, Globus |

### Sizing (DesignSafe project, measured October 2026)

* HDF5 volume: 20,609 + 3,912,638 + 34,313 MB ≈ **3.97 TB**, roughly **2 million traces**
  (~1.9 MB per trace in the nc73819746 example), ~500k of them reviewed.
* With `--reviewed-only` (used by `run_batch.sh`), extraction is ~50–60 core-hours in
  total: a 400 GB batch (~10% of the data) takes ~5–6 core-hours, i.e. minutes on one
  48-core node. **The Globus transfers, not the compute, set the pace.** Expect each batch
  job to finish well before the next batch has arrived.
* Each task reads only the parts of a file gmcr needs (~36% of its bytes) into `/tmp`, at
  most one file per task at a time.
* Outputs are small: `traces_all.parquet` well under 1 GB and `curves_all.parquet` ~1 GB.

### Where to stage: `$SCRATCH` (recommended) or `$WORK`

Both work (`STAGE_ROOT` / `--dest-root` below). `$SCRATCH` is recommended:

* `$WORK` has a per-user quota (check with `/usr/local/etc/taccinfo`; it is typically on
  the order of 1 TB and shared across TACC systems). Two 400 GB batches may not fit.
* TACC asks that many-node jobs read from `$SCRATCH`, not `$WORK`.
* `$SCRATCH` is purged after files go unused for a while (about 10 days at the time of
  writing). That doesn't matter for staged batches, which live for hours, but copy the
  results off `$SCRATCH` at the end (step 4).

If you stage on `$WORK`, use smaller batches (`--batch-gb 200`) and run one at a time.

Check the things this runbook can't know for sure: quotas, purge policy, and the names and
paths of the Globus collections.

---

## 0. One-time setup

### 0.1 Code and environment

```bash
cd $WORK
git clone <your copy of gm_corner_review>   # or copy the folder with Globus/scp
cd gm_corner_review
idev -p skx -N 1 -t 01:00:00 -A BCS22014     # interactive node for the build
bash tacc/setup_env.sh                       # Miniforge in $WORK, conda env "gmcr", packed copy
exit
```

`setup_env.sh` installs the package into the env, so **rerun it after changing gmcr code**.
The env includes the Globus CLI.

`conda` is not a TACC module: every new login needs Miniforge's setup script before
`conda activate`. Add a shortcut once (rather than `conda init`, which edits your login
shell and can interfere with TACC modules):

```bash
echo "alias gmcr='source \$WORK/miniforge3/etc/profile.d/conda.sh && conda activate gmcr'" >> ~/.bashrc
source ~/.bashrc
gmcr          # activates the env; the commands below assume it is active
```

### 0.2 Globus CLI

**Prerequisite (once): link your CILogon identity to your TACC account**, or Globus can't
reach DesignSafe on Corral (DesignSafe Globus guide, steps 1-3): log in at
https://cilogon.org with your institutional account and copy the **ePPN** under *User
Attributes*; paste it under **ePPN** at https://accounts.tacc.utexas.edu and save; allow
~30 minutes to propagate.

On a login node:

```bash
gmcr                                      # activate the env (alias from 0.1)
globus login --no-local-server            # prints a URL; log in in a browser, paste the code back
globus endpoint search "TACC Corral3" --format json --jmespath "DATA[].[id, display_name, owner_string]"
globus endpoint search "Stampede3"    --format json --jmespath "DATA[].[id, display_name, owner_string]"
```

* DesignSafe: use **"TACC Corral3 with CILogon Authentication"** (DesignSafe's CLI guide).
  Searching "DesignSafe" finds nothing.
* Stampede3: use the **mapped collection "TACC Stampede3 GCS v5.4 Filesystems"**
  (`1e9ddd41-fe4b-406f-95ff-f3d79f9cb523` as of October 2026). The entries named
  "...Endpoint" are the servers behind it, not collections.

Put the ids in `~/.bashrc`:

```bash
echo 'export DESIGNSAFE_COLLECTION=<TACC Corral3 id>' >> ~/.bashrc
echo 'export STAMPEDE3_COLLECTION=1e9ddd41-fe4b-406f-95ff-f3d79f9cb523' >> ~/.bashrc
source ~/.bashrc
```

Your project's Corral path is `/corral/projects/NHERI/projects/<project-uuid>/`, the same
path in the Globus collection and on the login nodes, so `--globus-root` equals
`--corral-root` below. The UUID is shown when you click the blue text "Learn how to
transfer data to this project" under the project description in the Data Depot.

Check both sides. The first `ls` on each collection asks for consent: run the
`globus session consent ...` command it prints, adding `--no-local-server`, then repeat
the `ls`.

```bash
globus ls "$STAMPEDE3_COLLECTION:$SCRATCH/"
globus ls "$DESIGNSAFE_COLLECTION:/corral/projects/NHERI/projects/<project-uuid>/"
```

### 0.3 Batch lists

On a login node (they can see `/corral`). The workspaces are in the gmprocess project
`4525470084104262126-242ac117-0001-012`, in three folders of event directories: `data/`,
`data2/`, and `data_new/` (the ~20 GB, ~3.9 TB, and ~34 GB of the sizing above). Check
first, e.g. `ls /corral/projects/NHERI/projects/$PRJ/data2 | head` and
`ls /corral/projects/NHERI/projects/$PRJ/data2/<a comcatid>/`.

Split them into batches of whole events:

```bash
cd $WORK/gm_corner_review
PRJ=/corral/projects/NHERI/projects/4525470084104262126-242ac117-0001-012
python tacc/globus_batches.py \
    --root $PRJ/data --root $PRJ/data2 --root $PRJ/data_new \
    --dest-root $SCRATCH/gm_stage \
    --batch-gb 400 \
    --out $SCRATCH/gmcr_run/globus
```

* Each root is searched at any depth: every folder that directly holds `.h5` files is an
  event folder (e.g. `data/<comcatid>/`, the nested `data/data/<comcatid>/`, and
  `data/data/<group>/<comcatid>/`). Only those `.h5` files are listed (one Globus line per
  file), so `raw/` waveforms, plots, and reports are not transferred. `.h5` files in
  subfolders *of* an event folder are not transferred either; the script lists them in
  `nested_not_transferred.csv` and warns.
* Staged files go to `$SCRATCH/gm_stage/<batch>/<location>__<comcatid>/<file>.h5`, where
  the location is the root plus any folders above the event (`data`, `data-data`, ...),
  so events that exist in more than one place stay apart.
* Events are often split over several workspace files (the nc73819746 example holds 7 of
  the event's stations). After extraction, `summary.json` → `duplicate_traces` and
  `duplicate_traces.csv` show traces that came out of more than one file (overlapping
  workspaces or copies in different folders).
* If an event is in more than one folder, the script warns and lists it in
  `duplicates.csv`. Each copy is extracted (keyed by folder), and cross-validation groups
  by the event id inside the files, so copies never straddle a training/test split. They
  would count twice in training, though: decide which folder is authoritative before
  training (e.g. drop the other copies from `traces_all.parquet`).

**Choose the files to transfer by their reviews.** Event folders hold many kinds of
`.h5` files (`workspace.h5`, `<id>.h5`, per-analyst copies such as `workspace-Shako.h5`,
`-Copy1`, Frontera runs, `p_arrival_shift` experiments, `-old`/`-corrupt` files), the same
event can sit in several locations (`data/`, `data/data/`, ...), and many events were never
reviewed. On the login node, scan every file's review records and rebuild the batches from
the selection:

```bash
python tacc/scan_reviews.py $SCRATCH/gmcr_run/globus            # --workers 8 by default
python tacc/globus_batches.py --root $PRJ/data --root $PRJ/data2 --root $PRJ/data_new     --dest-root $SCRATCH/gm_stage --batch-gb 400 --out $SCRATCH/gmcr_run/globus     --select $SCRATCH/gmcr_run/globus/select.csv
```

`scan_reviews.py` reads, for each reviewed trace, the decision and the analyst's corners,
and marks each file `unreadable`, `unreviewed` (automatic output only), `redundant` (all of
its reviews are in another file of the event; ties go to `workspace.h5`/`<id>.h5` and the
top-level location), or `selected`. Files with different reviews of the same traces (e.g.
two analysts) are all selected. It prints the selected events, files, TB, and reviewed
traces; `files_reviews.csv` has one row per file. Check it before transferring.

It prints the number of event folders and files per root and the batches (about 10 for
4 TB), and writes `b001.txt`, `b002.txt`, ... (Globus batch files), `manifest.csv` (root,
event, batch, size), and `commands.sh`. Look at the first lines of `b001.txt`: each line is
a quoted source path under `$PRJ/...` and a destination under `$SCRATCH/gm_stage/b001/`.
Check that the total matches the ~3.97 TB measured.

## 1. Test on a few files

Stage the first batch, but extract only 20 files and keep them (no `--delete-source`), so
you can inspect the outputs and rerun if needed:

```bash
tmux new -s gmcr                                   # survives logout
cd $WORK/gm_corner_review
source $WORK/miniforge3/etc/profile.d/conda.sh && conda activate gmcr
globus transfer $DESIGNSAFE_COLLECTION $STAMPEDE3_COLLECTION \
    --batch $SCRATCH/gmcr_run/globus/b001.txt --label gmcr-b001 --verify-checksum
# wait for the e-mail (or: globus task wait <task id>), then
python tacc/inventory.py $SCRATCH/gm_stage/b001 --sample 20
# the 20 smallest files, one per task (the default order is largest first)
python tacc/make_tasks.py $SCRATCH/gm_stage/b001 $SCRATCH/gmcr_test --limit 20 --smallest-first \
    --files-per-task 1 "--extra-args=--reviewed-only --stage-dir /tmp/gmcr_stage"
bash tacc/submit.sh extract RUN_DIR=$SCRATCH/gmcr_test -- -N 1 -t 01:00:00
# if the job hits its time limit, merge what finished (fine on a login node):
#   python scripts/merge_outputs.py $SCRATCH/gmcr_test/features
# when it finishes:
cat $SCRATCH/gmcr_test/features/summary.json
head $SCRATCH/gmcr_test/features/errors.csv
```

Look at `summary.json`: `file_errors` and `trace_errors` should be 0 or close to it,
`reviewed` > 0, `fchp_source` should show gmprocess / gui_auto / manual counts,
`gmprocess_versions` / `review_usernames` should look right, and `staging` shows how fast
files were copied to `/tmp`. If something fails, the per-file logs are in
`features/logs/` and the job's output in `$SCRATCH/gmcr_logs/gmcr_extract.<jobid>.out`.

## 2. Stage and extract, batch after batch

Batch b001 is already staged from the test, so its extraction can start right away:

```bash
SKIP_TRANSFER=1 bash tacc/run_batch.sh b001
```

From then on, one command per batch, inside tmux on a login node:

```bash
bash tacc/run_batch.sh b002      # transfer, wait, check, submit; hours, mostly the transfer
bash tacc/run_batch.sh b003
...
```

`run_batch.sh` starts the Globus transfer of the batch, waits for it (printing the status
every hour), runs `inventory.py` on it, writes `tasks_<batch>.txt`, and submits
`extract.slurm` for it (`NODES=1` by default). Because the extraction finishes long before
the next transfer does, running the batches one after another in the same tmux window
keeps at most two batches on disk. To be faster, run two `run_batch.sh` in two tmux
windows (two transfers at once).

After each batch job: `ls $SCRATCH/gm_stage/<batch>` should be empty. Files left there had
errors and were kept on purpose; see `features/errors.csv` and `features/logs/`. To retry
them, rerun `SKIP_TRANSFER=1 bash tacc/run_batch.sh <batch>`; once they are fixed or
written off, delete the folder.

Each batch job merges all outputs so far, so `$SCRATCH/gmcr_run/features/summary.json`
always reflects everything extracted. Batches can be rerun safely: files already extracted
are skipped (and deleted from the stage if they were transferred again).

`extract.slurm` deletes nothing on Corral; the originals stay on DesignSafe.

## 3. Database metadata and model training

On a **login node** (compute nodes have no internet). Put your nga-gm.org token in a
file only you can read, or use `--login` to be prompted (tokens expire after 120 min):

```bash
chmod 600 ~/nga_token.txt
python scripts/fetch_flatfile.py $SCRATCH/gmcr_run/features --token-file ~/nga_token.txt
```

Then train on a compute node:

```bash
bash tacc/submit.sh train FEATURES_DIR=$SCRATCH/gmcr_run/features
```

The job runs `join_metadata.py` and five model comparisons; results go to
`$SCRATCH/gmcr_run/models/`:

| folder | what |
|---|---|
| `fchp/` | high-pass corner: gmprocess and GUI baselines, simple log-linear (Tobit for picks at the 1.2 Hz cap), LightGBM |
| `fchp_curves/` | same, LightGBM also given the SNR/spectral/displacement-ratio curves |
| `fclp/` | low-pass corner: probability the analyst lowered it × how much |
| `accept/`, `accept_curves/` | accept/reject: constant rate, logistic regression, LightGBM |

Each folder has `report.md` (start here), `report.json`, out-of-fold predictions,
figures, and `models.joblib` (models refit on all data). The reports state whether the
ML model clears the bar for replacing the simple one (>= 10% lower RMSE for corners,
>= 5% lower log loss for accept/reject, with the event-bootstrap 95% interval excluding
zero; change with `--min-gain`).

Remember when reading the accept/reject report: analysts only reviewed records that passed
the automatic checks, so it can only tell you how to make the criteria stricter.

### Applying a model to new records

`scripts/predict.py` applies the model a run recommends (its `report.json`), or another one with
`--model`, to new records, either from workspace files or from extracted features:

```bash
python scripts/predict.py $SCRATCH/gmcr_run/models/fchp --workspaces /path/to/<comcatid>/*.h5 --out fchp_pred.csv
python scripts/predict.py $SCRATCH/gmcr_run/models/accept --features $SCRATCH/new_run/features     --flag-share 80 --out accept_pred.csv
```

It also takes an `auto_fchp` calibration folder (calibrate_auto_fchp.py output); calibrations that
chose a `polyvar_*` curve family need `--fchp-variants` with `--workspaces`. For the best
predictions, give it features that went through `join_metadata.py` (finite-fault distances, site
data), as the training data did; otherwise those predictors are missing and the script says so.

### Bundling the trained models with the package

The trained models exist only in the training output folders on Stampede3, and `$SCRATCH`
is purged. To ship them, export the chosen folders into `gmcr/trained/<generation>/` (version-independent
JSON + LightGBM text files, checked against `models.joblib`), then commit. The generation
(`gmcr/generations.py`: the gmprocess version the models go with) comes from the training report,
which records the generation of the features; training runs from before generations existed are
`gmprocess-1.3`, and `--generation` overrides it:

```bash
cd $WORK/gm_corner_review && git pull
gmcr                                   # activate the env
python scripts/export_models.py $SCRATCH/gmcr_run/models/fchp $SCRATCH/gmcr_run/models/fclp     $SCRATCH/gmcr_run/models/accept    # add an auto_fchp calibration folder if you use one
git add gmcr/trained && git commit -m "Bundle trained models" && git push
```

Users then run `python scripts/predict.py fchp --workspaces ... --out ...` (or `--list-models`)
without any files from Stampede3; `fchp` is the bundled model of the workspaces' generation.

## 4. Keep the results

`$SCRATCH` is purged. Copy the outputs to `$WORK`, and anything worth keeping long-term to
DesignSafe with Globus:

```bash
mkdir -p $WORK/gmcr_results
cp -r $SCRATCH/gmcr_run/models $WORK/gmcr_results/
cp $SCRATCH/gmcr_run/features/{traces_joined.parquet,summary.json,errors.csv,grids.json} $WORK/gmcr_results/
cp $SCRATCH/gmcr_run/globus/manifest.csv $WORK/gmcr_results/
```

`curves_all.parquet` (~1 GB) is needed to retrain with `--curves` and to calibrate auto_fchp
(section 5); copy it too if you may do either.

## 5. Calibrating auto_fchp

`scripts/calibrate_auto_fchp.py` scores a calibrated auto_fchp against the analysts' fchp the same
way as the trained models (out-of-fold, event-grouped CV). It reads only the feature tables, so it
runs anywhere with about 8 GB of memory: a DesignSafe JupyterHub session, a laptop, or a Stampede3 login
or compute node. About 6 minutes on one core.

### 5.1 Files

From the extraction and training run (`$SCRATCH/gmcr_run`, or `$WORK/gmcr_results` for what
section 4 copied):

| file | needed | notes |
|---|---|---|
| `features/traces_joined.parquet` | yes | `traces_all.parquet` also works (no database metadata needed) |
| `features/curves_all.parquet` | yes | ~1 GB; not in the section 4 copy, so take it from `$SCRATCH` before it is purged |
| `models/fchp/oof_predictions.parquet` | for `--compare-oof` | head-to-head comparison with the simple and LightGBM models |
| `features/grids.json` | no | records the trial-fchp grid (features.FCHP_GRID) |

Keep them in one folder per role, e.g. on a DesignSafe project with Globus (Stampede3 collection to
the DesignSafe collection, the reverse of the staging transfers):

```
gmcr_results/features/traces_joined.parquet
gmcr_results/features/curves_all.parquet
gmcr_results/models/fchp/oof_predictions.parquet
```

### 5.2 Run

```bash
git clone https://github.com/sjbrandenberg/gm_corner_review.git
cd gm_corner_review
pip install --user numpy scipy pandas pyarrow   # not in the gmcr env (see below)
R=/path/to/gmcr_results        # e.g. ~/MyProjects/PRJ-XXXX/gmcr_results on DesignSafe JupyterHub
python scripts/calibrate_auto_fchp.py $R/features --out $R/models/auto_fchp     --compare-oof $R/models/fchp/oof_predictions.parquet
```

The `pip install` is for a Python without the gmcr env, such as DesignSafe JupyterHub. On
Stampede3, activate the gmcr env instead (`gmcr`) and skip the `pip install`. The script imports gmcr
from the cloned folder, so after a `git pull` it runs without rebuilding the env.

### 5.3 Outputs (in `--out`)

* `report.md`: RMSE, mean and sd of the residuals, tau, phi, and fraction within a factor of 1.5 for
  the baselines, `gui_grid`, `auto`, `auto_recal`, `gmp_max`, and the trained models; the selected
  parameters; and event-bootstrap 95% intervals of the RMSE differences (negative favors the first
  model). Check first that `gui_grid` is close to the GUI's value (median |ln ratio| near 0.01).
* `report.json`: the same, plus the parameters selected in each fold (their spread shows how well
  determined they are).
* `oof_predictions.parquet`: out-of-fold ln(fchp) of every model, keyed by `trace_key`.
* `calibration_grid.parquet`: full-data RMSE of every parameter combination, for plotting the
  RMSE against the targets and fchp_max.

### 5.4 Optional: other polynomial orders and fit windows (needs the HDF5 files again)

The curves for other orders and event-tied windows come from the waveforms, so the workspace files
must be staged and extracted again (section 2), into a new run folder so finished files are not
skipped, and without PSA:

```bash
RUN_DIR=$SCRATCH/gmcr_variants EXTRACT_ARGS='--reviewed-only --no-psa --fchp-variants --stage-dir /tmp/gmcr_stage --delete-source'     bash tacc/run_batch.sh b001        # and so on for each batch
python scripts/merge_outputs.py $SCRATCH/gmcr_variants/features
```

Extraction jobs run the gmcr installed in the packed env, so rerun `bash tacc/setup_env.sh` (section
0.1) after pulling this code, or `--fchp-variants` is not recognized.

This adds about 0.2 s per trace to the extraction. Then pass the new curves to the calibration;
searching the 20 extra curve families takes about 6 minutes per family per core, so use `--jobs`
(on a compute node, or a JupyterHub session with several cores):

```bash
python scripts/calibrate_auto_fchp.py $R/features --variants-dir $SCRATCH/gmcr_variants/features     --out $R/models/auto_fchp_variants --compare-oof $R/models/fchp/oof_predictions.parquet --jobs 21
```

Each process holds one family's curves (~25 MB) plus the shared arrays.

## 6. Retraining for gmprocess 2.9 (model generation gmprocess-2.9)

The models above were trained on gmprocess 1.3 workspaces, and gmprocess 2.9 makes different automatic
decisions on the same raw data (its low-pass corners are ~30% higher), so new records processed with 2.9
need models trained on 2.9 output (`gmcr/generations.py`). The training events are reprocessed with
gmprocess 2.9 and the generation's config (`gmcr/conf/gmprocess-2.9/nga.yml`, which the nga-gm server
uses too), and the analysts' decisions are carried over by trace id:

| step | where | script |
|---|---|---|
| 6.0 env with gmprocess 2.9, STREC data | login node (internet) | `tacc/setup_gmp29_env.sh` |
| 6.1 per batch: stage, prepare (StationXML, STREC, SIS metadata) | login node (internet), tmux | `tacc/run_reprocess_batch.sh` |
| 6.1 per batch: export raw data, gmprocess 2.9, extract | 1 compute node | `tacc/reprocess.slurm` (submitted by the above) |
| 6.2 labels from the 1.3 extraction | login node | `scripts/transfer_labels.py` |
| 6.3 train, compare, bundle | 1 compute node | `tacc/train.slurm`, `scripts/export_models.py` |

What `gmcr.reprocess` does with each event (all of its staged files: events split over several
workspace files are reprocessed as one, and a station reviewed in two files once):

* only instruments with a reviewed trace (e.g. `CI.CRY..HN`) are reprocessed;
* `prepare` writes the event's origin and magnitude **from the 1.3 workspace** (so the signal windows
  are those the analysts saw, not today's ComCat values), the 1.3 StationXML, STREC's tectonic region,
  and the event's ANSS SIS snapshot (`sis/`), which gmprocess applies offline when it assembles;
* `run` hands the 1.3 *unprocessed* traces to gmprocess 2.9 unchanged, by their process level in the
  1.3 workspace:
  * raw counts (FDSN data) are written as MiniSEED and go through gmprocess's own assemble, which
    applies the SIS snapshot. 1.3 stored some of them as floats (segments it merged or
    interpolated); these are written as floats, not rounded, and gmprocess's "raw data type is not
    integer" check is waived for exactly those traces (`float_counts` in the report);
  * data a format reader had already converted (CESMD's DMG and COSMOS files: cm/s², response
    removed, often network `--`) can't be written as raw data. They are read from the 1.3 workspace
    with gmprocess 2.9's own reader, with their units, process level, and response-removal mark, and
    added to the workspace after assembling, as 2.9 reads such files itself (`physical_units`);
  * non-free-field stations are kept. With SIS metadata, gmprocess 2.9 knows each instrument's housing
    and its own assemble drops every one that isn't free field (large vaults, building bases,
    tunnels, dams, downhole arrays: about 3% of the training reviews); NGA-West3 keeps them, labelled
    by station type, so gmcr's assemble (`gmcr.reprocess.assemble_event`) keeps them and the
    generation's config doesn't reject them (`check_free_field: reject_non_free_field: False`);
* then `gmrecords process_waveforms` with the generation's config, gmcr's extraction (auto_fchp
  3.1.0), and cleanup: the MiniSEED files, the 2.9 workspace, and the staged 1.3 files go; the event
  folder keeps its metadata, `sis/`, logs, `export.json`, and `reprocess.json`.

### 6.0 One-time setup

```bash
cd $WORK/gm_corner_review && git pull
bash tacc/setup_gmp29_env.sh       # login node: conda env "gmp29", $WORK/gmp29_env.tar.gz, $WORK/strec_data
```

The 1.3 extraction's merged features (`traces_all.parquet`, `curves_all.parquet`, and `db/`) supply the
labels. They were on `$SCRATCH/gmcr_run/features`; if `$SCRATCH` was purged, use the copy kept in
step 4. The batch files (`$SCRATCH/gmcr_run/globus/b*.txt`) can be rebuilt with the two commands of 0.3
(`scan_reviews.py`, then `globus_batches.py --select`): retraining uses the same files as the 1.3 run.

### 6.1 Reprocess, batch after batch

Test on a few events first: prepare the first batch without submitting, run the first three events,
and look at the report:

```bash
tmux new -s gmcr29
cd $WORK/gm_corner_review
SKIP_SUBMIT=1 bash tacc/run_reprocess_batch.sh b001     # stages b001 (skip with SKIP_TRANSFER=1), prepares it
head -n 3 $SCRATCH/gmcr29_run/tasks_b001.txt > $SCRATCH/gmcr29_run/tasks_test.txt
GMCR_PACK_SRC=$WORK/gmp29_env.tar.gz bash tacc/submit.sh reprocess \
    RUN_DIR=$SCRATCH/gmcr29_run TASKS=$SCRATCH/gmcr29_run/tasks_test.txt -- -t 01:00:00
# when it finishes:
python tacc/make_reprocess_tasks.py $SCRATCH/gmcr29_run --report
cat $SCRATCH/gmcr29_run/events/b001/<event id>/reprocess.json
```

The first three tasks are the batch's largest events, so the test also shows the worst-case memory
and time. Check `$SCRATCH/gmcr29_run/events/b001/prepare.json` (SIS outcomes per event: `usable` for
most CI/NC/BK stations, `no_coverage` for networks outside ANSS SIS) and the report: status `ok`, `n_ok`
close to `n_traces`, and the time per stage, which sets the node-hours of the
rest. Then the batches:

```bash
SKIP_TRANSFER=1 bash tacc/run_reprocess_batch.sh b001   # the rest of b001 (finished events are skipped)
bash tacc/run_reprocess_batch.sh b002
...
```

Each job runs one event per task, 24 tasks per node (gmprocess keeps an event's streams in memory),
largest events first; `NODES=2` speeds up a batch. The job goes to `skx` for 8 hours unless
`PARTITION` and `TIME` say otherwise: `skx-dev` (at most 2 hours, one job at a time) usually starts
within minutes, where `skx` can wait a day. b001 (120 events, 69,000 traces) took 31.8 core-hours,
about 1.3 hours spread over 24 tasks, so a batch like it should fit in 2 hours on one node, unless one
very large event runs longer than that by itself:

```bash
PARTITION=skx-dev TIME=02:00:00 bash tacc/run_reprocess_batch.sh b002
```

If a job hits its limit, the events it finished are kept; rerun with `SKIP_TRANSFER=1` for the rest.
After each job, the batch's staging folder is empty except for the files of events that failed or had
nothing to reprocess. To retry the failures, rerun
`SKIP_TRANSFER=1 bash tacc/run_reprocess_batch.sh <batch>`. To fetch an event's SIS metadata again,
delete its `sis/` folder and rerun.

To rerun some events (e.g. after a fix), `tacc/redo_events.py` clears their outputs and writes a Globus
batch file that stages their 1.3 files again; `--unmatched` picks the events whose reviewed
instruments are missing from the reprocessed features (after 6.2). Its docstring has the steps.

### 6.2 Labels

```bash
gmcr
L=$SCRATCH/gmcr_run/features          # the 1.3 extraction (or its copy from step 4)
python scripts/transfer_labels.py --labels $L --features $SCRATCH/gmcr29_run/features \
    --out $SCRATCH/gmcr29_run/train
cp -r $L/db $SCRATCH/gmcr29_run/train/     # database metadata: same events, no need to fetch again
```

`summary.json` counts the reviews matched to reprocessed traces, the ones left out and why
(`unmatched_reviews.csv`), the reviewed traces gmprocess 2.9 rejects automatically (dropped), how
often the analyst kept gmprocess 1.3's corner, and the shifts between the versions' automatic corners.

Labels are the analysts' corners in Hz (`--kept absolute`). Where the analyst kept gmprocess 1.3's
corner, `--kept relative` assumes they would have kept 2.9's instead; that matters most for fclp,
where analysts kept the automatic value most of the time (15 of 18 traces on nc73819746). Train both
ways to see the difference if in doubt (separate `--out` folders).

### 6.3 Train, compare, bundle

```bash
bash tacc/submit.sh train FEATURES_DIR=$SCRATCH/gmcr29_run/train     # models in $SCRATCH/gmcr29_run/models
```

Compare `$SCRATCH/gmcr29_run/models/summary.md` with the 1.3 run's. The training reports record the
generation (`gmprocess-2.9`), so the export puts the models in `gmcr/trained/gmprocess-2.9/`:

```bash
python scripts/export_models.py $SCRATCH/gmcr29_run/models/fchp $SCRATCH/gmcr29_run/models/fclp \
    $SCRATCH/gmcr29_run/models/accept
git add gmcr/trained && git commit -m "Bundle the gmprocess-2.9 models" && git push
```

`predict.py fchp --workspaces ...` then uses them for 2.9 workspaces and the 1.3 models for 1.3 ones.

### 6.4 List of every record and its outcome (for distance limits)

Each reprocessing task also lists every trace of the event's 1.3 files, including those gmprocess
rejected (which the reviewed-only extraction of steps 1-2 left out) and those the analysts rejected,
into `$SCRATCH/gmcr29_run/records13` (`gmcr.extract --metadata-only`: no waveforms are read, about
5 ms per trace). The job merges it. When all batches are done:

```bash
gmcr
R=$SCRATCH/gmcr29_run/records13
cp -r $SCRATCH/gmcr_run/features/db $R/          # database metadata (or run fetch_flatfile.py on $R)
python scripts/merge_outputs.py $R && python scripts/join_metadata.py $R
python scripts/record_status.py $R               # -> $R/records/records.csv, rejected.csv, summary.json
```

`records.csv` has one row per trace and file, with `status` (`accepted`, `analyst_rejected`,
`auto_rejected` with the gmprocess step in `auto_fail_module`, `colocated_not_used`, `not_reviewed`),
magnitude, style of faulting from the rake (NGA-West2 bins; `ev_rake` kept), sensor class, Rrup/Rjb
(finite fault, else hypocentral/epicentral: `d_source`), the SNR band edges, and the corners; see the
docstring of `scripts/record_status.py`. The `db/` folder must cover these events: the 1.3 run's does
if it was fetched for all extracted events.

These are the files selected for training (`select.csv`): events with reviews. To list the other files
too (unreviewed events, redundant copies), run the extraction alone on their batches, e.g. with
`EXTRACT_ARGS='--metadata-only --stage-dir /tmp/gmcr_stage --delete-source' bash tacc/run_batch.sh <batch>`
and a separate `RUN_DIR`.

## Troubleshooting

* **Jobs "CANCELLED by 0" after ~2:02 with no log files, or `Input/output error` on a
  file in `$WORK`**: something in the job used `$WORK` from a compute node (e.g. it was
  submitted with plain `sbatch` from `$WORK`). Submit with `tacc/submit.sh`. If it still
  happens, report it to TACC with the job ids, nodes (`sacct -X -j <id> -o JobID,NodeList`),
  and the error.
* **Logs**: `ls -t $SCRATCH/gmcr_logs | head`; `tail -f $SCRATCH/gmcr_logs/<name>.<id>.out`
  follows a running job (Python output is unbuffered).
* **`module load pylauncher` fails or pylauncher can't be imported with the conda Python**:
  `conda activate gmcr && pip install pylauncher`, then resubmit.
* **Jobs slow to start Python / file-system warnings from TACC**: make sure
  `$WORK/gmcr_env.tar.gz` exists (made by `setup_env.sh`); `extract.slurm` then unpacks
  the env to each node's `/tmp`.
* **Out of memory**: use fewer tasks per node: `bash tacc/submit.sh extract ... -- --ntasks-per-node 24`
  (in `run_batch.sh`, add `--ntasks-per-node 24` after the `--` of its `submit.sh` line).
* **Globus transfer fails or stalls**: `globus task show $(cat $SCRATCH/gmcr_run/globus/<batch>.task)`;
  rerunning the same `globus transfer` command resumes (add `--sync-level checksum`).
* **`globus transfer` rejects the batch file**: check the line format with
  `globus transfer --help` (lines are `--recursive SOURCE_PATH DEST_PATH`) and that the
  source paths match what the web app shows for the DesignSafe collection.
* **`fetch_flatfile.py` returns 401**: the token expired; rerun with `--login`.
