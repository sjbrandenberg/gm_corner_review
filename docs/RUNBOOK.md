# Runbook: staging, extraction, and model training on Stampede3

Allocation: **BCS22014** (already set in `tacc/*.slurm`).

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

* Only the `.h5` files directly inside each event folder are listed (one Globus line per
  file), so `raw/` waveforms, plots, and reports are not transferred.
* Staged files go to `$SCRATCH/gm_stage/<batch>/<root>__<comcatid>/<file>.h5`; the root
  name keeps events that exist in more than one folder apart.
* If an event is in more than one folder, the script warns and lists it in
  `duplicates.csv`. Each copy is extracted (keyed by folder), and cross-validation groups
  by the event id inside the files, so copies never straddle a training/test split. They
  would count twice in training, though: decide which folder is authoritative before
  training (e.g. drop the other copies from `traces_all.parquet`).

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
python tacc/make_tasks.py $SCRATCH/gm_stage/b001 $SCRATCH/gmcr_test --limit 20 --files-per-task 2 \
    "--extra-args=--reviewed-only --stage-dir /tmp/gmcr_stage"
sbatch -N 1 -t 00:30:00 --export=ALL,RUN_DIR=$SCRATCH/gmcr_test tacc/extract.slurm
# when it finishes:
cat $SCRATCH/gmcr_test/features/summary.json
head $SCRATCH/gmcr_test/features/errors.csv
```

Look at `summary.json`: `file_errors` and `trace_errors` should be 0 or close to it,
`reviewed` > 0, `fchp_source` should show gmprocess / gui_auto / manual counts,
`gmprocess_versions` / `review_usernames` should look right, and `staging` shows how fast
files were copied to `/tmp`. If something fails, the per-file logs are in
`features/logs/` and the job's output in `gmcr_extract.<jobid>.out`.

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
sbatch --export=ALL,FEATURES_DIR=$SCRATCH/gmcr_run/features tacc/train.slurm
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

## 4. Keep the results

`$SCRATCH` is purged. Copy the outputs to `$WORK`, and anything worth keeping long-term to
DesignSafe with Globus:

```bash
mkdir -p $WORK/gmcr_results
cp -r $SCRATCH/gmcr_run/models $WORK/gmcr_results/
cp $SCRATCH/gmcr_run/features/{traces_joined.parquet,summary.json,errors.csv,grids.json} $WORK/gmcr_results/
cp $SCRATCH/gmcr_run/globus/manifest.csv $WORK/gmcr_results/
```

`curves_all.parquet` (~1 GB) is needed only to retrain with `--curves`.

## Troubleshooting

* **`module load pylauncher` fails or pylauncher can't be imported with the conda Python**:
  `conda activate gmcr && pip install pylauncher`, then resubmit.
* **Jobs slow to start Python / file-system warnings from TACC**: make sure
  `$WORK/gmcr_env.tar.gz` exists (made by `setup_env.sh`); `extract.slurm` then unpacks
  the env to each node's `/tmp`.
* **Out of memory**: use fewer tasks per node: `sbatch --ntasks-per-node 24 ...`
  (in `run_batch.sh`, add it to the `sbatch` line).
* **Globus transfer fails or stalls**: `globus task show $(cat $SCRATCH/gmcr_run/globus/<batch>.task)`;
  rerunning the same `globus transfer` command resumes (add `--sync-level checksum`).
* **`globus transfer` rejects the batch file**: check the line format with
  `globus transfer --help` (lines are `--recursive SOURCE_PATH DEST_PATH`) and that the
  source paths match what the web app shows for the DesignSafe collection.
* **`fetch_flatfile.py` returns 401**: the token expired; rerun with `--login`.
