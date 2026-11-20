"""Extract features and labels from one or more gmprocess workspace files.

Usage:
    python -m gmcr.extract FILE.h5 [FILE2.h5 ...] --out OUTDIR [--reviewed-only] [--no-psa]
                           [--fchp-variants]
                           [--stage-dir /tmp --stage-mode slim|copy]

For each input file writes
    OUTDIR/traces/<stem>.parquet   one row per processed trace (scalars)
    OUTDIR/curves/<stem>.parquet   one row per trace, fixed-grid float32 curves
    OUTDIR/logs/<stem>.json        timing, counts, per-trace errors
Files whose traces parquet already exists are skipped (safe to resubmit).
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import fchp_variants as fv
from . import features as ft
from . import generations
from . import tables
from .asdf_reader import Workspace

CURVE_LENGTHS = {
    "snr": len(ft.FREQ_GRID), "fas_signal": len(ft.FREQ_GRID), "fas_noise": len(ft.FREQ_GRID),
    "poly_ratio": len(ft.FCHP_GRID), "pre_disp_ratio": len(ft.FCHP_GRID),
}


def _curve_row(key, curves, psa_len, fchp_variants=False):
    row = {"trace_key": key}
    lengths = dict(CURVE_LENGTHS, psa_gmp=psa_len)
    if fchp_variants:
        lengths.update({name: len(ft.FCHP_GRID) for name in fv.NAMES})
    for name, n in lengths.items():
        vals = curves.get(name)
        if vals is None or len(vals) != n:
            vals = np.full(n, np.nan, np.float32)
        for i, v in enumerate(vals):
            row[f"{name}_{i:03d}"] = np.float32(v)
    return row


def _stage(path, stage_dir, mode):
    """Local copy of path in a per-process folder of stage_dir ("copy": the whole file;
    "slim": only what gmcr reads). Returns (local path, seconds, bytes copied)."""
    t0 = time.time()
    local_dir = Path(stage_dir) / f"gmcr_{os.getpid()}"
    local_dir.mkdir(parents=True, exist_ok=True)
    local = local_dir / path.name
    if mode == "slim":
        from .slim import slim

        _, nbytes = slim(path, local)
    else:
        tmp = local.with_suffix(".tmp")
        shutil.copyfile(path, tmp)
        tmp.replace(local)
        nbytes = local.stat().st_size
    return local, time.time() - t0, nbytes


def _delete_source(path):
    """Remove a staged input file (and its event directory if that is now empty)."""
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def process_file(path, outdir, reviewed_only=False, compute_psa=True, label=None,
                 stage_dir=None, stage_mode="slim", delete_source=False, fchp_variants=False,
                 generation=None, metadata_only=False):
    """Extract one workspace file. With stage_dir (e.g. node-local /tmp), the file is
    first copied there (whole, or only the parts gmcr reads with stage_mode="slim"),
    read from the copy, and the copy is deleted afterwards, also on errors.

    delete_source=True deletes the input file itself once its outputs are written and
    every trace was extracted without error (for staged batches; never use it on the
    only copy of the data). Files with errors are kept for a rerun.

    fchp_variants=True adds the auto_fchp calibration curves (gmcr.fchp_variants).

    metadata_only=True reads no waveforms (gmcr.features.trace_features(waveforms=False)): for lists
    of every record and its outcome (scripts/record_status.py), at milliseconds per trace.

    generation: the model generation (gmcr.generations) whose auto_fchp computes fchp_gui and the
    displacement-ratio curves. By default, the one for the workspace's gmprocess version; a file
    that no generation covers is an error (give generation= to extract it anyway)."""
    path = Path(path)
    outdir = Path(outdir)
    # an event can have several workspace files, so prefix the ComCat-ID directory
    stem = f"{path.parent.name}__{path.stem}"
    out_tr = outdir / "traces" / f"{stem}.parquet"
    if out_tr.exists():
        if delete_source:
            _delete_source(path)
        return "skipped"
    for sub in ("traces", "curves", "logs"):
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    psa_len = len(ft.NGAW2_T)

    t0 = time.time()
    log = {"file": str(path), "errors": {}, "n_traces": 0, "n_ok": 0}
    rows, crows = [], []
    local = None
    try:
        read_path = path
        if stage_dir:
            local, log["stage_seconds"], log["stage_bytes"] = _stage(path, stage_dir, stage_mode)
            read_path = local
        with Workspace(read_path, label=label) as ws:
            version = ws.gmprocess_version()
            gen = generation or generations.for_gmprocess(version)
            log.update(event_id=ws.event_id, label=ws.label, gmprocess_version=version)
            if gen is None and not metadata_only:  # the record list uses no auto_fchp
                raise ValueError(f"no model generation covers gmprocess {version}; known: "
                                 + ", ".join(generations.GENERATIONS))
            af_version = generations.auto_fchp_version(gen) if gen else None
            log.update(generation=gen,
                       auto_fchp=f"ucla_geotech_tools {af_version}" if af_version else None)
            for tid in ws.processed_trace_ids():
                if reviewed_only:
                    rv = ws.review(tid)
                    if not rv or rv.get("reviewed", 0) != 1:
                        continue
                log["n_traces"] += 1
                key = f"{ws.event_id}|{tid}|{stem}"  # stem = <event folder>__<file stem>
                try:
                    rec, curves = ft.trace_features(ws, tid, compute_psa=compute_psa,
                                                    fchp_variants=fchp_variants,
                                                    auto_fchp=af_version,
                                                    waveforms=not metadata_only)
                    log["n_ok"] += 1
                except Exception:
                    rec, curves = {"event_id": ws.event_id, "trace_id": tid}, {}
                    rec["error"] = traceback.format_exc(limit=3)
                    log["errors"][tid] = rec["error"]
                rec["trace_key"] = key
                rec["gmprocess_version"], rec["generation"] = version, gen
                rows.append(rec)
                crows.append(_curve_row(key, curves, psa_len, fchp_variants))
    except Exception:
        log["file_error"] = traceback.format_exc()
    finally:
        if local is not None:
            for f in (local, local.with_suffix(".tmp")):
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass

    if rows:
        df = tables.parquet_safe(pd.DataFrame(rows))
        tmp = out_tr.with_suffix(".tmp")
        pd.DataFrame(crows).to_parquet(outdir / "curves" / f"{stem}.parquet", index=False)
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out_tr)  # traces file written last = completion marker
    clean = "file_error" not in log and log["n_ok"] == log["n_traces"]
    if not rows and clean:
        # nothing to extract (e.g. no reviewed traces): write an empty marker so reruns skip it
        pd.DataFrame({"trace_key": pd.Series(dtype=str)}).to_parquet(out_tr, index=False)
        pd.DataFrame({"trace_key": pd.Series(dtype=str)}).to_parquet(
            outdir / "curves" / f"{stem}.parquet", index=False)
    if delete_source and clean:
        _delete_source(path)
        log["source_deleted"] = True
    log["seconds"] = round(time.time() - t0, 2)
    (outdir / "logs" / f"{stem}.json").write_text(json.dumps(log, indent=1, default=str))
    return "ok" if "file_error" not in log else "error"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="workspace .h5 files, or a text file of paths with --list")
    p.add_argument("--out", required=True)
    p.add_argument("--list", action="store_true", help="treat each positional arg as a file listing paths")
    p.add_argument("--reviewed-only", action="store_true")
    p.add_argument("--no-psa", action="store_true")
    p.add_argument("--label", default=None, help="processing label (default: auto-detect)")
    p.add_argument("--stage-dir", default=None,
                   help="copy each file here (e.g. node-local /tmp) before reading it, and "
                        "delete the copy afterwards")
    p.add_argument("--stage-mode", choices=["slim", "copy"], default="slim",
                   help="slim: copy only what gmcr reads (default); copy: the whole file")
    p.add_argument("--fchp-variants", action="store_true",
                   help="add polynomial-fit ratio curves for other orders and fit windows "
                        "(for scripts/calibrate_auto_fchp.py)")
    p.add_argument("--metadata-only", action="store_true",
                   help="read no waveforms: metadata, gmprocess's decisions, reviews, cached SNR "
                        "(for record lists, scripts/record_status.py; use without --reviewed-only)")
    p.add_argument("--generation", default=None, choices=list(generations.GENERATIONS),
                   help="model generation whose auto_fchp to use (default: the one for each "
                        "workspace's gmprocess version)")
    p.add_argument("--delete-source", action="store_true",
                   help="delete each input file after it is extracted without errors "
                        "(for staged copies only)")
    a = p.parse_args(argv)


    files = a.files
    if a.list:
        files = [ln.strip() for fn in a.files for ln in Path(fn).read_text().splitlines() if ln.strip()]
    status = {}
    for fn in files:
        if not Path(fn).exists():
            status["missing"] = status.get("missing", 0) + 1
            print(f"missing  {fn}", flush=True)
            continue
        s = process_file(fn, a.out, a.reviewed_only, not a.no_psa, a.label,
                         a.stage_dir, a.stage_mode, a.delete_source, a.fchp_variants,
                         a.generation, a.metadata_only)
        status[s] = status.get(s, 0) + 1
        print(f"{s:8s} {fn}", flush=True)
    print(status)
    return 0 if "error" not in status else 1


if __name__ == "__main__":
    sys.exit(main())
