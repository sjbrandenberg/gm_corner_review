"""Reprocess the NGA-West3 training workspaces (gmprocess 1.3) with gmprocess 2.9, for retraining.

The analysts reviewed records processed with gmprocess 1.3. To train the gmprocess-2.9 model generation
(gmcr.generations), the same raw data are processed again with gmprocess 2.9 and the generation's config
(gmcr/conf/gmprocess-2.9), and the analysts' decisions are joined to the new features by trace id
(scripts/transfer_labels.py). Two steps per batch of staged workspace files:

    prepare  (login node: needs the internet for ANSS SIS metadata, and STREC's data)
        python -m gmcr.reprocess prepare STAGE_DIR OUT_DIR
    For each event in STAGE_DIR (any depth; files grouped by the event id inside them), writes
    OUT_DIR/<event id>/ with event.json (the 1.3 workspace's origin and magnitude, so windows match
    what the analysts saw), strec_results.json, raw/<NET.STA>.xml (the 1.3 StationXML of the
    reviewed stations), the event's SIS snapshot (sis/: gmprocess applies it when assembling,
    offline), and sources.json (which file holds each instrument's raw data). Light: it reads only
    StationXML and review records from the workspace files.

    run  (compute nodes, one task per list of events; no internet)
        python -m gmcr.reprocess run EVENT_DIR [...] --features FEATURES_DIR [--records RECORDS_DIR]
                                     [--delete-source]
    For each event: with --records, first lists every trace of the event's 1.3 files with its outcome
    (gmcr.extract --metadata-only into RECORDS_DIR, for scripts/record_status.py: all traces, also
    those gmprocess or the analyst rejected); then writes the 1.3 unprocessed traces of the reviewed instruments to raw/ as
    MiniSEED, runs `gmrecords assemble` and `process_waveforms` with the generation's config,
    extracts gmcr's features from the new workspace (generation gmprocess-2.9) into FEATURES_DIR,
    then deletes the MiniSEED files and the workspace (--keep keeps them) and, with
    --delete-source, the staged 1.3 files. Writes reprocess.json (status, counts, timings); events
    whose features exist are skipped, so tasks can be rerun.

Only instruments (NET.STA.LOC + band/instrument code, e.g. CI.CRY..HN) with at least one reviewed trace
are reprocessed: the others have no labels. gmprocess 2.9's MiniSEED reader requires integer raw data,
and some 1.3 unprocessed traces are floats: half-count values from interpolation when gappy segments
were merged. They are rounded to whole counts when the instrument's response says the data are counts
(sensitivity >= MIN_COUNTS_SENSITIVITY); traces in physical units (e.g. converted by a format reader)
can't go through MiniSEED and are left out, counted in reprocess.json as "physical_units".
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import generations
from .asdf_reader import UNPROCESSED, Workspace

GENERATION = "gmprocess-2.9"
MIN_COUNTS_SENSITIVITY = 1e3  # counts per physical unit; real digitizers are 1e5 to 1e9
SOURCES = "sources.json"
STATUS = "reprocess.json"


def instrument(trace_id):
    """NET.STA.LOC.II of a trace id (II = band and instrument codes): one gmprocess stream."""
    net, sta, loc, cha = trace_id.split(".")
    return f"{net}.{sta}.{loc}.{cha[:2]}"


def _file_order(path):
    # the main workspace first (as scan_reviews.py prefers it), then by name
    return (path.stem not in ("workspace", path.parent.name.split("__")[-1]), str(path))


# ------------------------------------------------------------------------------ prepare


def group_files(stage_dir):
    """{event id: [workspace files]} of every .h5 file under stage_dir."""
    groups, bad = {}, {}
    for f in sorted(Path(stage_dir).rglob("*.h5")):
        try:
            with Workspace(f) as ws:
                groups.setdefault(ws.event_id, []).append(f)
        except Exception as exc:  # noqa: BLE001
            bad[str(f)] = repr(exc)
    return {e: sorted(fs, key=_file_order) for e, fs in groups.items()}, bad


def reviewed_instruments(ws):
    """Instruments with at least one reviewed trace in workspace ws."""
    out = set()
    for tid in ws.processed_trace_ids():
        rv = ws.review(tid)
        if rv and rv.get("reviewed", 0) == 1:
            out.add(instrument(tid))
    return out


def has_unprocessed(ws, inst):
    tag = f"{ws.event_id}_{UNPROCESSED}"
    return any(t == tag and instrument(tid) == inst for tid, t in ws.waveforms)


def write_event_json(event_dir, event_id, ev):
    """event.json as gmprocess's ScalarEvent.to_json writes it, from the 1.3 workspace's QuakeML."""
    data = {"id": event_id, "time": ev["event_time"], "latitude": ev["event_lat"],
            "longitude": ev["event_lon"], "depth_km": ev["event_depth_km"],
            "magnitude": ev["magnitude"], "magnitude_type": ev["magnitude_type"] or "Mw"}
    (event_dir / "event.json").write_text(json.dumps(data))
    return data


def prepare_event(event_id, files, event_dir, config, sis=True):
    """Write an event's folder for `run` (see the module docstring). Returns a summary dict."""
    from gmprocess.core.scalar_event import ScalarEvent
    from gmprocess.extended_xml import sis_metadata, sis_snapshot
    from gmprocess.utils.strec import STREC

    event_dir = Path(event_dir)
    raw = event_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sources, xml_from, event = {}, {}, None
    for f in files:
        with Workspace(f) as ws:
            if event is None:
                event = write_event_json(event_dir, event_id, ws.event())
            for inst in sorted(reviewed_instruments(ws)):
                if inst in sources or not has_unprocessed(ws, inst):
                    continue
                sources[inst] = str(f)
                netsta = ".".join(inst.split(".")[:2])
                if netsta not in xml_from:
                    raw_xml = np.asarray(ws.f[f"Waveforms/{netsta}/StationXML"][()]).tobytes()
                    (raw / f"{netsta}.xml").write_bytes(raw_xml)
                    xml_from[netsta] = str(f)
    summary = {"event_id": event_id, "files": [str(f) for f in files],
               "instruments": len(sources), "stations": len(xml_from)}
    (event_dir / SOURCES).write_text(json.dumps(
        {**summary, "generation": GENERATION, "sources": sources, "stationxml_from": xml_from},
        indent=1))
    if event is None or not sources:
        summary["skipped"] = "no reviewed instruments with raw data"
        return summary

    from obspy import UTCDateTime

    ev = ScalarEvent.from_params(event_id, UTCDateTime(event["time"]), event["latitude"], event["longitude"],
                                 event["depth_km"], event["magnitude"], event["magnitude_type"])
    STREC.from_event(ev).to_file(event_dir / "strec_results.json")
    if sis:
        options = sis_metadata.sis_options_from_config(
            config["fetchers"]["FDSNFetcher"].get("sis"), force=True)
        stations = sorted(tuple(ns.split(".")) for ns in xml_from)
        manifest = sis_snapshot.refresh_snapshot(event_dir, stations, **options)
        outcomes = [((s.get("latest") or {}).get("outcome")) for s in
                    (manifest.get("stations") or {}).values()] if isinstance(manifest, dict) else []
        summary["sis"] = {o: outcomes.count(o) for o in set(outcomes)}
    return summary


def prepare(stage_dir, out_dir, sis=True):
    from gmprocess.utils.config import get_config

    config = get_config(generations.config_dir(GENERATION))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups, bad = group_files(stage_dir)
    print(f"{sum(map(len, groups.values()))} files of {len(groups)} events in {stage_dir}"
          + (f"; {len(bad)} unreadable" if bad else ""), flush=True)
    rows = []
    for i, (event_id, files) in enumerate(sorted(groups.items()), 1):
        event_dir = out_dir / event_id
        if (event_dir / SOURCES).exists() and (not sis or (event_dir / "sis").exists()):
            rows.append(json.loads((event_dir / SOURCES).read_text()))
            continue
        t0 = time.time()
        try:
            s = prepare_event(event_id, files, event_dir, config, sis=sis)
        except Exception:  # noqa: BLE001
            s = {"event_id": event_id, "error": traceback.format_exc(limit=3)}
            (event_dir / SOURCES).unlink(missing_ok=True)
        s["seconds"] = round(time.time() - t0, 1)
        rows.append(s)
        print(f"[{i}/{len(groups)}] {event_id}: {s.get('instruments', 0)} instruments, "
              f"{s.get('stations', 0)} stations, SIS {s.get('sis', '-')}"
              + (f", {s['skipped']}" if "skipped" in s else "")
              + (f", ERROR {s['error'].strip().splitlines()[-1]}" if "error" in s else ""), flush=True)
    report = {"stage_dir": str(stage_dir), "events": rows, "unreadable_files": bad}
    (out_dir / "prepare.json").write_text(json.dumps(report, indent=1, default=str))
    ready = [r["event_id"] for r in rows if "error" not in r and "skipped" not in r]
    print(f"{len(ready)} events ready to run; {sum('error' in r for r in rows)} errors "
          f"(see {out_dir / 'prepare.json'})", flush=True)
    return ready


# ---------------------------------------------------------------------------------- run


def export_waveforms(event_dir):
    """Write the 1.3 unprocessed traces of the event's instruments to raw/ as MiniSEED.
    Returns counts by kind: integer, rounded (counts stored as floats), physical_units (left out)."""
    import obspy

    event_dir = Path(event_dir)
    src = json.loads((event_dir / SOURCES).read_text())
    by_file = {}
    for inst, f in src["sources"].items():
        by_file.setdefault(f, set()).add(inst)
    counts = {"integer": 0, "rounded": 0, "physical_units": 0}
    left_out = []
    for f, insts in by_file.items():
        with Workspace(f) as ws:
            tag = f"{ws.event_id}_{UNPROCESSED}"
            for (tid, t), path in sorted(ws.waveforms.items()):
                if t != tag or instrument(tid) not in insts:
                    continue
                ds = ws.f[path]
                data = np.asarray(ds[()])
                net, sta, loc, cha = tid.split(".")
                if data.dtype.kind in "iu":
                    kind = "integer"
                else:
                    kind = "rounded"
                    inv = ws.inventory(f"{net}.{sta}")
                    try:
                        sens = inv.select(location=loc, channel=cha)[0][0][0].response \
                            .instrument_sensitivity.value
                    except Exception:  # noqa: BLE001
                        sens = None
                    if not sens or abs(sens) < MIN_COUNTS_SENSITIVITY:
                        counts["physical_units"] += 1
                        left_out.append(tid)
                        continue
                    if np.allclose(data, np.round(data), rtol=0, atol=1e-6):
                        kind = "integer"
                counts[kind] += 1
                tr = obspy.Trace(data=np.round(data).astype(np.int32), header={
                    "network": net, "station": sta, "location": loc, "channel": cha,
                    "sampling_rate": float(ds.attrs["sampling_rate"]),
                    "starttime": obspy.UTCDateTime(ns=int(ds.attrs["starttime"]))})
                t0 = tr.stats.starttime.strftime("%Y%m%dT%H%M%SZ")
                t1 = tr.stats.endtime.strftime("%Y%m%dT%H%M%SZ")
                tr.write(str(event_dir / "raw" / f"{tid}__{t0}__{t1}.mseed"), format="MSEED", encoding="INT32")
    return counts, left_out


def gmrecords(event_dir, subcommand):
    """Run a gmrecords subcommand on one event folder with the generation's config (one process,
    overwriting earlier output); raises on failure."""
    event_dir = Path(event_dir)
    cmd = [sys.executable, "-m", "gmprocess.apps.gmrecords", "--datadir", str(event_dir.parent),
           "--confdir", str(generations.config_dir(GENERATION)), "-o",
           "--log", str(event_dir / f"gmrecords_{subcommand}.log"), "-e", event_dir.name, subcommand]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:])
        raise RuntimeError(f"gmrecords {subcommand} failed ({r.returncode}):\n{tail}")


def run_event(event_dir, features_dir, keep=False, delete_source=False, records_dir=None):
    """Reprocess and extract one prepared event folder (after listing its 1.3 records into
    records_dir, if given). Returns its status dict."""
    from .extract import process_file

    event_dir = Path(event_dir)
    batch = event_dir.parent.name
    workspace = event_dir / f"workspace_{batch}.h5"  # unique file stem across batches
    stem = f"{event_dir.name}__{workspace.stem}"
    status_path = event_dir / STATUS
    if (Path(features_dir) / "traces" / f"{stem}.parquet").exists():
        return {"event_id": event_dir.name, "status": "skipped"}
    st = {"event_id": event_dir.name, "batch": batch, "generation": GENERATION}
    t = time.time()
    try:
        src = json.loads((event_dir / SOURCES).read_text())
        if records_dir:
            st["records"] = {f: process_file(f, records_dir, metadata_only=True) for f in src["files"]}
            st["records_seconds"] = round(time.time() - t, 1)
        if not src.get("sources"):
            st["status"] = "nothing to run"
            return st
        st["export"], st["left_out_physical_units"] = export_waveforms(event_dir)
        st["export_seconds"] = round(time.time() - t, 1)
        t1 = time.time()
        (event_dir / "workspace.h5").unlink(missing_ok=True)
        gmrecords(event_dir, "assemble")
        st["assemble_seconds"] = round(time.time() - t1, 1)
        t1 = time.time()
        gmrecords(event_dir, "process_waveforms")
        st["process_seconds"] = round(time.time() - t1, 1)
        (event_dir / "workspace.h5").replace(workspace)
        t1 = time.time()
        st["extract"] = process_file(workspace, features_dir, generation=GENERATION)
        st["extract_seconds"] = round(time.time() - t1, 1)
        log = Path(features_dir) / "logs" / f"{stem}.json"
        if log.exists():
            lg = json.loads(log.read_text())
            st.update(n_traces=lg.get("n_traces"), n_ok=lg.get("n_ok"),
                      gmprocess_version=lg.get("gmprocess_version"))
        st["status"] = "ok" if st["extract"] == "ok" else "extract error"
    except Exception:  # noqa: BLE001
        st["status"] = "error"
        st["error"] = traceback.format_exc(limit=4)
    finally:
        st["seconds"] = round(time.time() - t, 1)
        if st.get("status") == "ok" and not keep:
            for f in (event_dir / "raw").glob("*.mseed"):
                f.unlink()
            workspace.unlink(missing_ok=True)
        records_ok = all(v in ("ok", "skipped") for v in (st.get("records") or {}).values())
        if st.get("status") in ("ok", "nothing to run") and delete_source and records_ok:
            for f in json.loads((event_dir / SOURCES).read_text())["files"]:
                Path(f).unlink(missing_ok=True)
        status_path.write_text(json.dumps(st, indent=1, default=str))
    return st


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare", help="login node: event folders, StationXML, STREC, SIS snapshot")
    pp.add_argument("stage_dir", help="folder of staged 1.3 workspace files (searched at any depth)")
    pp.add_argument("out_dir", help="folder for the event folders (one batch)")
    pp.add_argument("--no-sis", action="store_true", help="skip the SIS snapshot (no internet)")
    pr = sub.add_parser("run", help="compute node: export raw data, gmprocess 2.9, extract")
    pr.add_argument("event_dirs", nargs="+", help="prepared event folders (or files listing them, with --list)")
    pr.add_argument("--list", action="store_true", help="each argument is a text file of event folders")
    pr.add_argument("--features", required=True, help="gmcr features folder (shared by all tasks)")
    pr.add_argument("--records", default=None,
                    help="first list every trace of the 1.3 files with its outcome (gmcr.extract "
                         "--metadata-only) into this folder, for scripts/record_status.py")
    pr.add_argument("--keep", action="store_true", help="keep the MiniSEED files and the 2.9 workspace")
    pr.add_argument("--delete-source", action="store_true",
                    help="delete the staged 1.3 files of each event that finished (staged copies only)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if a.cmd == "prepare":
        prepare(a.stage_dir, a.out_dir, sis=not a.no_sis)
        return 0
    dirs = a.event_dirs
    if a.list:
        dirs = [ln.strip() for fn in dirs for ln in Path(fn).read_text().splitlines() if ln.strip()]
    n_err = 0
    for d in dirs:
        st = run_event(d, a.features, keep=a.keep, delete_source=a.delete_source,
                       records_dir=a.records)
        n_err += st["status"] not in ("ok", "skipped", "nothing to run")
        print(f"{st['status']:14s} {d} {st.get('seconds', '')}s "
              f"{st.get('n_ok', '')}/{st.get('n_traces', '')} traces {st.get('export', '')}"
              + (f"\n  {st['error'].strip().splitlines()[-1]}" if "error" in st else ""), flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
