"""Reprocess the NGA-West3 training workspaces (gmprocess 1.3) with gmprocess 2.9, for retraining.

The analysts reviewed records processed with gmprocess 1.3. To train the gmprocess-2.9 model generation
(gmcr.generations), the same raw data are processed again with gmprocess 2.9 and the generation's config
(gmcr/conf/gmprocess-2.9), and the analysts' decisions are joined to the new features by trace id
(scripts/transfer_labels.py). Two steps per batch of staged workspace files:

    prepare  (login node: needs the internet for ANSS SIS metadata, and STREC's data)
        python -m gmcr.reprocess prepare STAGE_DIR OUT_DIR [--sis-cache DIR] [--workers 8]
    For each event in STAGE_DIR (any depth; files grouped by the event id inside them), writes
    OUT_DIR/<event id>/ with event.json (the 1.3 workspace's origin and magnitude, so windows match
    what the analysts saw), strec_results.json, raw/<NET.STA>.xml (the 1.3 StationXML of the
    reviewed stations), the event's SIS snapshot (sis/: gmprocess applies it when assembling,
    offline), and sources.json (which file holds each instrument's raw data). Light: it reads only
    StationXML and review records from the workspace files, and asks SIS about each station only
    once (SISCache: a station's answer is reused for every event, in every batch), a few events at
    a time.

    run  (compute nodes, one task per list of events; no internet)
        python -m gmcr.reprocess run EVENT_DIR [...] --features FEATURES_DIR [--records RECORDS_DIR]
                                     [--delete-source]
    For each event: with --records, first lists every trace of the event's 1.3 files with its outcome
    (gmcr.extract --metadata-only into RECORDS_DIR, for scripts/record_status.py: all traces, also
    those gmprocess or the analyst rejected); then hands the 1.3 unprocessed traces of the reviewed
    instruments to gmprocess 2.9 unchanged (export_waveforms, assemble_event), runs gmprocess's
    assemble and `gmrecords process_waveforms` with the generation's config,
    extracts gmcr's features from the new workspace (generation gmprocess-2.9) into FEATURES_DIR,
    then deletes the MiniSEED files and the workspace (--keep keeps them) and, with
    --delete-source, the staged 1.3 files. Writes reprocess.json (status, counts, timings); events
    whose features exist are skipped, so tasks can be rerun.

Only instruments (NET.STA.LOC + band/instrument code, e.g. CI.CRY..HN) with at least one reviewed trace
are reprocessed: the others have no labels. The 1.3 unprocessed traces are of two kinds (their process
level in the workspace): raw counts from FDSN data, which go through MiniSEED and gmprocess's own
assemble (with the SIS snapshot), unrounded even when 1.3 stored them as floats (merged or
interpolated segments); and data a format reader already converted to physical units (CESMD's DMG and
COSMOS files: cm/s^2, response removed), which no raw file can carry and are added from the 1.3
workspace with gmprocess 2.9's own reader. reprocess.json counts both (integer, float_counts,
physical_units).
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


class SISCache:
    """A `retrieve` function for gmprocess's refresh_snapshot that asks SIS about each station once.

    SIS metadata belong to a station, not an event, and the same stations recur in many events, so
    each station's answer is kept in cache_dir (its response bytes, gzipped, and the outcome) and
    reused for every later event, also in later batches and reruns. Only definite answers are kept:
    usable, unusable, and no coverage (HTTP 404); a failed download is tried again next time. A
    reused answer is rebuilt the way gmprocess builds it (parse_station_xml, select_sis_station),
    with the time of the original download. Thread-safe: one download per station at a time.
    """

    def __init__(self, cache_dir):
        import threading

        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._guard = threading.Lock()
        self._locks = {}
        self.hits = self.downloads = 0

    def _paths(self, net, sta):
        return self.dir / f"{net}.{sta}.json", self.dir / f"{net}.{sta}.xml.gz"

    def _load(self, net, sta):
        import gzip

        from gmprocess.extended_xml import sis_metadata as sm
        from gmprocess.extended_xml.extended_xml_parser import parse_station_xml

        meta_path, content_path = self._paths(net, sta)
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        content = gzip.decompress(content_path.read_bytes()) if content_path.exists() else None
        r = sm.SISRetrieval(**meta, content=content)
        if r.outcome == sm.SIS_USABLE:
            r.station_xml = parse_station_xml(content)
            r.sis_station, _ = sm.select_sis_station(r.station_xml, net, sta)
        return r

    def _save(self, r):
        import gzip

        meta_path, content_path = self._paths(r.network, r.station)
        if r.content is not None:
            tmp = content_path.with_name(content_path.name + ".tmp")
            tmp.write_bytes(gzip.compress(r.content))
            tmp.replace(content_path)
        meta = {k: getattr(r, k) for k in ("network", "station", "url", "outcome", "retrieved_at",
                                            "attempts", "reason")}
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        tmp.write_text(json.dumps(meta))
        tmp.replace(meta_path)  # written last: marks the entry complete

    def seed(self, event_dirs):
        """Add the answers already saved in events' SIS snapshots (sis/manifest.json and its
        records), so a rerun or a cache started after some events were done doesn't ask again.
        Returns the number of stations added."""
        import gzip

        from gmprocess.extended_xml import sis_metadata as sm

        added = 0
        for d in event_dirs:
            mpath = Path(d) / "sis" / "manifest.json"
            if not mpath.exists():
                continue
            for sid, entry in (json.loads(mpath.read_text()).get("stations") or {}).items():
                net, sta = sid.split(".", 1)
                if self._paths(net, sta)[0].exists():
                    continue
                latest, active = entry.get("latest") or {}, entry.get("active") or {}
                outcome = latest.get("outcome")
                if outcome == sm.SIS_USABLE and active.get("file"):
                    rec = Path(d) / "sis" / "records" / active["file"]
                    if not rec.exists():
                        continue
                    r = sm.SISRetrieval(net, sta, active.get("url") or latest.get("url", ""), outcome,
                                        active.get("retrieved_at", ""), latest.get("attempts", 1),
                                        content=gzip.decompress(rec.read_bytes()))
                elif outcome == sm.SIS_NO_COVERAGE:
                    r = sm.SISRetrieval(net, sta, latest.get("url", ""), outcome,
                                        latest.get("attempted_at", ""), latest.get("attempts", 1),
                                        reason=latest.get("reason", ""))
                else:
                    continue
                self._save(r)
                added += 1
        return added

    def __call__(self, net, sta, timeout, retries=0, retry_delay=0.0):
        from gmprocess.extended_xml import sis_metadata as sm

        with self._guard:
            lock = self._locks.setdefault((net, sta), __import__("threading").Lock())
        with lock:
            r = self._load(net, sta)
            if r is not None:
                with self._guard:
                    self.hits += 1
                return r
            r = sm.retrieve_sis_station(net, sta, timeout, retries=retries, retry_delay=retry_delay)
            with self._guard:
                self.downloads += 1
            if r.outcome in (sm.SIS_USABLE, sm.SIS_UNUSABLE, sm.SIS_NO_COVERAGE):
                self._save(r)
            return r


def sis_event(event_dir, options, retrieve):
    """Save an event's SIS snapshot (sis/) for the stations in its sources.json. Returns the
    outcome counts."""
    from gmprocess.extended_xml import sis_snapshot

    event_dir = Path(event_dir)
    src = json.loads((event_dir / SOURCES).read_text())
    stations = sorted(tuple(ns.split(".")) for ns in src["stationxml_from"])
    manifest = sis_snapshot.refresh_snapshot(event_dir, stations, retrieve=retrieve, **options)
    outcomes = [((s.get("latest") or {}).get("outcome")) for s in
                (manifest.get("stations") or {}).values()] if isinstance(manifest, dict) else []
    return {o: outcomes.count(o) for o in set(outcomes)}


def prepare_event(event_id, files, event_dir):
    """Write an event's folder for `run` (see the module docstring), except its SIS snapshot
    (sis_event). Returns a summary dict."""
    from gmprocess.core.scalar_event import ScalarEvent
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
    return summary


def prepare(stage_dir, out_dir, sis=True, sis_cache=None, workers=8):
    """Prepare every event staged in stage_dir: first each event's folder, one event after another
    (reads the workspaces' reviews and StationXML, runs STREC), then the SIS snapshots, `workers`
    events at a time (network waits only), through a SISCache in sis_cache (default
    <out_dir>/../sis_cache, shared by the batches). Events already done are skipped."""
    from concurrent.futures import ThreadPoolExecutor

    from gmprocess.extended_xml import sis_metadata
    from gmprocess.utils.config import get_config

    config = get_config(generations.config_dir(GENERATION))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups, bad = group_files(stage_dir)
    print(f"{sum(map(len, groups.values()))} files of {len(groups)} events in {stage_dir}"
          + (f"; {len(bad)} unreadable" if bad else ""), flush=True)

    rows = {}
    t_events = time.time()
    for i, (event_id, files) in enumerate(sorted(groups.items()), 1):
        event_dir = out_dir / event_id
        src = event_dir / SOURCES
        if src.exists():
            info = json.loads(src.read_text())
            if not info.get("sources") or (event_dir / "strec_results.json").exists():
                rows[event_id] = {k: info[k] for k in ("event_id", "files", "instruments", "stations")}
                if not info.get("sources"):
                    rows[event_id]["skipped"] = "no reviewed instruments with raw data"
                continue
        t0 = time.time()
        try:
            s = prepare_event(event_id, files, event_dir)
        except Exception:  # noqa: BLE001
            s = {"event_id": event_id, "error": traceback.format_exc(limit=3)}
            src.unlink(missing_ok=True)
        s["seconds"] = round(time.time() - t0, 1)
        rows[event_id] = s
        print(f"[{i}/{len(groups)}] {event_id}: {s.get('instruments', 0)} instruments, "
              f"{s.get('stations', 0)} stations, {s['seconds']} s"
              + (f", {s['skipped']}" if "skipped" in s else "")
              + (f", ERROR {s['error'].strip().splitlines()[-1]}" if "error" in s else ""), flush=True)
    t_events = time.time() - t_events

    todo = [e for e, s in rows.items() if "error" not in s and "skipped" not in s
            and not (out_dir / e / "sis" / "manifest.json").exists()]
    t_sis = time.time()
    if sis and todo:
        options = sis_metadata.sis_options_from_config(
            config["fetchers"]["FDSNFetcher"].get("sis"), force=True)
        cache = SISCache(sis_cache or out_dir.parent / "sis_cache")
        seeded = cache.seed(p for p in out_dir.iterdir() if p.is_dir())
        print(f"SIS metadata for {len(todo)} events, {workers} at a time (station cache {cache.dir}"
              + (f", {seeded} stations added from earlier snapshots" if seeded else "") + ")",
              flush=True)

        def one(event_id):
            try:
                return event_id, sis_event(out_dir / event_id, options, cache)
            except Exception:  # noqa: BLE001
                return event_id, {"error": traceback.format_exc(limit=3)}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for k, (event_id, res) in enumerate(pool.map(one, todo), 1):
                if "error" in res:
                    rows[event_id]["sis_error"] = res["error"]
                else:
                    rows[event_id]["sis"] = res
                print(f"[SIS {k}/{len(todo)}] {event_id}: {res if 'error' not in res else 'ERROR ' + res['error'].strip().splitlines()[-1]}"
                      f"  (cache: {cache.downloads} downloaded, {cache.hits} reused)", flush=True)
    t_sis = time.time() - t_sis
    rows = list(rows.values())
    report = {"stage_dir": str(stage_dir), "events": rows, "unreadable_files": bad,
              "seconds": {"events": round(t_events, 1), "sis": round(t_sis, 1)}}
    (out_dir / "prepare.json").write_text(json.dumps(report, indent=1, default=str))
    ready = [r["event_id"] for r in rows if "error" not in r and "skipped" not in r
             and (not sis or (out_dir / r["event_id"] / "sis" / "manifest.json").exists())]
    n_sis_err = sum("sis_error" in r for r in rows)
    print(f"{len(ready)} events ready to run; {sum('error' in r for r in rows)} errors"
          + (f", {n_sis_err} SIS errors (rerun to retry)" if n_sis_err else "")
          + f"; {t_events / 60:.1f} min for the events, {t_sis / 60:.1f} min for SIS "
          f"(see {out_dir / 'prepare.json'})", flush=True)
    return ready


# ---------------------------------------------------------------------------------- run


EXPORT = "export.json"
NOT_INTEGER = "Raw data type is not integer."


def process_levels(ws):
    """{instrument: gmprocess process level} of a workspace's unprocessed streams ('raw counts' for
    FDSN data; 'uncorrected physical units' etc. for data a format reader converted, e.g. DMG/COSMOS
    from CESMD), from AuxiliaryData/StreamSupplementalStats."""
    out = {}
    grp = ws.f.get("AuxiliaryData/StreamSupplementalStats")
    if grp is None:
        return out
    suffix = f"_{ws.event_id}_{UNPROCESSED}"
    for netsta in grp:
        for name in grp[netsta]:
            if name.endswith(suffix):
                d = json.loads(np.asarray(grp[netsta][name][()]).tobytes().decode())
                out[name[: -len(suffix)]] = (d.get("standard") or {}).get("process_level")
    return out


def _counts_by_response(ws, tid):
    """Fallback without a recorded process level: counts if the response's sensitivity is that of
    a digitizer."""
    net, sta, loc, cha = tid.split(".")
    try:
        sens = ws.inventory(f"{net}.{sta}").select(location=loc, channel=cha)[0][0][0] \
            .response.instrument_sensitivity.value
    except Exception:  # noqa: BLE001
        return False
    return bool(sens) and abs(sens) >= MIN_COUNTS_SENSITIVITY


def export_waveforms(event_dir):
    """Hand the 1.3 unprocessed traces of the event's instruments to gmprocess 2.9 unchanged:

    raw counts (FDSN data): written to raw/ as MiniSEED for `assemble` (which also applies the SIS
        snapshot), as int32, or as float64 when 1.3 stored them as floats (merged or interpolated
        segments); those are listed in export.json, and `assemble_event` waives gmprocess's "raw data
        type is not integer" check for exactly them instead of rounding the data;
    physical units (data a format reader converted, e.g. CESMD's DMG files): no raw file can carry
        them, so `assemble_event` adds these streams from the 1.3 workspace with gmprocess 2.9's own
        reader, keeping their units, process level, and response-removal provenance, as 2.9 would
        have read them.

    Returns counts of traces: integer, float_counts, physical_units."""
    import obspy

    event_dir = Path(event_dir)
    src = json.loads((event_dir / SOURCES).read_text())
    by_file = {}
    for inst, f in src["sources"].items():
        by_file.setdefault(f, set()).add(inst)
    counts = {"integer": 0, "float_counts": 0, "physical_units": 0}
    float_ids, inject = [], {}
    for f, insts in by_file.items():
        with Workspace(f) as ws:
            levels = process_levels(ws)
            tag = f"{ws.event_id}_{UNPROCESSED}"
            for (tid, t), path in sorted(ws.waveforms.items()):
                inst = instrument(tid)
                if t != tag or inst not in insts:
                    continue
                ds = ws.f[path]
                data = np.asarray(ds[()])
                level = levels.get(inst)
                is_counts = (level == "raw counts") if level else \
                    (data.dtype.kind in "iu" or _counts_by_response(ws, tid))
                if not is_counts:
                    counts["physical_units"] += 1
                    inject.setdefault(f, set()).add(inst)
                    continue
                net, sta, loc, cha = tid.split(".")
                if data.dtype.kind in "iu" or np.array_equal(data, np.round(data)):
                    counts["integer"] += 1
                    data, encoding = data.astype(np.int32), "INT32"
                else:
                    counts["float_counts"] += 1
                    float_ids.append(tid)
                    data, encoding = data.astype(np.float64), "FLOAT64"
                tr = obspy.Trace(data=data, header={
                    "network": net, "station": sta, "location": loc, "channel": cha,
                    "sampling_rate": float(ds.attrs["sampling_rate"]),
                    "starttime": obspy.UTCDateTime(ns=int(ds.attrs["starttime"]))})
                t0 = tr.stats.starttime.strftime("%Y%m%dT%H%M%SZ")
                t1 = tr.stats.endtime.strftime("%Y%m%dT%H%M%SZ")
                tr.write(str(event_dir / "raw" / f"{tid}__{t0}__{t1}.mseed"), format="MSEED",
                         encoding=encoding)
    (event_dir / EXPORT).write_text(json.dumps(
        {"counts": counts, "float_counts": float_ids,
         "inject": {f: sorted(v) for f, v in inject.items()}}, indent=1))
    return counts


def assemble_event(event_dir):
    """gmprocess 2.9's assemble for one exported event folder, run in this process (see
    export_waveforms): the MiniSEED traces through gmprocess's assemble, with its "raw data type is
    not integer" check waived only for the float count traces export.json lists, then the
    physical-unit streams from the 1.3 workspaces added as unprocessed streams. Writes
    <event_dir>/workspace.h5 and gmrecords_assemble.log."""
    import logging as log
    from importlib.metadata import version

    from gmprocess.core.stationtrace import StationTrace
    from gmprocess.core.streamcollection import StreamCollection
    from gmprocess.io.asdf.stream_workspace import StreamWorkspace
    from gmprocess.utils import assemble_utils
    from gmprocess.utils.config import get_config

    event_dir = Path(event_dir)
    log.basicConfig(filename=str(event_dir / "gmrecords_assemble.log"), level=log.INFO, force=True)
    exp = json.loads((event_dir / EXPORT).read_text())
    waived = {t.replace(".--.", "..") for t in exp["float_counts"]}
    fail = StationTrace.fail

    def fail_unless_waived(trace, reason):
        tid = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}"
        if reason == NOT_INTEGER and tid.replace(".--.", "..") in waived:
            return  # 1.3 stored these counts as floats; keep them unrounded
        return fail(trace, reason)

    config = get_config(generations.config_dir(GENERATION))
    gmp_version = version("gmprocess")
    (event_dir / "workspace.h5").unlink(missing_ok=True)
    StationTrace.fail = fail_unless_waived
    try:
        ws = assemble_utils.assemble(event_dir.name, None, config, event_dir.parent, gmp_version,
                                     "default")
    finally:
        StationTrace.fail = fail
    if ws is None:
        raise RuntimeError(f"gmprocess's assemble found no event in {event_dir}")
    try:
        event = ws.get_event(event_dir.name)
        for f, insts in exp["inject"].items():
            w13 = StreamWorkspace.open(f)
            try:
                stations = sorted({".".join(i.split(".")[:2]) for i in insts})
                streams = [st for st in w13.get_streams(event_dir.name, stations=stations,
                                                        labels=[UNPROCESSED], config=config)
                           if instrument(st[0].id) in set(insts)]
            finally:
                w13.close()
            if streams:
                ws.add_streams(event, StreamCollection(streams, config=config),
                               label=UNPROCESSED, gmprocess_version=gmp_version)
                log.info("added %d physical-unit streams from %s", len(streams), f)
    finally:
        ws.close()


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
        for f in (event_dir / "raw").glob("*.mseed"):  # leftovers of an earlier attempt
            f.unlink()
        t1 = time.time()
        st["export"] = export_waveforms(event_dir)
        st["export_seconds"] = round(time.time() - t1, 1)
        t1 = time.time()
        (event_dir / "workspace.h5").unlink(missing_ok=True)
        r = subprocess.run([sys.executable, "-m", "gmcr.reprocess", "assemble", str(event_dir)],
                           capture_output=True, text=True)
        if r.returncode:
            tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:])
            raise RuntimeError(f"assemble failed ({r.returncode}):\n{tail}")
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
                try:
                    Path(f).parent.rmdir()  # the staged event folder, once empty
                except OSError:
                    pass
        status_path.write_text(json.dumps(st, indent=1, default=str))
    return st


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare", help="login node: event folders, StationXML, STREC, SIS snapshot")
    pp.add_argument("stage_dir", help="folder of staged 1.3 workspace files (searched at any depth)")
    pp.add_argument("out_dir", help="folder for the event folders (one batch)")
    pp.add_argument("--no-sis", action="store_true", help="skip the SIS snapshot (no internet)")
    pp.add_argument("--sis-cache", default=None,
                    help="folder of each station's SIS answer, reused across events and batches "
                         "(default: OUT_DIR/../sis_cache)")
    pp.add_argument("--workers", type=int, default=8,
                    help="events whose SIS metadata are fetched at a time (default 8)")
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
    pa = sub.add_parser("assemble", help="(used by run) gmprocess 2.9's assemble of one exported event")
    pa.add_argument("event_dir")
    a = p.parse_args(argv)
    if a.cmd == "assemble":
        assemble_event(a.event_dir)
        return 0
    logging.basicConfig(level=logging.WARNING)

    if a.cmd == "prepare":
        prepare(a.stage_dir, a.out_dir, sis=not a.no_sis, sis_cache=a.sis_cache, workers=a.workers)
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
