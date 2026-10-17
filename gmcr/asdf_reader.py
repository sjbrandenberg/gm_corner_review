"""Lightweight, read-only access to gmprocess workspace files using h5py.

gmprocess itself is not required. The layout handled here is the one written by
gmprocess 1.3.x plus the NGA-West3 review GUI:

    Waveforms/<NET.STA>/<NET.STA.LOC.CHA>__<start>__<end>__<eventid>_<label>
    Waveforms/<NET.STA>/StationXML
    QuakeML
    Provenance/<NET.STA.LOC.CHA>_<eventid>_<label>
    AuxiliaryData/TraceProcessingParameters/<NET.STA>/<NET.STA.LOC.CHA>_<eventid>_<label>
    AuxiliaryData/Cache/<CacheName>/<NET.STA>/<NET.STA.LOC.CHA>_<eventid>_<label>
    AuxiliaryData/review/<NET.STA>/<NET.STA.LOC.CHA>_<eventid>_<label>/{accepted,colocation,corner_frequencies/*}
"""

import io
import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

UNPROCESSED = "unprocessed"
PROV_NS = "{http://www.w3.org/ns/prov#}"
SEIS_NS = "{http://seisprov.org/seis_prov/0.1/#}"


def _bytes_to_str(ds):
    return bytes(np.asarray(ds[()]).tobytes()).decode("utf-8", errors="replace")


def _attrs(ds):
    out = {}
    for k, v in ds.attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


def _to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


class Workspace:
    """Read-only view of one gmprocess workspace file."""

    def __init__(self, path, label=None):
        self.path = Path(path)
        self.f = h5py.File(self.path, "r")
        self.waveforms = self._index_waveforms()
        self.event_id = self._find_event_id()
        self.label = label or self._choose_label()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ index
    def _index_waveforms(self):
        """Map (trace_id, tag) -> h5 path for every waveform dataset."""
        index = {}
        for netsta, grp in self.f["Waveforms"].items():
            for name in grp:
                if name == "StationXML":
                    continue
                parts = name.split("__")
                if len(parts) != 4:
                    continue
                trace_id, _, _, tag = parts
                index[(trace_id, tag)] = f"Waveforms/{netsta}/{name}"
        return index

    def _find_event_id(self):
        for path in self.waveforms.values():
            eid = self.f[path].attrs.get("event_id")
            if eid is not None:
                return eid.decode() if isinstance(eid, bytes) else str(eid)
        return self.path.stem

    def labels(self):
        """Processing labels present besides 'unprocessed'."""
        prefix = self.event_id + "_"
        labels = set()
        for _, tag in self.waveforms:
            if tag.startswith(prefix) and tag != prefix + UNPROCESSED:
                labels.add(tag[len(prefix):])
        return sorted(labels)

    def _choose_label(self):
        labels = self.labels()
        if not labels:
            raise ValueError(f"{self.path}: no processed waveforms found")
        if "review" in self.f.get("AuxiliaryData", {}):
            reviewed = {
                k.rsplit("_", 1)[-1]
                for sta in self.f["AuxiliaryData/review"].values()
                for k in sta
            }
            for lab in labels:
                if lab in reviewed:
                    return lab
        return labels[0]

    def processed_trace_ids(self):
        tag = f"{self.event_id}_{self.label}"
        return sorted(tid for tid, t in self.waveforms if t == tag)

    def aux_key(self, trace_id):
        return f"{trace_id}_{self.event_id}_{self.label}"

    @staticmethod
    def netsta(trace_id):
        net, sta = trace_id.split(".")[:2]
        return f"{net}.{sta}"

    # -------------------------------------------------------------- waveforms
    def waveform(self, trace_id, processed=True):
        """Return (data, sampling_rate, starttime_ns) or None."""
        tag = f"{self.event_id}_{self.label}" if processed else f"{self.event_id}_{UNPROCESSED}"
        path = self.waveforms.get((trace_id, tag))
        if path is None:
            return None
        ds = self.f[path]
        return (
            np.asarray(ds[()], dtype=float),
            float(ds.attrs["sampling_rate"]),
            int(ds.attrs["starttime"]),
        )

    def obspy_trace(self, trace_id, processed=True):
        import obspy

        wf = self.waveform(trace_id, processed)
        if wf is None:
            return None
        data, sr, t0 = wf
        net, sta, loc, cha = trace_id.split(".")
        header = {
            "network": net, "station": sta, "location": loc, "channel": cha,
            "sampling_rate": sr, "starttime": obspy.UTCDateTime(ns=t0),
        }
        return obspy.Trace(data=data, header=header)

    @lru_cache(maxsize=None)
    def inventory(self, netsta):
        import obspy

        raw = np.asarray(self.f[f"Waveforms/{netsta}/StationXML"][()]).tobytes()
        return obspy.read_inventory(io.BytesIO(raw), format="STATIONXML")

    @lru_cache(maxsize=1)
    def event(self):
        """Origin/magnitude summary from QuakeML (falls back to regex-free XML parse)."""
        import obspy

        raw = np.asarray(self.f["QuakeML"][()]).tobytes()
        cat = obspy.read_events(io.BytesIO(raw), format="QUAKEML")
        ev = cat[0]
        org = ev.preferred_origin() or ev.origins[0]
        mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        return {
            "event_time": str(org.time),
            "event_lat": org.latitude,
            "event_lon": org.longitude,
            "event_depth_km": (org.depth or np.nan) / 1000.0,
            "magnitude": mag.mag if mag else np.nan,
            "magnitude_type": mag.magnitude_type if mag else None,
        }

    # ---------------------------------------------------------- aux metadata
    def processing_parameters(self, trace_id):
        path = f"AuxiliaryData/TraceProcessingParameters/{self.netsta(trace_id)}/{self.aux_key(trace_id)}"
        if path not in self.f:
            return {}
        return json.loads(_bytes_to_str(self.f[path]))

    def stream_parameters(self, trace_id):
        inst = trace_id.split(".")[3][:2]
        netsta = self.netsta(trace_id)
        loc = trace_id.split(".")[2]
        path = f"AuxiliaryData/StreamProcessingParameters/{netsta}/{netsta}.{loc}.{inst}_{self.event_id}_{self.label}"
        if path not in self.f:
            return {}
        return json.loads(_bytes_to_str(self.f[path]))

    def cached(self, trace_id, name):
        """Cached array, e.g. name='SnrFreq', 'SmoothSignalSpectrumSpec'."""
        path = f"AuxiliaryData/Cache/{name}/{self.netsta(trace_id)}/{self.aux_key(trace_id)}"
        if path not in self.f:
            return None
        return np.asarray(self.f[path][()], dtype=float)

    def review(self, trace_id):
        """Analyst review record, or None if the trace was never written by the GUI."""
        path = f"AuxiliaryData/review/{self.netsta(trace_id)}/{self.aux_key(trace_id)}"
        if path not in self.f:
            return None
        g = self.f[path]
        out = {}
        if "accepted" in g:
            out.update(_attrs(g["accepted"]))
        if "colocation" in g:
            for k, v in _attrs(g["colocation"]).items():
                out.setdefault(k, v)
        for which in ("highpass", "lowpass"):
            key = f"corner_frequencies/{which}"
            out[f"{which}"] = float(np.asarray(g[key][()]).ravel()[0]) if key in g else np.nan
        return out

    def provenance(self, trace_id):
        """List of (step_label, {attribute: value}) in processing order."""
        path = f"Provenance/{self.aux_key(trace_id)}"
        if path not in self.f:
            return []
        root = ET.fromstring(np.asarray(self.f[path][()]).tobytes())
        steps = []
        for act in root.iter(f"{PROV_NS}activity"):
            lab = act.find(f"{PROV_NS}label")
            attrs = {
                child.tag.replace(SEIS_NS, ""): _to_float(child.text)
                for child in act
                if child.tag.startswith(SEIS_NS)
            }
            steps.append((lab.text if lab is not None else None, attrs))
        return steps

    def config(self):
        path = "AuxiliaryData/config/config"
        return json.loads(_bytes_to_str(self.f[path])) if path in self.f else {}

    def gmprocess_version(self):
        path = "AuxiliaryData/gmprocess_version/version"
        return _bytes_to_str(self.f[path]) if path in self.f else None
