"""Every record in the workspaces with its outcome (accepted, or rejected by gmprocess or the analyst) and
the metadata for setting distance limits by magnitude, style of faulting, and sensor type.

    python scripts/record_status.py FEATURES_DIR [--out OUT_DIR]

FEATURES_DIR is a merged extraction of every trace (gmcr.extract --metadata-only, without
--reviewed-only; merge_outputs.py; and join_metadata.py for distances and event metadata). Writes to
OUT_DIR (default FEATURES_DIR/records):

    records.parquet / records.csv   one row per trace and workspace file
    rejected.csv                    the rows with status auto_rejected or analyst_rejected
    summary.json                    counts by status, rejection step, sensor, and style of faulting

status (one per row):
    auto_rejected         gmprocess failed the trace (auto_fail_module / auto_fail_reason say where:
                          snr_check, check_clipping, ...); analysts never saw these
    analyst_rejected      reviewed and rejected
    accepted              reviewed and accepted
    colocated_not_used    passed gmprocess, not reviewed, at a site where a co-located instrument was
                          selected instead (colocated = 1, selected = 0): not a rejection of the record
    not_reviewed          passed gmprocess and never reviewed (e.g. the analyst stopped at a distance)

Columns, besides the trace's identity: the event (magnitude, type, hypocenter, from the workspace's
QuakeML; ev_rake, ev_dip, ev_strike, ev_ztor from the database), style of faulting from the rake,
sensor class from the SEED channel codes, distances (repi_km and rhypo_km from the workspace; d_rrup,
d_rjb, d_rx, d_ry0 from the database's finite fault, or the hypocenter without one: d_source), the
outcome, gmprocess's corners and SNR band edges (snrN_flo / snrN_fhi: the band where SNR > N), the
analyst's corners, and site data (st_vs30, ...) where the database has the station. The database has
stations only if one of their records was accepted for some event, so site data are missing more often
for rejected records (st_matched); use a proxy that covers every station if site terms matter.

Style of faulting (sof) uses the NGA-West2 rake bins (Ancheta et al. 2014): strike_slip
(|rake| <= 30 or >= 150), normal (-120 <= rake <= -60), reverse (60 <= rake <= 120),
reverse_oblique (30 < rake < 60 or 120 < rake < 150), normal_oblique (-150 < rake < -120 or
-60 < rake < -30); unknown without a rake. The rake is kept for other binnings.

Sensor class: accelerometer (instrument code N), broadband (high- or low-gain seismometer with band code
H or B), short_period (band E or S seismometer), other.

A trace reviewed in several workspace files (e.g. by two analysts) has one row per file: n_files > 1.
Records outside gmprocess's download radius, or with no data at all, are not in the workspaces.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STATUS_ORDER = ["accepted", "analyst_rejected", "auto_rejected", "colocated_not_used", "not_reviewed"]

COLUMNS = [
    # identity
    "event_id", "trace_id", "network", "station", "location", "channel", "component", "source_file",
    "trace_key", "n_files",
    # outcome
    "status", "rejected", "auto_fail_module", "auto_fail_reason", "stream_any_failure",
    "y_reviewed", "y_accepted", "colocated", "selected", "review_username", "review_timestamp",
    # event
    "magnitude", "magnitude_type", "event_time", "event_lat", "event_lon", "event_depth_km",
    "ev_rake", "ev_dip", "ev_strike", "ev_ztor", "sof", "ev_event_type_id",
    # sensor and station
    "sensor", "band_code", "inst_code", "sampling_rate", "sta_lat", "sta_lon", "sta_elev_m",
    "st_matched", "st_vs30", "st_vs30_code_id", "st_cosmos_station_type", "st_housing",
    # distance
    "repi_km", "rhypo_km", "d_rrup", "d_rjb", "d_rx", "d_ry0", "d_source",
    # signal and corners
    "snr2_flo", "snr2_fhi", "snr3_flo", "snr3_fhi", "snr5_flo", "snr5_fhi", "snr_max_0p1_10",
    "snr_at_0.1hz", "snr_at_1hz", "snr_at_10hz", "clipping_prob", "signal_duration",
    "fchp_gmp", "fclp_gmp", "y_fchp", "y_fclp", "y_fclp_none",
    "gmprocess_version",
]


def style_of_faulting(rake):
    r = pd.to_numeric(rake, errors="coerce")
    out = pd.Series("unknown", index=r.index, dtype=object)
    a = r.abs()
    out[(a <= 30) | (a >= 150)] = "strike_slip"
    out[(r >= -120) & (r <= -60)] = "normal"
    out[(r >= 60) & (r <= 120)] = "reverse"
    out[((r > 30) & (r < 60)) | ((r > 120) & (r < 150))] = "reverse_oblique"
    out[((r > -150) & (r < -120)) | ((r > -60) & (r < -30))] = "normal_oblique"
    return out


def sensor_class(band, inst):
    band, inst = band.astype(str), inst.astype(str)
    out = pd.Series("other", index=band.index, dtype=object)
    seis = inst.isin(["H", "L"])
    out[inst == "N"] = "accelerometer"
    out[seis & band.isin(["H", "B"])] = "broadband"
    out[seis & band.isin(["E", "S"])] = "short_period"
    return out


def status(df):
    failed = df["auto_failed"].astype("boolean").fillna(False).astype(bool)
    reviewed = pd.to_numeric(df["y_reviewed"], errors="coerce").fillna(0) == 1
    acc = pd.to_numeric(df["y_accepted"], errors="coerce")
    coloc = (pd.to_numeric(df.get("colocated"), errors="coerce") == 1) \
        & (pd.to_numeric(df.get("selected"), errors="coerce") == 0)
    st = np.select([failed, reviewed & (acc == 1), reviewed & (acc == 0), coloc],
                   ["auto_rejected", "accepted", "analyst_rejected", "colocated_not_used"],
                   default="not_reviewed")
    return pd.Series(st, index=df.index)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features_dir")
    p.add_argument("--out", default=None, help="output folder (default FEATURES_DIR/records)")
    a = p.parse_args()
    feat = Path(a.features_dir)
    out = Path(a.out) if a.out else feat / "records"
    src = feat / "traces_joined.parquet"
    if not src.exists():
        src = feat / "traces_all.parquet"
        print("no traces_joined.parquet (join_metadata.py): no database distances, rake, or site "
              "data; repi_km / rhypo_km only", file=sys.stderr)
    df = pd.read_parquet(src)
    if "error" in df:
        bad = df["error"].notna()
        if bad.any():
            print(f"{int(bad.sum())} traces could not be read and are left out", file=sys.stderr)
        df = df[~bad]

    df["status"] = status(df)
    df["rejected"] = df["status"].isin(["auto_rejected", "analyst_rejected"])
    df["sof"] = style_of_faulting(df["ev_rake"]) if "ev_rake" in df else "unknown"
    df["sensor"] = sensor_class(df["band_code"], df["inst_code"])
    df["n_files"] = df.groupby(["event_id", "trace_id"])["trace_key"].transform("size")
    if "st_station_id" in df:
        df["st_matched"] = df["st_station_id"].notna()
    cols = [c for c in COLUMNS if c in df]
    rec = df[cols].sort_values(["event_id", "station", "channel", "source_file"]).reset_index(drop=True)

    out.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(out / "records.parquet", index=False)
    rec.to_csv(out / "records.csv", index=False)
    rec[rec["rejected"]].to_csv(out / "rejected.csv", index=False)

    by = lambda c: rec.groupby([c, "status"]).size().unstack(fill_value=0) \
        .reindex(columns=STATUS_ORDER, fill_value=0).to_dict(orient="index")
    summary = {
        "source": str(src), "records": len(rec), "events": int(rec["event_id"].nunique()),
        "status": rec["status"].value_counts().reindex(STATUS_ORDER, fill_value=0).to_dict(),
        "auto_rejected_by_step": rec.loc[rec["status"] == "auto_rejected", "auto_fail_module"]
        .value_counts().to_dict(),
        "by_sensor": by("sensor"), "by_sof": by("sof"),
        "traces_in_several_files": int((rec["n_files"] > 1).sum()),
    }
    if "d_source" in rec:
        summary["distance_source"] = rec["d_source"].value_counts().to_dict()
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("records", "events", "status", "auto_rejected_by_step")},
                     indent=1, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
