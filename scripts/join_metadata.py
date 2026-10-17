"""Attach database metadata and computed distances to the extracted traces.

    python scripts/join_metadata.py FEATURES_DIR

Reads FEATURES_DIR/traces_all.parquet and FEATURES_DIR/db/*.parquet (from
fetch_flatfile.py); writes FEATURES_DIR/traces_joined.parquet.

Column groups added
  ev_*   event + finite-fault attributes (events/finiteFaults tables)      all events in DB
  d_*    Rrup, Rjb (and Rx, Ry0 for single-segment faults) computed from
         the finite fault and StationXML coordinates                        all traces
  analyst_id  the event's analyst (user_id of its accepted motions)         events with >=1 accepted motion
  st_*   station + site attributes (stations/sites tables)                 stations with an accepted record
                                                                            for at least one event
  ff_*   flatfile row of the motion (accepted records only)
  ff_hpass_fc / ff_lpass_fc  analyst's final corners, for label cross-check only

Leakage for the accept/reject model:
  safe       ev_*, d_*, analyst_id (event level), all HDF5-derived features
  NOT safe   ff_* (present <=> accepted) and st_* (a station whose records were
             always rejected is absent from the DB, so missing st_* hints at
             rejection). Site terms for accept/reject need a source that covers
             every station, e.g. proxy Vs30 sampled at StationXML coordinates.
The printed station-match rates for accepted vs rejected traces quantify the
st_* bias. All groups are fine for the corner-frequency model (accepted only).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmcr import distances as gd  # noqa: E402

KEYS = ["event_id", "network", "station"]


def _prefix(df, p, keep=()):
    return df.rename(columns={c: f"{p}{c}" for c in df.columns if c not in keep})


def event_table(db):
    ev = pd.read_parquet(db / "events.parquet")
    ff = pd.read_parquet(db / "finiteFaults.parquet")
    nseg = pd.read_parquet(db / "finiteFaultSegments.parquet")["finite_fault_id"].value_counts()
    ff["fault_nseg"] = ff["finite_fault_id"].map(nseg).fillna(0).astype(int)
    ev = ev.merge(ff, on="event_id", how="left")
    ev = ev.rename(columns={"event_id": "db_event_id", "comcat_id": "event_id"})
    return _prefix(ev, "ev_", keep=("event_id", "db_event_id"))


def station_table(db):
    st = pd.read_parquet(db / "stations.parquet").merge(
        pd.read_parquet(db / "networks.parquet"), on="network_id", how="left")
    st = st.merge(pd.read_parquet(db / "sites.parquet"), on="site_id", how="left")
    st = st.rename(columns={"network_code": "network", "station_code": "station"})
    return _prefix(st, "st_", keep=("network", "station"))


def attach_stations(tr, st):
    """Match on network+station; if several DB stations share the code, take the
    one nearest the StationXML coordinates."""
    m = tr[["trace_key", "network", "station", "sta_lat", "sta_lon"]].merge(st, on=["network", "station"], how="left")
    dx = (m["st_station_latitude"] - m["sta_lat"]) ** 2 + \
         ((m["st_station_longitude"] - m["sta_lon"]) * np.cos(np.radians(m["sta_lat"]))) ** 2
    m["st_match_km"] = np.sqrt(dx) * 111.32
    m = m.sort_values("st_match_km").drop_duplicates("trace_key", keep="first")
    m = m.drop(columns=["network", "station", "sta_lat", "sta_lon"])
    return tr.merge(m, on="trace_key", how="left")


def attach_distances(tr, db):
    segs = pd.read_parquet(db / "finiteFaultSegments.parquet")
    by_fault = {k: g.to_dict("records") for k, g in segs.groupby("finite_fault_id")}
    out = {k: np.full(len(tr), np.nan) for k in ("d_rrup", "d_rjb", "d_rx", "d_ry0", "d_gc2_u")}
    src = np.array(["none"] * len(tr), dtype=object)
    for fid, idx in tr.groupby("ev_finite_fault_id").indices.items():
        seg = by_fault.get(fid)
        if not seg:
            continue
        # one call per event: stations are de-duplicated (3 components share a site)
        sta = tr.iloc[idx][["sta_lat", "sta_lon"]].drop_duplicates()
        d = gd.event_distances(seg, sta["sta_lat"], sta["sta_lon"])
        look = {(a, b): i for i, (a, b) in enumerate(zip(sta["sta_lat"], sta["sta_lon"]))}
        for i in idx:
            j = look.get((tr["sta_lat"].iat[i], tr["sta_lon"].iat[i]))
            if j is None or not np.isfinite(d["rrup"][j]):
                continue
            for k, src_k in (("d_rrup", "rrup"), ("d_rjb", "rjb"), ("d_rx", "rx"),
                             ("d_ry0", "ry0"), ("d_gc2_u", "gc2_u")):
                out[k][i] = d[src_k][j]
            src[i] = "finite_fault"
    # fallback: point source
    nofault = src == "none"
    out["d_rrup"][nofault] = tr.loc[nofault, "rhypo_km"]
    out["d_rjb"][nofault] = tr.loc[nofault, "repi_km"]
    src[nofault & tr["rhypo_km"].notna().to_numpy()] = "hypocenter"
    for k, v in out.items():
        tr[k] = v
    tr["d_source"] = src
    return tr


def attach_flatfile(tr, db):
    ff = pd.read_parquet(db / "flatfile_features.parquet")
    lab = pd.read_parquet(db / "flatfile_labels.parquet")
    analyst = (ff.groupby("comcat_id")["user_id"]
               .agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
               .rename("analyst_id").reset_index().rename(columns={"comcat_id": "event_id"}))
    tr = tr.merge(analyst, on="event_id", how="left")

    ff = ff.drop(columns=["event_id"]).rename(
        columns={"comcat_id": "event_id", "network_code": "network", "station_code": "station"})
    ff = _prefix(ff, "ff_", keep=KEYS)
    ff["_n"] = ff.groupby(KEYS)["ff_motion_id"].transform("size")
    m = tr[["trace_key", "sampling_rate", *KEYS]].merge(ff, on=KEYS, how="left")
    m = m[(m["_n"].fillna(1) == 1) | (m["ff_sampling_rate"] == m["sampling_rate"])]
    m = m.drop_duplicates("trace_key").drop(columns=["_n", "sampling_rate", *KEYS])
    tr = tr.merge(m, on="trace_key", how="left")
    tr["ff_matched"] = tr["ff_motion_id"].notna()

    # component: Z -> v; horizontals by azimuth
    comp = np.where(tr["is_vertical"].astype(bool), "v", None).astype(object)
    for c in ("h1", "h2"):
        dd = ((tr["channel_azimuth"] - tr[f"ff_azimuth_{c}"] + 180) % 360 - 180).abs()
        comp = np.where((comp == None) & (dd < 15), c, comp)  # noqa: E711
    tr["ff_component"] = comp

    lab = lab.rename(columns={"motion_id": "ff_motion_id"})
    tr = tr.merge(lab, on="ff_motion_id", how="left")
    for kind in ("hpass", "lpass"):
        vals = np.full(len(tr), np.nan)
        for c in ("h1", "h2", "v"):
            sel = (tr["ff_component"] == c).to_numpy()
            vals[sel] = tr.loc[sel, f"{kind}_fc_{c}"].to_numpy(dtype=float)
        tr[f"ff_{kind}_fc"] = vals
    return tr.drop(columns=[c for c in lab.columns if c != "ff_motion_id"])


def main(feat_dir):
    feat = Path(feat_dir)
    db = feat / "db"
    tr = pd.read_parquet(feat / "traces_all.parquet")
    n0 = len(tr)
    tr = tr.merge(event_table(db), on="event_id", how="left")
    tr = attach_stations(tr, station_table(db))
    tr = attach_distances(tr, db)
    tr = attach_flatfile(tr, db)
    assert len(tr) == n0, "join changed row count"

    rev = tr["y_reviewed"] == 1
    acc, rej = rev & (tr["y_accepted"] == 1), rev & (tr["y_accepted"] == 0)
    both = tr[acc & tr["ff_hpass_fc"].notna()]
    agree = np.isclose(both["y_fchp"], both["ff_hpass_fc"], rtol=1e-3)
    pct = lambda s: f"{100 * s.mean():.1f}%" if len(s) else "n/a"
    print(f"traces {len(tr)}  reviewed {int(rev.sum())}  accepted {int(acc.sum())}  rejected {int(rej.sum())}")
    print(f"event matched: {pct(tr['db_event_id'].notna())}  analyst_id: {pct(tr['analyst_id'].notna())}")
    print(f"station/site matched: all {pct(tr['st_station_id'].notna())}, accepted {pct(tr.loc[acc, 'st_station_id'].notna())}, "
          f"rejected {pct(tr.loc[rej, 'st_station_id'].notna())}  (gap = st_* leakage for accept/reject)")
    print(f"vs30 available: accepted {pct(tr.loc[acc, 'st_vs30'].notna())}, rejected {pct(tr.loc[rej, 'st_vs30'].notna())}")
    print(f"distances: {tr['d_source'].value_counts().to_dict()}")
    if "ff_rrup" in tr:
        ok = tr["ff_rrup"].notna() & tr["d_rrup"].notna()
        print(f"Rrup computed vs flatfile: median |diff| {np.median(np.abs(tr.loc[ok, 'd_rrup'] - tr.loc[ok, 'ff_rrup'])):.3f} km (n={int(ok.sum())})")
    print(f"label check: accepted with flatfile fchp {len(both)}, agree {int(agree.sum())}")
    if (~agree).any():
        print(both.loc[~agree, ["trace_key", "y_fchp", "ff_hpass_fc", "ff_component"]].head(20).to_string())
    tr.to_parquet(feat / "traces_joined.parquet", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
