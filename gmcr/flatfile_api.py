"""Fetch NGA-West3 flatfile metadata from the nga-gm.org API.

The flatfile endpoint returns one row per motion (event-station) holding
accepted records only. Columns derived from the analyst's processing
(corner frequencies, intensity measures) are excluded by default so they can't
leak into model predictors; LABEL_COLUMNS are fetched separately for
cross-checking.

Token: read from a file (never printed). Tokens expire after 120 minutes; use
--login to obtain a fresh one interactively.
"""

import getpass
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://nga-gm.org"

# Predictors safe to use: they don't depend on the analyst's processing choices
FEATURE_COLUMNS = [
    "motion_id", "event_id", "station_id", "comcat_id", "network_code", "station_code",
    "user_id", "time_series_metadata_id",
    # event / source
    "magnitude", "magnitude_type", "hypocenter_depth", "strike", "dip", "rake",
    "event_type", "ztor", "fault_width", "fault_length",
    # path
    "epicentral_distance", "hypocentral_distance", "rjb", "rrup", "rx", "ry0", "hanging_wall",
    # station / site
    "housing", "cosmos_station_type", "sensor_depth", "vs30", "vs30_code_id",
    "z1p0_preferred", "z2p5_preferred", "geological_unit", "slope_gradient", "terrain_class",
    "basin_geospatial_category",
    # instrument
    "instrument_type", "colocated_instrument", "sampling_rate", "azimuth_h1", "azimuth_h2",
    "late_p_trigger", "late_s_trigger", "multiple_wave_train",
]
# Reference tables cover rejected records too (stations/sites are shared across
# events), so these are safe predictors for the accept/reject model.
REFERENCE_TABLES = {
    "events": ["event_id", "comcat_id", "magnitude", "magnitude_type", "hypocenter_latitude",
               "hypocenter_longitude", "hypocenter_depth", "strike", "dip", "rake", "event_type_id"],
    "networks": ["network_id", "network_code"],
    "stations": ["station_id", "site_id", "network_id", "station_code", "station_latitude",
                 "station_longitude", "station_elevation", "housing", "cosmos_station_type",
                 "sensor_depth", "installation_date", "removal_date"],
    "sites": ["site_id", "vs30", "vs30_lnstd", "vs30_code_id", "z1p0_preferred", "z1p0_code_id",
              "z2p5_preferred", "z2p5_code_id", "geological_unit", "slope_gradient",
              "terrain_class", "basin_geomorphic_category", "basin_geospatial_category"],
    "finiteFaults": ["finite_fault_id", "event_id", "ztor", "fault_length", "fault_width",
                     "ffm_model", "ffm_complexity"],
    "finiteFaultSegments": None,  # all columns
}

# The analyst's final values (labels) -- for validation only, never predictors
LABEL_COLUMNS = [
    "motion_id", "hpass_fc_h1", "hpass_fc_h2", "hpass_fc_v",
    "lpass_fc_h1", "lpass_fc_h2", "lpass_fc_v",
    "hpass_applied_h1", "hpass_applied_h2", "hpass_applied_v",
    "lpass_applied_h1", "lpass_applied_h2", "lpass_applied_v",
    "processing_type", "filter_type", "order_hpass", "order_lpass",
]


def login():
    user = input("nga-gm.org username: ")
    r = requests.get(f"{BASE}/users/login", headers={"Accept": "application/json"},
                     auth=(user, getpass.getpass("password: ")), timeout=60)
    r.raise_for_status()
    return r.json()["token"]


class Client:
    def __init__(self, token):
        self.s = requests.Session()
        self.s.headers.update({"Accept": "application/json", "Authorization": f"Bearer {token}"})

    @classmethod
    def from_file(cls, path):
        return cls(Path(path).read_text().strip())

    def _post(self, endpoint, body, retries=3):
        for attempt in range(retries):
            try:
                r = self.s.post(f"{BASE}/{endpoint}", json=body, timeout=600)
                if r.status_code == 401:
                    raise PermissionError("token missing/expired: rerun with --login")
                r.raise_for_status()
                out = r.json()
                # table/flatfile endpoints return errors as a JSON string with status 200
                if isinstance(out, str):
                    if "did not return a result" in out:
                        return {}
                    raise RuntimeError(f"{endpoint}: {out[:300]}")
                return out
            except (requests.ConnectionError, requests.Timeout):
                if attempt == retries - 1:
                    raise
                time.sleep(5 * (attempt + 1))

    def paged(self, endpoint, where, select, limit=2000):
        frames, page = [], 1
        while True:
            body = {"select": ",".join(select), "limit": limit, "page": page}
            if where:
                body["where"] = where
            out = self._post(endpoint, body)
            df = pd.DataFrame(out)
            if df.empty:
                break
            frames.append(df)
            if len(df) < limit:
                break
            page += 1
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=select)

    def table(self, endpoint, select=None, where=None, limit=20000):
        """Whole table (all pages). Reference tables are small (< ~20k rows)."""
        frames, page = [], 1
        while True:
            body = {"limit": limit, "page": page}
            if select:
                body["select"] = ",".join(select)
            if where:
                body["where"] = where
            df = pd.DataFrame(self._post(endpoint, body))
            if df.empty:
                break
            frames.append(df)
            if len(df) < limit:
                break
            page += 1
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return df[[c for c in select if c in df.columns]] if select and not df.empty else df

    def event_ids(self, comcat_ids):
        """Map ComCat IDs -> database event_id.

        `IN (...)` on string fields returns no rows on the API (seen Oct 2026),
        so fetch the whole (small) events table and match locally."""
        ev = self.paged("events", None, ["event_id", "comcat_id"], limit=20000)
        return ev[ev["comcat_id"].isin(set(comcat_ids))].reset_index(drop=True)

    def flatfile(self, event_ids, columns, chunk=50):
        frames = []
        ids = sorted(set(int(e) for e in event_ids))
        for i in range(0, len(ids), chunk):
            lst = ",".join(map(str, ids[i:i + chunk]))
            frames.append(self.paged("flatfile", f"event_id IN ({lst})", columns))
            print(f"  flatfile: {min(i + chunk, len(ids))}/{len(ids)} events", flush=True)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        # the endpoint adds primary keys of joined tables; keep only what was asked
        return df[[c for c in columns if c in df.columns]]
