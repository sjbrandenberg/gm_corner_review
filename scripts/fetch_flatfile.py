"""Download database metadata for every event in the extracted traces table.

    python scripts/fetch_flatfile.py FEATURES_DIR --token-file ~/Documents/token.txt
    python scripts/fetch_flatfile.py FEATURES_DIR --login       # prompts, token not saved

Writes to FEATURES_DIR/db/:
    flatfile_features.parquet  flatfile rows (accepted records only)
    flatfile_labels.parquet    analyst's final corners, for validation
    <table>.parquet            reference tables: events, networks, stations,
                               sites, finiteFaults, finiteFaultSegments
Run where there is internet access (laptop or a TACC login node, not compute nodes).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmcr import flatfile_api as fa  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("features_dir")
    p.add_argument("--token-file", default=str(Path.home() / "Documents" / "token.txt"))
    p.add_argument("--login", action="store_true")
    a = p.parse_args()

    feat = Path(a.features_dir)
    db = feat / "db"
    db.mkdir(exist_ok=True)
    comcat = pd.read_parquet(feat / "traces_all.parquet", columns=["event_id"])["event_id"].dropna().unique()
    print(f"{len(comcat)} ComCat IDs in traces table")

    client = fa.Client(fa.login()) if a.login else fa.Client.from_file(a.token_file)

    for name, cols in fa.REFERENCE_TABLES.items():
        df = client.table(name, cols)
        df.to_parquet(db / f"{name}.parquet", index=False)
        print(f"  {name}: {len(df)} rows")

    ev = pd.read_parquet(db / "events.parquet")
    ev = ev[ev["comcat_id"].isin(set(comcat))]
    missing = sorted(set(comcat) - set(ev["comcat_id"]))
    print(f"{len(ev)} events matched in database, {len(missing)} not found")
    pd.Series(missing, name="comcat_id").to_csv(db / "missing_events.csv", index=False)

    client.flatfile(ev["event_id"], fa.FEATURE_COLUMNS).to_parquet(db / "flatfile_features.parquet", index=False)
    client.flatfile(ev["event_id"], fa.LABEL_COLUMNS).to_parquet(db / "flatfile_labels.parquet", index=False)
    print("done")


if __name__ == "__main__":
    main()
