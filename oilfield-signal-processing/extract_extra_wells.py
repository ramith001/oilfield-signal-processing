"""
Part 6, data step: more Volve wells.

The f0nzie mirror used so far holds only three wells (9-F-4, 9-F-7, 9-F-9; see extract_witsml.py), so more wells
have to come from somewhere else. The only public source of further Volve real-time drilling data that can be
fetched without registration is the converted store published with DrillDash
(github.com/Sky370/DrillDash, MIT code, data CC BY-NC-SA 4.0 from the Equinor Volve release: Equinor ASA as
publisher; data owners Equinor, ExxonMobil, Bayerngas and the Norwegian Petroleum Directorate). It carries the
same Baker Hughes real-time log as the mirror, merged into one table per wellbore with SI units:
WOB in N, torque in N.m, rotary speed in rev/s, bit depth in m, ROP in m/s, MSE in Pa. The rig activity code is
kept, but TORQUE_MAX is not (the saturation rule in this repo uses TORQUE_MAX, so it is applied to the mean torque
for these wells).

This script downloads
  - the demo store file for well 15/9-F-1 C (a small raw file in the DrillDash repository), and
  - store-bakerhughes-2.zip from the DrillDash v1.0.0 release (about 700 MB, checked against the published
    SHA-256) for wells F-12, F-14, F-15 and F-5,
and writes one slim gzipped CSV per well to data/extra_wells/, plus _inventory.csv with each well's full record length. Only rows with rotation (surface rpm above 0.1 rev/s,
6 rpm) are kept, with the columns the analysis uses, so the CSVs are small and the large download is needed only
once. Rows without rotation are not used by any analysis in Part 6.

Reading parquet needs pyarrow, or polars where pyarrow has no wheel (Windows ARM).

    python extract_extra_wells.py
"""
import hashlib
import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests

REL = "https://github.com/Sky370/DrillDash/releases/download/v1.0.0/"
ZIP_NAME = "store-bakerhughes-2.zip"
ZIP_SHA256 = "64c47a9b4c221035642ae9445950591c665ab83c2cdff13e4429cf27ce22cede"
DEMO_URL = ("https://raw.githubusercontent.com/Sky370/DrillDash/HEAD/demo/store/"
            "Norway-Statoil-NO-15-9-F-1-C/3/drilling.parquet")
OUT_DIR = "data/extra_wells"
MIN_ROT = 0.1    # rev/s
KEEP = ["t", "rig_code", "bit_depth", "wob", "rpm", "torque", "hookload", "flow_in", "rop", "mse"]
# store folder -> label used in this repo
STORE_WELLS = {
    "Norway-Statoil-NO-15-9-F-12/1": "9-F-12",
    "Norway-Statoil-NO-15-9-F-14/1": "9-F-14",
    "Norway-Statoil-NO-15-9-F-15/1": "9-F-15a",
    "Norway-Statoil-NO-15-9-F-15/2": "9-F-15b",
    "Norway-Statoil-NO-15-9-F-5/1": "9-F-5",
}


def read_parquet_bytes(b):
    try:
        return pd.read_parquet(io.BytesIO(b))
    except ImportError:
        import polars as pl
        d = pl.read_parquet(io.BytesIO(b))
        return pd.DataFrame({c: d[c].to_numpy() for c in d.columns})


INVENTORY = []


def slim(df, label):
    df = df.copy()
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    dt = df["t"].diff().dt.total_seconds().clip(upper=10).fillna(0)
    INVENTORY.append({"well": label, "first_time": df["t"].min(), "last_time": df["t"].max(),
                      "record_hours": dt.sum() / 3600, "rows": len(df),
                      "rotating_hours": dt[df["rpm"] > MIN_ROT].sum() / 3600})
    out = df[KEEP].rename(columns={"t": "Time"})
    out = out[out["rpm"] > MIN_ROT].reset_index(drop=True)
    for c, nd in (("bit_depth", 1), ("wob", 0), ("torque", 0), ("hookload", 0), ("rpm", 3), ("flow_in", 5),
                  ("rop", 6), ("mse", 0)):
        out[c] = out[c].round(nd)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    r = requests.get(DEMO_URL, timeout=120)
    r.raise_for_status()
    out = slim(read_parquet_bytes(r.content), "9-F-1C")
    out.to_csv(f"{OUT_DIR}/9-F-1C.csv.gz", index=False)
    print(f"9-F-1C: {len(out)} rotating rows")

    zpath = os.path.join(OUT_DIR, "_" + ZIP_NAME)
    if not os.path.exists(zpath):
        print(f"downloading {ZIP_NAME} (about 700 MB)...")
        with requests.get(REL + ZIP_NAME, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
    h = hashlib.sha256()
    with open(zpath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != ZIP_SHA256:
        raise SystemExit("checksum mismatch: delete the zip and run again")
    z = zipfile.ZipFile(zpath)
    for folder, label in STORE_WELLS.items():
        out = slim(read_parquet_bytes(z.read(f"store/{folder}/drilling.parquet")), label)
        out.to_csv(f"{OUT_DIR}/{label}.csv.gz", index=False)
        print(f"{label}: {len(out)} rotating rows")
    z.close()
    os.remove(zpath)
    pd.DataFrame(INVENTORY).to_csv(f"{OUT_DIR}/_inventory.csv", index=False)


if __name__ == "__main__":
    main()
