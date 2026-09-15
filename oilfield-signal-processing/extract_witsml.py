"""
Download and parse Volve field WITSML real-time drilling logs (raw XML, as
released by Equinor and mirrored with the original folder structure at
github.com/f0nzie/volve-drilling) into a single clean time-indexed CSV.

No API key needed -- this pulls the 5 raw log-chunk files directly from
raw.githubusercontent.com. Run this once to (re)populate data/ before
running signal_analysis.py.

WITSML quirk handled here: the <indexCurve> (Time) is not necessarily first
in the declared <logCurveInfo> mnemonic list, but its VALUE always leads
each <data> row. Verified against this file: mnemonic order lists
RigActivityCode first, but data row field 0 is an ISO timestamp and field 1
is "0" -- checked against RigActivityCode's own curveDescription
("NoMonitor[0] Drilling[1] Reaming[2] OffBottom[3] ...") a 0 at the very
first timestamp of the run (rig not yet monitoring/drilling) is exactly
what's physically expected, confirming the mapping: column 0 = Time,
columns 1..N = the other declared curves in their original document order.
"""
import os
import re
import pandas as pd
import requests

RAW_BASE = (
    "https://raw.githubusercontent.com/f0nzie/volve-drilling/master/witsml/"
    "Norway-Statoil-NO%2015_%2447%24_9-F-9/1/log/1/9/1"
)
CHUNK_FILES = [f"0000{i}.xml" for i in range(1, 6)]

KEEP_CURVES = [
    "RigActivityCode", "Depth", "BITDEP",
    "WOB_AVG", "WOB_MIN", "WOB_MAX",
    "TORQUE_AVG", "TORQUE_MIN", "TORQUE_MAX",
    "SURF_RPM", "BIT_RPM_AVG", "BIT_RPM_MIN", "BIT_RPM_MAX",
    "ROP_AVG", "HKLD_AVG", "MSE", "FLOWIN",
]


def download_chunks(dest_dir="raw_xml"):
    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for fname in CHUNK_FILES:
        dest = os.path.join(dest_dir, fname)
        if not os.path.exists(dest):
            resp = requests.get(f"{RAW_BASE}/{fname}", timeout=30)
            resp.raise_for_status()
            with open(dest, "w", encoding="utf-8") as f:
                f.write(resp.text)
        paths.append(dest)
    return paths


def parse_log_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    mnemonics = re.findall(r"<mnemonic>(.*?)</mnemonic>", content)
    index_curve = re.findall(r"<indexCurve[^>]*>(.*?)</indexCurve>", content)[0]

    # column order: Time first, then the other declared curves in document order
    other_curves = [m for m in mnemonics if m != index_curve]
    columns = [index_curve] + other_curves

    rows_raw = re.findall(r"<data>(.*?)</data>", content)
    rows = [r.split(",") for r in rows_raw]

    df = pd.DataFrame(rows, columns=columns)
    return df


def main():
    os.makedirs("data", exist_ok=True)
    files = download_chunks()
    print(f"Downloaded/found {len(files)} log chunk files")
    frames = []
    for fp in files:
        df = parse_log_file(fp)
        frames.append(df)
        print(f"  {fp}: {len(df)} rows, {df['Time'].iloc[0]} to {df['Time'].iloc[-1]}")

    full = pd.concat(frames, ignore_index=True)
    full["Time"] = pd.to_datetime(full["Time"])
    full = full.sort_values("Time").drop_duplicates(subset="Time").reset_index(drop=True)

    keep = ["Time"] + [c for c in KEEP_CURVES if c in full.columns]
    out = full[keep].copy()
    for c in keep:
        if c != "Time":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    print("\nFinal shape:", out.shape)
    print("Time range:", out["Time"].min(), "to", out["Time"].max())
    print("\nRigActivityCode value counts:")
    print(out["RigActivityCode"].value_counts().sort_index())
    print("\nSample physical ranges (non-zero only):")
    for c in ["WOB_AVG", "TORQUE_AVG", "SURF_RPM", "BIT_RPM_AVG", "ROP_AVG", "HKLD_AVG"]:
        nz = out[out[c] != 0][c]
        if len(nz):
            print(f"  {c}: n={len(nz)}, min={nz.min():.2f}, median={nz.median():.2f}, max={nz.max():.2f}")

    out.to_csv("data/f9_drilling_telemetry.csv", index=False)
    print("\nSaved data/f9_drilling_telemetry.csv")


if __name__ == "__main__":
    main()
