"""
Download and parse Volve field WITSML real-time drilling logs (raw XML, as
released by Equinor and mirrored with the original folder structure at
github.com/f0nzie/volve-drilling) into clean time-indexed CSVs, one well per
file.

No API key needed -- this pulls raw log-chunk files directly from
raw.githubusercontent.com. Run this once to (re)populate data/ before
running signal_analysis.py / bitdepth_analysis.py / multi_well_validation.py.

WITSML quirk handled here: the <indexCurve> (Time) is not necessarily first
in the declared <logCurveInfo> mnemonic list, but its VALUE always leads
each <data> row. Verified against well 9-F-9's file: mnemonic order lists
RigActivityCode first, but data row field 0 is an ISO timestamp and field 1
is "0" -- checked against RigActivityCode's own curveDescription
("NoMonitor[0] Drilling[1] Reaming[2] OffBottom[3] ...") a 0 at the very
first timestamp of the run (rig not yet monitoring/drilling) is exactly
what's physically expected, confirming the mapping: column 0 = Time,
columns 1..N = the other declared curves in their original document order.

Multi-well extension (Part 1 of 7): the mirror repo turns out to hold three
wells, not one -- found by cloning the repo shallow (api.github.com and
github.com/.../tree/... directory browsing are blocked in some sandboxed
environments, but a plain `git clone` of the whole repo is not, and
raw.githubusercontent.com file URLs built from the discovered paths are
reachable with no clone needed). The original script used well 9-F-9's
log object "1/9/1" under the assumption that log-object numbering is
consistent across wells -- it is NOT: each well's wellbore has its own set
of up to ~10 log objects, numbered independently, and only one of them
(per well) is the real-time surface-drilling-parameters log (date-time
indexed, carrying WOB/TORQUE/BITDEP together). The other log objects are
gas-detection logs, depth-indexed logs, or short logs with almost no rows.
Each well's correct log object below was identified by scanning every log
object's declared mnemonic list for the presence of TORQUE_AVG/TORQUE_MAX
together with WOB_AVG, not assumed from 9-F-9's numbering:
  - 9-F-9: log object "1/9/1" (5 chunks, ~33.6h, the original single well)
  - 9-F-7: log object "1/4" has two generations, "1/4/1" (1 chunk, a short
    lead-in segment) immediately followed in time by "1/4/2" (6 chunks) --
    confirmed contiguous (00001.xml of "1/4/1" ends 2016-07-28T10:57:21Z;
    00001.xml of "1/4/2" starts 2016-07-28T10:57:30Z), so both are
    downloaded and concatenated as one continuous ~2.4-day record.
  - 9-F-4: log object "1/6/1" (22 chunks, ~7 continuous days -- the
    longest of the three by a wide margin).

    python extract_witsml.py
"""
import os
import re
import pandas as pd
import requests

RAW_ROOT = "https://raw.githubusercontent.com/f0nzie/volve-drilling/master/witsml"

# (well label used in filenames, URL-encoded well folder name, list of log-object
# paths to download and concatenate in order, output CSV name)
WELLS = [
    ("f9", "Norway-Statoil-NO%2015_%2447%24_9-F-9", ["1/log/1/9/1"], "f9_drilling_telemetry.csv"),
    ("f7", "Norway-Statoil-NO%2015_%2447%24_9-F-7", ["1/log/1/4/1", "1/log/1/4/2"], "f7_drilling_telemetry.csv"),
    ("f4", "Norway-Statoil-NO%2015_%2447%24_9-F-4", ["1/log/1/6/1"], "f4_drilling_telemetry.csv"),
]

KEEP_CURVES = [
    "RigActivityCode", "Depth", "BITDEP",
    "WOB_AVG", "WOB_MIN", "WOB_MAX",
    "TORQUE_AVG", "TORQUE_MIN", "TORQUE_MAX",
    "SURF_RPM", "BIT_RPM_AVG", "BIT_RPM_MIN", "BIT_RPM_MAX",
    "ROP_AVG", "HKLD_AVG", "MSE", "FLOWIN",
]


def list_chunk_files(well_folder, log_path, dest_dir):
    """The repo has no directory-listing API reachable from this sandbox, so
    chunk filenames are discovered by probing 00001.xml, 00002.xml, ... until
    a 404, rather than assumed to be a fixed count like the original
    single-well version did."""
    chunks = []
    i = 1
    while True:
        fname = f"{i:05d}.xml"
        url = f"{RAW_ROOT}/{well_folder}/{log_path}/{fname}"
        dest = os.path.join(dest_dir, f"{log_path.replace('/', '_')}_{fname}")
        if os.path.exists(dest):
            chunks.append(dest)
            i += 1
            continue
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            break
        resp.raise_for_status()
        with open(dest, "w", encoding="utf-8") as f:
            f.write(resp.text)
        chunks.append(dest)
        i += 1
    return chunks


def parse_log_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    mnemonics = re.findall(r"<mnemonic>(.*?)</mnemonic>", content)
    index_curve = re.findall(r"<indexCurve[^>]*>(.*?)</indexCurve>", content)[0]

    other_curves = [m for m in mnemonics if m != index_curve]
    columns = [index_curve] + other_curves

    rows_raw = re.findall(r"<data>(.*?)</data>", content)
    rows = [r.split(",") for r in rows_raw]

    df = pd.DataFrame(rows, columns=columns)
    return df


def extract_well(label, well_folder, log_paths, out_name, dest_dir="raw_xml"):
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    frames = []
    for log_path in log_paths:
        files = list_chunk_files(well_folder, log_path, dest_dir)
        print(f"  [{label}] {log_path}: {len(files)} chunk file(s)")
        for fp in files:
            df = parse_log_file(fp)
            frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full["Time"] = pd.to_datetime(full["Time"])
    full = full.sort_values("Time").drop_duplicates(subset="Time").reset_index(drop=True)

    keep = ["Time"] + [c for c in KEEP_CURVES if c in full.columns]
    out = full[keep].copy()
    for c in keep:
        if c != "Time":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    print(f"  [{label}] final shape: {out.shape}, time range: {out['Time'].min()} to {out['Time'].max()}")
    out.to_csv(f"data/{out_name}", index=False)
    print(f"  [{label}] saved data/{out_name}")
    return out


def main():
    for label, well_folder, log_paths, out_name in WELLS:
        print(f"\nExtracting well {label} ({well_folder.replace('%20', ' ').replace('%24', '$')})...")
        extract_well(label, well_folder, log_paths, out_name)


if __name__ == "__main__":
    main()
