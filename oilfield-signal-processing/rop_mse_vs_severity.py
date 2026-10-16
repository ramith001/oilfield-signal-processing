"""
Part 3: does stronger stick-slip cost drilling efficiency? ROP and MSE against the stick-slip severity index.

The answer this data can give is "mostly: it cannot say", and this script is built to show exactly why,
step by step, rather than to produce a correlation from whatever is left. Four stages:

1. CHANNEL AUDIT. Part 1 found ROP_AVG and MSE identically zero in 9-F-9's log. This checks every log object
   the mirror holds for all three wells (not only the one Part 1 used), and every channel that could stand in
   for them (ROP_AVG, MSE, I_MSE, DMSE, DWOR, MIN_2M, MIN_5FT, ONBOTTOM_TIME). Reads raw_xml/ (download it
   first with extract_witsml.py / extract_f4_extra.py; it is git-ignored). -> data/rop_mse_channel_audit.csv

2. CHECK THE SEVERITY INDEX ITSELF, which Part 3 depends on and which Part 2's findings put in question.
   Part 1's index is a rolling max-min of TORQUE_AVG over 30 *rows* of the code-114 subset, labelled "60 s".
   Two things follow from Part 2's row-timing finding, and one more turns up here:
     (a) 30 rows are 29 gaps x 2.8 s = 81 s on average (the 3,3,3,3,2 s cadence), not 60 s. A constant, so it
         changes the label, not the ranking.
     (b) NOT noted before: the rolling window runs over the filtered code-114 rows, not over time, so a window
         that sits at the edge of a drilling block reaches across the gap into the next block, hours later in
         some cases. Such a "window" mixes separate drilling episodes. This script measures how many
         severity windows do that, and re-runs Part 1's WOB-severity and depth-severity correlations on
         time-contiguous windows only (span <= 120 s; a clean window is about 81 s).
     (c) TQ_IN_AMPS* are zero and log 8's torque is the same sensor as log 6's, so there is no independent
         torque channel to check the torque that severity (and MSE) is built from.
   -> data/severity_window_contiguity.csv

3. CAN ROP BE DERIVED? With ROP absent, the only route is the rate of change of bit depth (BITDEP, a real
   channel in metres; "Depth", the hole depth, is a constant that never updates). For each severity window
   this takes BITDEP at the two ends of the window over the window's real timestamps (not an assumed 2 s grid:
   the actual spacing is 2.8 s on average, so a nominal-2 s calculation would overstate ROP by about 40%).
   A window counts as "advancing" if BITDEP rises over the window at a rate of 0 to 150 m/h. -> data/rop_derivation_feasibility.csv

4. WHAT CAN STILL BE SAID, where there are enough advancing windows (only 9-F-9). ROP is that derived rate;
   MSE is the Teale mechanical specific energy with the bit area left out: MSE x A = WOB + 2 pi N T / ROP
   (WOB in N, N = SURF_RPM in rev/s, T = surface torque in N.m, ROP in m/s), which is a force in N. The bit
   diameter is not in this telemetry, and 1/A multiplies both terms, so Spearman correlations of MSE are
   unaffected by it within one hole section. Two cautions are built in: (i) severity (torque range) and the
   MSE index (mean torque) both come from the same torque channel, so a correlation between them is partly
   mechanical, and the correlation of severity with mean torque is reported alongside; (ii) the advancing
   windows are heavily overlapping and fall in a handful of separate segments, so the window-level n is
   not the real n, and the test that counts is at the segment level. -> data/rop_mse_9f9_windows.csv,
   data/rop_mse_9f9_segments.csv

    python rop_mse_vs_severity.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from multi_well_validation import WIN, clean_and_flag_live, rank_residual
from extract_witsml import parse_log_file

# dataviz skill palette (references/palette.md), as in multi_well_validation.py
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

WELLS = [
    ("9-F-9", "data/f9_drilling_telemetry.csv", BLUE),
    ("9-F-7", "data/f7_drilling_telemetry.csv", ORANGE),
    ("9-F-4", "data/f4_drilling_telemetry.csv", AQUA),
]
# raw log objects per well (see extract_witsml.py / extract_f4_extra.py); "log8" is 9-F-4's second log
RAW_LOGS = {
    "9-F-9 log 9/1": ["1_log_1_9_1"],
    "9-F-7 log 4/1+4/2": ["1_log_1_4_1", "1_log_1_4_2"],
    "9-F-4 log 6/1": ["1_log_1_6_1"],
    "9-F-4 log 8/1": ["1_log_1_8_1"],
}
AUDIT_CHANNELS = ["ROP_AVG", "MSE", "I_MSE", "DMSE", "DWOR", "MIN_2M", "MIN_5FT", "ONBOTTOM_TIME"]
MAX_CLEAN_SPAN_S = 120   # a clean 30-row window spans about 81 s; more than this means it crosses a gap
MAX_ROP_MH = 150         # derived rates above this are depth jumps (connections, resets), not drilling


def channel_audit():
    rows = []
    for name, prefixes in RAW_LOGS.items():
        files = []
        for p in prefixes:
            files += sorted(glob.glob(f"raw_xml/{p}_*.xml"))
        if not files:
            print(f"  {name}: raw_xml/ not found, skipped (run extract_witsml.py / extract_f4_extra.py first)")
            continue
        raw = pd.concat([parse_log_file(f) for f in files], ignore_index=True)
        raw = raw.drop_duplicates("Time")
        for ch in AUDIT_CHANNELS:
            if ch in raw.columns:
                v = pd.to_numeric(raw[ch], errors="coerce")
                rows.append({"log": name, "channel": ch, "rows": len(v), "nonzero_rows": int((v.fillna(0) != 0).sum()),
                             "max_abs": float(v.abs().max())})
    return pd.DataFrame(rows)


def window_table(df):
    """Part 1's severity windows (30 rows of the code-114 subset, centred), plus the time each one spans and the
    BITDEP behaviour across it. Returns the code-114 frame with one row per row/window."""
    d = df.copy()
    t = (d["Time"] - d["Time"].iloc[0]).dt.total_seconds()
    d["span_s"] = t.rolling(WIN, center=True).max() - t.rolling(WIN, center=True).min()
    r = d["TORQUE_AVG"].rolling(WIN, center=True)
    d["sev"] = r.max() - r.min()
    d["tq_mean"] = r.mean()
    d["wob60"] = d["WOB_AVG"].rolling(WIN, center=True).mean()
    d["rpm60"] = d["SURF_RPM"].rolling(WIN, center=True).mean()
    d["bd_first"] = d["BITDEP"].shift(WIN // 2)
    d["bd_last"] = d["BITDEP"].shift(-(WIN // 2 - 1))
    d["dbd"] = d["bd_last"] - d["bd_first"]
    d["segment"] = (d["Time"].diff().dt.total_seconds() > 10).cumsum()
    return d


def contiguity_and_correlations(label, d):
    v = d[d["is_live"]].dropna(subset=["sev", "wob60"])
    v = v[v["wob60"] > 0]
    clean = v[v["span_s"] <= MAX_CLEAN_SPAN_S]
    out = {"well": label, "severity_windows": len(v), "windows_span_over_120s": int((v["span_s"] > MAX_CLEAN_SPAN_S).sum()),
           "pct_over_120s": 100 * (v["span_s"] > MAX_CLEAN_SPAN_S).mean(),
           "median_span_s": v["span_s"].median(), "max_span_s": v["span_s"].max(),
           "clean_windows": len(clean),
           "wob_r_part1_windows": v["wob60"].corr(v["sev"], method="spearman"),
           "wob_r_clean_windows": clean["wob60"].corr(clean["sev"], method="spearman")}
    # depth-severity, same rules as multi_well_validation.severity_correlations (BITDEP > 0, WOB > 0)
    for tag, frame in (("part1_windows", v), ("clean_windows", clean)):
        b = frame[frame["BITDEP"] > 0].dropna(subset=["BITDEP"])
        if len(b) >= 20:
            rs, rb = rank_residual(b["sev"], b["wob60"]), rank_residual(b["BITDEP"], b["wob60"])
            out[f"depth_n_{tag}"] = len(b)
            out[f"depth_r_{tag}"] = b["BITDEP"].corr(b["sev"], method="spearman")
            out[f"depth_partial_r_{tag}"] = float(np.corrcoef(rs, rb)[0, 1])
        else:
            out[f"depth_n_{tag}"] = len(b)
            out[f"depth_r_{tag}"] = out[f"depth_partial_r_{tag}"] = float("nan")
    return out, v, clean


def rop_feasibility(label, v, clean):
    w = clean.dropna(subset=["dbd"]).copy()
    w["rop_mh"] = w["dbd"] / w["span_s"] * 3600
    adv = w[(w["dbd"] > 0) & (w["rop_mh"] <= MAX_ROP_MH)]
    live_all = int(v["Time"].size)
    return ({"well": label, "severity_windows": live_all, "clean_windows": len(clean),
             "clean_windows_bitdep_flat": int((w["dbd"] == 0).sum()), "clean_windows_bitdep_up": int((w["dbd"] > 0).sum()),
             "clean_windows_bitdep_down": int((w["dbd"] < 0).sum()),
             "advancing_windows": len(adv), "advancing_segments": int(adv["segment"].nunique()),
             "advancing_minutes_of_clock_time": float(adv["Time"].diff().dt.total_seconds().clip(upper=10).sum() / 60)}, adv)


def mse_index(wob, rpm, torque, rop_ms):
    """Teale MSE times bit area, in N: WOB + 2 pi N T / ROP. (Divide by bit area in m^2 for Pa.)"""
    return wob + 2 * np.pi * rpm * torque / rop_ms


def segment_analysis(adv):
    a = adv.copy()
    a["rop_ms"] = a["rop_mh"] / 3600
    a["mse_x_area_N"] = mse_index(a["wob60"], a["rpm60"], a["tq_mean"], a["rop_ms"])
    seg = a.groupby("segment").agg(start=("Time", "first"), windows=("sev", "size"), rop_mh=("rop_mh", "median"),
                                    severity=("sev", "median"), mean_torque=("tq_mean", "median"),
                                    wob_N=("wob60", "median"), rpm_cs=("rpm60", "median"),
                                    mse_x_area_N=("mse_x_area_N", "median")).reset_index()
    return a, seg


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs("data", exist_ok=True)

    print("=" * 78)
    print("1. CHANNEL AUDIT: is ROP / MSE populated anywhere?")
    print("=" * 78)
    audit = channel_audit()
    if len(audit):
        audit.to_csv("data/rop_mse_channel_audit.csv", index=False)
        piv = audit.pivot(index="channel", columns="log", values="nonzero_rows")
        print(piv.to_string())
        print(f"\nNon-zero rows across all {audit['log'].nunique()} log objects checked and all {len(AUDIT_CHANNELS)} "
              f"ROP/MSE-type channels: {int(audit['nonzero_rows'].sum())}")

    print("\n" + "=" * 78)
    print("2. SEVERITY WINDOWS: how many cross a gap, and does that matter?")
    print("=" * 78)
    contig, feas, adv_by_well = [], [], {}
    for label, path, _ in WELLS:
        df = pd.read_csv(path, parse_dates=["Time"])
        _, d_all, _, _, _ = clean_and_flag_live(df)
        d = window_table(d_all)
        c, v, clean = contiguity_and_correlations(label, d)
        contig.append(c)
        f, adv = rop_feasibility(label, v, clean)
        feas.append(f)
        adv_by_well[label] = (adv, d)
        live = d[d["is_live"]]
        f["live_rows"] = len(live)
        f["live_rows_bitdep_le_0"] = int((live["BITDEP"] <= 0).sum())
    cdf = pd.DataFrame(contig)
    cdf.to_csv("data/severity_window_contiguity.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(cdf[["well", "severity_windows", "windows_span_over_120s", "pct_over_120s", "median_span_s", "max_span_s",
               "clean_windows"]].to_string(index=False))
    print("\nWOB-severity Spearman, Part 1's windows vs. time-contiguous windows only:")
    print(cdf[["well", "wob_r_part1_windows", "wob_r_clean_windows"]].to_string(index=False))
    print("\nDepth-severity (BITDEP > 0, WOB > 0), Part 1's windows vs. contiguous windows only "
          "(raw r and partial r net of WOB):")
    print(cdf[["well", "depth_n_part1_windows", "depth_r_part1_windows", "depth_partial_r_part1_windows",
               "depth_n_clean_windows", "depth_r_clean_windows", "depth_partial_r_clean_windows"]].to_string(index=False))

    print("\n" + "=" * 78)
    print("3. CAN ROP BE DERIVED FROM BITDEP?")
    print("=" * 78)
    fdf = pd.DataFrame(feas)
    fdf.to_csv("data/rop_derivation_feasibility.csv", index=False)
    print(fdf.to_string(index=False))

    print("\n" + "=" * 78)
    print("4. 9-F-9: severity vs. derived ROP and MSE index (the only well with advancing windows)")
    print("=" * 78)
    adv, _ = adv_by_well["9-F-9"]
    a, seg = segment_analysis(adv)
    a[["Time", "segment", "sev", "tq_mean", "wob60", "rpm60", "rop_mh", "mse_x_area_N"]].to_csv(
        "data/rop_mse_9f9_windows.csv", index=False)
    seg.to_csv("data/rop_mse_9f9_segments.csv", index=False)
    print(seg.to_string(index=False))
    n_seg = len(seg)
    print(f"\n{len(a)} advancing windows, {n_seg} segments, derived ROP {a['rop_mh'].min():.1f} to {a['rop_mh'].max():.1f} m/h "
          f"(median {a['rop_mh'].median():.1f}); SURF_RPM {a['rpm60'].min():.3f} to {a['rpm60'].max():.3f} rev/s "
          f"({a['rpm60'].min()*60:.0f} to {a['rpm60'].max()*60:.0f} rpm)")

    def sp(x, y):
        r, p = spearmanr(x, y)
        return r, p

    print("\nSpearman correlation with stick-slip severity:")
    print(f"{'':<28}{'windows (n=%d, overlapping)' % len(a):>34}{'segments (n=%d)' % n_seg:>24}")
    res_rows = []
    for name, wcol, scol in (("derived ROP (m/h)", "rop_mh", "rop_mh"), ("MSE index (MSE x area, N)", "mse_x_area_N", "mse_x_area_N"),
                             ("mean torque (shared channel)", "tq_mean", "mean_torque"), ("WOB", "wob60", "wob_N")):
        rw, pw = sp(a["sev"], a[wcol])
        rs, ps = sp(seg["severity"], seg[scol])
        # leave-one-segment-out range of the segment-level r
        loo = [spearmanr(seg["severity"].drop(i), seg[scol].drop(i))[0] for i in seg.index]
        print(f"{name:<28}{rw:>+14.3f} (p={pw:.3f}){rs:>+14.3f} (p={ps:.3f})   leave-one-segment-out range {min(loo):+.2f} to {max(loo):+.2f}")
        res_rows.append({"variable": name, "window_r": rw, "window_p": pw, "segment_r": rs, "segment_p": ps,
                         "segment_loo_min": min(loo), "segment_loo_max": max(loo), "n_windows": len(a), "n_segments": n_seg})
    pd.DataFrame(res_rows).to_csv("data/rop_mse_9f9_correlations.csv", index=False)

    # ---------------- chart ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    labels = [w for w, _, _ in WELLS]
    stages = [("severity_windows", "Part 1 severity windows"), ("clean_windows", "of which time-contiguous"),
              ("advancing_windows", "of which BITDEP advancing")]
    x = np.arange(len(labels))
    width = 0.26
    for i, (col, lab) in enumerate(stages):
        vals = fdf.set_index("well").loc[labels, col].values
        bars = ax.bar(x + (i - 1) * width, vals, width, color=[BLUE, ORANGE, AQUA][i], label=lab)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 25, f"{int(val)}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK)
    ax.set_ylabel("Windows", color=SECONDARY_INK)
    ax.set_title("How many windows can say anything about ROP", color=INK, fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=GRIDLINE, lw=0.8)
    ax.set_axisbelow(True)
    ax2 = axes[1]
    ax2.scatter(a["rop_mh"], a["sev"], s=18, color=BLUE, alpha=0.45, label=f"windows (n={len(a)}, overlapping)")
    ax2.scatter(seg["rop_mh"], seg["severity"], s=40 + 12 * seg["windows"], color=ORANGE, edgecolor=SURFACE, lw=1,
                label=f"segment medians (n={n_seg}; size = windows)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Derived ROP from BITDEP (m/h, log scale)", color=SECONDARY_INK)
    ax2.set_ylabel("Stick-slip severity (torque range, N.m)", color=SECONDARY_INK)
    ax2.set_title("9-F-9: severity vs. derived ROP", color=INK, fontsize=11.5, loc="left")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(color=GRIDLINE, lw=0.8)
    ax2.set_axisbelow(True)
    for a_ in axes:
        a_.set_facecolor(SURFACE)
        for sp_ in ("top", "right"):
            a_.spines[sp_].set_visible(False)
        a_.spines["left"].set_color(BASELINE)
        a_.spines["bottom"].set_color(BASELINE)
        a_.tick_params(colors=MUTED, labelsize=9)
    fig.text(0.01, -0.03, "ROP_AVG, MSE and every related channel are zero in every log object of all three wells; ROP here is derived from the "
             "rate of change of bit depth, so it exists only where the bit visibly advanced.", fontsize=8.5, color=MUTED, ha="left")
    plt.tight_layout()
    fig.savefig("data/chart_rop_mse.png", dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_rop_mse.png")


if __name__ == "__main__":
    main()
