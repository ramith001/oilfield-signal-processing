"""
Does the single-well stick-slip story (23.8s dominant period; WOB-severity
Spearman r = 0.27; BITDEP-severity Spearman r = -0.25 over a narrow ~69m
interval) hold up on other wells, as the README's own "Possible extensions"
asks? The mirror repo turns out to contain three wells, not one -- see
extract_witsml.py's docstring for how 9-F-7's and 9-F-4's real-time
drilling-parameters log objects were identified (each well numbers its log
objects independently, so this isn't just reusing 9-F-9's "log 9").

This script reruns the identical cleaning + FFT + severity-index pipeline
signal_analysis.py built for 9-F-9 -- same TORQUE_MAX>=15000 saturation
clip, same code-114 (rotating+circulating+on-bottom) filter, same 60s/30-
sample rolling-window live-vs-frozen torque classification -- against all
three wells, refactored into shared functions so the method is provably
identical rather than re-implemented per well. Per-well results are
reported as-is, including where they don't line up, rather than only
keeping the wells that confirm the single-well finding.

Well 9-F-4 is a particularly useful addition on its own: it's a ~7-day
continuous record (vs. 9-F-9's 33.6h) whose live rows span BITDEP
1m-2616m, a genuine depth sweep rather than the ~69m single interval
9-F-9 offered -- the exact limitation bitdepth_analysis.py flagged as
needing more wells/sections to resolve.

    python multi_well_validation.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataviz skill palette (references/palette.md). Categorical hues in fixed
# order for the 3 wells (<=3 slots, all-pairs CVD-safe).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
NEUTRAL = "#f0efec"
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

WIN = 30          # 30 samples = 60s rolling window at nominal 2s sampling
LIVE_THRESH = 10  # min distinct TORQUE_AVG values in the 60s window to count as "live"
CLIP = 15000      # TORQUE_MAX >= this -> 16-bit sensor-saturation artifact, same as signal_analysis.py


def clean_and_flag_live(df):
    df = df.sort_values("Time").reset_index(drop=True)
    n_clipped = int((df["TORQUE_MAX"] >= CLIP).sum())
    df.loc[df["TORQUE_MAX"] >= CLIP, ["TORQUE_AVG", "TORQUE_MIN", "TORQUE_MAX"]] = np.nan

    df["block"] = (df["RigActivityCode"] != df["RigActivityCode"].shift()).cumsum()
    blocks = df.groupby("block").agg(code=("RigActivityCode", "first"), start=("Time", "first"),
                                      end=("Time", "last"), n=("Time", "size"))
    drilling_blocks = blocks[blocks["code"] == 114].sort_values("n", ascending=False)

    d_all = df[df["RigActivityCode"] == 114].copy().reset_index(drop=True)
    if len(d_all) == 0:
        return df, d_all, drilling_blocks, n_clipped, float("nan")
    d_all["torque_nunique_60s"] = d_all["TORQUE_AVG"].rolling(WIN, center=True).apply(
        lambda x: pd.Series(x).nunique(), raw=False)
    frozen_frac = float((d_all["torque_nunique_60s"] <= 1).mean())
    d_all["is_live"] = d_all["torque_nunique_60s"] >= LIVE_THRESH

    live_ids = set(d_all.loc[d_all["is_live"], "Time"])
    df["is_live"] = df["Time"].isin(live_ids)
    return df, d_all, drilling_blocks, n_clipped, frozen_frac


def fft_longest_live_block(df, drilling_blocks):
    if "is_live" not in df.columns or not df["is_live"].any():
        return None
    live_block_counts = df[df["is_live"]].groupby("block").size().sort_values(ascending=False)
    if len(live_block_counts) == 0:
        return None
    best_block_id = live_block_counts.index[0]
    longest = drilling_blocks.loc[best_block_id]
    seg = df[(df["Time"] >= longest["start"]) & (df["Time"] <= longest["end"])].copy()

    t = (seg["Time"] - seg["Time"].iloc[0]).dt.total_seconds().values
    torque = seg["TORQUE_AVG"].values
    mask = ~np.isnan(torque)
    t, torque = t[mask], torque[mask]
    if len(t) < 30:
        return None

    grid = np.arange(t[0], t[-1], 2.0)
    if len(grid) < 16:
        return None
    torque_uniform = np.interp(grid, t, torque)
    torque_detrended = torque_uniform - np.polyval(np.polyfit(grid, torque_uniform, 1), grid)

    fs = 0.5
    n = len(torque_detrended)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    spectrum = np.abs(np.fft.rfft(torque_detrended * np.hanning(n))) ** 2
    min_freq = 2 / (grid[-1] - grid[0])
    valid = freqs > min_freq
    if not valid.any():
        return None
    peak_freq = freqs[valid][np.argmax(spectrum[valid])]
    peak_period = 1 / peak_freq
    return {
        "period_s": peak_period,
        "segment_min": (grid[-1] - grid[0]) / 60,
        "n_samples": n,
        "live_block_n": int(live_block_counts.iloc[0]),
        "n_live_blocks": len(live_block_counts),
        "freqs": freqs, "spectrum": spectrum,
    }


def rank_residual(y, x):
    ry, rx = y.rank(), x.rank()
    beta = np.polyfit(rx, ry, 1)
    return ry - np.polyval(beta, rx)


def severity_correlations(d_all):
    d = d_all.copy()
    d["torque_range_60s"] = d["TORQUE_AVG"].rolling(WIN, center=True).max() - d["TORQUE_AVG"].rolling(WIN, center=True).min()
    d["wob_60s"] = d["WOB_AVG"].rolling(WIN, center=True).mean()

    wob_valid = d[d["is_live"]].dropna(subset=["torque_range_60s", "wob_60s"])
    wob_valid = wob_valid[wob_valid["wob_60s"] > 0]
    wob_corr = wob_valid["wob_60s"].corr(wob_valid["torque_range_60s"], method="spearman") if len(wob_valid) >= 20 else float("nan")

    bd_corr = partial_corr = float("nan")
    bd_min = bd_max = float("nan")
    n_bd = 0
    if "BITDEP" in d.columns:
        bd_valid = d[d["is_live"]].dropna(subset=["torque_range_60s", "wob_60s", "BITDEP"])
        bd_valid = bd_valid[(bd_valid["wob_60s"] > 0) & (bd_valid["BITDEP"] > 0)]
        n_bd = len(bd_valid)
        if n_bd >= 20:
            bd_corr = bd_valid["BITDEP"].corr(bd_valid["torque_range_60s"], method="spearman")
            resid_sev = rank_residual(bd_valid["torque_range_60s"], bd_valid["wob_60s"])
            resid_bd = rank_residual(bd_valid["BITDEP"], bd_valid["wob_60s"])
            partial_corr = float(np.corrcoef(resid_sev, resid_bd)[0, 1])
            bd_min, bd_max = bd_valid["BITDEP"].min(), bd_valid["BITDEP"].max()

    return {
        "n_wob": len(wob_valid), "wob_corr": wob_corr,
        "n_bitdep": n_bd, "bitdep_corr": bd_corr, "bitdep_partial_corr": partial_corr,
        "bitdep_min": bd_min, "bitdep_max": bd_max,
    }


def main():
    results = {}
    for label, path, _ in WELLS:
        print(f"\n=== {label} ===")
        df = pd.read_csv(path, parse_dates=["Time"])
        df, d_all, drilling_blocks, n_clipped, frozen_frac = clean_and_flag_live(df)
        print(f"{len(df)} total rows, {len(d_all)} code-114 (rotating+circulating+on-bottom) rows, "
              f"{n_clipped} rows excluded for torque-saturation (TORQUE_MAX >= {CLIP}), "
              f"{frozen_frac*100:.1f}% of code-114 rows frozen/stale")

        fft_res = fft_longest_live_block(df, drilling_blocks)
        if fft_res:
            print(f"Longest live block: {fft_res['segment_min']:.1f} min ({fft_res['n_samples']} samples), "
                  f"dominant period = {fft_res['period_s']:.1f}s "
                  f"({'in the 5-120s stick-slip range' if 5 <= fft_res['period_s'] <= 120 else 'OUTSIDE the typical stick-slip range'})")
        else:
            print("No live block long enough for a usable FFT.")

        corrs = severity_correlations(d_all)
        print(f"WOB check: n={corrs['n_wob']}, Spearman r = {corrs['wob_corr']:.3f}" if corrs["n_wob"] else "WOB check: not enough rows")
        if corrs["n_bitdep"]:
            print(f"BITDEP check: n={corrs['n_bitdep']}, Spearman r = {corrs['bitdep_corr']:.3f}, "
                  f"partial r (net of WOB) = {corrs['bitdep_partial_corr']:.3f}, "
                  f"depth range {corrs['bitdep_min']:.0f}-{corrs['bitdep_max']:.0f}m "
                  f"(span {corrs['bitdep_max']-corrs['bitdep_min']:.0f}m)")
        else:
            print("BITDEP check: not enough rows")

        results[label] = {"fft": fft_res, "corrs": corrs, "n_total": len(df), "n_drilling": len(d_all)}

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    rows = []
    for label, _, _ in WELLS:
        r = results[label]
        rows.append({
            "well": label,
            "total_rows": r["n_total"],
            "code114_rows": r["n_drilling"],
            "stickslip_period_s": round(r["fft"]["period_s"], 1) if r["fft"] else None,
            "wob_n": r["corrs"]["n_wob"],
            "wob_spearman_r": round(r["corrs"]["wob_corr"], 3) if not np.isnan(r["corrs"]["wob_corr"]) else None,
            "bitdep_n": r["corrs"]["n_bitdep"],
            "bitdep_spearman_r": round(r["corrs"]["bitdep_corr"], 3) if not np.isnan(r["corrs"]["bitdep_corr"]) else None,
            "bitdep_partial_r": round(r["corrs"]["bitdep_partial_corr"], 3) if not np.isnan(r["corrs"]["bitdep_partial_corr"]) else None,
            "bitdep_depth_span_m": round(r["corrs"]["bitdep_max"] - r["corrs"]["bitdep_min"], 0) if not np.isnan(r["corrs"]["bitdep_min"]) else None,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv("data/multi_well_summary.csv", index=False)
    print("\nCross-well summary:")
    print(summary.to_string(index=False))
    print("\nSaved data/multi_well_summary.csv")

    # ------------------------------------------------------------------
    # Comparison chart: 3 panels, one per well-level metric
    # ------------------------------------------------------------------
    labels = [w[0] for w in WELLS]
    colors = [w[2] for w in WELLS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=SURFACE)

    # Panel 1: stick-slip period per well. Log scale -- 9-F-4's dominant period
    # comes out nearly two orders of magnitude above the other two wells (see
    # printed caveat below), so a linear axis would flatten 9-F-9/9-F-7 to
    # invisibility.
    ax = axes[0]
    periods = [results[l]["fft"]["period_s"] if results[l]["fft"] else 0 for l in labels]
    ax.axhspan(5, 120, color=NEUTRAL, alpha=0.6, zorder=0, label="5-120s expected range")
    bars = ax.bar(labels, periods, color=colors, width=0.55, zorder=2)
    for b, p in zip(bars, periods):
        ax.text(b.get_x() + b.get_width() / 2, p * 1.15, f"{p:.1f}s" if p else "n/a",
                ha="center", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_ylabel("Dominant torque-oscillation period (s, log scale)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Stick-slip period by well", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.set_ylim(3, max(periods) * 2.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # Panel 2: WOB-severity correlation per well
    ax = axes[1]
    wob_rs = [results[l]["corrs"]["wob_corr"] for l in labels]
    bars = ax.bar(labels, wob_rs, color=colors, width=0.55)
    ax.axhline(0, color=BASELINE, lw=1)
    for b, r in zip(bars, wob_rs):
        if not np.isnan(r):
            ax.text(b.get_x() + b.get_width() / 2, r + (0.02 if r >= 0 else -0.05), f"{r:.2f}",
                    ha="center", fontsize=9, color=INK)
    ax.set_ylabel("Spearman r (severity vs. WOB)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("WOB-severity correlation by well", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.set_ylim(-0.5, 0.6)

    # Panel 3: BITDEP-severity correlation per well (raw + partial net of WOB)
    ax = axes[2]
    x = np.arange(len(labels))
    w = 0.32
    bd_rs = [results[l]["corrs"]["bitdep_corr"] for l in labels]
    bd_partials = [results[l]["corrs"]["bitdep_partial_corr"] for l in labels]
    ax.bar(x - w / 2, bd_rs, width=w, color=colors, label="raw Spearman r")
    ax.bar(x + w / 2, bd_partials, width=w, color=colors, alpha=0.45, label="partial r (net of WOB)")
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Correlation (severity vs. BITDEP)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Depth-severity correlation by well", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.set_ylim(-0.5, 0.5)

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(BASELINE)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRIDLINE, lw=0.8, axis="y")
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)

    fig.suptitle("Does the single-well (9-F-9) stick-slip story hold up on other Volve wells?",
                 fontsize=13, color=INK, x=0.01, ha="left", y=1.04)
    fig.text(0.01, -0.04,
              "9-F-9: 33.6h, 9-F-7: ~2.4 days, 9-F-4: ~7 days (also the only well with a genuine depth sweep in its "
              "live rows: BITDEP 1-2616m, vs. 9-F-9's single ~69m interval). Same cleaning/filtering pipeline applied "
              "to all three -- see extract_witsml.py and this script's docstring for how each well's log object was found.",
              fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig("data/chart_multi_well.png", dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_multi_well.png")


if __name__ == "__main__":
    main()
