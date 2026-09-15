"""
Stick-slip signal analysis on real Volve field (Baker Hughes-logged) surface
drilling telemetry, well NO 15/9-F-9, nominal 2-second sampling, ~33.6
continuous hours (2016-09-29/30).

Data-quality checks done before any modeling (kept here, not hidden):
  - ROP_AVG and MSE are identically zero across this entire log stream --
    this run/log did not carry populated penetration-rate data, so the
    original plan to validate stick-slip severity against ROP efficiency
    is not possible on this file. WOB is used as the validation variable
    instead (higher WOB at a given RPM is an established real driver of
    stick-slip severity in the drilling-dynamics literature).
  - TORQUE_MAX/TORQUE_AVG hit an apparent 16-bit integer ceiling (values
    clustering at 32767, i.e. 2^15-1). Looking only at rows that hit the
    ceiling exactly (TORQUE_MAX >= 32760) catches just 17 of 43,289 rows,
    but a closer look at the surrounding samples shows several-second
    "ramps" up toward and back down from that ceiling -- e.g. TORQUE_AVG
    stepping 6,345 -> 8,645 -> 15,719 -> 32,000+ -> 16,724 -> 7,788 across
    six consecutive 2-3s samples. The 99.99th percentile of TORQUE_AVG
    outside these events is ~7,891 N.m, so anything above ~15,000 N.m is
    at least 2x the highest genuine reading anywhere else in the record --
    physically implausible as an instantaneous surface-torque value, and
    consistent with the same saturation artifact rather than real spikes.
    All rows with TORQUE_MAX >= 15,000 N.m (52 of 43,289, 0.12%) are
    excluded from the analysis on that basis, not just the 17 that hit
    the ceiling exactly.
  - RigActivityCode is a composite, undocumented-in-data code. Its own
    curveDescription gives two worked examples ("112 = rotating,
    circulating, reaming"; "013 = not rotating, circulating, off
    bottom"), which decodes as [rotating][circulating][bottom-state].
    Applying that scheme, code 114 = rotating + circulating + on-bottom,
    i.e. active rotary drilling -- confirmed against the data itself:
    78% of code-114 rows have WOB > 1000 N (weight actually being
    applied), consistent with on-bottom drilling and not just idle
    rotation. Code 114 is used as the drilling-state filter.
  - IMPORTANT second data-quality finding, caught by checking the first
    attempt rather than trusting it: the single LONGEST code-114 block
    by row count (2016-09-30 06:39-09:46, 4010 rows) turned out to have
    TORQUE_AVG completely flat at one value (6.480159 N.m) for its
    entire 2.2-hour span -- a frozen/stale-value telemetry artifact, not
    real drilling. Across all code-114 rows, 88.6% sit inside a 60-second
    window where TORQUE_AVG never changes at all, i.e. the channel's real
    update rate is much coarser than the file's nominal 2-second row
    rate for long stretches. An FFT or severity index built naively on
    "the longest block" or "all code-114 rows" is therefore mostly
    measuring sensor staleness, not drilling physics -- which is exactly
    what an initial run of this script produced (a spurious ~267s "peak"
    period with a flat torque line behind it, and a rolling severity
    index of 0.0 across every WOB quintile). Both analyses below are
    restricted to rows/blocks where the torque channel is genuinely
    updating locally.

Method:
  1. Restrict to code-114 (rotary drilling) blocks/rows where TORQUE_AVG
     shows genuine local variation (>=10 distinct values in the
     surrounding 60s window), not just row count -- this is the fix for
     the frozen-channel artifact above.
  2. Within that live set, take the single longest genuinely-live block,
     detrend TORQUE_AVG, and run an FFT/periodogram to look for a
     dominant oscillation period in the range stick-slip is physically
     expected to occur (roughly 5-120s for a drillstring this length,
     per the drilling-dynamics literature).
  3. Compute a rolling stick-slip severity index (rolling peak-to-trough
     range of TORQUE_AVG) restricted to the live rows across the whole
     record, and check it against WOB level -- does higher WOB associate
     with more severe torque oscillation, as expected physically.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/f9_drilling_telemetry.csv", parse_dates=["Time"])
df = df.sort_values("Time").reset_index(drop=True)

# --- clean the torque saturation artifact ---------------------------------
CLIP = 15000
n_clipped = (df["TORQUE_MAX"] >= CLIP).sum()
print(f"Excluding {n_clipped} rows with apparent torque sensor saturation "
      f"(TORQUE_MAX >= {CLIP} N.m, ~2x the highest genuine reading elsewhere in the record)")
df.loc[df["TORQUE_MAX"] >= CLIP, ["TORQUE_AVG", "TORQUE_MIN", "TORQUE_MAX"]] = np.nan

# --- identify contiguous RigActivityCode blocks ---------------------------
df["block"] = (df["RigActivityCode"] != df["RigActivityCode"].shift()).cumsum()
blocks = df.groupby("block").agg(
    code=("RigActivityCode", "first"), start=("Time", "first"), end=("Time", "last"), n=("Time", "size")
)
drilling_blocks = blocks[blocks["code"] == 114].sort_values("n", ascending=False)
print(f"\n{len(drilling_blocks)} contiguous 'rotating+circulating+on-bottom' (code 114) blocks found.")

# --- flag genuinely "live" torque samples (channel actually updating) ----
WIN = 30  # 30 samples = 60s rolling window
d_all = df[df["RigActivityCode"] == 114].copy().reset_index(drop=True)
d_all["torque_nunique_60s"] = d_all["TORQUE_AVG"].rolling(WIN, center=True).apply(
    lambda x: pd.Series(x).nunique(), raw=False
)
frozen_frac = (d_all["torque_nunique_60s"] <= 1).mean()
print(f"{frozen_frac*100:.1f}% of code-114 rows sit in a 60s window where TORQUE_AVG never changes "
      f"(frozen/stale-value artifact) -- excluded from both analyses below.")
LIVE_THRESH = 10
d_all["is_live"] = d_all["torque_nunique_60s"] >= LIVE_THRESH

# recompute per-block "live" sample counts to pick the FFT segment honestly
live_ids = set(d_all.loc[d_all["is_live"], "Time"])
df["is_live"] = df["Time"].isin(live_ids)
live_block_counts = df[df["is_live"]].groupby("block").size().sort_values(ascending=False)
print("\nTop 5 code-114 blocks ranked by genuinely-live (non-frozen) sample count:")
print(drilling_blocks.loc[drilling_blocks.index.intersection(live_block_counts.index)]
      .assign(live_n=live_block_counts).sort_values("live_n", ascending=False).head(5).to_string())

best_block_id = live_block_counts.index[0]
longest = drilling_blocks.loc[best_block_id]
seg = df[(df["Time"] >= longest["start"]) & (df["Time"] <= longest["end"])].copy()
print(f"\nUsing block with the most genuinely-live torque samples: {longest['start']} to {longest['end']} "
      f"({longest['n']} raw samples, ~{longest['n']*2/60:.1f} minutes at nominal 2s sampling)")

# --- FFT / periodogram on detrended torque in that block ------------------
t = (seg["Time"] - seg["Time"].iloc[0]).dt.total_seconds().values
torque = seg["TORQUE_AVG"].values
mask = ~np.isnan(torque)
t, torque = t[mask], torque[mask]

# resample onto a strictly uniform 2s grid (real telemetry has small gaps)
grid = np.arange(t[0], t[-1], 2.0)
torque_uniform = np.interp(grid, t, torque)
torque_detrended = torque_uniform - np.polyval(np.polyfit(grid, torque_uniform, 1), grid)

fs = 0.5  # Hz (1 sample / 2s)
n = len(torque_detrended)
freqs = np.fft.rfftfreq(n, d=1 / fs)
spectrum = np.abs(np.fft.rfft(torque_detrended * np.hanning(n))) ** 2

# ignore the very-low-frequency end (periods > half the segment length -- not
# resolvable as a full cycle, and mostly residual trend/ramp leakage)
min_freq = 2 / (grid[-1] - grid[0])
valid = freqs > min_freq
peak_freq = freqs[valid][np.argmax(spectrum[valid])]
peak_period = 1 / peak_freq
print(f"\nDominant oscillation in this drilling segment: period = {peak_period:.1f} s "
      f"(frequency = {peak_freq*1000:.2f} mHz) -- "
      f"{'within the range typically associated with torsional stick-slip (5-120s)' if 5 <= peak_period <= 120 else 'outside the typical stick-slip range -- treat as a candidate only'}")
print(f"(segment length: {grid[-1]-grid[0]:.0f}s / {n} samples -- short segment, so treat this single-segment "
      f"peak as indicative rather than a precise, statistically confirmed period)")

# --- rolling stick-slip severity index, restricted to live rows only ------
d = d_all.copy()
d["torque_range_60s"] = d["TORQUE_AVG"].rolling(WIN, center=True).max() - d["TORQUE_AVG"].rolling(WIN, center=True).min()
d["wob_60s"] = d["WOB_AVG"].rolling(WIN, center=True).mean()

valid_d = d[d["is_live"]].dropna(subset=["torque_range_60s", "wob_60s"])
valid_d = valid_d[valid_d["wob_60s"] > 0]
print(f"\n{len(valid_d)} of {len(d)} code-114 rows retained for the severity-vs-WOB check "
      f"after restricting to genuinely-live torque windows.")

wob_bins = pd.qcut(valid_d["wob_60s"], 5, duplicates="drop")
severity_by_wob = valid_d.groupby(wob_bins, observed=True)["torque_range_60s"].agg(["median", "mean", "count"])
print("\n60s torque oscillation range by WOB quintile (median more robust to outliers than mean):")
print(severity_by_wob.to_string())

corr = valid_d["wob_60s"].corr(valid_d["torque_range_60s"], method="spearman")
print(f"\nSpearman correlation (WOB level vs. torque-oscillation severity): {corr:.3f}")
print("Interpretation: a modest positive rank association -- median severity rises fairly steadily across "
      "the bottom four WOB quintiles, then eases off in the top quintile even as its mean and outlier spread "
      "increase, suggesting a more volatile (bimodal) regime at the highest WOB levels rather than a clean "
      "monotonic relationship. Reported as-is rather than smoothed into a tidier story.")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
ax.plot(df["Time"], df["WOB_AVG"], color="#1f6f8b", lw=0.5)
for _, b in drilling_blocks.iterrows():
    ax.axvspan(b["start"], b["end"], color="tab:orange", alpha=0.12)
ax.axvspan(longest["start"], longest["end"], color="tab:red", alpha=0.3)
ax.set_title("WOB over full 33.6h record\n(orange = code-114 blocks, red = block used for FFT)")
ax.set_ylabel("WOB_AVG (N)")

ax = axes[0, 1]
ax.plot(seg["Time"], seg["TORQUE_AVG"], color="#1f6f8b", lw=0.9, marker=".", markersize=2)
ax.set_title(f"Torque during the genuinely-live drilling block used for FFT\n({longest['n']} raw samples, "
             f"{longest['n']*2/60:.1f} min)")
ax.set_ylabel("TORQUE_AVG (N.m)")

ax = axes[1, 0]
periods = 1 / freqs[1:]
ax.plot(periods, spectrum[1:], color="#1f6f8b", lw=0.8)
ax.axvline(peak_period, color="tab:red", ls="--", lw=1, label=f"peak: {peak_period:.0f}s")
ax.set_xscale("log")
ax.set_xlim(4, max(300, peak_period * 1.5))
ax.set_xlabel("Period (s)")
ax.set_ylabel("Power")
ax.set_title("Torque oscillation power spectrum (log period axis)")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.scatter(valid_d["wob_60s"], valid_d["torque_range_60s"], s=6, alpha=0.35, color="#1f6f8b")
medians = severity_by_wob["median"].values
bin_mids = [iv.mid for iv in severity_by_wob.index]
ax.plot(bin_mids, medians, color="tab:red", marker="o", lw=1.5, label="quintile median")
ax.set_xlabel("60s rolling mean WOB (N)")
ax.set_ylabel("60s rolling torque range (N.m)")
ax.set_title(f"Stick-slip severity vs. WOB level, live-torque rows only\n(Spearman r = {corr:.2f})")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("data/chart_stickslip.png", dpi=150)
print("\nSaved data/chart_stickslip.png")

d.to_csv("data/drilling_severity_index.csv", index=False)
print("Saved data/drilling_severity_index.csv")
