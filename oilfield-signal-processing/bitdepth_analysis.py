"""
Does stick-slip severity also vary with hole depth (BITDEP), alongside the
WOB relationship signal_analysis.py already found? Listed as a possible
extension in the README: "drilling-dynamics theory predicts [stick-slip
severity varies] with hole depth, for a longer, more compliant drillstring."

Reuses data/drilling_severity_index.csv (signal_analysis.py's output), which
already carries BITDEP alongside the same live-torque-row filter and
torque_range_60s severity index used for the WOB check, so this is an
additional cut of the same cleaned data, not a new fetch or a new cleaning
pass.

Important scope caveat, reported rather than hidden: this run's live rows
span BITDEP 127-196m, a single ~69m interval within one drilling block of
one well. A depth relationship found here is local to that interval, not
evidence about depth effects across a well's full length -- see
multi_well_validation.py, which re-checks this question on two more wells,
one of which (9-F-4) does span a genuine multi-hundred-metre depth range.

    python bitdepth_analysis.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataviz skill palette (references/palette.md). Single hue per panel, no
# legend needed (each panel's title names its own variable).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

d = pd.read_csv("data/drilling_severity_index.csv", parse_dates=["Time"])
live = d[d["is_live"]].dropna(subset=["torque_range_60s", "wob_60s", "BITDEP"]).copy()
live = live[(live["wob_60s"] > 0) & (live["BITDEP"] > 0)]
print(f"{len(live)} genuinely-live rows with valid BITDEP "
      f"(same 953-row set signal_analysis.py used for the WOB check).")
print(f"BITDEP range in this run: {live['BITDEP'].min():.1f}m to {live['BITDEP'].max():.1f}m "
      f"({live['BITDEP'].max() - live['BITDEP'].min():.1f}m span) -- a single interval within one "
      f"drilling block of one well, not a well-length depth sweep. Treat any pattern below as local.")

# --- severity vs depth, same quintile-breakdown style as the WOB check ---
depth_bins = pd.qcut(live["BITDEP"], 5, duplicates="drop")
severity_by_depth = live.groupby(depth_bins, observed=True)["torque_range_60s"].agg(["median", "mean", "count"])
print("\n60s torque oscillation range by BITDEP quintile:")
print(severity_by_depth.to_string())

corr_depth = live["BITDEP"].corr(live["torque_range_60s"], method="spearman")
print(f"\nSpearman correlation (BITDEP vs. torque-oscillation severity): {corr_depth:.3f}")

# --- does depth add anything once WOB is accounted for? partial Spearman ---
# (rank-residual approach: regress each variable's ranks on the other
# predictor's rank, correlate the residuals -- a simple partial-correlation
# proxy, not a full model, reported as exactly that)
def rank_residual(y, x):
    ry, rx = y.rank(), x.rank()
    beta = np.polyfit(rx, ry, 1)
    return ry - np.polyval(beta, rx)

resid_severity_given_wob = rank_residual(live["torque_range_60s"], live["wob_60s"])
resid_depth_given_wob = rank_residual(live["BITDEP"], live["wob_60s"])
partial_corr = np.corrcoef(resid_severity_given_wob, resid_depth_given_wob)[0, 1]
print(f"Partial correlation of BITDEP with severity, net of WOB (rank-residual proxy): {partial_corr:.3f}")
print(
    "Interpretation: " +
    ("a depth relationship survives even after accounting for WOB, so it isn't just WOB and depth moving "
     "together." if abs(partial_corr) > 0.1 and abs(partial_corr) > abs(corr_depth) * 0.5
     else "once WOB is accounted for, depth adds little on its own in this interval -- consistent with WOB "
          "being the dominant driver here, not depth independently.")
)

live.to_csv("data/depth_severity_data.csv", index=False)

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)

ax = axes[0]
ax.scatter(live["BITDEP"], live["torque_range_60s"], s=10, alpha=0.35, color=BLUE)
mids = [iv.mid for iv in severity_by_depth.index]
ax.plot(mids, severity_by_depth["median"].values, color=ORANGE, marker="o", lw=1.5, label="quintile median")
ax.set_xlabel("Bit depth, BITDEP (m)", fontsize=10, color=SECONDARY_INK)
ax.set_ylabel("60s rolling torque range (N·m)", fontsize=10, color=SECONDARY_INK)
ax.set_title(f"Stick-slip severity vs. bit depth\n(Spearman r = {corr_depth:.2f}, single ~69m interval)",
             fontsize=11, color=INK, loc="left", pad=10, fontweight="bold")
ax.legend(fontsize=8, frameon=False)

ax = axes[1]
ax.scatter(resid_depth_given_wob, resid_severity_given_wob, s=10, alpha=0.35, color=BLUE)
ax.set_xlabel("BITDEP rank, residual after WOB", fontsize=10, color=SECONDARY_INK)
ax.set_ylabel("Severity rank, residual after WOB", fontsize=10, color=SECONDARY_INK)
ax.set_title(f"Depth vs. severity net of WOB\n(partial corr = {partial_corr:.2f})",
             fontsize=11, color=INK, loc="left", pad=10, fontweight="bold")

for ax in axes:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRIDLINE, lw=0.8)
    ax.set_facecolor(SURFACE)

fig.suptitle("Does stick-slip severity vary with hole depth, independent of WOB?",
             fontsize=12, color=INK, x=0.01, ha="left", y=1.03)
fig.text(0.01, -0.03,
          "Live rows only (same filter as the WOB check). BITDEP spans a single ~69m interval in this "
          "run -- a local result, not a well-length depth sweep.", fontsize=8.5, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
fig.savefig("data/chart_depth_severity.png", dpi=180, facecolor=SURFACE, bbox_inches="tight")
print("\nSaved data/chart_depth_severity.png")
