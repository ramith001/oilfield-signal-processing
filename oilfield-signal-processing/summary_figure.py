"""
Part 7: one consolidated figure for the repo. Reads the CSVs the earlier parts saved and runs no analysis.

    python summary_figure.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
HATCH_EDGE = "#ffffff"


def main():
    inv = pd.read_csv("data/extra_wells_inventory.csv")
    scal = pd.read_csv("data/f4_period_stability_window_scaling.csv")
    cont = pd.read_csv("data/severity_window_contiguity.csv")
    pers = pd.read_csv("data/spectrogram_persistence.csv")
    det3 = pd.read_csv("data/detector_validation.csv")
    det6 = pd.read_csv("data/extra_wells_detector.csv")
    sev = pd.read_csv("data/extra_wells_severity_correlations.csv")
    multi = pd.read_csv("data/multi_well_summary.csv")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.8), facecolor=SURFACE)

    ax = axes[0, 0]
    names = inv["well"].tolist()
    x = np.arange(len(names))
    ax.bar(x - 0.2, inv["record_hours"], 0.4, color=MUTED, label="hours of record")
    ax.bar(x + 0.2, inv["hours_rotating_drilling"].clip(lower=0.02), 0.4, color=ORANGE, label="hours of rotating drilling")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    ax.axvline(2.5, color=INK, lw=0.8, ls=":")
    ax.text(1.0, 2500, "Parts 1-5 wells", ha="center", fontsize=8.5, color=SECONDARY_INK)
    ax.set_ylabel("Hours (log scale)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Part 6: the first three wells barely rotate", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax = axes[0, 1]
    ax.plot(scal["window_min"], scal["median_period_s"] / 60, "o-", color=BLUE, lw=1.8, ms=7, label="dominant period found")
    ax.plot(scal["window_min"], scal["window_min"] / 3, "--", color=ORANGE, lw=1.4, label="window length / 3")
    ax.set_xlabel("Window length (min)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Period (min)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Part 2: the 2,911 s period follows the window", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 2]
    x = np.arange(len(cont))
    ax.bar(x - 0.2, cont["wob_r_part1_windows"], 0.4, color=BLUE, label="Part 1 windows")
    ax.bar(x + 0.2, cont["wob_r_clean_windows"], 0.4, color=ORANGE, label="time-contiguous windows only")
    for xi, p in zip(x, cont["pct_over_120s"]):
        ax.text(xi, -0.07, f"{p:.0f}% of windows\ncross a gap", ha="center", va="top", fontsize=8, color=SECONDARY_INK)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cont["well"])
    ax.set_ylim(-0.2, 0.45)
    ax.set_ylabel("Spearman, WOB vs severity", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Part 3: severity windows that crossed gaps", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax = axes[1, 0]
    c = pers[(pers["well"] == "9-F-4") & (pers["band"] == "clean 20-120 s")].set_index("channel")
    ch = ["torque", "wob", "hookload"]
    xs = np.arange(3)
    ax.bar(xs - 0.2, [c.loc[k, "median_abs_dlnT"] for k in ch], 0.4, color=BLUE, label="real")
    ax.bar(xs + 0.2, [c.loc[k, "null_median"] for k in ch], 0.4, color=MUTED, label="noise (median)")
    ax.set_xticks(xs)
    ax.set_xticklabels(["torque", "WOB", "hookload"])
    ax.set_ylabel("Median |change in ln period| (lower = steadier)", fontsize=9, color=SECONDARY_INK)
    ax.set_ylim(0, 0.5)
    ax.set_title("Part 4: the steady ~25 s period is in all three channels", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 1]
    wells = list(det3["well"]) + list(det6["well"])
    real = list(det3["events_per_live_hour"]) + list(det6["events_per_hour"])
    surr = list(det3["surrogate_events_per_hour_median"]) + list(det6["surrogate_events_per_hour_median"])
    x = np.arange(len(wells))
    ax.bar(x - 0.2, real, 0.4, color=BLUE, label="real torque")
    ax.bar(x + 0.2, surr, 0.4, color=MUTED, label="same-spectrum noise")
    ax.axvline(2.5, color=INK, lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(wells, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Regular events per hour", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Part 5: the detector finds as many events in noise", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax = axes[1, 2]
    labs = ["Part 1\n" + w for w in multi["well"]] + list(sev["well"])
    x = np.arange(len(labs))
    vals_abs = [np.nan] * 3 + list(sev["abs_vs_wob"])
    vals_rel = [np.nan] * 3 + list(sev["rel_vs_wob"])
    ax.bar(x[:3], multi["wob_spearman_r"], 0.5, color=BLUE, hatch="//", edgecolor=HATCH_EDGE, label="Part 1 (non-rotating), torque range")
    ax.bar(x[3:] - 0.2, vals_abs[3:], 0.4, color=BLUE, label="rotating, torque range")
    ax.bar(x[3:] + 0.2, vals_rel[3:], 0.4, color=ORANGE, label="rotating, relative severity")
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Spearman with WOB", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Part 6: WOB against severity on rotating wells", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.set_ylim(-0.45, 0.66)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    for a in axes.ravel():
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.spines["left"].set_color(BASELINE)
        a.spines["bottom"].set_color(BASELINE)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig("data/chart_summary.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("Saved data/chart_summary.png")


if __name__ == "__main__":
    main()
