"""
Part 4: how does the dominant stick-slip frequency move over time within each well?

Parts 1 and 2 reduced each well to one FFT period. This script slides a 5-minute window (50% overlap) along every
contiguous run of live torque and keeps the whole spectrum of each window, so the dominant period can be followed
through time instead of reported once.

It reuses the Part 2 machinery (f4_period_stability.py) unchanged: the same live-row definition, the same 2 s
re-gridding, the same prewhitened periodogram divided by a fitted red-noise background, and the same bootstrap null
for "is the largest in-band ratio bigger than noise gives". Torque is tested together with WOB and hookload, which
should not carry a torsional oscillation, so a peak that shows up in all three is not torque physics.

Two questions are asked of the sequence of windows:
  1. Is there a peak at all? Per window, and in total against what 5% false positives would give.
  2. Does the dominant period persist from one window to the next? The statistic is the median absolute change in
     log period between consecutive windows of the same live section. Its null comes from simulating a continuous
     noise series for each real section (same length, same fitted spectral slope), cutting it into the same
     overlapping windows and running the identical pipeline, so the overlap between neighbouring windows, which makes
     even noise look persistent, is part of the null.

How much each well can say is limited by how much live torque it has, and that is reported first: only 9-F-4 has
more than a handful of windows.

    python spectrogram_analysis.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from multi_well_validation import clean_and_flag_live
from f4_period_stability import (
    DT, BAND_FULL, BAND_CLEAN, CADENCE_S, ALPHA, detrend, uniform_series, live_sections, band_mask,
    whitened_periodogram, fit_background, gaussian_noise, ratios_with_null, single_test,
)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
NEUTRAL = "#f0efec"

WIN_MIN = 5
STEP_MIN = 2.5
L = int(WIN_MIN * 60 / DT)
STEP = int(STEP_MIN * 60 / DT)
N_SIM_WINDOW = 200      # bootstrap draws per window for the per-window p-value
N_SIM_PERSIST = 300     # simulated continuous series per real dataset for the persistence null
SEED = 4
WELLS = [("9-F-4", "data/f4_drilling_telemetry.csv"),
         ("9-F-9", "data/f9_drilling_telemetry.csv"),
         ("9-F-7", "data/f7_drilling_telemetry.csv")]
CHANNELS = [("TORQUE_AVG", "torque"), ("WOB_AVG", "wob"), ("HKLD_AVG", "hookload")]


def window_ratio(x):
    """Observed-to-background spectrum of one detrended window (no bootstrap)."""
    fr, pw = whitened_periodogram(x)
    bg, _ = fit_background(fr, pw, (len(x) - 1) * DT)
    return fr, pw / bg


def dominant_in_band(fr, ratio, band):
    m = band_mask(fr, band)
    i = int(np.argmax(ratio[m]))
    return float(1 / fr[m][i]), float(ratio[m][i])


def section_windows(sec, rng, with_null):
    """Sliding windows over one live section, for torque (tested) and the control channels (peak only)."""
    out = []
    series = {}
    for col, name in CHANNELS:
        grid, x = uniform_series(sec["Time"], sec[col])
        if x is None:
            return out
        series[name] = (grid, x)
    n = len(series["torque"][1])
    for s in range(0, n - L + 1, STEP):
        row = {"start": sec["Time"].iloc[0] + pd.Timedelta(seconds=float(series["torque"][0][s]))}
        ok = True
        for name, (grid, x) in series.items():
            xs = detrend(x[s:s + L], grid[s:s + L])
            if np.ptp(np.diff(xs)) == 0:
                ok = False
                break
            if name == "torque" and with_null:
                fr, r_obs, null = ratios_with_null(xs, rng, n_sim=N_SIM_WINDOW)
                for label, band in (("full", BAND_FULL), ("clean", BAND_CLEAN)):
                    t = single_test(r_obs, null, band_mask(fr, band))
                    row[f"p_{label}"] = t["p"]
                    row[f"thr_{label}"] = t["threshold"]
                row["ratio_spectrum"] = r_obs
                row["freqs"] = fr
            else:
                fr, r_obs = window_ratio(xs)
                if name == "torque":
                    row["ratio_spectrum"], row["freqs"] = r_obs, fr
            for label, band in (("full", BAND_FULL), ("clean", BAND_CLEAN)):
                per, rat = dominant_in_band(fr, r_obs, band)
                row[f"{name}_period_{label}_s"] = per
                row[f"{name}_ratio_{label}"] = rat
        if not ok:
            continue
        out.append(row)
    return out


def persistence_stat(periods_by_section):
    """Median absolute change in ln(period) between consecutive windows within a section."""
    d = []
    for p in periods_by_section:
        if len(p) > 1:
            d.extend(np.abs(np.diff(np.log(p))))
    return float(np.median(d)) if d else np.nan


def simulated_persistence(sections_x, rng, band, n_sim):
    """Null distribution of the persistence statistic: noise with each real section's own length and slope."""
    stats = []
    params = []
    for grid, x in sections_x:
        xs = detrend(x, grid)
        fr, pw = whitened_periodogram(xs)
        _, slope_w = fit_background(fr, pw, (len(xs) - 1) * DT)
        params.append((len(x), float(np.clip(slope_w - 2.0, -4.0, 0.0))))
    for _ in range(n_sim):
        per_sec = []
        for n, slope in params:
            noise = gaussian_noise(n, slope, rng)
            t = np.arange(n) * DT
            ps = []
            for s in range(0, n - L + 1, STEP):
                xs = detrend(noise[s:s + L], t[s:s + L])
                fr, r = window_ratio(xs)
                ps.append(dominant_in_band(fr, r, band)[0])
            per_sec.append(ps)
        stats.append(persistence_stat(per_sec))
    return np.array(stats)


def main():
    rng = np.random.default_rng(SEED)
    inventory, win_tables, spectra = [], [], {}
    persist_rows = []
    persist_null = {}

    for well, path in WELLS:
        df = pd.read_csv(path, parse_dates=["Time"])
        df, d_all, blocks, _, _ = clean_and_flag_live(df)
        secs = live_sections(df)
        dur = np.array([(s["Time"].iloc[-1] - s["Time"].iloc[0]).total_seconds() / 60 for s in secs])
        usable = [s for s, d in zip(secs, dur) if d >= WIN_MIN]
        n_win_overlap = int(sum(int((d - WIN_MIN) // STEP_MIN) + 1 for d in dur if d >= WIN_MIN))
        inventory.append({"well": well, "live_sections": len(secs), "live_minutes": round(float(dur.sum()), 1),
                          "sections_ge_5min": len(usable), "minutes_in_those": round(float(dur[dur >= WIN_MIN].sum()), 1),
                          "longest_section_min": round(float(dur.max()), 1), "windows_50pct_overlap": n_win_overlap})
        print(f"\n=== {well} ===")
        print(f"{len(secs)} live sections, {dur.sum():.0f} min of live torque; {len(usable)} sections of at least "
              f"{WIN_MIN} min ({dur[dur >= WIN_MIN].sum():.0f} min) give {n_win_overlap} windows at 50% overlap")

        rows = []
        for si, sec in enumerate(usable):
            w = section_windows(sec, rng, with_null=True)
            for r in w:
                r["section"] = si
            rows.extend(w)
        if not rows:
            continue
        wdf = pd.DataFrame(rows)
        # drilling context of each window: mean WOB, RPM, bit depth over the rows it covers
        ctx = []
        for _, r in wdf.iterrows():
            seg = df[(df["Time"] >= r["start"]) & (df["Time"] < r["start"] + pd.Timedelta(minutes=WIN_MIN)) & df["is_live"]]
            ctx.append({"wob_mean_n": seg["WOB_AVG"].mean(), "rpm_mean": seg["SURF_RPM"].mean() if "SURF_RPM" in seg else np.nan,
                        "bitdep_mean_m": seg["BITDEP"].mean() if "BITDEP" in seg else np.nan,
                        "torque_range_n_m": seg["TORQUE_AVG"].max() - seg["TORQUE_AVG"].min()})
        wdf = pd.concat([wdf, pd.DataFrame(ctx)], axis=1)
        wdf["well"] = well
        spectra[well] = wdf[["start", "section", "ratio_spectrum", "freqs", "p_clean", "p_full", "thr_full", "thr_clean"]].copy()
        wdf = wdf.drop(columns=["ratio_spectrum", "freqs"])
        win_tables.append(wdf)

        n = len(wdf)
        n_indep = max(1, int(sum(int((d - WIN_MIN) // WIN_MIN) + 1 for d in dur if d >= WIN_MIN)))
        for label in ("clean", "full"):
            sig = (wdf[f"p_{label}"] < ALPHA)
            wob_sig = (wdf[f"wob_ratio_{label}"] > wdf[f"thr_{label}"])
            hk_sig = (wdf[f"hookload_ratio_{label}"] > wdf[f"thr_{label}"])
            print(f"  {label} band: {int(sig.sum())} of {n} overlapping windows have a torque peak above the 5% line "
                  f"(about {ALPHA*n_indep:.1f} expected by chance among {n_indep} non-overlapping windows); "
                  f"of those, WOB also exceeds the same line in {int((sig & wob_sig).sum())} and hookload in {int((sig & hk_sig).sum())}")
        # persistence
        for label, band in (("clean 20-120 s", BAND_CLEAN), ("full 5-120 s", BAND_FULL)):
            key = "clean" if "clean" in label else "full"
            if n < 3:
                persist_rows.append({"well": well, "band": label, "n_windows": n, "median_abs_dlnT": np.nan})
                continue
            periods = [g[f"torque_period_{key}_s"].values for _, g in wdf.groupby("section")]
            real = persistence_stat(periods)
            sections_x = []
            for sec in usable:
                grid, x = uniform_series(sec["Time"], sec["TORQUE_AVG"])
                sections_x.append((grid, x))
            nullv = simulated_persistence(sections_x, rng, band, N_SIM_PERSIST)
            p_low = float((1 + np.sum(nullv <= real)) / (len(nullv) + 1))
            persist_null[(well, key)] = (real, nullv)
            persist_rows.append({"well": well, "channel": "torque", "band": label, "n_windows": n, "median_abs_dlnT": real,
                                 "null_median": float(np.median(nullv)), "null_5pct": float(np.percentile(nullv, 5)),
                                 "p_more_persistent_than_noise": p_low})
            # the same statistic for the control channels, with their own simulated nulls
            for ccol, cname in CHANNELS[1:]:
                cper = [g[f"{cname}_period_{key}_s"].values for _, g in wdf.groupby("section")]
                creal = persistence_stat(cper)
                csecs = [uniform_series(sec["Time"], sec[ccol]) for sec in usable]
                cnull = simulated_persistence(csecs, rng, band, N_SIM_PERSIST)
                cp = float((1 + np.sum(cnull <= creal)) / (len(cnull) + 1))
                persist_rows.append({"well": well, "channel": cname, "band": label, "n_windows": n, "median_abs_dlnT": creal,
                                     "null_median": float(np.median(cnull)), "null_5pct": float(np.percentile(cnull, 5)),
                                     "p_more_persistent_than_noise": cp})
                print(f"    control {cname}: {creal:.2f} (noise median {np.median(cnull):.2f}), p = {cp:.3f}; "
                      f"its strongest period has median {wdf[f'{cname}_period_{key}_s'].median():.0f} s")
            print(f"  {label}: median |change in ln period| between consecutive windows = {real:.2f} "
                  f"(noise: median {np.median(nullv):.2f}, 5th percentile {np.percentile(nullv, 5):.2f}); "
                  f"p(as persistent as this under noise) = {p_low:.3f}")

        # dominant period vs drilling context
        sig_clean = wdf[wdf["p_clean"] < ALPHA]
        print(f"  Dominant clean-band period over all windows: median {wdf['torque_period_clean_s'].median():.0f} s, "
              f"IQR {wdf['torque_period_clean_s'].quantile(.25):.0f} to {wdf['torque_period_clean_s'].quantile(.75):.0f} s; "
              f"over the {len(sig_clean)} significant ones: "
              + (f"median {sig_clean['torque_period_clean_s'].median():.0f} s" if len(sig_clean) else "none"))

    inv = pd.DataFrame(inventory)
    inv.to_csv("data/spectrogram_live_inventory.csv", index=False)
    print("\nLive torque available for a spectrogram:")
    print(inv.to_string(index=False))

    wins = pd.concat(win_tables, ignore_index=True)
    wins.to_csv("data/spectrogram_windows.csv", index=False)
    pers = pd.DataFrame(persist_rows)
    pers.to_csv("data/spectrogram_persistence.csv", index=False)

    # context correlations for 9-F-4, the only well with enough windows
    w4 = wins[wins["well"] == "9-F-4"].copy()
    ctx_rows = []
    for label in ("clean", "full"):
        for var in ("bitdep_mean_m", "wob_mean_n", "rpm_mean"):
            ok = w4[["torque_period_%s_s" % label, var]].dropna()
            if len(ok) >= 10 and ok[var].nunique() > 3:
                rho, p = spearmanr(ok["torque_period_%s_s" % label], ok[var])
                ctx_rows.append({"well": "9-F-4", "band": label, "context_variable": var, "n_windows": len(ok),
                                 "spearman_period_vs_variable": rho, "p_value_naive": p})
        for var in ("bitdep_mean_m", "wob_mean_n", "rpm_mean"):
            ok = w4[["torque_ratio_%s" % label, var]].dropna()
            if len(ok) >= 10 and ok[var].nunique() > 3:
                rho, p = spearmanr(ok["torque_ratio_%s" % label], ok[var])
                ctx_rows.append({"well": "9-F-4", "band": label, "context_variable": var + " (peak strength)", "n_windows": len(ok),
                                 "spearman_period_vs_variable": rho, "p_value_naive": p})
    ctx_df = pd.DataFrame(ctx_rows)
    ctx_df.to_csv("data/spectrogram_context_correlations.csv", index=False)
    print("\nDominant period and peak strength against drilling context, 9-F-4 (overlapping windows, so p-values are not valid; descriptive):")
    print(ctx_df.round(3).to_string(index=False))

    # ------------------------------------------------------------------------------------------ chart
    s4 = spectra["9-F-4"]
    fr = s4["freqs"].iloc[0]
    per = 1 / fr
    band_sel = (per >= BAND_FULL[0]) & (per <= BAND_FULL[1])
    img = np.vstack([r[band_sel] for r in s4["ratio_spectrum"]]).T
    pers_axis = per[band_sel]
    order = np.argsort(pers_axis)

    fig = plt.figure(figsize=(15, 10.6), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, width_ratios=[2.1, 1], height_ratios=[1.35, 1, 0.9], hspace=0.5, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    nwin = img.shape[1]
    xs = np.arange(nwin)
    mesh = ax.pcolormesh(np.arange(nwin), pers_axis[order], np.log10(np.maximum(img[order], 1e-3)),
                         cmap="viridis", shading="nearest", vmin=-0.5, vmax=1.2)
    ax.set_yscale("log")
    ax.set_ylim(BAND_FULL[0], BAND_FULL[1])
    sec = s4["section"].values
    for i in range(1, nwin):
        if sec[i] != sec[i - 1]:
            ax.axvline(i - 0.5, color="white", lw=1.6)
    dom = wins[wins["well"] == "9-F-4"]["torque_period_clean_s"].values
    sig = (wins[wins["well"] == "9-F-4"]["p_clean"] < ALPHA).values
    ax.plot(xs[~sig], dom[~sig], "o", mfc="none", mec="white", ms=5, label="strongest peak, 20-120 s (not significant)")
    ax.plot(xs[sig], dom[sig], "o", color=ORANGE, mec="white", ms=7, label="strongest peak, 20-120 s (p < 0.05)")
    ax.axhline(CADENCE_S, color="white", lw=1, ls=":")
    ax.text(-0.4, CADENCE_S * 0.93, "14 s sampling cycle", color="white", fontsize=8, ha="left", va="top")
    ax.set_xlabel("Window number in time order (white lines: gaps between live sections)", fontsize=9.5, color=SECONDARY_INK)
    ax.set_ylabel("Period (s)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("A. 9-F-4: torque power relative to red-noise background, 5-minute windows",
                 fontsize=11, color=INK, loc="left", fontweight="bold")
    cb = fig.colorbar(mesh, ax=ax, pad=0.01)
    cb.set_label("log10(power / background)", fontsize=9, color=SECONDARY_INK)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    ax.legend(fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=BASELINE, loc="upper right", framealpha=0.9)

    ax = fig.add_subplot(gs[0, 1])
    for lab, band_key, col in (("20-120 s", "clean", BLUE), ("5-120 s", "full", ORANGE)):
        ax.plot(xs, wins[wins["well"] == "9-F-4"][f"torque_period_{band_key}_s"].values, "o-", color=col, ms=4, lw=0.9, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Window number", fontsize=9.5, color=SECONDARY_INK)
    ax.set_ylabel("Strongest period (s)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("B. Strongest period by window", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 0])
    for j, (key, col, lab) in enumerate((("clean", BLUE, "20-120 s"), ("full", ORANGE, "5-120 s"))):
        if ("9-F-4", key) not in persist_null:
            continue
        real, nullv = persist_null[("9-F-4", key)]
        ax.hist(nullv, bins=30, color=col, alpha=0.45, label=f"noise, {lab}")
        ax.axvline(real, color=col, lw=2.2, label=f"9-F-4 torque, {lab}: {real:.2f}")
    ax.set_xlabel("Median |change in ln(period)| between consecutive windows (lower = more persistent)", fontsize=9, color=SECONDARY_INK)
    ax.set_ylabel("Simulated series", fontsize=10, color=SECONDARY_INK)
    ax.set_title("C. Does the dominant period persist more than noise would?", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    cnt = pd.DataFrame({"well": inv["well"], "min": inv["live_minutes"], "min5": inv["minutes_in_those"]})
    xw = np.arange(len(cnt))
    ax.bar(xw - 0.2, cnt["min"], 0.4, color=BLUE, label="all live torque")
    ax.bar(xw + 0.2, cnt["min5"], 0.4, color=AQUA, label="in sections of 5 min or more")
    for x_, a, b in zip(xw, cnt["min"], cnt["min5"]):
        ax.text(x_ - 0.2, a + 3, f"{a:.0f}", ha="center", fontsize=8.5, color=INK)
        ax.text(x_ + 0.2, b + 3, f"{b:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(xw)
    ax.set_xticklabels(cnt["well"])
    ax.set_ylabel("Minutes", fontsize=10, color=SECONDARY_INK)
    ax.set_title("D. Live torque available per well", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[2, :])
    cols = {"9-F-9": ORANGE, "9-F-7": AQUA}
    for w_, col in cols.items():
        if w_ in spectra:
            for _, r in spectra[w_].iterrows():
                ax.plot(1 / r["freqs"], r["ratio_spectrum"], color=col, lw=1.2, alpha=0.85)
            ax.plot([], [], color=col, label=f"{w_}: {len(spectra[w_])} window(s)")
            thr = spectra[w_]["thr_full"].iloc[0]
            ax.axhline(thr, color=col, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(5, 120)
    ax.axvspan(10, 17, color=NEUTRAL, alpha=0.9, zorder=0)
    ax.set_xlabel("Period (s)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Power / background", fontsize=10, color=SECONDARY_INK)
    ax.set_title("E. 9-F-9 and 9-F-7: too little live torque for a spectrogram (dashed: 5% line)",
                 fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    for a in fig.axes:
        if a.get_label() == "<colorbar>":
            continue
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)
        a.spines["left"].set_color(BASELINE)
        a.spines["bottom"].set_color(BASELINE)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.savefig("data/chart_spectrogram.png", dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_spectrogram.png")


if __name__ == "__main__":
    main()
