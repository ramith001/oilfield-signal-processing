"""
Does well 9-F-4's ~2911 s "stick-slip period" (multi_well_validation.py, Part 1) hold up, or was it an artifact
of the single section it came from?

Part 1 took the one drilling block with the most live rows (a 145-minute stretch, half of whose rows are
frozen/stale torque) and FFT'd the whole thing, reporting the largest peak above a minimum frequency of
2 / (segment length). That period, 2911 s, is flagged there as outside the 5-120 s stick-slip band and "likely
slower drilling-state variation". This script tests that explanation directly rather than assuming it.

"More sections" turned out not to mean more data: the mirror holds no further torque/WOB sections for 9-F-4
(see extract_f4_extra.py for the scan of every log object), so the test uses what exists, much more
thoroughly, with the same cleaning and live-row definition as Part 1 (imported from multi_well_validation.py):

  0. The sampling cadence. Rows are not 2 s apart: in every well the spacing repeats 3,3,3,3,2 s (a 14 s cycle).
     Part 1 interpolates this onto a uniform 2 s grid, and any feature near 14 s (and its 7 s harmonic) has to
     be treated with suspicion.
  A. Diagnose the original peak: where does it sit relative to the allowed search range, and how much of the
     block's power is at slow periods?
  B. Window-length scaling: if 2911 s were a physical oscillation, shorter windows cut from the same block
     should keep finding it. If it is slow drift landing on the lowest allowed bins, the unrestricted
     "dominant period" should track the window length instead.
  C. Every contiguous live section, in short non-overlapping windows (5-minute primary, 3-minute sensitivity):
     is any in-band peak larger than a smooth red-noise background explains, in single windows and pooled across
     windows (a stable period should recur at the same frequency)? The same windows are run for WOB and
     hookload as controls, since a torsional stick-slip peak should be far stronger in torque than in them.
     The in-band search is run over the full 5-120 s band and over a 20-120 s "clean" band that excludes the
     14 s cadence and its harmonics.
  D. The single longest contiguous live section (73 minutes), pooled across its own windows.
  E. Part 1's own segments for all three wells, tested the same way, to say what the result implies for the
     23.8 s (9-F-9) and 40.9 s (9-F-7) periods as well.

The test. Each window is prewhitened (first difference) so spectral leakage from the steeply red spectrum does
not swamp the band, a robust power law is fitted to its own periodogram as the "no peak" background, and the
statistic is the largest ratio of observed power to that background across the band. P-values come from a
parametric bootstrap: simulated Gaussian noise with that window's own fitted spectral slope is pushed through
the identical whitening, background fit and statistic. That makes the p-values self-calibrating (textbook
exponential/gamma tails were tried first and were badly liberal on simulated noise), and calibrate_test()
re-checks the false-positive rate and the detection power on known noise and an injected sinusoid before any
real data is trusted. Pooled p-values combine windows that share a frequency grid, so a stable period (one
that recurs at the same frequency) counts for more than a one-off peak. The simulated noise is sampled
uniformly, so it does not contain the real 14 s sampling cadence; that is deliberate, and the WOB and hookload
controls are how a shared sampling artifact is told apart from torque physics.

    python f4_period_stability.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_well_validation import clean_and_flag_live, fft_longest_live_block

# dataviz skill palette (references/palette.md), as in the other scripts
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

DT = 2.0                      # s, the uniform grid Part 1 interpolates onto
BAND_FULL = (5.0, 120.0)      # s, the stick-slip band used throughout Part 1
BAND_CLEAN = (20.0, 120.0)    # s, excludes the 14 s sampling cadence, its 7 s harmonic, and the 10-17 s region around it
CADENCE_S = 14.0              # s, the 3,3,3,3,2 row-spacing cycle
WINDOWS_MIN = (5, 3)          # minutes; the first is the primary analysis, the second a sensitivity check
PRIMARY_WINDOW_MIN = WINDOWS_MIN[0]
ALPHA = 0.05
EULER = 0.5772156649
N_SIM = 300                   # bootstrap simulations per window
SEED = 0
WELLS = [("9-F-4", "data/f4_drilling_telemetry.csv", BLUE),
         ("9-F-9", "data/f9_drilling_telemetry.csv", ORANGE),
         ("9-F-7", "data/f7_drilling_telemetry.csv", AQUA)]
CHANNELS = [("TORQUE_AVG", "Torque", BLUE), ("WOB_AVG", "WOB (control)", ORANGE), ("HKLD_AVG", "Hookload (control)", AQUA)]


# ------------------------------------------------------------------------------------ signal handling
def detrend(x, t):
    return x - np.polyval(np.polyfit(t, x, 1), t)


def periodogram(x):
    """Hann-windowed periodogram of a uniform series; returns (freqs, power), DC excluded."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=DT)
    power = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    return freqs[1:], power[1:]


def dominant_period(freqs, power, min_bin_freq=None):
    """Dominant period above Part 1's minimum frequency (used to reproduce Part 1's search rule)."""
    mask = np.ones_like(freqs, dtype=bool)
    if min_bin_freq is not None:
        mask &= freqs > min_bin_freq
    return 1 / freqs[np.argmax(np.where(mask, power, -np.inf))]


def uniform_series(seg_times, seg_values):
    """Part 1's preprocessing: interpolate onto a uniform 2 s grid on the true timestamps."""
    t = (seg_times - seg_times.iloc[0]).dt.total_seconds().values
    v = seg_values.values
    ok = ~np.isnan(v)
    t, v = t[ok], v[ok]
    if len(t) < 30:
        return None, None
    grid = np.arange(t[0], t[-1], DT)
    if len(grid) < 16:
        return None, None
    return grid, np.interp(grid, t, v)


def live_sections(df):
    """Maximal runs of consecutive live rows (same drilling block, no gap > 10 s), as DataFrames."""
    d = df[df["is_live"]].copy()
    gap = d["Time"].diff().dt.total_seconds().fillna(0) > 10
    new_block = d["block"] != d["block"].shift()
    d["section"] = (gap | new_block).cumsum()
    return [g for _, g in d.groupby("section")]


def part1_segment_series(df, blocks, col="TORQUE_AVG"):
    """The exact segment multi_well_validation.fft_longest_live_block analyses: the drilling block with the most
    live rows, interpolated onto the 2 s grid. (Rebuilt here only because that function returns the spectrum,
    not the series; main() asserts the dominant period matches Part 1's.)"""
    live_counts = df[df["is_live"]].groupby("block").size().sort_values(ascending=False)
    longest = blocks.loc[live_counts.index[0]]
    seg = df[(df["Time"] >= longest["start"]) & (df["Time"] <= longest["end"])]
    return uniform_series(seg["Time"], seg[col])


# ------------------------------------------------------------------------------- the peak-vs-background test
def band_mask(fr, band):
    return (fr >= 1 / band[1]) & (fr <= 1 / band[0])


def whitened_periodogram(x):
    """Periodogram of the first difference of a detrended series (prewhitening: flattens a steeply red
    spectrum so leakage from the slowest bins does not swamp the band; a peak is multiplied by the same filter
    gain as the background, so its ratio to the background is unchanged)."""
    return periodogram(np.diff(x))


def fit_background(fr, pw, t_win):
    """Robust power-law background for one periodogram: log P regressed on log f over periods from t_win/3 down
    to 6 s, bins more than 4x the (bias-corrected) fit trimmed and the fit repeated so peaks do not drag it up,
    shifted by Euler's constant because the mean of log of an exponential variate is log(mean) - gamma.
    Returns (background at every frequency, fitted log-log slope)."""
    fit = (fr >= 3 / t_win) & (fr <= 1 / 6.0)
    lf, lp = np.log(fr[fit]), np.log(np.maximum(pw[fit], 1e-300))
    keep = np.ones_like(lf, dtype=bool)
    for _ in range(4):
        b = np.polyfit(lf[keep], lp[keep], 1)
        keep = lp < np.polyval(b, lf) + EULER + np.log(4.0)
    return np.exp(np.polyval(b, np.log(fr)) + EULER), float(b[0])


def gaussian_noise(n, slope, rng):
    """Gaussian noise on the 2 s grid with a power-law power spectrum f^slope (random amplitudes and phases)."""
    f = np.fft.rfftfreq(n, d=DT)[1:]
    spec = np.zeros(n // 2 + 1, dtype=complex)
    spec[1:] = f ** (slope / 2) * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    return np.fft.irfft(spec, n=n)


def ratios_with_null(x, rng, n_sim=N_SIM):
    """Observed ratio-to-background spectrum for a detrended window x, plus n_sim null spectra from simulated
    noise with the window's own fitted slope, each passed through the identical whitening and background fit.
    Returns (freqs, observed ratios, null ratios (n_sim x bins))."""
    n = len(x)
    t = np.arange(n) * DT
    t_win = (n - 1) * DT
    fr, pw = whitened_periodogram(x)
    bg, slope_w = fit_background(fr, pw, t_win)
    slope = float(np.clip(slope_w - 2.0, -4.0, 0.0))   # undo the +2 added by differencing
    null = np.empty((n_sim, len(fr)))
    for i in range(n_sim):
        _, pws = whitened_periodogram(detrend(gaussian_noise(n, slope, rng), t))
        bgs, _ = fit_background(fr, pws, t_win)
        null[i] = pws / bgs
    return fr, pw / bg, null


def single_test(r_obs, null, mask):
    """Largest observed/background ratio in the band vs. the same statistic on the simulated nulls."""
    stat = r_obs[mask].max()
    null_stats = null[:, mask].max(axis=1)
    return {"p": float((1 + np.sum(null_stats >= stat)) / (len(null_stats) + 1)), "stat": float(stat),
            "peak_index": int(np.argmax(r_obs[mask])), "threshold": float(np.percentile(null_stats, 100 * (1 - ALPHA)))}


def pooled_test(r_obs_list, null_list, mask):
    """Combine windows that share a frequency grid: the mean ratio across windows at each frequency, then its
    maximum. A stable period recurs at the same frequency, so it survives the averaging; one-off peaks do not."""
    mean_obs = np.mean([r[mask] for r in r_obs_list], axis=0)
    null_stats = np.mean([nl[:, mask] for nl in null_list], axis=0).max(axis=1)
    stat = mean_obs.max()
    return {"p": float((1 + np.sum(null_stats >= stat)) / (len(null_stats) + 1)), "stat": float(stat),
            "peak_index": int(np.argmax(mean_obs)), "mean_ratio": mean_obs,
            "threshold": float(np.percentile(null_stats, 100 * (1 - ALPHA)))}


def calibrate_test(rng, n_windows=120, n_sim=200, slope=-2.0, win_min=PRIMARY_WINDOW_MIN):
    """Does the test do what it claims? False-positive rate on simulated noise (should be about 5%) and
    detection rate for an injected 40 s sinusoid, for single windows."""
    L = int(win_min * 60 / DT)
    t = np.arange(L) * DT
    out = {}
    for label, amp in (("noise only", 0.0), ("40 s sinusoid, amplitude = 0.5 x noise sd", 0.5)):
        hits = {"full": 0, "clean": 0}
        for _ in range(n_windows):
            x = gaussian_noise(L, slope, rng)
            x = x / x.std() + amp * np.sin(2 * np.pi * t / 40.0 + rng.random() * 6.28)
            fr, r_obs, null = ratios_with_null(detrend(x, t), rng, n_sim=n_sim)
            for name, band in (("full", BAND_FULL), ("clean", BAND_CLEAN)):
                hits[name] += single_test(r_obs, null, band_mask(fr, band))["p"] < ALPHA
        out[label] = {f"single_{k}": v / n_windows for k, v in hits.items()}
    return out


def window_tests(secs, col, win_min, rng):
    """Non-overlapping windows of every live section: per-window tests (both bands), plus the observed/null
    ratio spectra kept for pooling. Windows share one frequency grid because they share one length."""
    L = int(win_min * 60 / DT)
    rows, r_obs_list, null_list, fr = [], [], [], None
    for si, sec in enumerate(secs):
        grid, x = uniform_series(sec["Time"], sec[col])
        if x is None:
            continue
        for s in range(0, len(x) - L + 1, L):
            xs = detrend(x[s:s + L], grid[s:s + L])
            if np.ptp(np.diff(xs)) == 0:
                continue
            fr, r_obs, null = ratios_with_null(xs, rng)
            row = {"section": si, "window_start": sec["Time"].iloc[0] + pd.Timedelta(seconds=float(grid[s]))}
            for name, band in (("full", BAND_FULL), ("clean", BAND_CLEAN)):
                mask = band_mask(fr, band)
                tst = single_test(r_obs, null, mask)
                row.update({f"p_{name}": tst["p"], f"stat_{name}": tst["stat"],
                            f"peak_period_{name}_s": float(1 / fr[mask][tst["peak_index"]])})
            rows.append(row)
            r_obs_list.append(r_obs)
            null_list.append(null)
    return pd.DataFrame(rows), r_obs_list, null_list, fr


def main():
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------------ 0. the sampling cadence
    print("=== 0. Sampling cadence in the live rows ===")
    wells = {}
    for well, path, _ in WELLS:
        df = pd.read_csv(path, parse_dates=["Time"])
        df, d_all, blocks, _, _ = clean_and_flag_live(df)
        wells[well] = (df, blocks)
        secs = live_sections(df)
        dts = np.concatenate([sec["Time"].diff().dropna().dt.total_seconds().values for sec in secs])
        share_32 = np.mean(np.isin(dts, (2.0, 3.0)))
        longest = max(secs, key=len)
        t = (longest["Time"] - longest["Time"].iloc[0]).dt.total_seconds().values
        k = np.arange(len(t))
        resid = t - np.polyval(np.polyfit(k, t, 1), k)
        fr_j = np.fft.rfftfreq(len(resid), d=1.0)[1:]
        line_rows = 1 / fr_j[np.argmax(np.abs(np.fft.rfft(resid * np.hanning(len(resid))))[1:])]
        print(f"  {well}: {share_32*100:.0f}% of live-row gaps are 2 s or 3 s (mean {dts.mean():.2f} s); strongest line in the "
              f"timestamp jitter has a period of {line_rows:.1f} rows = {line_rows*dts.mean():.1f} s")
    print(f"  => the sampling cycle is {CADENCE_S:.0f} s (3,3,3,3,2). Nyquist for the true average spacing is about "
          f"{2*2.8:.1f} s, so the bottom of the 5-120 s band is at the sampling limit.")

    # ----------------------------------------------------------------- A. diagnose the 2911 s peak
    df4, blocks4 = wells["9-F-4"]
    orig = fft_longest_live_block(df4, blocks4)
    n, T = orig["n_samples"], orig["n_samples"] * DT
    freqs, spec = orig["freqs"][1:], orig["spectrum"][1:]
    live_counts = df4[df4["is_live"]].groupby("block").size().sort_values(ascending=False)
    best_block = live_counts.index[0]
    block_rows, block_live = int(blocks4.loc[best_block, "n"]), int(live_counts.iloc[0])
    slow_share = spec[1 / freqs > 300].sum() / spec.sum()
    print("\n=== A. The original 2911 s peak ===")
    print(f"Segment: {T/60:.1f} min ({n} samples on the 2 s grid); {block_live} of the block's {block_rows} rows "
          f"are live ({block_live/block_rows*100:.0f}%), the rest are frozen/stale torque filled by interpolation")
    print(f"Reported dominant period: {orig['period_s']:.1f} s = segment length / {T/orig['period_s']:.2f}, "
          f"i.e. FFT bin {int(round(T/orig['period_s']))} of the segment")
    print("Search rule (Part 1): freqs > 2/T, which excludes bins 1 and 2, so bin 3 is the lowest allowed bin.")
    print("Top 5 bins in the full spectrum (bin, period s, share of total power):")
    for i in np.argsort(spec)[::-1][:5]:
        print(f"   bin {i+1}: {1/freqs[i]:8.0f} s   {spec[i]/spec.sum()*100:5.1f}%")
    print(f"Share of spectral power at periods above 300 s: {slow_share*100:.0f}%")

    # ------------------------------------------------ B. does the period scale with the window length?
    seg = df4[(df4["Time"] >= blocks4.loc[best_block, "start"]) & (df4["Time"] <= blocks4.loc[best_block, "end"])]
    grid, x_full = uniform_series(seg["Time"], seg["TORQUE_AVG"])
    print("\n=== B. Unrestricted dominant period vs. window length (same block, Part 1's search rule) ===")
    b_rows = []
    for win_min in (10, 20, 36, 72, 145):
        L = int(win_min * 60 / DT)
        periods = []
        for s in range(0, len(x_full) - L + 1, L):
            fr, pw = periodogram(detrend(x_full[s:s + L], grid[s:s + L]))
            periods.append(dominant_period(fr, pw, min_bin_freq=2 / (L * DT)))
        bins = [L * DT / p for p in periods]
        b_rows.append({"window_min": win_min, "n_windows": len(periods), "median_period_s": np.median(periods),
                       "median_bin": np.median(bins), "max_bin": np.max(bins)})
        print(f"  {win_min:4d}-min windows (n={len(periods):2d}): median dominant period {np.median(periods):7.0f} s, "
              f"= bin {np.median(bins):.1f} (max {np.max(bins):.1f}) of the window; the lowest bin Part 1's rule allows is 3")
    bdf = pd.DataFrame(b_rows)
    bdf.to_csv("data/f4_period_stability_window_scaling.csv", index=False)

    # ------------------------------------------------------------------------- calibrate the test
    print("\n=== Calibration of the peak test on simulated noise (single windows, 5 min, slope -2) ===")
    cal = calibrate_test(rng)
    cal_rows = []
    for label, vals in cal.items():
        print(f"  {label}: " + ", ".join(f"{k} {v*100:.0f}%" for k, v in vals.items()) + " of windows flagged at p < 0.05")
        cal_rows.append({"case": label, **vals})
    pd.DataFrame(cal_rows).to_csv("data/f4_period_stability_calibration.csv", index=False)

    # -------------------------- C. every live section, short windows, single and pooled, torque + controls
    print("\n=== C. Contiguous live sections, short windows ===")
    win_rows, pool_rows, store = [], [], {}
    for well, _, _ in WELLS:
        df, _ = wells[well]
        secs = live_sections(df)
        dur = np.array([(s["Time"].iloc[-1] - s["Time"].iloc[0]).total_seconds() / 60 for s in secs])
        print(f"  {well}: {len(secs)} contiguous live sections, {df['is_live'].sum()*2/60:.0f} min of live torque in total; "
              f"longest section {dur.max():.1f} min, {int((dur >= 5).sum())} sections >= 5 min")
        for col, cname, _ in CHANNELS:
            for win_min in WINDOWS_MIN:
                wdf, r_list, n_list, fr_w = window_tests(secs, col, win_min, rng)
                if wdf.empty:
                    continue
                wdf.insert(0, "window_min", win_min)
                wdf.insert(0, "channel", col)
                wdf.insert(0, "well", well)
                win_rows.append(wdf)
                for name, band, label in (("full", BAND_FULL, "full 5-120 s"), ("clean", BAND_CLEAN, "clean 20-120 s")):
                    mask = band_mask(fr_w, band)
                    pt = pooled_test(r_list, n_list, mask)
                    pool_rows.append({"well": well, "channel": col, "window_min": win_min, "band": label,
                                      "n_windows": len(wdf), "n_single_significant": int((wdf[f"p_{name}"] < ALPHA).sum()),
                                      "expected_by_chance": round(ALPHA * len(wdf), 1),
                                      "pooled_peak_period_s": float(1 / fr_w[mask][pt["peak_index"]]),
                                      "pooled_stat": pt["stat"], "pooled_threshold": pt["threshold"], "pooled_p": pt["p"]})
                    if well == "9-F-4" and win_min == PRIMARY_WINDOW_MIN:
                        store[(col, label)] = (fr_w[mask], pt)
    wins = pd.concat(win_rows, ignore_index=True)
    wins.to_csv("data/f4_period_stability_windows.csv", index=False)
    pools = pd.DataFrame(pool_rows)
    pools.to_csv("data/f4_period_stability_summary.csv", index=False)
    print("\nSingle-window and pooled results (5-minute windows):")
    show = pools[pools["window_min"] == PRIMARY_WINDOW_MIN].drop(columns=["window_min"])
    print(show.round(3).to_string(index=False))
    print("\n  (3-minute sensitivity check is in data/f4_period_stability_summary.csv)")

    # ------------------------------------- D. the longest contiguous live section, pooled over its windows
    secs4 = live_sections(df4)
    longest = max(secs4, key=lambda s: (s["Time"].iloc[-1] - s["Time"].iloc[0]).total_seconds())
    dur_l = (longest["Time"].iloc[-1] - longest["Time"].iloc[0]).total_seconds() / 60
    wdf_l, r_l, n_l, fr_l = window_tests([longest], "TORQUE_AVG", PRIMARY_WINDOW_MIN, rng)
    d_rows = []
    for name, band, label in (("full", BAND_FULL, "full 5-120 s"), ("clean", BAND_CLEAN, "clean 20-120 s")):
        mask = band_mask(fr_l, band)
        pt = pooled_test(r_l, n_l, mask)
        d_rows.append({"band": label, "n_windows": len(r_l), "n_single_significant": int((wdf_l[f"p_{name}"] < ALPHA).sum()),
                       "pooled_peak_period_s": float(1 / fr_l[mask][pt["peak_index"]]), "pooled_stat": pt["stat"],
                       "pooled_threshold": pt["threshold"], "pooled_p": pt["p"]})
    d_df = pd.DataFrame(d_rows)
    d_df.to_csv("data/f4_period_stability_longest_section.csv", index=False)
    print(f"\n=== D. Longest contiguous live section of 9-F-4 ({dur_l:.1f} min), pooled over its "
          f"{len(r_l)} five-minute windows ===")
    print(d_df.round(3).to_string(index=False))

    # ------------------- E. do Part 1's own dominant periods stand out against a red-noise background?
    print("\n=== E. Part 1's own segments, tested the same way ===")
    e_rows, e_spectra = [], {}
    for well, _, _ in WELLS:
        df, blocks = wells[well]
        part1 = fft_longest_live_block(df, blocks)
        grid_w, x_w = part1_segment_series(df, blocks)
        x_w = detrend(x_w, grid_w)
        fr0, pw0 = periodogram(x_w)
        assert abs(dominant_period(fr0, pw0, min_bin_freq=2 / (len(x_w) * DT)) - part1["period_s"]) < 1e-6, "segment mismatch"
        fr, r_obs, null = ratios_with_null(x_w, rng)
        near = np.argmin(np.abs(1 / fr - part1["period_s"]))
        row = {"well": well, "segment_min": len(x_w) * DT / 60, "part1_period_s": part1["period_s"],
               "part1_ratio_to_background": float(r_obs[near])}
        for name, band in (("full", BAND_FULL), ("clean", BAND_CLEAN)):
            mask = band_mask(fr, band)
            tst = single_test(r_obs, null, mask)
            row.update({f"{name}_peak_period_s": float(1 / fr[mask][tst["peak_index"]]), f"{name}_stat": tst["stat"],
                        f"{name}_threshold": tst["threshold"], f"{name}_p": tst["p"]})
        e_rows.append(row)
        e_spectra[well] = (fr, r_obs, row["full_threshold"])
        print(f"  {well}: segment {row['segment_min']:.1f} min, Part 1 period {part1['period_s']:.1f} s "
              f"({row['part1_ratio_to_background']:.1f}x its background); strongest peak over 5-120 s at "
              f"{row['full_peak_period_s']:.1f} s ({row['full_stat']:.1f}x, p = {row['full_p']:.3f}), over 20-120 s at "
              f"{row['clean_peak_period_s']:.1f} s ({row['clean_stat']:.1f}x, p = {row['clean_p']:.3f})")
    pd.DataFrame(e_rows).to_csv("data/f4_period_stability_part1_segments.csv", index=False)

    # ------------------------------------------------------------------------------------- the verdict
    p4 = pools[(pools["well"] == "9-F-4") & (pools["window_min"] == PRIMARY_WINDOW_MIN)]
    print("\n=== Verdict inputs for 9-F-4 ===")
    print(f"- The original 2911 s peak is bin {T/orig['period_s']:.0f} of its segment, the lowest bin Part 1's search allowed; "
          f"{slow_share*100:.0f}% of the segment's power is slower than 300 s, so it was set by the segment, not by physics.")
    for _, r in p4.iterrows():
        print(f"- {r['channel']}, {r['band']}: {int(r['n_single_significant'])} of {int(r['n_windows'])} windows significant "
              f"(about {r['expected_by_chance']:.1f} by chance); pooled peak at {r['pooled_peak_period_s']:.1f} s, "
              f"pooled p = {r['pooled_p']:.3f}")

    # ---------------------------------------------------------------------------------------- chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.8), facecolor=SURFACE)

    ax = axes[0, 0]
    ax.loglog(1 / freqs, spec / spec.sum(), color=BLUE, lw=1)
    ax.axvspan(BAND_FULL[0], BAND_FULL[1], color=NEUTRAL, alpha=0.9, zorder=0, label="5-120 s stick-slip band")
    ax.axvline(orig["period_s"], color=ORANGE, lw=1.5, ls="--", label=f"reported peak, {orig['period_s']:.0f} s (bin 3)")
    ax.invert_xaxis()
    ax.set_xlabel("Period (s)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Share of total power", fontsize=10, color=SECONDARY_INK)
    ax.set_title(f"A. Spectrum of the original {T/60:.1f}-min block:\n{slow_share*100:.0f}% of power is slower than 300 s",
                 fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 1]
    ax.plot(bdf["window_min"], bdf["median_period_s"] / 60, "o-", color=BLUE, lw=1.8, ms=7, label="median dominant period found")
    ax.plot(bdf["window_min"], bdf["window_min"] / 3, "--", color=ORANGE, lw=1.4, label="window length / 3 (lowest allowed bin)")
    ax.axhline(orig["period_s"] / 60, color=MUTED, lw=1, ls=":", label=f"original: {orig['period_s']/60:.0f} min")
    ax.set_xlabel("Window length (min)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Dominant period found (min)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("B. The 'period' follows the window length,\nnot a fixed physical value", fontsize=11, color=INK,
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 0]
    for col, cname, color in CHANNELS:
        per, pt = store[(col, "full 5-120 s")]
        ax.plot(per ** -1 if False else 1 / per, pt["mean_ratio"], color=color, lw=1.6, label=cname)
    thr = store[("TORQUE_AVG", "full 5-120 s")][1]["threshold"]
    ax.axhline(thr, color=INK, lw=1, ls="--", label="5% threshold (pooled, bootstrap)")
    ax.axvspan(10, 17, color=NEUTRAL, alpha=0.9, zorder=0)
    ax.axvline(CADENCE_S, color=MUTED, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Period (s)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Power / red-noise background\n(mean over windows)", fontsize=10, color=SECONDARY_INK)
    ax.set_title(f"C. 9-F-4, {int(p4.iloc[0]['n_windows'])} live {PRIMARY_WINDOW_MIN}-min windows pooled (shaded: 10-17 s,\n"
                 "around the 14 s sampling cycle)", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="center right")

    ax = axes[1, 1]
    for well, color in (("9-F-9", ORANGE), ("9-F-7", AQUA)):
        fr_e, r_e, thr_e = e_spectra[well]
        m = band_mask(fr_e, BAND_FULL)
        p1 = [r for r in e_rows if r["well"] == well][0]["part1_period_s"]
        ax.plot(1 / fr_e[m], r_e[m], color=color, lw=1.2, label=f"{well} (Part 1: {p1:.1f} s)")
        ax.axhline(thr_e, color=color, lw=0.9, ls="--")
        ax.axvline(p1, color=color, lw=1, ls=":")
    ax.axvspan(10, 17, color=NEUTRAL, alpha=0.9, zorder=0)
    ax.axvline(CADENCE_S, color=MUTED, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Period (s)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Power / red-noise background", fontsize=10, color=SECONDARY_INK)
    ax.set_title("D. Part 1's own segments for the other two wells\n(dashed = each segment's 5% threshold)", fontsize=11, color=INK,
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    for ax in axes.flat:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(BASELINE)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRIDLINE, lw=0.8)
        ax.set_facecolor(SURFACE)

    fig.suptitle("Does 9-F-4's ~2911 s stick-slip period hold up?", fontsize=13, color=INK, x=0.01, ha="left", y=1.02)
    fig.text(0.01, -0.02,
              "Same cleaning and live-row definition as Part 1. No further torque sections exist in the mirror, so the test uses every contiguous "
              "live section of the existing week. Background = robust power law fitted to each window's own periodogram.",
              fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    fig.savefig("data/chart_f4_period_stability.png", dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_f4_period_stability.png and the data/f4_period_stability_*.csv files")


if __name__ == "__main__":
    main()
