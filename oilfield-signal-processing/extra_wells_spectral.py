"""
Part 6, second half: Parts 4 and 5 repeated on wells that actually rotated while drilling.

extra_wells_validation.py shows that the three wells behind Parts 1 to 5 are almost entirely non-rotating. Here the
spectral test (Part 2/4 machinery) and the event detector (Part 5) are run on contiguous runs of rotating drilling
rows (same definition as extra_wells_validation.py) in the wells that have enough of them.

  1. Spectral peaks. Non-overlapping 5-minute windows inside runs of at least 5 minutes; up to 150 windows per
     well, drawn at random with a fixed seed. Each window gets the calibrated peak-versus-red-noise test over
     20-120 s and 5-120 s, with WOB and hookload as controls. The share of windows with a significant torque peak
     is compared with the 5% expected by chance, and for significant windows the strongest period is compared
     with bit depth. A torsional stick-slip period should rise with the length of the string (fundamental period
     about 4 L / c, with c near 3,000 m/s for steel drillstring, 1.3 to 6 s for 1,000 to 4,700 m), and would be
     at or below this data's sampling limit of 5.6 s for most depths.
  2. The Part 5 detector, on every drilling run of at least 120 s, against phase-randomised copies of the same
     runs, and against the 60 s relative-severity windows of extra_wells_validation.py.

    python extra_wells_spectral.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

from f4_period_stability import (DT, BAND_FULL, BAND_CLEAN, ALPHA, detrend, uniform_series, band_mask,
                                  ratios_with_null, single_test)
from stickslip_detector import detect_section, phase_randomise, synthetic_test, severity_class, SOS, EDGE, A_ON, CLASSES
from extra_wells_validation import (load_extra, drilling_rows, severity_windows, BLUE, ORANGE, AQUA, RED, PURPLE, INK,
                                    SECONDARY_INK, MUTED, BASELINE, SURFACE, COLORS)
from scipy.signal import sosfiltfilt

WIN_S = 300
L = int(WIN_S / DT)
MAX_WINDOWS = 150
N_SIM = 200
N_SURR = 20
SEED = 7
WELLS = ["9-F-1C", "9-F-15b", "9-F-12", "9-F-14", "9-F-5", "9-F-15a"]
C_WOB_COL, C_HK_COL = "wob", "hookload"


def runs_of(x, min_s):
    out = []
    for _, g in x.groupby("run"):
        if (g["Time"].iloc[-1] - g["Time"].iloc[0]).total_seconds() >= min_s:
            out.append(g)
    return out


def peak_ratio(xs):
    from f4_period_stability import whitened_periodogram, fit_background
    fr, pw = whitened_periodogram(xs)
    bg, _ = fit_background(fr, pw, (len(xs) - 1) * DT)
    return fr, pw / bg


def main():
    rng = np.random.default_rng(SEED)
    spec_rows, det_rows, depth_rows = [], [], []
    win_store, det_store = {}, {}

    for label in WELLS:
        d = load_extra(label)
        x = drilling_rows(d)
        runs = runs_of(x, WIN_S)
        # candidate 5-minute windows (non-overlapping within each run)
        cands = []
        for ri, g in enumerate(runs):
            t0 = g["Time"].iloc[0]
            grids = {}
            for col in ("torque", C_WOB_COL, C_HK_COL):
                grids[col] = uniform_series(g["Time"], g[col])
            if grids["torque"][1] is None:
                continue
            n = len(grids["torque"][1])
            for s in range(0, n - L + 1, L):
                cands.append((ri, s))
        print(f"\n=== {label} ===")
        print(f"{len(runs)} drilling runs of at least 5 min, {len(cands)} non-overlapping 5-minute windows")
        if len(cands) < 10:
            print("  too few windows for the spectral test")
            continue
        pick = rng.choice(len(cands), size=min(MAX_WINDOWS, len(cands)), replace=False)
        rows = []
        for ci in sorted(pick):
            ri, s = cands[ci]
            g = runs[ri]
            gt, xt = uniform_series(g["Time"], g["torque"])
            xs = detrend(xt[s:s + L], gt[s:s + L])
            if np.ptp(np.diff(xs)) == 0:
                continue
            fr, r_obs, null = ratios_with_null(xs, rng, n_sim=N_SIM)
            row = {"well": label, "run": ri, "start": g["Time"].iloc[0] + pd.Timedelta(seconds=float(gt[s])),
                   "bit_depth": g["bit_depth"].iloc[min(len(g) - 1, int(s * DT / 2.8)):int((s + L) * DT / 2.8) + 1].mean(),
                   "wob": g["wob"].iloc[min(len(g) - 1, int(s * DT / 2.8)):int((s + L) * DT / 2.8) + 1].mean()}
            for name, band in (("full", BAND_FULL), ("clean", BAND_CLEAN)):
                m = band_mask(fr, band)
                t = single_test(r_obs, null, m)
                row[f"p_{name}"] = t["p"]
                row[f"period_{name}_s"] = float(1 / fr[m][t["peak_index"]])
                row[f"stat_{name}"] = t["stat"]
                row[f"thr_{name}"] = t["threshold"]
                for cname, ccol in (("wob", C_WOB_COL), ("hookload", C_HK_COL)):
                    gc, xc = uniform_series(g["Time"], g[ccol])
                    xcs = detrend(xc[s:s + L], gc[s:s + L])
                    if np.ptp(np.diff(xcs)) == 0:
                        row[f"{cname}_exceeds_{name}"] = np.nan
                        continue
                    frc, rc = peak_ratio(xcs)
                    row[f"{cname}_exceeds_{name}"] = float(rc[band_mask(frc, band)].max() > t["threshold"])
            rows.append(row)
        w = pd.DataFrame(rows)
        win_store[label] = w
        for name in ("clean", "full"):
            sig = w[f"p_{name}"] < ALPHA
            both = sig & (w[f"wob_exceeds_{name}"] == 1)
            ctl_rate = float(np.nanmean(w[f"wob_exceeds_{name}"]))
            print(f"  {name}: {int(sig.sum())} of {len(w)} windows significant ({sig.mean()*100:.0f}%, {ALPHA*100:.0f}% expected); "
                  f"median strongest period {w[f'period_{name}_s'].median():.0f} s; WOB exceeds the same line in {ctl_rate*100:.0f}% of all windows; "
                  f"hookload in {np.nanmean(w[f'hookload_exceeds_{name}'])*100:.0f}%")
            spec_rows.append({"well": label, "band": name, "windows": len(w), "significant": int(sig.sum()),
                              "share_significant": float(sig.mean()), "median_strongest_period_s": float(w[f"period_{name}_s"].median()),
                              "wob_exceeds_line_share": ctl_rate, "hookload_exceeds_line_share": float(np.nanmean(w[f"hookload_exceeds_{name}"]))})
            if sig.sum() >= 12 and w.loc[sig, "bit_depth"].nunique() > 3:
                rho, p = spearmanr(w.loc[sig, f"period_{name}_s"], w.loc[sig, "bit_depth"])
                depth_rows.append({"well": label, "band": name, "n_significant": int(sig.sum()), "spearman_period_vs_depth": rho, "p_value": p})
                print(f"    significant windows: period vs bit depth Spearman {rho:+.2f} (p = {p:.3f}, n = {int(sig.sum())})")
            # all windows
            rho, p = spearmanr(w[f"period_{name}_s"], w["bit_depth"])
            depth_rows.append({"well": label, "band": name, "n_significant": len(w), "spearman_period_vs_depth": rho, "p_value": p, "note": "all windows"})

        # ------------------------------------------------------------------------------- detector
        sections = runs_of(x, 120)
        secs_x = []
        for g in sections:
            gt, xt = uniform_series(g["Time"], g["torque"])
            if xt is not None and len(xt) >= 60:
                secs_x.append((g, gt, xt))
        hours = sum(len(xt) * DT for _, _, xt in secs_x) / 3600
        events, A_series = [], []
        for g, gt, xt in secs_x:
            bp, a, ev = detect_section(xt)
            lab = np.zeros(len(xt), dtype=int)
            for e in ev:
                if e["regular"]:
                    lab[e["i0"]:e["i1"]] = e["label"]
                e["well"] = label
                events.append(e)
            A_series.append(pd.DataFrame({"Time": g["Time"].iloc[0] + pd.to_timedelta(gt, unit="s"), "A": a, "label": lab}))
        reg = [e for e in events if e["regular"]]
        sur = []
        for _ in range(N_SURR):
            n_s = 0
            for g, gt, xt in secs_x:
                n_s += sum(1 for e in detect_section(phase_randomise(xt, rng))[2] if e["regular"])
            sur.append(n_s / hours)
        a_all = pd.concat(A_series, ignore_index=True)
        rel_sd = float(np.median([np.std(sosfiltfilt(SOS, xt)[EDGE:-EDGE]) / xt.mean() for _, _, xt in secs_x]))
        cls = {CLASSES[k]: sum(1 for e in reg if e["label"] == k) for k in range(1, 5)}
        # window-level agreement with 60 s relative severity
        sw = severity_windows(x)
        T = a_all["Time"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
        meanA = []
        for t0 in sw["Time"]:
            lo, hi = np.searchsorted(T, t0.value / 1e9), np.searchsorted(T, t0.value / 1e9 + 60.0, side="right")
            meanA.append(np.nanmean(a_all["A"].values[lo:hi]) if hi - lo >= 20 else np.nan)
        sw["detA"] = meanA
        ok = sw.dropna(subset=["detA"])
        rho = spearmanr(ok["detA"], ok["rel_sev"])[0] if len(ok) > 20 else np.nan
        det_rows.append({"well": label, "drilling_hours_on_grid": hours, "candidates": len(events), "regular_events": len(reg),
                         "events_per_hour": len(reg) / hours, "surrogate_events_per_hour_median": float(np.median(sur)),
                         "surrogate_5pct": float(np.percentile(sur, 5)), "surrogate_95pct": float(np.percentile(sur, 95)),
                         "share_time_in_regular_event": float((a_all["label"] > 0).mean()),
                         "labels": str(cls), "median_period_s": float(np.median([e["period_s"] for e in reg])) if reg else np.nan,
                         "band_passed_sd_over_mean": rel_sd, "windows_matched": len(ok), "spearman_detA_vs_rel_severity": rho})
        print(f"  detector: {len(reg)} regular events in {hours:.1f} drilling hours ({len(reg)/hours:.1f} per hour) vs. surrogates "
              f"{np.median(sur):.1f} per hour ({np.percentile(sur, 5):.1f} to {np.percentile(sur, 95):.1f}); labels {cls}; "
              f"band-passed sd / mean torque {rel_sd:.2f}; Spearman of detector A with 60 s relative severity {rho:.2f} (n = {len(ok)})")
        det_store[label] = (a_all, reg, rel_sd)

    pd.DataFrame(spec_rows).to_csv("data/extra_wells_spectral.csv", index=False)
    pd.DataFrame(depth_rows).to_csv("data/extra_wells_period_vs_depth.csv", index=False)
    det = pd.DataFrame(det_rows)
    det.to_csv("data/extra_wells_detector.csv", index=False)
    pd.concat([w for w in win_store.values()], ignore_index=True).drop(columns=[]).to_csv("data/extra_wells_spectral_windows.csv", index=False)

    # synthetic detection thresholds at the noise level seen in these wells
    med_sd = float(det["band_passed_sd_over_mean"].median())
    syn = synthetic_test(med_sd, rng, n_rep=100)
    syn.to_csv("data/extra_wells_detector_synthetic.csv", index=False)
    print(f"\nSynthetic test at the median band-passed noise level of these wells ({med_sd:.2f} of mean torque):")
    print(syn.round(2).to_string(index=False))

    # ------------------------------------------------------------------------------------------ chart
    sp = pd.DataFrame(spec_rows)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.4), facecolor=SURFACE)
    ax = axes[0, 0]
    sub = sp[sp["band"] == "clean"]
    x_ = np.arange(len(sub))
    ax.bar(x_, sub["share_significant"] * 100, 0.5, color=[COLORS[w] for w in sub["well"]])
    ax.axhline(5, color=INK, lw=1, ls="--", label="5% expected by chance")
    ax.set_xticks(x_)
    ax.set_xticklabels(sub["well"], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Windows with a significant torque peak (%)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("A. Spectral peaks, 20-120 s", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 1]
    for label, w in win_store.items():
        sg = w[w["p_clean"] < ALPHA]
        ax.scatter(sg["bit_depth"], sg["period_clean_s"], s=14, alpha=0.7, color=COLORS[label], label=f"{label} (n={len(sg)})")
    dd = np.linspace(0, 4800, 50)
    ax.fill_between(dd, 4 * dd / 3200, 4 * dd / 2800, color=BASELINE, alpha=0.5, label="4 L / c, c = 2,800-3,200 m/s")
    ax.set_yscale("log")
    ax.set_ylim(1, 130)
    ax.set_xlabel("Bit depth (m)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Strongest period, significant windows (s)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("B. Period against depth", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    ax = axes[0, 2]
    x_ = np.arange(len(det))
    ax.bar(x_ - 0.2, det["events_per_hour"], 0.4, color=BLUE, label="real torque")
    ax.bar(x_ + 0.2, det["surrogate_events_per_hour_median"], 0.4, color=MUTED, label="phase-randomised (median)")
    ax.set_xticks(x_)
    ax.set_xticklabels(det["well"], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Regular events per drilling hour", fontsize=10, color=SECONDARY_INK)
    ax.set_title("C. Detector: real vs same-spectrum noise", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 0]
    for label in ("9-F-1C", "9-F-15b"):
        a_all, reg, _ = det_store[label]
        ax.hist(np.clip(a_all["A"].dropna(), 0, 1.5), bins=60, histtype="step", lw=1.6, color=COLORS[label], density=True, label=label)
    ax.axvline(A_ON, color=INK, lw=0.8, ls="--")
    ax.set_xlabel("Relative amplitude A (band-passed 15-60 s)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Density", fontsize=10, color=SECONDARY_INK)
    ax.set_title("D. Distribution of A in drilling", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 1]
    ax.plot(syn["injected_A"], syn["regular_event_rate"], "o-", color=BLUE, lw=1.8, label=f"noise {med_sd:.2f} of mean torque")
    ax.set_xlabel("Injected 30 s oscillation, A", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Share of 5-minute series with an event", fontsize=10, color=SECONDARY_INK)
    ax.set_title("E. Synthetic detection at these wells' noise", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 2]
    ax.bar(np.arange(len(det)), det["spearman_detA_vs_rel_severity"], 0.5, color=[COLORS[w] for w in det["well"]])
    ax.set_xticks(np.arange(len(det)))
    ax.set_xticklabels(det["well"], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Spearman, detector A vs 60 s relative severity", fontsize=9.5, color=SECONDARY_INK)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_title("F. Detector vs severity windows", fontsize=11, color=INK, loc="left", fontweight="bold")
    for a in axes.ravel():
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
        a.spines["left"].set_color(BASELINE)
        a.spines["bottom"].set_color(BASELINE)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig("data/chart_extra_wells_spectral.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_extra_wells_spectral.png")


if __name__ == "__main__":
    main()
