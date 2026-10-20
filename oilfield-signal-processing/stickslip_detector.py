"""
Part 5: an automatic stick-slip event detector with severity labels, checked against the existing severity index.

Detector, per contiguous live section of torque (>= 120 s, no gap over 10 s, same drilling block):
  1. Re-grid to 2 s and band-pass 15-60 s (zero-phase Butterworth). The band is set by the periods seen in this
     repo (23 to 41 s) and stays clear of the 14 s sampling cycle; the 5-120 s theoretical band is not used
     because a 120 s section holds only one cycle of the slow end.
  2. Relative amplitude A(t) = Hilbert envelope of the band-passed torque / 150 s centred mean torque, with the
     mean floored at 500 N.m. For a sinusoid, A is the amplitude as a fraction of mean torque, and A = 1 means
     torque swings to zero. The floor is needed because the live torque is mostly small (median 196 N.m on
     9-F-4, 637 on 9-F-9, 1,290 on 9-F-7, against a few thousand under load), and a ratio to a near-zero mean
     is not meaningful.
  3. A candidate event is a run with A >= 0.15 (hysteresis: ends when A < 0.10) lasting at least 90 s, with
     the first and last 20 s of each section ignored (filter and envelope edge effects).
  4. A candidate is kept as a stick-slip event only if it is regular: at least 3 full cycles between
     successive upward zero crossings of the band-passed signal, with a coefficient of variation of the cycle
     length of at most 0.35. Irregular torque variation that is large but not periodic is counted separately.
  5. Severity label from the median A over the event: low 0.15-0.30, moderate 0.30-0.60, high 0.60-1.00,
     severe 1.00 and over. These cut-offs are fixed conventions set before looking at the validation, not
     tuned to the index.

Checks:
  A. Window by window against the existing severity index (Part 1's index, on the time-contiguous windows
     Part 3 identified): rank correlation, how well detector output separates the index's top fifth (AUC),
     precision and recall of "window overlaps an event", and agreement of severity classes (weighted kappa) with
     the index mapped onto the same scale (range / mean torque / 2 equals A for a sinusoid).
  B. A surrogate null: each live section is replaced by phase-randomised noise with the same spectrum, so any
     narrowband oscillation is destroyed but the spectrum and amplitudes stay. How many events does the detector
     report there? If about as many, the detector is finding large red-noise variation, not stick-slip.
  C. A synthetic test: noise at the level seen in these wells plus an injected 30 s oscillation of known
     amplitude. Detection rate and label accuracy by amplitude, and the false alarm rate with no oscillation.
  D. The same detector on WOB and hookload (controls).

    python stickslip_detector.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import spearmanr, rankdata

from multi_well_validation import WIN, clean_and_flag_live
from f4_period_stability import DT, live_sections, uniform_series, gaussian_noise, detrend
from rop_mse_vs_severity import window_table, MAX_CLEAN_SPAN_S

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
NEUTRAL = "#f0efec"

BAND_S = (15.0, 60.0)
MIN_SECTION_S = 120
MEAN_WIN = 75            # samples, 150 s
MEAN_FLOOR = 500.0       # N.m; the mean torque used to normalise is never taken below this
EDGE = 10                # samples (20 s) ignored at each section end
A_ON, A_OFF = 0.15, 0.10
MIN_EVENT_S = 90
MIN_CYCLES = 3
MAX_CV = 0.35
CLASSES = ["none", "low", "moderate", "high", "severe"]
CLASS_EDGES = [0.15, 0.30, 0.60, 1.00]    # lower edges of low, moderate, high, severe
SOS = butter(2, [1 / BAND_S[1], 1 / BAND_S[0]], btype="bandpass", fs=1 / DT, output="sos")
WELLS = [("9-F-4", "data/f4_drilling_telemetry.csv", AQUA),
         ("9-F-9", "data/f9_drilling_telemetry.csv", BLUE),
         ("9-F-7", "data/f7_drilling_telemetry.csv", ORANGE)]
SEED = 5


def severity_class(a):
    """0 = none, 1 low, 2 moderate, 3 high, 4 severe, from a relative amplitude."""
    return int(np.searchsorted(CLASS_EDGES, a, side="right"))


def relative_amplitude(x):
    bp = sosfiltfilt(SOS, x)
    env = np.abs(hilbert(bp))
    mean = pd.Series(x).rolling(MEAN_WIN, center=True, min_periods=1).mean().values
    a = env / np.maximum(mean, MEAN_FLOOR)
    return bp, a


def detect_section(x, min_cycles=MIN_CYCLES, max_cv=MAX_CV):
    """Events in one uniform-grid section. Returns (bp, A, list of event dicts) with indices on the grid."""
    bp, a = relative_amplitude(x)
    n = len(x)
    valid = np.zeros(n, dtype=bool)
    valid[EDGE:n - EDGE] = True
    a_use = np.where(valid, a, 0.0)
    a_use = np.nan_to_num(a_use)
    events, cands = [], []
    i = 0
    while i < n:
        if a_use[i] >= A_ON:
            j = i
            while j < n and a_use[j] >= A_OFF:
                j += 1
            cands.append((i, j))
            i = j
        else:
            i += 1
    for i, j in cands:
        dur = (j - i) * DT
        if dur < MIN_EVENT_S:
            continue
        seg = bp[i:j]
        up = np.where((seg[:-1] < 0) & (seg[1:] >= 0))[0]
        cyc = np.diff(up) * DT
        regular = len(cyc) >= min_cycles and (np.std(cyc) / np.mean(cyc)) <= max_cv
        amp = float(np.nanmedian(a[i:j]))
        events.append({"i0": i, "i1": j, "duration_s": dur, "n_cycles": int(len(cyc)),
                       "period_s": float(np.median(cyc)) if len(cyc) else np.nan,
                       "cycle_cv": float(np.std(cyc) / np.mean(cyc)) if len(cyc) >= 2 else np.nan,
                       "A_median": amp, "A_peak": float(np.nanmax(a[i:j])), "regular": bool(regular),
                       "label": severity_class(amp) if regular else 0})
    return bp, a, events


def run_well(df, col="TORQUE_AVG", **gate):
    """Detector over every long enough live section of a well. Returns the per-sample table and the events."""
    secs = [s for s in live_sections(df) if (s["Time"].iloc[-1] - s["Time"].iloc[0]).total_seconds() >= MIN_SECTION_S]
    samples, evs = [], []
    for si, sec in enumerate(secs):
        grid, x = uniform_series(sec["Time"], sec[col])
        if x is None or len(x) < 60:
            continue
        bp, a, events = detect_section(x, **gate)
        t0 = sec["Time"].iloc[0]
        times = t0 + pd.to_timedelta(grid, unit="s")
        lab = np.zeros(len(x), dtype=int)
        cand = np.zeros(len(x), dtype=bool)
        for e in events:
            e.update({"section": si, "start": times[e["i0"]], "end": times[min(e["i1"], len(x) - 1)]})
            cand[e["i0"]:e["i1"]] = True
            if e["regular"]:
                lab[e["i0"]:e["i1"]] = e["label"]
            evs.append(e)
        samples.append(pd.DataFrame({"Time": times, "A": a, "label": lab, "candidate": cand, "section": si}))
    s = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=["Time", "A", "label", "candidate", "section"])
    return s, pd.DataFrame(evs), secs


def phase_randomise(x, rng):
    xs = x - x.mean()
    f = np.fft.rfft(xs)
    ph = rng.uniform(0, 2 * np.pi, len(f))
    ph[0] = 0
    if len(xs) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=len(xs)) + x.mean()


def weighted_kappa(a, b, k=5):
    a, b = np.asarray(a), np.asarray(b)
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    return 1 - (W * O).sum() / (W * E).sum(), O


def auc(score, positive):
    r = rankdata(score)
    n1, n0 = positive.sum(), (~positive).sum()
    return (r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def window_join(d_clean, samples):
    """Detector output over each index window's time span. Needs >= 80% of the span covered by detector samples."""
    T = samples["Time"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
    A, L = samples["A"].values, samples["label"].values
    out = []
    for _, r in d_clean.iterrows():
        t0, t1 = r["t_first"].value / 1e9, r["t_last"].value / 1e9
        lo, hi = np.searchsorted(T, t0), np.searchsorted(T, t1, side="right")
        if hi - lo < 0.8 * (t1 - t0) / DT:
            continue
        a, l = A[lo:hi], L[lo:hi]
        out.append({"Time": r["Time"], "sev": r["sev"], "tq_mean": r["tq_mean"], "rel_index": r["sev"] / max(r["tq_mean"], MEAN_FLOOR) / 2,
                    "det_A_mean": float(np.nanmean(a)), "det_label_max": int(l.max()), "det_event_frac": float((l > 0).mean())})
    return pd.DataFrame(out)


def synthetic_test(noise_rel_sd, rng, n_rep=150, length_s=300, **gate):
    """Noise at the level seen in the real wells plus an injected 30 s oscillation of known amplitude (as a
    fraction of mean torque). Slopes -2 (red) for the noise, as in Part 2's calibration."""
    n = int(length_s / DT)
    t = np.arange(n) * DT
    rows = []
    mean_torque = 5000.0
    for amp in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.20):
        det, ok_label, any_event = 0, 0, 0
        for _ in range(n_rep):
            z = gaussian_noise(n, -2.0, rng)
            bp_z = sosfiltfilt(SOS, z)
            z = z / np.std(bp_z) * noise_rel_sd * mean_torque
            x = mean_torque + z + amp * mean_torque * np.sin(2 * np.pi * t / 30.0 + rng.uniform(0, 6.28))
            _, _, ev = detect_section(x, **gate)
            reg = [e for e in ev if e["regular"]]
            any_event += len(ev) > 0
            if reg:
                det += 1
                best = max(reg, key=lambda e: e["duration_s"])
                ok_label += best["label"] == max(severity_class(amp), 0) if amp > 0 else 0
        rows.append({"injected_A": amp, "regular_event_rate": det / n_rep, "any_candidate_rate": any_event / n_rep,
                     "label_matches_injected": (ok_label / det) if det and amp > 0 else np.nan,
                     "injected_class": CLASSES[severity_class(amp)]})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    all_events, well_res = [], {}
    join_tables, summ_rows, surr_rows, ctrl_rows = [], [], [], []

    for well, path, color in WELLS:
        df = pd.read_csv(path, parse_dates=["Time"])
        df, d_all, _, _, _ = clean_and_flag_live(df)
        samples, events, secs = run_well(df)
        live_hours = len(samples) * DT / 3600
        print(f"\n=== {well} ===")
        print(f"{len(secs)} live sections of at least {MIN_SECTION_S} s, {live_hours*60:.0f} min on the detector's grid")
        ev = events.copy()
        n_cand = len(ev)
        reg = ev[ev["regular"]] if n_cand else ev
        print(f"{n_cand} candidate events (A >= {A_ON} for at least {MIN_EVENT_S} s); {len(reg)} regular "
              f"(at least {MIN_CYCLES} cycles, CV <= {MAX_CV})")
        if n_cand:
            ev["well"] = well
            all_events.append(ev)
        cls_counts = {CLASSES[k]: int((reg["label"] == k).sum()) for k in range(1, 5)} if len(reg) else {}
        print("  severity labels of regular events:", cls_counts)
        share_event = float((samples["label"] > 0).mean()) if len(samples) else float("nan")
        share_cand = float(samples["candidate"].mean()) if len(samples) else float("nan")
        print(f"  share of live time inside a regular event: {share_event*100:.0f}%; inside any candidate: {share_cand*100:.0f}%")
        if len(reg):
            print(f"  regular events: median duration {reg['duration_s'].median():.0f} s, median period {reg['period_s'].median():.0f} s, "
                  f"median A {reg['A_median'].median():.2f}")

        # ---- A. against the existing index on time-contiguous windows
        d = window_table(d_all)
        d["t_first"] = d["Time"].shift(WIN // 2)
        d["t_last"] = d["Time"].shift(-(WIN // 2 - 1))
        v = d[d["is_live"]].dropna(subset=["sev", "wob60", "t_first", "t_last"])
        v = v[(v["wob60"] > 0) & (v["span_s"] <= MAX_CLEAN_SPAN_S) & (v["tq_mean"] > 0)]
        jt = window_join(v, samples)
        jt["well"] = well
        join_tables.append(jt)
        n_win = len(jt)
        res = {"well": well, "live_minutes_on_grid": round(live_hours * 60, 1), "candidates": n_cand, "regular_events": len(reg),
               "events_per_live_hour": len(reg) / live_hours if live_hours else np.nan,
               "share_time_in_regular_event_%": 100 * share_event, "index_windows_matched": n_win}
        if n_win >= 20:
            rho_rel = spearmanr(jt["det_A_mean"], jt["rel_index"])[0]
            rho_raw = spearmanr(jt["det_A_mean"], jt["sev"])[0]
            top = jt["rel_index"] >= jt["rel_index"].quantile(0.8)
            flag = jt["det_label_max"] > 0
            prec = float((flag & top).sum() / max(flag.sum(), 1))
            rec = float((flag & top).sum() / max(top.sum(), 1))
            idx_class = np.array([severity_class(a) for a in jt["rel_index"]])
            kap, conf = weighted_kappa(jt["det_label_max"].values, idx_class)
            res.update({"spearman_detA_vs_relindex": rho_rel, "spearman_detA_vs_rawindex": rho_raw,
                        "auc_detA_vs_index_top20": auc(jt["det_A_mean"].values, top.values),
                        "flag_share_of_windows": float(flag.mean()), "precision_top20": prec, "recall_top20": rec,
                        "weighted_kappa_classes": kap})
            print(f"  vs. index, {n_win} matched windows: Spearman {rho_rel:.2f} (relative index), {rho_raw:.2f} (raw range); "
                  f"AUC for the index's top fifth {res['auc_detA_vs_index_top20']:.2f}; event flag in {flag.mean()*100:.0f}% of windows, "
                  f"precision {prec:.2f}, recall {rec:.2f}; weighted kappa of classes {kap:.2f}")
        # ---- B. surrogate null
        n_sur = 40
        sur_events = []
        for _ in range(n_sur):
            tot_h, reg_n = 0.0, 0
            for sec in secs:
                grid, x = uniform_series(sec["Time"], sec["TORQUE_AVG"])
                if x is None or len(x) < 60:
                    continue
                xs = phase_randomise(x, rng)
                _, _, e = detect_section(xs)
                tot_h += len(x) * DT / 3600
                reg_n += sum(1 for k in e if k["regular"])
            sur_events.append(reg_n / tot_h if tot_h else np.nan)
        res["surrogate_events_per_hour_median"] = float(np.nanmedian(sur_events))
        res["surrogate_events_per_hour_5_95"] = f"{np.nanpercentile(sur_events, 5):.1f} to {np.nanpercentile(sur_events, 95):.1f}"
        print(f"  phase-randomised surrogates: {np.nanmedian(sur_events):.1f} regular events per live hour "
              f"(5th to 95th percentile {np.nanpercentile(sur_events, 5):.1f} to {np.nanpercentile(sur_events, 95):.1f}); real: {res['events_per_live_hour']:.1f}")
        # ---- D. controls
        for ccol, cname in (("WOB_AVG", "WOB"), ("HKLD_AVG", "hookload")):
            s_c, e_c, _ = run_well(df, col=ccol)
            regc = e_c[e_c["regular"]] if len(e_c) else e_c
            h_c = len(s_c) * DT / 3600
            ctrl_rows.append({"well": well, "channel": cname, "regular_events_per_live_hour": len(regc) / h_c if h_c else np.nan,
                              "share_time_in_event_%": 100 * float((s_c["label"] > 0).mean()) if len(s_c) else np.nan})
        ctrl_rows.append({"well": well, "channel": "torque", "regular_events_per_live_hour": res["events_per_live_hour"],
                          "share_time_in_event_%": res["share_time_in_regular_event_%"]})
        summ_rows.append(res)
        well_res[well] = (samples, events, jt)

    summ = pd.DataFrame(summ_rows)
    summ.to_csv("data/detector_validation.csv", index=False)
    pd.concat(all_events, ignore_index=True).drop(columns=["i0", "i1"]).to_csv("data/detector_events.csv", index=False)
    pd.DataFrame(ctrl_rows).to_csv("data/detector_controls.csv", index=False)
    pooled = pd.concat(join_tables, ignore_index=True)
    pooled.to_csv("data/detector_window_join.csv", index=False)

    # pooled agreement of severity classes with the index
    idx_class = np.array([severity_class(a) for a in pooled["rel_index"]])
    kap, conf = weighted_kappa(pooled["det_label_max"].values, idx_class)
    conf_df = pd.DataFrame(conf.astype(int), index=[f"detector: {c}" for c in CLASSES], columns=[f"index: {c}" for c in CLASSES])
    conf_df.to_csv("data/detector_class_confusion.csv")
    print(f"\nPooled over wells: weighted kappa {kap:.2f}; confusion matrix (rows detector, columns index):")
    print(conf_df.to_string())

    # ---- C. synthetic
    rel_sd = []
    for well, path, _ in WELLS:
        df = pd.read_csv(path, parse_dates=["Time"])
        df, *_ = clean_and_flag_live(df)
        for sec in live_sections(df):
            if (sec["Time"].iloc[-1] - sec["Time"].iloc[0]).total_seconds() < MIN_SECTION_S:
                continue
            g, x = uniform_series(sec["Time"], sec["TORQUE_AVG"])
            if x is None or len(x) < 60 or x.mean() <= 0:
                continue
            rel_sd.append(np.std(sosfiltfilt(SOS, x)[EDGE:-EDGE]) / max(x.mean(), MEAN_FLOOR))
    noise_sd = float(np.median(rel_sd))
    print(f"\nMedian standard deviation of band-passed torque / mean torque over live sections: {noise_sd:.3f}")
    syn = synthetic_test(noise_sd, rng)
    syn_low = synthetic_test(noise_sd / 3, rng, n_rep=100)
    syn["noise_level"] = f"real ({noise_sd:.2f})"
    syn_low["noise_level"] = f"one third ({noise_sd/3:.2f})"
    syn_all = pd.concat([syn, syn_low], ignore_index=True)
    syn_all.to_csv("data/detector_synthetic.csv", index=False)
    print(syn_all.round(2).to_string(index=False))

    # ---- gate sweep: how strict must the regularity gate be before noise stops producing events?
    print("\nRegularity-gate sweep (synthetic noise at the real level; real wells vs. surrogates):")
    var_rows = []
    well_secs = {}
    for well, path, _ in WELLS:
        dfw_ = pd.read_csv(path, parse_dates=["Time"])
        dfw_, *_ = clean_and_flag_live(dfw_)
        well_secs[well] = [uniform_series(sec["Time"], sec["TORQUE_AVG"]) for sec in live_sections(dfw_)
                           if (sec["Time"].iloc[-1] - sec["Time"].iloc[0]).total_seconds() >= MIN_SECTION_S]
    for mc, cv in ((3, 0.35), (4, 0.30), (5, 0.25), (6, 0.20), (8, 0.15)):
        syn_g = synthetic_test(noise_sd, rng, n_rep=100, min_cycles=mc, max_cv=cv)
        row = {"min_cycles": mc, "max_cv": cv,
               "noise_only_false_alarm": float(syn_g.loc[syn_g["injected_A"] == 0.0, "regular_event_rate"].iloc[0]),
               "detect_A0.30": float(syn_g.loc[syn_g["injected_A"] == 0.30, "regular_event_rate"].iloc[0]),
               "detect_A0.60": float(syn_g.loc[syn_g["injected_A"] == 0.60, "regular_event_rate"].iloc[0])}
        for well, _, _ in WELLS:
            real_n, hours = 0, 0.0
            for grid_, x_ in well_secs[well]:
                if x_ is None or len(x_) < 60:
                    continue
                hours += len(x_) * DT / 3600
                real_n += sum(1 for e in detect_section(x_, mc, cv)[2] if e["regular"])
            sur = []
            for _ in range(25):
                n_s = 0
                for grid_, x_ in well_secs[well]:
                    if x_ is None or len(x_) < 60:
                        continue
                    n_s += sum(1 for e in detect_section(phase_randomise(x_, rng), mc, cv)[2] if e["regular"])
                sur.append(n_s / hours)
            row[f"{well}_real_per_h"] = real_n / hours
            row[f"{well}_surrogate_per_h"] = float(np.median(sur))
        var_rows.append(row)
        print("  " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    pd.DataFrame(var_rows).to_csv("data/detector_gate_sweep.csv", index=False)

    # -------------------------------------------------------------------------------------- chart
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), facecolor=SURFACE)
    ax = axes[0, 0]
    samples4, events4, _ = well_res["9-F-4"]
    longest = samples4.groupby("section")["label"].apply(lambda v: (v > 0).sum()).idxmax()
    sec_df = samples4[samples4["section"] == longest]
    dfw = pd.read_csv("data/f4_drilling_telemetry.csv", parse_dates=["Time"])
    seg = dfw[(dfw["Time"] >= sec_df["Time"].iloc[0]) & (dfw["Time"] <= sec_df["Time"].iloc[-1])]
    ax2 = ax.twinx()
    ax.plot(seg["Time"], seg["TORQUE_AVG"], color=MUTED, lw=0.8)
    ax.set_ylabel("Torque (N.m)", fontsize=10, color=SECONDARY_INK)
    ax2.plot(sec_df["Time"], sec_df["A"], color=BLUE, lw=1.1, label="relative amplitude A")
    for k, c in zip(range(1, 5), (AQUA, "#e0b000", ORANGE, RED)):
        m = (sec_df["label"] == k).values
        if m.any():
            ax2.fill_between(sec_df["Time"], 0, 1.6, where=m, color=c, alpha=0.25, label=CLASSES[k])
    ax2.axhline(A_ON, color=INK, lw=0.8, ls="--")
    ax2.set_ylim(0, 1.6)
    ax2.set_ylabel("A (fraction of mean torque)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("A. 9-F-4, section with most event time: torque, A, labels", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax2.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)
    ax.tick_params(axis="x", labelrotation=30)

    ax = axes[0, 1]
    for well, _, c in WELLS:
        jt = well_res[well][2]
        if len(jt):
            ax.scatter(jt["rel_index"], jt["det_A_mean"], s=7, alpha=0.35, color=c, label=f"{well} (n={len(jt)})")
    lim = [0, max(pooled["rel_index"].max(), pooled["det_A_mean"].max())]
    ax.plot(lim, lim, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("Existing index / mean torque (floored) / 2, per window", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Detector mean A over the same window", fontsize=10, color=SECONDARY_INK)
    ax.set_title("B. Detector amplitude vs. existing index", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 2]
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(CLASSES, fontsize=8)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=8.5, color="white" if conf[i, j] > conf.max() / 2 else INK)
    ax.set_xlabel("Index class (range / mean / 2)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Detector class", fontsize=10, color=SECONDARY_INK)
    ax.set_title(f"C. Severity classes, all wells (weighted kappa {kap:.2f})", fontsize=11, color=INK, loc="left", fontweight="bold")

    ax = axes[1, 0]
    x = np.arange(len(summ))
    ax.bar(x - 0.2, summ["events_per_live_hour"], 0.4, color=BLUE, label="real torque")
    ax.bar(x + 0.2, summ["surrogate_events_per_hour_median"], 0.4, color=MUTED, label="phase-randomised surrogate (median)")
    ax.set_xticks(x); ax.set_xticklabels(summ["well"])
    ax.set_ylabel("Regular events per live hour", fontsize=10, color=SECONDARY_INK)
    ax.set_title("D. Real vs. same-spectrum noise", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 1]
    for nl, c in zip(syn_all["noise_level"].unique(), (BLUE, AQUA)):
        s_ = syn_all[syn_all["noise_level"] == nl]
        ax.plot(s_["injected_A"], s_["regular_event_rate"], "o-", color=c, lw=1.6, label=f"regular event found, noise {nl}")
    ax.set_xlabel("Injected 30 s oscillation, A (fraction of mean torque)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Share of 5-minute series with an event", fontsize=10, color=SECONDARY_INK)
    ax.set_title("E. Synthetic detection rate", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 2]
    cdf = pd.DataFrame(ctrl_rows)
    for k, (ch, c) in enumerate((("torque", BLUE), ("WOB", ORANGE), ("hookload", AQUA))):
        s_ = cdf[cdf["channel"] == ch].set_index("well").reindex(summ["well"])
        ax.bar(np.arange(len(summ)) + (k - 1) * 0.27, s_["share_time_in_event_%"], 0.27, color=c, label=ch)
    ax.set_xticks(np.arange(len(summ))); ax.set_xticklabels(summ["well"])
    ax.set_ylabel("Live time inside a regular event (%)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("F. Same detector on WOB and hookload", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    for a in list(axes.ravel()) + [ax2]:
        for sp in ("top",):
            a.spines[sp].set_visible(False)
        a.spines["left"].set_color(BASELINE)
        a.spines["bottom"].set_color(BASELINE)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig("data/chart_detector.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_detector.png")


if __name__ == "__main__":
    main()
