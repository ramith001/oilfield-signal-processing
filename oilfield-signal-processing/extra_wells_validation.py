"""
Part 6: do the WOB and bit-depth findings generalize to more Volve wells?

Adding wells (extract_extra_wells.py) first showed a problem with the three wells used in Parts 1 to 5: they are
almost entirely not rotating. In 9-F-4, 9-F-7 and 9-F-9 the surface rotary speed is above 0.5 rev/s (30 rpm) for a
few minutes in each well, out of 34 to 166 hours of record, and the "live" torque used for the severity index
sits at a median of 196 to 1,290 N.m with the pipe stationary. Stick-slip needs a rotating drillstring, so
Parts 1 to 5 describe torque variation on a non-rotating string, not stick-slip. This script documents that and
then repeats the severity analysis on wells that did rotate while drilling.

Definitions used here (the same for every well):
  - rotating drilling row: surface rpm > 0.5 rev/s, 1,000 < torque < 32,000 N.m (the upper cut removes the
    16-bit saturation ramp near 32,767 and larger spikes), 5,000 < WOB < 1,000,000 N (bit loaded; the cap removes spikes of up to 1.6e9 N, and 1 MN is above the 99.5th percentile in every well).
  - severity window: 60 s of clock time, non-overlapping, inside one contiguous run of rotating drilling rows
    (no gap over 10 s), at least 18 rows (nominal 21 at 2.8 s), and at least 10 distinct torque values (a
    frozen sensor is not a window). Windows are cut from the start of each run, so they never cross a gap.
  - severity = torque range in the window (N.m, the Part 1 definition) and relative severity = range / (2 x mean
    torque), the amplitude of a sinusoid with that range as a fraction of mean torque. Range grows with mean
    torque, which grows with WOB, so the relative measure is the one that can speak about stick-slip itself.
Confidence intervals resample whole contiguous runs (cluster bootstrap), because windows within a run are
serially dependent.

Then, on the two wells whose log carries ROP and MSE (9-F-1C and 9-F-15b): does severity cost drilling
efficiency (Part 3's question, which the original three wells could not answer)?

    python extra_wells_validation.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from multi_well_validation import rank_residual

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
PURPLE = "#7b5bd1"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

RPM_MIN, TQ_MIN, TQ_MAX, WOB_MIN, WOB_MAX = 0.5, 1000.0, 32000.0, 5000.0, 1000000.0
WIN_S = 60.0
MIN_ROWS = 18
MIN_DISTINCT = 10
MAX_GAP_S = 10.0
N_BOOT = 1000
SEED = 6

ORIGINAL = [("9-F-9", "data/f9_drilling_telemetry.csv"), ("9-F-7", "data/f7_drilling_telemetry.csv"),
            ("9-F-4", "data/f4_drilling_telemetry.csv")]
EXTRA = ["9-F-1C", "9-F-15b", "9-F-15a", "9-F-12", "9-F-14", "9-F-5"]
COLORS = {"9-F-1C": BLUE, "9-F-15b": ORANGE, "9-F-15a": AQUA, "9-F-12": RED, "9-F-14": PURPLE, "9-F-5": MUTED}


def load_original(path):
    d = pd.read_csv(path, parse_dates=["Time"]).sort_values("Time")
    return pd.DataFrame({"Time": d["Time"], "bit_depth": d["BITDEP"], "wob": d["WOB_AVG"], "rpm": d["SURF_RPM"],
                         "torque": d["TORQUE_AVG"], "hookload": d["HKLD_AVG"], "rop": d["ROP_AVG"], "mse": d["MSE"]}), len(d)


def load_extra(label):
    d = pd.read_csv(f"data/extra_wells/{label}.csv.gz", parse_dates=["Time"]).sort_values("Time")
    return d.reset_index(drop=True)


def drilling_rows(d):
    m = (d["rpm"] > RPM_MIN) & (d["torque"] > TQ_MIN) & (d["torque"] < TQ_MAX) & (d["wob"] > WOB_MIN) & (d["wob"] < WOB_MAX)
    x = d[m].copy()
    gap = x["Time"].diff().dt.total_seconds().fillna(0) > MAX_GAP_S
    x["run"] = gap.cumsum()
    return x


def severity_windows(x):
    """Non-overlapping 60 s windows from the start of each contiguous run."""
    rows = []
    for run, g in x.groupby("run"):
        t = (g["Time"] - g["Time"].iloc[0]).dt.total_seconds().values
        if t[-1] < WIN_S * 0.9:
            continue
        bins = (t // WIN_S).astype(int)
        for b in np.unique(bins):
            w = g[bins == b]
            if len(w) < MIN_ROWS or w["torque"].nunique() < MIN_DISTINCT:
                continue
            tw = (w["Time"].iloc[-1] - w["Time"].iloc[0]).total_seconds()
            if tw < WIN_S * 0.9:
                continue
            mean_t = w["torque"].mean()
            rng_t = w["torque"].max() - w["torque"].min()
            rows.append({"run": run, "Time": w["Time"].iloc[0], "range": rng_t, "rel_sev": rng_t / (2 * mean_t),
                         "tq_mean": mean_t, "wob": w["wob"].mean(), "rpm": w["rpm"].mean(),
                         "bit_depth": w["bit_depth"].mean(), "rop_mh": w["rop"].mean() * 3600 if "rop" in w else np.nan,
                         "mse": w["mse"].mean() if "mse" in w else np.nan})
    return pd.DataFrame(rows)


def spearman(a, b):
    ok = a.notna() & b.notna()
    return spearmanr(a[ok], b[ok])[0] if ok.sum() >= 20 and a[ok].nunique() > 2 and b[ok].nunique() > 2 else np.nan


def partial(y, x, controls):
    """Spearman partial correlation of y and x given controls (rank residuals on the controls, jointly)."""
    d = pd.concat([y, x] + controls, axis=1).dropna()
    if len(d) < 30:
        return np.nan
    R = d.rank()
    C = np.column_stack([np.ones(len(R))] + [R.iloc[:, k].values for k in range(2, R.shape[1])])
    ry = R.iloc[:, 0].values - C @ np.linalg.lstsq(C, R.iloc[:, 0].values, rcond=None)[0]
    rx = R.iloc[:, 1].values - C @ np.linalg.lstsq(C, R.iloc[:, 1].values, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


def cluster_boot_ci(w, stat, rng, n_boot=N_BOOT):
    runs = [g for _, g in w.groupby("run")]
    if len(runs) < 5:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(runs), len(runs))
        vals.append(stat(pd.concat([runs[i] for i in pick], ignore_index=True)))
    vals = np.array(vals, dtype=float)
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))


def main():
    rng = np.random.default_rng(SEED)
    inv = pd.read_csv("data/extra_wells/_inventory.csv", parse_dates=["first_time", "last_time"])
    inv_rows, windows = [], {}

    # ----------------------------------------------------------------------- 0. inventory, all nine wells
    for label, path in ORIGINAL:
        d, n_rows = load_original(path)
        dt = d["Time"].diff().dt.total_seconds().clip(upper=10).fillna(0)
        rot = d["rpm"] > RPM_MIN
        x = drilling_rows(d)
        w = severity_windows(x)
        inv_rows.append({"well": label, "source": "f0nzie mirror (Parts 1-5)", "record_hours": dt.sum() / 3600,
                         "hours_rpm_over_0.5": dt[rot].sum() / 3600,
                         "hours_rotating_drilling": x["Time"].diff().dt.total_seconds().clip(upper=10).fillna(0).sum() / 3600,
                         "severity_windows": len(w), "runs_with_windows": int(w["run"].nunique()) if len(w) else 0,
                         "depth_min_m": x["bit_depth"].min() if len(x) else np.nan,
                         "depth_max_m": x["bit_depth"].max() if len(x) else np.nan,
                         "rop_populated": False})
        windows[label] = w
    for label in EXTRA:
        d = load_extra(label)
        x = drilling_rows(d)
        w = severity_windows(x)
        meta = inv[inv["well"] == label].iloc[0]
        inv_rows.append({"well": label, "source": "DrillDash store (this part)", "record_hours": meta["record_hours"],
                         "hours_rpm_over_0.5": (d["Time"].diff().dt.total_seconds().clip(upper=10).fillna(0)[d["rpm"] > RPM_MIN]).sum() / 3600,
                         "hours_rotating_drilling": x["Time"].diff().dt.total_seconds().clip(upper=10).fillna(0).sum() / 3600,
                         "severity_windows": len(w), "runs_with_windows": int(w["run"].nunique()) if len(w) else 0,
                         "depth_min_m": x["bit_depth"].min() if len(x) else np.nan,
                         "depth_max_m": x["bit_depth"].max() if len(x) else np.nan,
                         "rop_populated": bool((x["rop"] > 0).sum() > 0.5 * len(x))})
        windows[label] = w
        windows[label + "_rows"] = x
    invdf = pd.DataFrame(inv_rows)
    invdf.to_csv("data/extra_wells_inventory.csv", index=False)
    pd.set_option("display.width", 230)
    print("=== 0. Rotation inventory ===")
    print(invdf.round(1).to_string(index=False))

    # ---------------------------------------------------------- 1. WOB, rpm, depth against severity
    print("\n=== 1. Severity against WOB, rpm and depth, 60 s windows of rotating drilling ===")
    res_rows = []
    for label in EXTRA:
        w = windows[label]
        if len(w) < 40:
            print(f"{label}: only {len(w)} windows, skipped")
            continue
        row = {"well": label, "windows": len(w), "runs": int(w["run"].nunique()),
               "median_range_Nm": w["range"].median(), "median_rel_severity": w["rel_sev"].median(),
               "depth_span_m": w["bit_depth"].max() - w["bit_depth"].min()}
        for sev_name, sev in (("rel", "rel_sev"), ("abs", "range")):
            for var in ("wob", "rpm", "bit_depth"):
                f = lambda g, sev=sev, var=var: spearman(g[sev], g[var])
                est = f(w)
                lo, hi = cluster_boot_ci(w, f, rng, n_boot=400)
                row[f"{sev_name}_vs_{var}"] = est
                row[f"{sev_name}_vs_{var}_lo"] = lo
                row[f"{sev_name}_vs_{var}_hi"] = hi
            f = lambda g, sev=sev: partial(g[sev], g["bit_depth"], [g["wob"], g["rpm"]])
            est = f(w)
            lo, hi = cluster_boot_ci(w, f, rng, n_boot=400)
            row[f"{sev_name}_vs_depth_partial"] = est
            row[f"{sev_name}_vs_depth_partial_lo"] = lo
            row[f"{sev_name}_vs_depth_partial_hi"] = hi
        res_rows.append(row)
        print(f"{label}: {len(w)} windows in {row['runs']} runs, depth span {row['depth_span_m']:.0f} m, median range {row['median_range_Nm']:.0f} N.m, "
              f"median relative severity {row['median_rel_severity']:.2f}")
        for sev_name, nm in (("rel", "relative severity"), ("abs", "torque range")):
            print(f"   {nm}: WOB {row[f'{sev_name}_vs_wob']:+.2f} [{row[f'{sev_name}_vs_wob_lo']:+.2f}, {row[f'{sev_name}_vs_wob_hi']:+.2f}], "
                  f"rpm {row[f'{sev_name}_vs_rpm']:+.2f} [{row[f'{sev_name}_vs_rpm_lo']:+.2f}, {row[f'{sev_name}_vs_rpm_hi']:+.2f}], "
                  f"depth {row[f'{sev_name}_vs_bit_depth']:+.2f} [{row[f'{sev_name}_vs_bit_depth_lo']:+.2f}, {row[f'{sev_name}_vs_bit_depth_hi']:+.2f}], "
                  f"depth|WOB,rpm {row[f'{sev_name}_vs_depth_partial']:+.2f} [{row[f'{sev_name}_vs_depth_partial_lo']:+.2f}, {row[f'{sev_name}_vs_depth_partial_hi']:+.2f}]")
    res = pd.DataFrame(res_rows)
    res.to_csv("data/extra_wells_severity_correlations.csv", index=False)

    # ------------------------------------------------------------------- 2. ROP and MSE against severity
    print("\n=== 2. ROP and MSE against severity (wells whose log carries them) ===")
    ropm_rows = []
    for label in ("9-F-1C", "9-F-15b"):
        w = windows[label]
        w = w[(w["rop_mh"] > 0.5) & (w["rop_mh"] < 200) & (w["mse"] > 0)].copy()
        w["mse_MPa"] = w["mse"] / 1e6
        row = {"well": label, "windows": len(w), "runs": int(w["run"].nunique()), "median_rop_m_per_h": w["rop_mh"].median()}
        for sev_name, sev in (("rel", "rel_sev"), ("abs", "range")):
            f = lambda g, sev=sev: spearman(g[sev], g["rop_mh"])
            row[f"{sev_name}_vs_rop"] = f(w)
            row[f"{sev_name}_vs_rop_lo"], row[f"{sev_name}_vs_rop_hi"] = cluster_boot_ci(w, f, rng, n_boot=400)
            f = lambda g, sev=sev: partial(g[sev], g["rop_mh"], [g["wob"], g["rpm"]])
            row[f"{sev_name}_vs_rop_partial"] = f(w)
            row[f"{sev_name}_vs_rop_partial_lo"], row[f"{sev_name}_vs_rop_partial_hi"] = cluster_boot_ci(w, f, rng, n_boot=400)
            f = lambda g, sev=sev: spearman(g[sev], g["mse"])
            row[f"{sev_name}_vs_mse"] = f(w)
            row[f"{sev_name}_vs_mse_lo"], row[f"{sev_name}_vs_mse_hi"] = cluster_boot_ci(w, f, rng, n_boot=400)
        row["tq_mean_vs_mse"] = spearman(w["tq_mean"], w["mse"])
        ropm_rows.append(row)
        print(f"{label}: {len(w)} windows in {row['runs']} runs, median ROP {row['median_rop_m_per_h']:.1f} m/h")
        for sev_name, nm in (("rel", "relative severity"), ("abs", "torque range")):
            print(f"   {nm}: ROP {row[f'{sev_name}_vs_rop']:+.2f} [{row[f'{sev_name}_vs_rop_lo']:+.2f}, {row[f'{sev_name}_vs_rop_hi']:+.2f}], "
                  f"ROP | WOB, rpm {row[f'{sev_name}_vs_rop_partial']:+.2f} [{row[f'{sev_name}_vs_rop_partial_lo']:+.2f}, {row[f'{sev_name}_vs_rop_partial_hi']:+.2f}], "
                  f"MSE {row[f'{sev_name}_vs_mse']:+.2f} [{row[f'{sev_name}_vs_mse_lo']:+.2f}, {row[f'{sev_name}_vs_mse_hi']:+.2f}]")
        print(f"   (mean torque vs MSE {row['tq_mean_vs_mse']:+.2f}: MSE contains torque, so a positive link with severity is partly mechanical)")
        windows[label + "_ropm"] = w
    pd.DataFrame(ropm_rows).to_csv("data/extra_wells_rop_mse.csv", index=False)

    # ------------------------------------------------------------------------------------------ chart
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.6), facecolor=SURFACE)
    ax = axes[0, 0]
    names = invdf["well"].tolist()
    x = np.arange(len(names))
    ax.bar(x - 0.27, invdf["record_hours"], 0.27, color=MUTED, label="record")
    ax.bar(x, invdf["hours_rpm_over_0.5"], 0.27, color=BLUE, label="rpm > 0.5 rev/s")
    ax.bar(x + 0.27, invdf["hours_rotating_drilling"], 0.27, color=ORANGE, label="rotating drilling rows")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Hours (log scale)", fontsize=10, color=SECONDARY_INK)
    ax.axvline(2.5, color=INK, lw=0.8, ls=":")
    ax.text(1.0, ax.get_ylim()[1] * 0.55, "Parts 1-5 wells", ha="center", fontsize=8.5, color=SECONDARY_INK)
    ax.set_title("A. Rotation in each well's record", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    for k, (var, title, ax) in enumerate((("wob", "B. Relative severity vs WOB", axes[0, 1]),
                                           ("bit_depth", "C. Relative severity vs bit depth", axes[0, 2]))):
        for label in ("9-F-1C", "9-F-15b"):
            w = windows[label]
            ax.scatter(w[var] / (1000 if var == "wob" else 1), w["rel_sev"], s=3, alpha=0.18, color=COLORS[label], label=label)
            qb = pd.qcut(w[var], 12, duplicates="drop")
            m = w.groupby(qb, observed=True).agg(v=(var, "median"), s=("rel_sev", "median"))
            ax.plot(m["v"] / (1000 if var == "wob" else 1), m["s"], "o-", color=COLORS[label], lw=2, ms=5, mec="white")
        ax.set_ylim(0, 1.0)
        if var == "wob":
            ax.set_xlim(0, 400)
        ax.set_xlabel("Mean WOB in window (kN)" if var == "wob" else "Mean bit depth (m)", fontsize=10, color=SECONDARY_INK)
        ax.set_ylabel("Range / (2 x mean torque)", fontsize=10, color=SECONDARY_INK)
        ax.set_title(title, fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.legend(fontsize=8, frameon=False, markerscale=3)

    ax = axes[1, 0]
    labs = res["well"].tolist()
    xx = np.arange(len(labs))
    for off, var, col, nm in ((-0.27, "rel_vs_wob", BLUE, "WOB"), (0.0, "rel_vs_rpm", ORANGE, "rpm"), (0.27, "rel_vs_bit_depth", AQUA, "depth")):
        v = res[var].values
        lo = v - res[var + "_lo"].values
        hi = res[var + "_hi"].values - v
        ax.bar(xx + off, v, 0.27, color=col, label=nm)
        ax.errorbar(xx + off, v, yerr=[lo, hi], fmt="none", ecolor=INK, lw=1, capsize=2)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(xx)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("Spearman, relative severity (95% run-bootstrap)", fontsize=9.5, color=SECONDARY_INK)
    ax.set_title("D. Relative severity vs WOB, rpm, depth", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 1]
    for off, var, col, nm in ((-0.2, "abs_vs_wob", BLUE, "WOB, torque range"), (0.2, "rel_vs_wob", ORANGE, "WOB, relative severity")):
        v = res[var].values
        ax.bar(xx + off, v, 0.4, color=col, label=nm)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(xx)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("Spearman with WOB", fontsize=10, color=SECONDARY_INK)
    ax.set_title("E. Absolute vs relative severity against WOB", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 2]
    rm = pd.DataFrame(ropm_rows)
    xr = np.arange(len(rm))
    for off, var, col, nm in ((-0.27, "rel_vs_rop", BLUE, "ROP"), (0.0, "rel_vs_rop_partial", ORANGE, "ROP | WOB, rpm"), (0.27, "rel_vs_mse", AQUA, "MSE")):
        v = rm[var].values
        lo = v - rm[var + "_lo"].values
        hi = rm[var + "_hi"].values - v
        ax.bar(xr + off, v, 0.27, color=col, label=nm)
        ax.errorbar(xr + off, v, yerr=[lo, hi], fmt="none", ecolor=INK, lw=1, capsize=2)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(xr)
    ax.set_xticklabels(rm["well"])
    ax.set_ylabel("Spearman with relative severity", fontsize=10, color=SECONDARY_INK)
    ax.set_title("F. Does severity track ROP and MSE?", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    for a in axes.ravel():
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.spines["left"].set_color(BASELINE)
        a.spines["bottom"].set_color(BASELINE)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig("data/chart_extra_wells.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("\nSaved data/chart_extra_wells.png")


if __name__ == "__main__":
    main()
