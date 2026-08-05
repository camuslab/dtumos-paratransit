# -*- coding: utf-8 -*-
"""Figure generation for the TR_A revision (2026-08).

Manuscript figure styling follows the dtumos-paratransit-report/make_paper_figure.ipynb conventions:
Times New Roman serif, seaborn whitegrid, paper context (font_scale=1.3), despine.

Outputs → OneDrive revision/drafts/figures/
  fig_validation_failure_hourly   new (validation 3.X.2) hourly failures: sim vs observed upper/lower bounds, 2023/2024
  fig_validation_waiting_hourly   new (validation 3.X.3) hourly conditional waiting: sim vs observed, 2023/2024
  fig_fleet_grid                  replaces old Fig 11 (4.2) fleet grid + official expansion targets + targeted staffing baselines
  fig_matched_staffing_hourly     replaces old Fig 12 (4.2) Base/A-782/Ct156 hourly failure rates
  fig_factorial_waterfall         new (4.4) factorial waterfall + mean marginal contributions
  fig_supply_2023_2024            new (4.7) vehicles on duty by hour, 2023 vs 2024
  fig_cancellations_by_hour       new (appendix, R2-7) hourly cancellations (pre-/post-dispatch), June weekdays

Data sources
  - campaign 10-run KPI: revision_runs_2026/campaign_fast_results.csv
  - 2025-scenario 10-run KPI: revision_runs_2026/stability_table_2025scenarios.csv
  - run-level passenger_marker: result/baseline, _simulation_archive_20260730/04_verification_data
    /verification_data_2025_paper/Scenario_A-1, revision_runs_2026/campaign_fast/simul_result/Ct156_r*
  - validation tidy CSV: OneDrive revision/analysis/10_empirical_validation/
  - fleet: 02_vehicle_inputs/paper_2025__613_782_870/Baseline_613__vehicle_0.csv,
    revision_runs_2026/fleet_files/vehicle24_636.csv
  - cancellations: data/seoul_paratransit_raw/ raw parquet (June weekdays, excluding the 6/6 holiday)
"""

import json
import os
from itertools import permutations
from pathlib import Path

from decimal import Decimal, ROUND_HALF_UP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REV = PROJECT_ROOT / "revision_runs_2026"
ARCH = PROJECT_ROOT / "_simulation_archive_20260730"
VAL = Path("/Users/jihoyeo/Library/CloudStorage/OneDrive-개인/research/장애인콜택시/paper/TR_A/revision/analysis/10_empirical_validation")
OUT = Path("/Users/jihoyeo/Library/CloudStorage/OneDrive-개인/research/장애인콜택시/paper/TR_A/revision/drafts/figures")
OUT.mkdir(parents=True, exist_ok=True)

HOURS = list(range(6, 24))
HOUR_LABELS = [f"{h:02d}-{h + 1:02d}" for h in HOURS]

NAVY = "#4B6B8A"
ORANGE = "#E76F51"
TEAL = "#2A9D8F"
GRAY = "#7F7F7F"
LIGHT = "#999999"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

STATS = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", format="png")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"saved: {name}")


def ci95(x):
    x = np.asarray(x, dtype=float)
    return sps.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


# ----------------------------------------------------------------------------
# Shared data
# ----------------------------------------------------------------------------
camp = pd.read_csv(REV / "campaign_fast_results.csv")
stab = pd.read_csv(REV / "stability_table_2025scenarios.csv")
stab_fail = stab[stab.kpi == "request_failure_%"].set_index("scenario")["mean"]
stab_sd = stab[stab.kpi == "request_failure_%"].set_index("scenario")["sd"]


def camp_fail(scen):
    sub = camp[camp.scenario == scen]["request_failure_pct"]
    return sub.mean(), ci95(sub)


def hourly_failure_rate(run_dirs):
    """Hourly failure rate (%) by request time from 10 passenger_marker.json runs (pooled)."""
    fail = np.zeros(len(HOURS))
    tot = np.zeros(len(HOURS))
    for d in run_dirs:
        pm = pd.read_json(Path(d) / "passenger_marker.json")
        req_hour = pm["timestamp"].str[0].floordiv(60).astype(int).clip(6, 23)
        for j, h in enumerate(HOURS):
            m = req_hour == h
            tot[j] += m.sum()
            fail[j] += (m & (pm["status"] == 0)).sum()
    return np.where(tot > 0, fail / tot * 100, np.nan)


# ============================================================================
# 0. Figure 9 redesign: hourly requests (bars) + failure/delay rates (lines), left/right ticks aligned
# ============================================================================
def fig_hourly_requests_rates():
    base_runs = [PROJECT_ROOT / "result" / "baseline" / f"simulation_{i}" for i in range(1, 11)]
    fail = np.zeros(len(HOURS)); tot = np.zeros(len(HOURS))
    served = np.zeros(len(HOURS)); delayed = np.zeros(len(HOURS))
    req_per_run = np.zeros((len(base_runs), len(HOURS)))
    for k, d in enumerate(base_runs):
        pm = pd.read_json(Path(d) / "passenger_marker.json")
        req_hour = pm["timestamp"].str[0].floordiv(60).astype(int).clip(6, 23)
        wait = pm["timestamp"].str[1] - pm["timestamp"].str[0]
        for j, h in enumerate(HOURS):
            m = req_hour == h
            tot[j] += m.sum(); req_per_run[k, j] = m.sum()
            f = m & (pm["status"] == 0); fail[j] += f.sum()
            s = m & (pm["status"] == 1); served[j] += s.sum()
            delayed[j] += (s & (wait > 30)).sum()
    req_mean = req_per_run.mean(axis=0)
    fail_pct = np.where(tot > 0, fail / tot * 100, 0.0)
    delay_pct = np.where(served > 0, delayed / served * 100, 0.0)
    STATS["fig_hourly_requests_rates"] = {
        "peak_hour_requests": {str(h): round(v, 1) for h, v in zip(HOURS, req_mean)},
        "combined_share_of_requests_14h_pct": round(float((fail[8] + delayed[8]) / tot[8] * 100), 2),
        "combined_share_of_requests_07h_pct": round(float((fail[1] + delayed[1]) / tot[1] * 100), 2),
    }

    x = np.arange(len(HOURS))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(x, req_mean, color="grey", alpha=0.5, label="Service Requests")
    ax1.set_xlabel("Time of day")
    ax1.set_ylabel("Number of service requests")
    ax1.set_xticks(x)
    ax1.set_xticklabels(HOUR_LABELS, rotation=45, ha="right")
    # Left 0–1000/200, right 0–100/20: ratio fixed at exactly 10x → gridlines match the ticks on both axes
    ax1.set_ylim(-40, 1040); ax1.set_yticks(np.arange(0, 1001, 200))

    ax2 = ax1.twinx()
    lf, = ax2.plot(x, fail_pct, marker="o", linestyle="-", color=NAVY,
                   linewidth=2, markersize=5, label="Request Failure Rate")
    ld, = ax2.plot(x, delay_pct, marker="s", linestyle="-", color=ORANGE,
                   linewidth=2, markersize=5, label="Service Delay Rate")
    ax2.set_ylabel("Rate (%)")
    ax2.set_ylim(-4, 104); ax2.set_yticks(np.arange(0, 101, 20))
    ax2.grid(False)

    ax1.legend([bars, lf, ld], ["Service Requests", "Request Failure Rate", "Service Delay Rate"],
               loc="upper right", frameon=True, framealpha=1.0, edgecolor="0.8")
    save(fig, "fig_hourly_requests_rates")


# ============================================================================
# 1. Validation: hourly failures — sim vs observed upper/lower bounds (2023/2024)
# ============================================================================
def fig_validation_failure_hourly():
    sim = pd.read_csv(VAL / "sim_hourly.csv").set_index("hour").reindex(HOURS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    rho = {}
    for ax, year in zip(axes, (2023, 2024)):
        real = (pd.read_csv(VAL / f"real_hourly_{year}.csv")
                .groupby("hour").agg(lo=("cancel_predisp", "mean"), hi=("nodisp30", "mean"))
                .reindex(HOURS))
        rho[year] = {
            "lo": sps.spearmanr(real["lo"], sim["sim_failure"])[0],
            "hi": sps.spearmanr(real["hi"], sim["sim_failure"])[0],
        }
        ax.fill_between(HOURS, real["lo"], real["hi"], color=NAVY, alpha=0.15,
                        label="Observed range between definitions")
        ax.plot(HOURS, real["hi"], "-^", color=NAVY, ms=4.5, lw=1.2,
                label="Observed upper bound: no dispatch within 30 min")
        ax.plot(HOURS, real["lo"], "-v", color=NAVY, ms=4.5, lw=1.2, ls="--",
                label="Observed lower bound: pre-dispatch cancellations")
        ax.plot(HOURS, sim["sim_failure"], "-o", color=ORANGE, ms=4.5, lw=1.6,
                label="Simulated request failures (Base, 10-run mean)")
        ax.set_xlabel("Time of Day", fontsize=12, labelpad=8)
        ax.set_xticks(HOURS[::2])
        ax.set_xticklabels([f"{h:02d}" for h in HOURS[::2]])
        ax.set_title(f"{year} records", fontsize=12)
        ax.text(0.97, 0.96,
                f"$\\rho_s$ = {rho[year]['lo']:.2f} (lower)\n$\\rho_s$ = {rho[year]['hi']:.2f} (upper)",
                transform=ax.transAxes, va="top", ha="right", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))
        sns.despine(ax=ax)
    axes[0].set_ylabel("Failures per hour (daily mean)", fontsize=12, labelpad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9.5, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.13), frameon=False)
    STATS["validation_failure_rho"] = rho
    save(fig, "fig_validation_failure_hourly")


# ============================================================================
# 2. Validation: hourly conditional waiting — sim vs observed (2023/2024)
# ============================================================================
def fig_validation_waiting_hourly():
    sim_w = pd.read_csv(PROJECT_ROOT / "data/verification/sim_baseline_waiting.csv")
    sim_w = sim_w[sim_w["hour"].isin(HOURS)]
    sh = sim_w.groupby("hour")["waiting_time"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    gaps = {}
    for ax, year in zip(axes, (2023, 2024)):
        w = pd.read_csv(VAL / f"real_waiting_cond_{year}.csv")
        rh = w.groupby("hour")["waiting_time"]
        ax.fill_between(HOURS, rh.quantile(0.25).reindex(HOURS), rh.quantile(0.75).reindex(HOURS),
                        color=NAVY, alpha=0.15, label=f"Observed {year} (IQR)")
        ax.fill_between(HOURS, sh.quantile(0.25).reindex(HOURS), sh.quantile(0.75).reindex(HOURS),
                        color=GRAY, alpha=0.18, label="Simulated (IQR)")
        ax.plot(HOURS, rh.mean().reindex(HOURS), "-o", color=NAVY, ms=4.5, lw=1.6,
                label=f"Observed {year} (mean)")
        ax.plot(HOURS, sh.mean().reindex(HOURS), "-s", color=GRAY, ms=4.5, lw=1.6,
                label="Simulated (mean)")
        day = np.abs(rh.mean().reindex(range(8, 18)) - sh.mean().reindex(range(8, 18))).mean()
        eve = (rh.mean().reindex(range(18, 24)) - sh.mean().reindex(range(18, 24))).mean()
        gaps[year] = {"daytime_mae": round(float(day), 1), "evening_underest": round(float(eve), 1)}
        ax.set_xlabel("Time of Day", fontsize=12, labelpad=8)
        ax.set_xticks(HOURS[::2])
        ax.set_xticklabels([f"{h:02d}" for h in HOURS[::2]])
        ax.set_title(f"{year} records", fontsize=12)
        ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Waiting time (min)", fontsize=12, labelpad=8)
    STATS["validation_waiting_gaps"] = gaps
    save(fig, "fig_validation_waiting_hourly")


# ============================================================================
# 3. Fleet grid + targeted staffing baselines (replaces old Fig 11)
# ============================================================================
def fig_fleet_grid():
    grid_sizes = [644, 674, 705, 736, 766, 797]
    xs, ys, es = [613], [stab_fail["baseline"]], [ci95_from_sd(stab_sd["baseline"], 10)]
    for n in grid_sizes:
        m, e = camp_fail(f"F{n}")
        xs.append(n); ys.append(m); es.append(e)
    a1, a1e = stab_fail["Scenario_A-1"], ci95_from_sd(stab_sd["Scenario_A-1"], 10)
    a2, a2e = stab_fail["Scenario_A-2"], ci95_from_sd(stab_sd["Scenario_A-2"], 10)
    ct100, ct100e = camp_fail("Ct100")
    ct156, ct156e = camp_fail("Ct156")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(xs, ys, yerr=es, fmt="-o", color=NAVY, ms=6, lw=1.8, capsize=3,
                label="Fleet expansion grid (613–797 vehicles)")
    ax.errorbar([782, 870], [a1, a2], yerr=[a1e, a2e], fmt="D", color=NAVY, mfc="white",
                ms=8, capsize=3, lw=1.4, label="Official expansion targets (Scenario A)")
    ax.annotate("A (782)\n9.2%", (782, a1), textcoords="offset points", xytext=(10, -26), fontsize=9.5)
    ax.annotate("A (870)\n5.3%", (870, a2), textcoords="offset points", xytext=(-6, -30),
                fontsize=9.5, ha="right")

    ax.axhline(ct100, color=ORANGE, ls="--", lw=1.6)
    ax.axhline(ct156, color=ORANGE, ls=":", lw=1.8)
    ax.text(867, ct100 + 0.35, "100 targeted shifts (C): 10.4%\n+900 veh-h/day, no vehicles purchased",
            color=ORANGE, fontsize=9.5, ha="right")
    ax.text(867, ct156 + 0.35, "156 targeted shifts: 6.6%\n+1,404 veh-h/day ≈ added hours of A (782)",
            color=ORANGE, fontsize=9.5, ha="right")

    ax.set_xlabel("Fleet size (vehicles)", fontsize=12, labelpad=8)
    ax.set_ylabel("Request failure rate (%)", fontsize=12, labelpad=8)
    ax.set_ylim(0, 27)
    ax.legend(loc="upper right", fontsize=10)
    sns.despine()
    STATS["fleet_grid"] = {"grid": dict(zip(map(str, xs), np.round(ys, 2))),
                           "A782": round(float(a1), 2), "A870": round(float(a2), 2),
                           "Ct100": round(ct100, 2), "Ct156": round(ct156, 2)}
    save(fig, "fig_fleet_grid")


def ci95_from_sd(sd, n):
    return sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n)


# ============================================================================
# 4. Base vs A-782 vs Ct156 hourly failure rates (replaces old Fig 12)
# ============================================================================
def fig_matched_staffing_hourly():
    base_dirs = [PROJECT_ROOT / f"result/baseline/simulation_{i}" for i in range(1, 11)]
    a1_dirs = [ARCH / f"04_verification_data/verification_data_2025_paper/Scenario_A-1/simulation_{i}"
               for i in range(1, 11)]
    ct_dirs = [REV / f"campaign_fast/simul_result/Ct156_r{i}/simulation_1" for i in range(10)]

    base_r = hourly_failure_rate(base_dirs)
    a1_r = hourly_failure_rate(a1_dirs)
    ct_r = hourly_failure_rate(ct_dirs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(7, 16, color=ORANGE, alpha=0.07)
    ax.text(11.5, 62, "targeted shift window (07:00–16:00)", color=ORANGE,
            fontsize=10, ha="center")
    ax.plot(HOURS, base_r, "-o", color=GRAY, ms=5, lw=1.6, label="Base (613 vehicles): 24.3%")
    ax.plot(HOURS, a1_r, "-s", color=NAVY, ms=5, lw=1.6,
            label="A (782) fleet expansion (+1,400 veh-h/day): 9.2%")
    ax.plot(HOURS, ct_r, "-^", color=ORANGE, ms=5.5, lw=1.8,
            label="156 targeted shifts (+1,404 veh-h/day): 6.6%")
    ax.set_xlabel("Time of Day", fontsize=12, labelpad=8)
    ax.set_ylabel("Request failure rate (%)", fontsize=12, labelpad=8)
    ax.set_xticks(HOURS)
    ax.set_xticklabels(HOUR_LABELS, rotation=45, ha="right")
    ax.set_ylim(0, 68)
    ax.legend(loc="upper right", fontsize=10)
    sns.despine()
    STATS["matched_hourly"] = {"base": np.round(base_r, 1).tolist(),
                               "A782": np.round(a1_r, 1).tolist(),
                               "Ct156": np.round(ct_r, 1).tolist()}
    save(fig, "fig_matched_staffing_hourly")


# ============================================================================
# 5. Factorial waterfall (new, 4.4)
# ============================================================================
def fig_factorial_waterfall():
    corners = {
        (): stab_fail["baseline"],
        ("B",): stab_fail["Scenario_B"],
        ("C",): camp_fail("Ct100")[0],
        ("D",): stab_fail["Scenario_D"],
        ("B", "C"): camp_fail("BCt100")[0],
        ("B", "D"): camp_fail("BD")[0],
        ("C", "D"): camp_fail("Ct100D")[0],
        ("B", "C", "D"): camp_fail("BCt100D")[0],
    }

    def val(s):
        return corners[tuple(sorted(s))]

    levers = ["B", "C", "D"]
    shap = {l: 0.0 for l in levers}
    for order in permutations(levers):
        cur = []
        for l in order:
            shap[l] += val(cur) - val(cur + [l])
            cur.append(l)
    for l in levers:
        shap[l] /= 6.0
    total = val([]) - val(levers)
    pct = {l: shap[l] / total * 100 for l in levers}
    inter = total - sum(val([]) - val([l]) for l in levers)

    steps = [("Base", val([]), None),
             ("+ Targeted staffing (C)", val(["C"]), val([]) - val(["C"])),
             ("+ Voucher program (D)", val(["C", "D"]), val(["C"]) - val(["C", "D"])),
             ("+ Dispatch re-optimization (B)", val(["B", "C", "D"]),
              val(["C", "D"]) - val(["B", "C", "D"]))]

    fig, ax = plt.subplots(figsize=(10, 5))
    xpos = np.arange(len(steps))
    prev = None
    for i, (label, level, drop) in enumerate(steps):
        if drop is None:
            ax.bar(i, level, width=0.55, color=GRAY, alpha=0.85)
            ax.text(i, level + 0.5, f"{level:.2f}%", ha="center", fontsize=11)
        else:
            ax.bar(i, drop, bottom=level, width=0.55, color=NAVY, alpha=0.35,
                   edgecolor=NAVY, hatch="//")
            ax.bar(i, level, width=0.55, color=NAVY, alpha=0.9)
            ax.plot([i - 1 + 0.28, i - 0.28], [prev, prev], color="black", lw=0.9, ls=":")
            ax.annotate(f"−{drop:.2f} pp", (i, level + drop / 2), ha="center",
                        fontsize=10, color="white" if drop > 3 else "black",
                        bbox=None if drop > 3 else dict(boxstyle="round,pad=0.15",
                                                        fc="white", ec="none", alpha=0.7))
            ax.text(i, level - 1.3 if level > 3 else level + drop + 0.5,
                    f"{level:.2f}%", ha="center", fontsize=11,
                    color="white" if level > 3 else "black")
        prev = level
    ax.set_xticks(xpos)
    ax.set_xticklabels([s[0] for s in steps], fontsize=10.5)
    ax.set_ylabel("Request failure rate (%)", fontsize=12, labelpad=8)
    ax.set_ylim(0, 27)
    sns.despine()
    STATS["waterfall"] = {"steps": {s[0]: round(float(s[1]), 2) for s in steps},
                          "shapley_pct": {k: round(v, 1) for k, v in pct.items()},
                          "interaction_pp": round(float(inter), 2)}
    save(fig, "fig_factorial_waterfall")


# ============================================================================
# 6. Vehicles on duty by hour, 2023 vs 2024 (new, 4.7)
# ============================================================================
def fig_supply_2023_2024():
    v23 = pd.read_csv(ARCH / "02_vehicle_inputs/paper_2025__613_782_870/Baseline_613__vehicle_0.csv")
    v24 = pd.read_csv(REV / "fleet_files/vehicle24_636.csv")
    c23 = [int(((v23.work_start <= h) & (v23.work_end > h)).sum()) for h in HOURS]
    c24 = [int(((v24.work_start <= h) & (v24.work_end > h)).sum()) for h in HOURS]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(HOURS))
    ax.bar(x - 0.2, c23, width=0.4, color=GRAY, alpha=0.75, label="June 2023 schedule (613 vehicles)")
    ax.bar(x + 0.2, c24, width=0.4, color=NAVY, alpha=0.9, label="June 2024 schedule (636 vehicles)")
    for h, hi in [(7, True), (16, True), (17, True)]:
        j = HOURS.index(h)
        ax.annotate(f"+{c24[j] - c23[j]}", (j + 0.2, c24[j] + 6), ha="center",
                    fontsize=9.5, color=NAVY, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(HOUR_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Time of Day", fontsize=12, labelpad=8)
    ax.set_ylabel("Vehicles on duty", fontsize=12, labelpad=8)
    ax.legend(loc="upper right", fontsize=10)
    sns.despine()
    STATS["supply_profile"] = {"2023": dict(zip(HOUR_LABELS, c23)), "2024": dict(zip(HOUR_LABELS, c24))}
    save(fig, "fig_supply_2023_2024")


# ============================================================================
# 7. Appendix: hourly cancellations (pre-/post-dispatch), June weekdays (R2-7)
# ============================================================================
def fig_cancellations_by_hour():
    cols = ["탑승유무", "희망일시", "배차시간", "출발시", "목적시"]
    src23 = PROJECT_ROOT / "data/seoul_paratransit_raw/seoul_2023_h1_raw.parquet"
    src24 = PROJECT_ROOT / "data/seoul_paratransit_raw/seoul_2024_raw.parquet"

    def june_weekday_cancels(src, year):
        df = pd.read_parquet(src, columns=cols)
        df = df[(df["출발시"] == "서울특별시") & (df["목적시"] == "서울특별시")]
        df["희망일시"] = pd.to_datetime(df["희망일시"])
        df = df[df["희망일시"].notna()]
        df = df[(df["희망일시"].dt.year == year) & (df["희망일시"].dt.month == 6)]
        df = df[df["희망일시"].dt.weekday < 5]
        df = df[df["희망일시"].dt.day != 6]  # Memorial Day holiday
        n_days = df["희망일시"].dt.date.nunique()
        c = df[df["탑승유무"] == "접수취소"].copy()
        c["hour"] = df["희망일시"].dt.hour
        c = c[c["hour"].isin(HOURS)]
        c["stage"] = np.where(c["배차시간"].notna(), "after", "before")
        g = c.groupby(["hour", "stage"]).size().unstack(fill_value=0).reindex(HOURS).fillna(0)
        return g / n_days, n_days, len(c) / n_days

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    totals = {}
    for ax, (src, year) in zip(axes, [(src23, 2023), (src24, 2024)]):
        g, n_days, per_day = june_weekday_cancels(src, year)
        x = np.arange(len(HOURS))
        ax.bar(x, g["before"], width=0.7, color=NAVY, alpha=0.9,
               label="Cancelled before dispatch")
        ax.bar(x, g["after"], width=0.7, bottom=g["before"], color=NAVY, alpha=0.35,
               label="Cancelled after dispatch")
        pre = g["before"].sum()
        totals[year] = {"days": int(n_days), "cancels_per_day": round(per_day, 1),
                        "predispatch_per_day": round(float(pre), 1)}
        ax.set_title(f"June {year} weekdays (n = {n_days} days)", fontsize=12)
        ax.set_xticks(x[::2])
        ax.set_xticklabels([f"{h:02d}" for h in HOURS[::2]])
        ax.set_xlabel("Time of Day", fontsize=12, labelpad=8)
        ax.text(0.03, 0.96, f"pre-dispatch total ≈ {pre:.0f}/day", transform=ax.transAxes,
                va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))
        sns.despine(ax=ax)
    axes[0].set_ylabel("Cancellations per hour (daily mean)", fontsize=12, labelpad=8)
    axes[1].legend(fontsize=9.5, loc="upper right")
    STATS["cancellations"] = totals
    save(fig, "fig_cancellations_by_hour")


if __name__ == "__main__":
    fig_hourly_requests_rates()
    fig_validation_failure_hourly()
    fig_validation_waiting_hourly()
    fig_fleet_grid()
    fig_matched_staffing_hourly()
    fig_factorial_waterfall()
    fig_supply_2023_2024()
    fig_cancellations_by_hour()
    (OUT / "figure_stats.json").write_text(json.dumps(STATS, indent=2, ensure_ascii=False))
    print(json.dumps(STATS, indent=2, ensure_ascii=False))
