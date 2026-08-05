"""Day-matched empirical validation — 2024-input rerun results vs observed data on the same dates.

passenger_2024/passenger_{i} holds all requests for an actual 2024 date (days.csv;
including cancellations, Seoul-internal, real coordinates), so the simulation_{i+1}
results are compared 1:1 with the observed data for that date.
A stronger validation than the pooled synthetic-day comparison (compare_validation.py).

Usage (after simulations finish) — campaign layout ({scenario}_r{N}/simulation_1/):
    python scripts/compare_validation_daymatched.py \
        --sim-dir revision_runs_2026/campaign_fast/simul_result --scenario base_y24

r{N} = passenger_{N} = the N-th date in days.csv.

Definitions (same as compare_validation.py):
- sim failure = no dispatch within 30 min (endogenous) → observed lower bound = pre-dispatch cancellations, upper bound = dispatch beyond 30 min or never dispatched
- sim waiting = request→pickup → observed = (pickup - desired time) of completed trips dispatched within 30 min

Outputs → revision/analysis/10_empirical_validation/daymatched_*
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = Path("/Users/jihoyeo/Library/CloudStorage/OneDrive-개인/research/장애인콜택시/paper/TR_A/revision/analysis/10_empirical_validation")
RAW_2024 = PROJECT_ROOT / "data/seoul_paratransit_raw/seoul_2024_raw.parquet"
DAYS_CSV = PROJECT_ROOT / "data/simulation-agent-data/passenger_2024/days.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--sim-dir", required=True, help="simul_result folder ({scenario}_r{N}/simulation_1 layout)")
ap.add_argument("--scenario", default="base_y24", help="scenario label (default base_y24)")
args = ap.parse_args()
SIM_DIR = Path(args.sim_dir) if Path(args.sim_dir).is_absolute() else PROJECT_ROOT / args.sim_dir

days = pd.read_csv(DAYS_CSV)
run_date = {n: pd.Timestamp(d).date() for n, d in enumerate(days["date"])}  # r{N} = passenger_{N}

# ---------- Simulation ----------
sim_rows = []
for run, date in run_date.items():
    f = SIM_DIR / f"{args.scenario}_r{run}" / "simulation_1" / "passenger_marker.json"
    if not f.exists():
        print(f"[skip] {f} missing — if runs are still in progress, rerun after they finish")
        continue
    p = pd.read_json(f)
    p["start"] = p["timestamp"].str[0]
    p["end"] = p["timestamp"].str[-1]
    p["hour"] = (p["start"] // 60).astype(int) % 24
    p["date"] = date
    p["run"] = run
    sim_rows.append(p)
if not sim_rows:
    raise SystemExit("No simulation results found.")
sim = pd.concat(sim_rows, ignore_index=True)
HOURS = sorted(sim["hour"].unique())
sim_succ = sim[sim["status"] == 1].copy()
sim_succ["waiting_time"] = sim_succ["end"] - sim_succ["start"]
sim_succ = sim_succ.dropna(subset=["waiting_time"])

# ---------- Observed data (same dates) ----------
df = pd.read_parquet(RAW_2024)
df = df[(df["출발시"] == "서울특별시") & (df["목적시"] == "서울특별시") & df["희망일시"].notna()]
df["date"] = df["희망일시"].dt.date
df = df[df["date"].isin(run_date.values())]
df["hour"] = df["희망일시"].dt.hour
df = df[df["hour"].isin(HOURS)]
df["disp_delay"] = (df["배차시간"] - df["희망일시"]) / pd.Timedelta(minutes=1)
df["cancel_predisp"] = (df["탑승유무"] == "접수취소") & df["배차시간"].isna()
df["nodisp30"] = df["배차시간"].isna() | (df["disp_delay"] > 30)
served = df[df["탑승시간"].notna() & (df["disp_delay"] <= 30)].copy()
served["waiting_time"] = (served["탑승시간"] - served["희망일시"]) / pd.Timedelta(minutes=1)
served = served[(served["waiting_time"] >= 0) & (served["waiting_time"] <= 240)]


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


# ---------- Day-level matched metrics ----------
daily = []
for run, date in run_date.items():
    s = sim[(sim["run"] == run)]
    if s.empty:
        continue
    r = df[df["date"] == date]
    sw = sim_succ[sim_succ["run"] == run]["waiting_time"]
    rw = served[served["date"] == date]["waiting_time"]
    sim_fail_h = s[s["status"] == 0].groupby("hour").size().reindex(HOURS, fill_value=0)
    real_lo_h = r.groupby("hour")["cancel_predisp"].sum().reindex(HOURS, fill_value=0)
    real_hi_h = r.groupby("hour")["nodisp30"].sum().reindex(HOURS, fill_value=0)
    daily.append({
        "run": run, "date": str(date),
        "sim_requests": len(s), "real_requests": len(r),
        "sim_failures": int((s["status"] == 0).sum()),
        "real_cancel_predisp": int(r["cancel_predisp"].sum()),
        "real_nodisp30": int(r["nodisp30"].sum()),
        "sim_wait_mean": sw.mean(), "real_wait_mean": rw.mean(),
        "sim_wait_median": sw.median(), "real_wait_median": rw.median(),
        "ks_stat": ks_2samp(sw, rw).statistic if len(sw) and len(rw) else np.nan,
        "rho_hourly_lo": spearmanr(sim_fail_h, real_lo_h)[0],
        "rho_hourly_hi": spearmanr(sim_fail_h, real_hi_h)[0],
    })
daily = pd.DataFrame(daily)
daily.to_csv(OUT / f"daymatched_{args.scenario}_daily.csv", index=False)

pooled_ks = ks_2samp(sim_succ["waiting_time"], served["waiting_time"])
lines = ["# Day-matched validation (2024 inputs)", "",
         f"sim dir: {SIM_DIR}  |  runs matched: {len(daily)}/10", "",
         daily.round(2).to_string(index=False), "",
         f"- pooled waiting KS = {pooled_ks.statistic:.3f} "
         f"(sim mean {sim_succ['waiting_time'].mean():.1f} / real {served['waiting_time'].mean():.1f})",
         f"- daily failures vs lower bound (pre-dispatch cancellations): R2 = {r2(daily['real_cancel_predisp'], daily['sim_failures']):.3f}, "
         f"rho = {spearmanr(daily['real_cancel_predisp'], daily['sim_failures'])[0]:.3f}",
         f"- daily failures vs upper bound (no dispatch within 30 min): R2 = {r2(daily['real_nodisp30'], daily['sim_failures']):.3f}, "
         f"rho = {spearmanr(daily['real_nodisp30'], daily['sim_failures'])[0]:.3f}",
         f"- hourly profile rho (mean over runs): lower {daily['rho_hourly_lo'].mean():.3f} / upper {daily['rho_hourly_hi'].mean():.3f}",
         f"- daily wait mean: sim {daily['sim_wait_mean'].mean():.1f} vs real {daily['real_wait_mean'].mean():.1f} "
         f"(paired diff {(daily['sim_wait_mean'] - daily['real_wait_mean']).mean():+.1f} min)"]
(OUT / f"daymatched_{args.scenario}_metrics.md").write_text("\n".join(lines), encoding="utf-8")

# ---------- Figures ----------
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, col, label in zip(axes, ("real_cancel_predisp", "real_nodisp30"),
                          ("Pre-dispatch cancellations", "No dispatch within 30 min")):
    lim = max(daily[col].max(), daily["sim_failures"].max()) * 1.15
    ax.scatter(daily[col], daily["sim_failures"], color="#4B6B8A", alpha=.8)
    ax.plot([0, lim], [0, lim], "--", color="#999999")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(f"Observed (same day): {label}")
    ax.text(.05, .95, f"$\\rho_s$ = {spearmanr(daily[col], daily['sim_failures'])[0]:.3f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=.8))
axes[0].set_ylabel("Simulated failures (same day)")
fig.tight_layout(); fig.savefig(OUT / f"daymatched_{args.scenario}_failure_scatter.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
rh = served.groupby("hour")["waiting_time"]
sh = sim_succ.groupby("hour")["waiting_time"]
ax.plot(HOURS, rh.mean().reindex(HOURS), "-o", color="#4B6B8A", label="Observed 2024 (mean)")
ax.fill_between(HOURS, rh.quantile(.25).reindex(HOURS), rh.quantile(.75).reindex(HOURS), color="#4B6B8A", alpha=.15)
ax.plot(HOURS, sh.mean().reindex(HOURS), "-s", color="gray", label="Simulated (mean)")
ax.fill_between(HOURS, sh.quantile(.25).reindex(HOURS), sh.quantile(.75).reindex(HOURS), color="gray", alpha=.15)
ax.set_xlabel("Time of Day"); ax.set_ylabel("Waiting Time (min)")
ax.set_xticks(HOURS); ax.set_xticklabels([f"{h:02d}" for h in HOURS])
ax.legend(fontsize=9); fig.tight_layout()
fig.savefig(OUT / f"daymatched_{args.scenario}_waiting_hourly.png", dpi=200); plt.close(fig)

print("\n".join(lines))
print("\nSaved:", OUT, "(daymatched_*)")
