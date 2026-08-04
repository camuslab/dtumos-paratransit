"""Day-matched empirical validation — 2024 입력 재실행 결과 vs 같은 날짜 실측.

passenger_2024/passenger_{i}는 실제 2024년 날짜(days.csv)의 전체 접수(취소 포함,
서울 내, 실좌표)이므로, simulation_{i+1} 결과를 그 날짜의 실측과 1:1로 비교한다.
기존 합성일 pooled 비교(compare_validation.py)보다 강한 검증.

사용법 (시뮬레이션 완료 후) — campaign 레이아웃 ({scenario}_r{N}/simulation_1/):
    python scripts/compare_validation_daymatched.py \
        --sim-dir revision_runs_2026/campaign_fast/simul_result --scenario base_y24

r{N} = passenger_{N} = days.csv N번째 날짜.

정의 (compare_validation.py와 동일):
- 시뮬 실패 = 30분 내 미배차 (내생) → 실측 하한 = 배차 전 접수취소, 상한 = 배차 30분 초과·미배차
- 시뮬 대기 = 요청→승차 → 실측 = 30분 내 배차된 완료 운행의 (승차 - 희망)

산출물 → revision/analysis/10_empirical_validation/daymatched_*
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
RAW_2024 = PROJECT_ROOT / "data/(서울시)특별교통수단/서울_2024_로우데이터.parquet"
DAYS_CSV = PROJECT_ROOT / "data/simulation-agent-data/passenger_2024/days.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--sim-dir", required=True, help="simul_result 폴더 ({scenario}_r{N}/simulation_1 구조)")
ap.add_argument("--scenario", default="base_y24", help="시나리오 라벨 (기본 base_y24)")
args = ap.parse_args()
SIM_DIR = Path(args.sim_dir) if Path(args.sim_dir).is_absolute() else PROJECT_ROOT / args.sim_dir

days = pd.read_csv(DAYS_CSV)
run_date = {n: pd.Timestamp(d).date() for n, d in enumerate(days["date"])}  # r{N} = passenger_{N}

# ---------- 시뮬레이션 ----------
sim_rows = []
for run, date in run_date.items():
    f = SIM_DIR / f"{args.scenario}_r{run}" / "simulation_1" / "passenger_marker.json"
    if not f.exists():
        print(f"[스킵] {f} 없음 — 아직 실행 중이면 완료 후 다시 실행")
        continue
    p = pd.read_json(f)
    p["start"] = p["timestamp"].str[0]
    p["end"] = p["timestamp"].str[-1]
    p["hour"] = (p["start"] // 60).astype(int) % 24
    p["date"] = date
    p["run"] = run
    sim_rows.append(p)
if not sim_rows:
    raise SystemExit("시뮬레이션 결과가 하나도 없습니다.")
sim = pd.concat(sim_rows, ignore_index=True)
HOURS = sorted(sim["hour"].unique())
sim_succ = sim[sim["status"] == 1].copy()
sim_succ["waiting_time"] = sim_succ["end"] - sim_succ["start"]
sim_succ = sim_succ.dropna(subset=["waiting_time"])

# ---------- 실측 (같은 날짜) ----------
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


# ---------- 일 단위 매칭 지표 ----------
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
         f"- daily failures vs 하한(배차 전 취소): R2 = {r2(daily['real_cancel_predisp'], daily['sim_failures']):.3f}, "
         f"rho = {spearmanr(daily['real_cancel_predisp'], daily['sim_failures'])[0]:.3f}",
         f"- daily failures vs 상한(30분 미배차): R2 = {r2(daily['real_nodisp30'], daily['sim_failures']):.3f}, "
         f"rho = {spearmanr(daily['real_nodisp30'], daily['sim_failures'])[0]:.3f}",
         f"- hourly profile rho (run별 평균): 하한 {daily['rho_hourly_lo'].mean():.3f} / 상한 {daily['rho_hourly_hi'].mean():.3f}",
         f"- daily wait mean: sim {daily['sim_wait_mean'].mean():.1f} vs real {daily['real_wait_mean'].mean():.1f} "
         f"(paired diff {(daily['sim_wait_mean'] - daily['real_wait_mean']).mean():+.1f}분)"]
(OUT / f"daymatched_{args.scenario}_metrics.md").write_text("\n".join(lines), encoding="utf-8")

# ---------- 그림 ----------
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
print("\n저장:", OUT, "(daymatched_*)")
