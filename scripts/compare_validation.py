"""Empirical validation: baseline 시뮬레이션 vs 실측 (2023, 2024).

정의를 맞춘 비교 (시뮬 규칙: 30분 내 미배차 = 실패, 대기 = 요청→승차):
- 실패 대응치  (a) real_cancel_predisp : 배차 전 접수취소   (기존 논문 그림의 정의)
               (b) real_nodisp30      : 배차까지 30분 초과 또는 미배차 (시뮬 규칙과 동일 정의)
- 대기 대응치  30분 내 배차된 완료 운행의 (승차 - 예정/희망) — 시뮬이 서비스하는 모집단과 일치

비교 모집단: 각 연도 수요 상위 10일(서울 내 O/D, 예정/희망시각 기준), 06~24시.
시뮬레이션: result/baseline 10-run (2025 입력 재현판, ETA ON — Gate A 통과 세대).

산출물 → {OUT}/  (README는 별도 작성)
- tidy CSV: real_hourly_{yr}.csv, real_waiting_cond_{yr}.csv, sim_hourly.csv
- metrics.md : KS, R², Spearman, 총량 비교
- figures: fig_waiting_hourly_{yr}.png, fig_failure_scatter_{yr}.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = Path("/Users/jihoyeo/Library/CloudStorage/OneDrive-개인/research/장애인콜택시/paper/TR_A/revision/analysis/10_empirical_validation")
OUT.mkdir(parents=True, exist_ok=True)

HOURS = list(range(6, 24))
N_DAYS = 10
SEOUL_GU = ['종로구','중구','용산구','성동구','광진구','동대문구','중랑구','성북구','강북구',
            '도봉구','노원구','은평구','서대문구','마포구','양천구','강서구','구로구','금천구',
            '영등포구','동작구','관악구','서초구','강남구','송파구','강동구']


def prep_real(year):
    """연도별 실측: (일×시간대 집계, 조건부 대기시간 rows)"""
    if year == 2023:
        df = pd.read_parquet(PROJECT_ROOT / "data/raw_data_2023.parquet")
        df = df[df['출발지구'].isin(SEOUL_GU) & df['목적지구'].isin(SEOUL_GU) & df['예정일시'].notna()]
        want, disp, ride = df['예정일시'], df['배차일시'], df['승차일시']
    else:
        df = pd.read_parquet(PROJECT_ROOT / "data/(서울시)특별교통수단/서울_2024_로우데이터.parquet")
        df = df[(df['접수일시'] >= '2024-05-01') & (df['접수일시'] < '2024-07-01')]
        df = df[(df['출발시'] == '서울특별시') & (df['목적시'] == '서울특별시') & df['희망일시'].notna()]
        want, disp, ride = df['희망일시'], df['배차시간'], df['탑승시간']
        df = df.assign(취소일시=pd.NaT)  # 2024 원본은 탑승유무로 취소 식별
    df = df.assign(want=want, disp=disp, ride=ride)
    df['hour'] = df['want'].dt.hour
    df = df[df['hour'].isin(HOURS)]
    df['date'] = df['want'].dt.date

    top = df.groupby('date').size().sort_values(ascending=False).head(N_DAYS).index
    df = df[df['date'].isin(top)]

    df['disp_delay'] = (df['disp'] - df['want']) / pd.Timedelta(minutes=1)
    if year == 2023:
        cancel_predisp = df['취소일시'].notna() & df['disp'].isna()
    else:
        cancel_predisp = (df['탑승유무'] == '접수취소') & df['disp'].isna()
    df['cancel_predisp'] = cancel_predisp
    df['nodisp30'] = df['disp'].isna() | (df['disp_delay'] > 30)

    hourly = df.groupby(['date', 'hour']).agg(
        request_count=('want', 'size'),
        cancel_predisp=('cancel_predisp', 'sum'),
        nodisp30=('nodisp30', 'sum'),
    ).reset_index()

    served = df[df['ride'].notna() & (df['disp_delay'] <= 30)].copy()
    served['waiting_time'] = (served['ride'] - served['want']) / pd.Timedelta(minutes=1)
    served = served[(served['waiting_time'] >= 0) & (served['waiting_time'] <= 240)]
    waiting = served[['date', 'hour', 'waiting_time']]
    return hourly, waiting


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


sim_w = pd.read_csv(PROJECT_ROOT / "data/verification/sim_baseline_waiting.csv")
sim_f = pd.read_csv(PROJECT_ROOT / "data/verification/sim_baseline_failure.csv")
sim_w = sim_w[sim_w['hour'].isin(HOURS)]
sim_hourly = sim_f[sim_f['hour'].isin(HOURS)].groupby('hour').agg(
    sim_request=('request_count', 'mean'),
    sim_failure=('failure_hour_request', 'mean'),
).reindex(HOURS).reset_index()
sim_wait_hour = sim_w.groupby('hour')['waiting_time'].agg(['mean', 'median']).reindex(HOURS)
sim_hourly.to_csv(OUT / "sim_hourly.csv", index=False)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
lines = ["# Empirical validation metrics", "",
         "Sim: result/baseline 10 runs (2025-input reproduction, ETA ON), fail rule = no dispatch within 30 min.",
         f"Real: top-{N_DAYS} demand days per year, Seoul-internal O/D, 06:00-24:00.", ""]

for year in (2023, 2024):
    hourly, waiting = prep_real(year)
    hourly.to_csv(OUT / f"real_hourly_{year}.csv", index=False)
    waiting.to_csv(OUT / f"real_waiting_cond_{year}.csv", index=False)

    real_hour = hourly.groupby('hour').agg(
        real_request=('request_count', 'mean'),
        real_cancel_predisp=('cancel_predisp', 'mean'),
        real_nodisp30=('nodisp30', 'mean'),
    ).reindex(HOURS)
    merged = sim_hourly.set_index('hour').join(real_hour)

    ks = ks_2samp(sim_w['waiting_time'], waiting['waiting_time'])
    rw, sw = waiting['waiting_time'], sim_w['waiting_time']

    hour_tbl = pd.DataFrame({
        'sim_mean': sim_wait_hour['mean'],
        'real_mean': waiting.groupby('hour')['waiting_time'].mean().reindex(HOURS),
    }).round(1)
    hour_tbl['diff'] = (hour_tbl['sim_mean'] - hour_tbl['real_mean']).round(1)
    day = hour_tbl.loc[8:17]
    day_gap = (day['sim_mean'] - day['real_mean']).abs().mean()

    m = {}
    for defn in ('real_cancel_predisp', 'real_nodisp30'):
        m[defn] = (r2(merged[defn].values, merged['sim_failure'].values),
                   spearmanr(merged[defn], merged['sim_failure'])[0])

    lines += [f"## vs {year}", "",
              f"- waiting (conditional, dispatch<=30): real mean {rw.mean():.1f} / median {rw.median():.1f} / p95 {rw.quantile(.95):.1f}"
              f" | sim mean {sw.mean():.1f} / median {sw.median():.1f} / p95 {sw.quantile(.95):.1f}",
              f"- waiting KS statistic = {ks.statistic:.3f} (p={ks.pvalue:.2e}, n_real={len(rw):,}, n_sim={len(sw):,})",
              f"- hourly waiting mean correlation (Pearson) = {np.corrcoef(sim_wait_hour['mean'], waiting.groupby('hour')['waiting_time'].mean().reindex(HOURS))[0,1]:.3f}",
              f"- failure vs pre-dispatch cancels: R2 = {m['real_cancel_predisp'][0]:.3f}, rho = {m['real_cancel_predisp'][1]:.3f}"
              f" (totals: sim {merged['sim_failure'].sum():.0f}/day vs real {merged['real_cancel_predisp'].sum():.0f}/day)",
              f"- failure vs no-dispatch-within-30min: R2 = {m['real_nodisp30'][0]:.3f}, rho = {m['real_nodisp30'][1]:.3f}"
              f" (totals: sim {merged['sim_failure'].sum():.0f}/day vs real {merged['real_nodisp30'].sum():.0f}/day)",
              f"- request volume: sim {merged['sim_request'].sum():.0f}/day vs real {merged['real_request'].sum():.0f}/day",
              f"- daytime (08-17h) waiting: mean absolute hourly gap = {day_gap:.1f} min; evening (18-23h) sim underestimates by ~20 min",
              "", "hourly waiting means (sim vs real):", "", "```", hour_tbl.to_string(), "```", ""]

    # Figure 1: hourly waiting time (mean lines + IQR band)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rh = waiting.groupby('hour')['waiting_time']
    ax.plot(HOURS, rh.mean().reindex(HOURS), '-o', color='#4B6B8A', label=f'Observed {year} (mean)')
    ax.fill_between(HOURS, rh.quantile(.25).reindex(HOURS), rh.quantile(.75).reindex(HOURS),
                    color='#4B6B8A', alpha=.15, label=f'Observed {year} (IQR)')
    sh = sim_w.groupby('hour')['waiting_time']
    ax.plot(HOURS, sh.mean().reindex(HOURS), '-s', color='gray', label='Simulated (mean)')
    ax.fill_between(HOURS, sh.quantile(.25).reindex(HOURS), sh.quantile(.75).reindex(HOURS),
                    color='gray', alpha=.15, label='Simulated (IQR)')
    ax.set_xlabel('Time of Day'); ax.set_ylabel('Waiting Time (min)')
    ax.set_xticks(HOURS); ax.set_xticklabels([f'{h:02d}' for h in HOURS])
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(OUT / f"fig_waiting_hourly_{year}.png", dpi=200); plt.close(fig)

    # Figure 2: hourly failure scatter (both definitions)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, defn, label in zip(axes, ('real_cancel_predisp', 'real_nodisp30'),
                               ('Pre-dispatch cancellations', 'No dispatch within 30 min')):
        x, y = merged[defn].values, merged['sim_failure'].values
        lim = max(x.max(), y.max()) * 1.1
        ax.scatter(x, y, color='gray', alpha=.7)
        ax.plot([0, lim], [0, lim], '--', color='#999999', label='Ideal fit')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f'Observed {year}: {label}')
        ax.text(.05, .95, f'$R^2$ = {m[defn][0]:.3f}\n$\\rho_s$ = {m[defn][1]:.3f}',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=.8))
    axes[0].set_ylabel('Simulated failures per hour')
    fig.tight_layout(); fig.savefig(OUT / f"fig_failure_scatter_{year}.png", dpi=200); plt.close(fig)

(OUT / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("저장 위치:", OUT)
