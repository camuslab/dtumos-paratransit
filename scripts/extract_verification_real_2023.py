"""Extract observed validation data from raw_data_2023 (for pooled comparison with existing simulation results).

The existing passenger_0~9 are synthetic days resampled from the 2023-05~06
demand distribution, so a 1:1 match to specific dates is impossible → store the
observed data day by day for the full period, and at comparison time filter to
e.g. top-demand weekdays and compare pooled distributions / hourly means.

1. real_waiting_time_2023.csv : waiting time per completed trip (pickup time - scheduled time, min)
   - immediate: whether receipt time == scheduled time (under 1 min apart) — proxy for immediate calls
2. real_failure_2023.csv : cancellation counts by date x hour (split pre-/post-dispatch)

To match the simulation's waiting-time definition (ride_time = appearance at the
scheduled time), waiting time is based on the scheduled time. Seoul-internal O/D
only (same condition as the passenger files).
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data/verification"

SEOUL_GU = ['종로구','중구','용산구','성동구','광진구','동대문구','중랑구','성북구','강북구',
            '도봉구','노원구','은평구','서대문구','마포구','양천구','강서구','구로구','금천구',
            '영등포구','동작구','관악구','서초구','강남구','송파구','강동구']

df = pd.read_parquet(PROJECT_ROOT / "data/raw_data_2023.parquet")
df = df[df['출발지구'].isin(SEOUL_GU) & df['목적지구'].isin(SEOUL_GU)]
df = df[df['예정일시'].notna()]
df['date'] = df['예정일시'].dt.date
df['hour'] = df['예정일시'].dt.hour
df['immediate'] = (df['예정일시'] - df['접수일시']).abs() < pd.Timedelta(minutes=1)
df['weekday'] = pd.to_datetime(df['date'].astype(str)).dt.dayofweek  # 0=Mon

# 1. Completed-trip waiting times
comp = df[df['승차일시'].notna()].copy()
comp['waiting_time'] = (comp['승차일시'] - comp['예정일시']) / pd.Timedelta(minutes=1)
waiting = comp[['date', 'hour', 'waiting_time', 'immediate', '차량구분', 'weekday']].rename(
    columns={'차량구분': 'vehicle_type'})

# 2. Cancellation counts (split pre-/post-dispatch)
canc = df[df['취소일시'].notna() & df['승차일시'].isna()].copy()
canc['cancel_stage'] = canc['배차일시'].notna().map({True: 'after_dispatch', False: 'before_dispatch'})
failure = canc.groupby(['date', 'hour', 'cancel_stage']).size().unstack(fill_value=0).reset_index()
failure['total_cancel'] = failure.get('before_dispatch', 0) + failure.get('after_dispatch', 0)

OUT_DIR.mkdir(parents=True, exist_ok=True)
waiting.to_csv(OUT_DIR / "real_waiting_time_2023.csv", index=False)
failure.to_csv(OUT_DIR / "real_failure_2023.csv", index=False)

print(f"Seoul-internal O/D {len(df):,} rows → completed {len(comp):,} / cancelled {len(canc):,}")
print(f"Saved: {OUT_DIR}/real_waiting_time_2023.csv, real_failure_2023.csv")
print("\nWaiting time summary (min, immediate calls, completed):")
print(waiting[waiting['immediate']]['waiting_time'].describe().round(1).to_string())
print("\nCancellations pre-/post-dispatch:", canc['cancel_stage'].value_counts().to_dict())
