"""raw_data_2023에서 검증용 실측 데이터 추출 (기존 시뮬레이션 결과와 pooled 비교용).

기존 passenger_0~9는 2023-05~06 수요 분포에서 재표집된 가상의 날이라
특정 날짜와 1:1 매칭이 불가 → 실측은 전체 기간을 일 단위로 저장해두고,
비교 시 상위 수요 평일 등으로 필터해 pooled 분포/시간대 평균으로 비교한다.

1. real_waiting_time_2023.csv : 완료 운행별 대기시간 (승차일시 - 예정일시, 분)
   - immediate: 접수일시==예정일시(1분 미만 차이) 여부 — 바로콜 프록시
2. real_failure_2023.csv : 날짜 x 시간대별 취소 건수 (배차 전/후 구분)

시뮬레이션 대기시간(ride_time = 예정일시 기준 등장)과 정의를 맞춰
대기시간은 예정일시 기준. 서울 내 O/D만 포함 (passenger 파일과 동일 조건).
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
df['weekday'] = pd.to_datetime(df['date'].astype(str)).dt.dayofweek  # 0=월

# 1. 완료 운행 대기시간
comp = df[df['승차일시'].notna()].copy()
comp['waiting_time'] = (comp['승차일시'] - comp['예정일시']) / pd.Timedelta(minutes=1)
waiting = comp[['date', 'hour', 'waiting_time', 'immediate', '차량구분', 'weekday']].rename(
    columns={'차량구분': 'vehicle_type'})

# 2. 취소 건수 (배차 전/후 구분)
canc = df[df['취소일시'].notna() & df['승차일시'].isna()].copy()
canc['cancel_stage'] = canc['배차일시'].notna().map({True: 'after_dispatch', False: 'before_dispatch'})
failure = canc.groupby(['date', 'hour', 'cancel_stage']).size().unstack(fill_value=0).reset_index()
failure['total_cancel'] = failure.get('before_dispatch', 0) + failure.get('after_dispatch', 0)

OUT_DIR.mkdir(parents=True, exist_ok=True)
waiting.to_csv(OUT_DIR / "real_waiting_time_2023.csv", index=False)
failure.to_csv(OUT_DIR / "real_failure_2023.csv", index=False)

print(f"서울 내 O/D {len(df):,}건 → 완료 {len(comp):,} / 취소 {len(canc):,}")
print(f"저장: {OUT_DIR}/real_waiting_time_2023.csv, real_failure_2023.csv")
print("\n대기시간 요약(분, 즉시콜·완료):")
print(waiting[waiting['immediate']]['waiting_time'].describe().round(1).to_string())
print("\n취소 배차 전/후:", canc['cancel_stage'].value_counts().to_dict())
