"""2024 원본에서 검증용 실측 데이터 추출.

passenger_2024 생성에 쓴 10일(days.csv)과 동일한 날짜에 대해:
1. real_waiting_time_2024.csv : 완료 운행별 대기시간 (탑승시간 - 희망일시, 분)
   - 콜구분(바로콜/전일접수/정기접수) 포함 → 그림 그릴 때 바로콜만 필터 가능
2. real_failure_2024.csv : 날짜 x 시간대별 취소(실패) 건수
   - 배차 전/후 취소 구분 포함 → 시뮬레이션 미배차와 비교할 땐 배차 전 취소가 대응

시뮬레이션 대기시간(승객 등장 ride_time = 희망일시 기준)과 정의를 맞추기 위해
대기시간은 접수일시가 아닌 희망일시 기준으로 계산한다 (바로콜은 둘이 동일).
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "data/(서울시)특별교통수단/서울_2024_로우데이터.parquet"
DAYS = PROJECT_ROOT / "data/simulation-agent-data/passenger_2024/days.csv"
OUT_DIR = PROJECT_ROOT / "data/verification"

days = pd.read_csv(DAYS)
target_dates = pd.to_datetime(days["date"]).dt.date.tolist()

df = pd.read_parquet(SRC)
df = df[(df["출발시"] == "서울특별시") & (df["목적시"] == "서울특별시")]
df = df[df["희망일시"].notna()]
df["date"] = df["희망일시"].dt.date
df = df[df["date"].isin(target_dates)]
df["hour"] = df["희망일시"].dt.hour

# 1. 완료 운행 대기시간
comp = df[df["탑승유무"] == "탑승완료"].copy()
comp["waiting_time"] = (comp["탑승시간"] - comp["희망일시"]) / pd.Timedelta(minutes=1)
waiting = comp[["date", "hour", "waiting_time", "콜구분", "차량구분", "휠체어"]].rename(
    columns={"콜구분": "call_type", "차량구분": "vehicle_type", "휠체어": "wheelchair"}
)
neg = (waiting["waiting_time"] < 0).sum()
if neg:
    print(f"[주의] 음수 대기시간 {neg}건 (예약 조기 픽업) — 그대로 저장, 사용 시 필터 판단")

# 2. 취소(실패) 건수: 날짜 x 시간대, 배차 전/후 구분
canc = df[df["탑승유무"] == "접수취소"].copy()
canc["cancel_stage"] = canc["배차시간"].notna().map({True: "after_dispatch", False: "before_dispatch"})
failure = (
    canc.groupby(["date", "hour", "cancel_stage"]).size().unstack(fill_value=0).reset_index()
)
failure["total_cancel"] = failure.get("before_dispatch", 0) + failure.get("after_dispatch", 0)

OUT_DIR.mkdir(parents=True, exist_ok=True)
waiting.to_csv(OUT_DIR / "real_waiting_time_2024.csv", index=False)
failure.to_csv(OUT_DIR / "real_failure_2024.csv", index=False)

print(f"대상 날짜 {len(target_dates)}일, 완료 {len(comp):,}건 / 취소 {len(canc):,}건")
print(f"저장: {OUT_DIR}/real_waiting_time_2024.csv, real_failure_2024.csv")
print("\n대기시간 요약(분, 바로콜만):")
b = waiting[waiting["call_type"] == "바로콜"]["waiting_time"]
print(b.describe().round(1).to_string())
print("\n시간대별 취소 합계(10일):")
print(failure.groupby("hour")["total_cancel"].sum().to_string())
