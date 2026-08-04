"""2024 원본(좌표 포함)으로 시뮬레이션용 passenger 데이터 생성.

기존 data/simulation-agent-data/passenger/ (2023 기반)와 동일한 방식:
- 수요 상위 10일(평일), 취소 건 포함한 전체 접수를 하루 단위 파일로 저장
- ride_time = 희망일시의 자정 기준 경과 분 (기존 파일이 예정일시 분포와 상관 0.997)
- 서울 출발 & 서울 도착 통행만 포함 (기존 파일 bbox가 서울 시내로 한정됨)

기존 방식 대비 개선점:
- 좌표: 행정동 내 랜덤 생성 → 실제 승하차 좌표
- type(휠체어): 랜덤 23/77 부여 → 실측 휠체어 컬럼 사용

출력: data/simulation-agent-data/passenger_2024/passenger_{0..9}.parquet
      + days.csv (파일별 대상 날짜와 건수)
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "data/(서울시)특별교통수단/서울_2024_로우데이터.parquet"
OUT_DIR = PROJECT_ROOT / "data/simulation-agent-data/passenger_2024"
N_DAYS = 10

df = pd.read_parquet(SRC)
df = df[(df["접수일시"] >= "2024-05-01") & (df["접수일시"] < "2024-07-01")]
print(f"2024-05~06 전체 접수: {len(df):,}")

df = df[(df["출발시"] == "서울특별시") & (df["목적시"] == "서울특별시")]
print(f"서울 내 O/D 필터 후: {len(df):,}")

valid_coord = (
    df[["출발위도", "출발경도", "목적위도", "목적경도"]].notna().all(axis=1)
    & (df["출발위도"] > 37) & (df["목적위도"] > 37)
)
df = df[valid_coord & df["희망일시"].notna()]
print(f"좌표/희망일시 유효 필터 후: {len(df):,}")

df["희망일자"] = df["희망일시"].dt.date
top_days = df.groupby("희망일자").size().sort_values(ascending=False).head(N_DAYS)
print("\n선정된 날짜 (희망일시 기준 수요 상위):")
print(top_days.to_string())

OUT_DIR.mkdir(parents=True, exist_ok=True)
meta = []
for i, (day, n) in enumerate(top_days.items()):
    d = df[df["희망일자"] == day].sort_values("희망일시").reset_index(drop=True)
    midnight = pd.Timestamp(day)
    out = pd.DataFrame({
        "ID": d.index,
        "ride_time": ((d["희망일시"] - midnight) / pd.Timedelta(minutes=1)).astype(int),
        "ride_lat": d["출발위도"],
        "ride_lon": d["출발경도"],
        "alight_lat": d["목적위도"],
        "alight_lon": d["목적경도"],
        "dispatch_time": 0,
        "type": d["휠체어"].astype(int),
    })
    out.to_parquet(OUT_DIR / f"passenger_{i}.parquet", index=False)
    meta.append({
        "file": f"passenger_{i}.parquet",
        "date": str(day),
        "weekday": ["월", "화", "수", "목", "금", "토", "일"][pd.Timestamp(day).dayofweek],
        "n_passengers": len(out),
        "n_cancelled_in_raw": (d["탑승유무"] == "접수취소").sum(),
        "wheelchair_ratio": round(out["type"].mean(), 3),
    })

meta = pd.DataFrame(meta)
meta.to_csv(OUT_DIR / "days.csv", index=False)
print(f"\n저장 완료: {OUT_DIR}")
print(meta.to_string(index=False))
