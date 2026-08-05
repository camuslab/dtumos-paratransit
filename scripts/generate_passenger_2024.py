"""Generate simulation passenger data from the 2024 raw data (with coordinates).

Same approach as the existing data/simulation-agent-data/passenger/ (2023-based):
- Top-10 demand days (weekdays); save all requests including cancellations, one file per day
- ride_time = minutes since midnight of the desired time (existing files correlate 0.997 with the scheduled-time distribution)
- Seoul-origin & Seoul-destination trips only (existing files' bbox is limited to Seoul)

Improvements over the existing approach:
- Coordinates: random points within admin dong → actual pickup/drop-off coordinates
- type (wheelchair): random 23/77 assignment → observed wheelchair column

Output: data/simulation-agent-data/passenger_2024/passenger_{0..9}.parquet
        + days.csv (target date and count per file)
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "data/seoul_paratransit_raw/seoul_2024_raw.parquet"
OUT_DIR = PROJECT_ROOT / "data/simulation-agent-data/passenger_2024"
N_DAYS = 10

df = pd.read_parquet(SRC)
df = df[(df["접수일시"] >= "2024-05-01") & (df["접수일시"] < "2024-07-01")]
print(f"All requests 2024-05~06: {len(df):,}")

df = df[(df["출발시"] == "서울특별시") & (df["목적시"] == "서울특별시")]
print(f"After Seoul-internal O/D filter: {len(df):,}")

valid_coord = (
    df[["출발위도", "출발경도", "목적위도", "목적경도"]].notna().all(axis=1)
    & (df["출발위도"] > 37) & (df["목적위도"] > 37)
)
df = df[valid_coord & df["희망일시"].notna()]
print(f"After valid coordinate/desired-time filter: {len(df):,}")

df["희망일자"] = df["희망일시"].dt.date
top_days = df.groupby("희망일자").size().sort_values(ascending=False).head(N_DAYS)
print("\nSelected dates (top demand by desired time):")
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
print(f"\nSaved: {OUT_DIR}")
print(meta.to_string(index=False))
