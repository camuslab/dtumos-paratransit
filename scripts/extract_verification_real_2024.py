"""Extract observed validation data from the 2024 raw data.

For the same 10 dates (days.csv) used to build passenger_2024:
1. real_waiting_time_2024.csv : waiting time per completed trip (boarding time - desired time, min)
   - includes call type (immediate / day-ahead / subscription) → immediate calls can be filtered when plotting
2. real_failure_2024.csv : cancellation (failure) counts by date x hour
   - split into pre-/post-dispatch cancellations → pre-dispatch cancellations correspond to simulated no-dispatch

To match the simulation's waiting-time definition (passenger appears at ride_time
= desired time), waiting time is computed from the desired time rather than the
receipt time (identical for immediate calls).
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "data/seoul_paratransit_raw/seoul_2024_raw.parquet"
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

# 1. Completed-trip waiting times
comp = df[df["탑승유무"] == "탑승완료"].copy()
comp["waiting_time"] = (comp["탑승시간"] - comp["희망일시"]) / pd.Timedelta(minutes=1)
waiting = comp[["date", "hour", "waiting_time", "콜구분", "차량구분", "휠체어"]].rename(
    columns={"콜구분": "call_type", "차량구분": "vehicle_type", "휠체어": "wheelchair"}
)
neg = (waiting["waiting_time"] < 0).sum()
if neg:
    print(f"[note] {neg} negative waiting times (early pickups for reservations) — saved as-is, filter at use time if needed")

# 2. Cancellation (failure) counts: date x hour, split pre-/post-dispatch
canc = df[df["탑승유무"] == "접수취소"].copy()
canc["cancel_stage"] = canc["배차시간"].notna().map({True: "after_dispatch", False: "before_dispatch"})
failure = (
    canc.groupby(["date", "hour", "cancel_stage"]).size().unstack(fill_value=0).reset_index()
)
failure["total_cancel"] = failure.get("before_dispatch", 0) + failure.get("after_dispatch", 0)

OUT_DIR.mkdir(parents=True, exist_ok=True)
waiting.to_csv(OUT_DIR / "real_waiting_time_2024.csv", index=False)
failure.to_csv(OUT_DIR / "real_failure_2024.csv", index=False)

print(f"{len(target_dates)} target dates, completed {len(comp):,} / cancelled {len(canc):,}")
print(f"Saved: {OUT_DIR}/real_waiting_time_2024.csv, real_failure_2024.csv")
print("\nWaiting time summary (min, immediate calls only):")
b = waiting[waiting["call_type"] == "바로콜"]["waiting_time"]
print(b.describe().round(1).to_string())
print("\nCancellation totals by hour (10 days):")
print(failure.groupby("hour")["total_cancel"].sum().to_string())
