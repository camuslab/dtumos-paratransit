"""baseline 시뮬레이션 10-run에서 검증용 시뮬 데이터 추출.

result/baseline/simulation_{1..10}/passenger_marker.json 기준:
- status 1 = 성공: timestamp = [요청시각, 승차시각] (분) → 대기시간 = end - start
- status 0 = 실패: timestamp = [요청시각, 실패판정시각]

출력 (data/verification/):
1. sim_baseline_waiting.csv : run, hour(요청시각 기준), waiting_time(분)
2. sim_baseline_failure.csv : run x hour별 요청/실패 건수
   - failure_hour_request : 요청시각 기준 실패 집계
   - failure_hour_fail    : 실패판정시각 기준 집계 (기존 논문 그림 방식)
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "result/baseline"
OUT_DIR = PROJECT_ROOT / "data/verification"

waiting_rows, count_rows = [], []
fail_durations = []
for sim_dir in sorted(BASE.glob("simulation_*"), key=lambda p: int(p.name.split("_")[1])):
    run = int(sim_dir.name.split("_")[1])
    p = pd.read_json(sim_dir / "passenger_marker.json")
    p["start"] = p["timestamp"].str[0]
    p["end"] = p["timestamp"].str[-1]
    p["hour"] = (p["start"] // 60).astype(int)

    succ = p[p["status"] == 1].copy()
    succ["waiting_time"] = succ["end"] - succ["start"]
    waiting_rows.append(pd.DataFrame({
        "run": run, "hour": succ["hour"], "waiting_time": succ["waiting_time"]}))

    fail = p[p["status"] == 0].copy()
    fail_durations.append(fail["end"] - fail["start"])
    fail["hour_fail"] = (fail["end"] // 60).astype(int)

    req_cnt = p.groupby("hour").size()
    fail_req_cnt = fail.groupby("hour").size()
    fail_fail_cnt = fail.groupby("hour_fail").size()
    hours = sorted(set(req_cnt.index) | set(fail_fail_cnt.index))
    for h in hours:
        count_rows.append({
            "run": run, "hour": h,
            "request_count": int(req_cnt.get(h, 0)),
            "failure_hour_request": int(fail_req_cnt.get(h, 0)),
            "failure_hour_fail": int(fail_fail_cnt.get(h, 0)),
        })

waiting = pd.concat(waiting_rows, ignore_index=True)
counts = pd.DataFrame(count_rows)
OUT_DIR.mkdir(parents=True, exist_ok=True)
waiting.to_csv(OUT_DIR / "sim_baseline_waiting.csv", index=False)
counts.to_csv(OUT_DIR / "sim_baseline_failure.csv", index=False)

fd = pd.concat(fail_durations)
n_req = counts.groupby("run")["request_count"].sum()
n_fail = counts.groupby("run")["failure_hour_request"].sum()
print(f"runs: {counts['run'].nunique()}, 요청/런 평균 {n_req.mean():.0f}, 실패율 평균 {(n_fail/n_req).mean()*100:.2f}%")
print(f"실패 판정까지 시간(분): min {fd.min():.1f} / median {fd.median():.1f} / max {fd.max():.1f}")
print(f"성공 대기시간(분): mean {waiting['waiting_time'].mean():.1f} / median {waiting['waiting_time'].median():.1f}")
print(f"저장: {OUT_DIR}/sim_baseline_waiting.csv, sim_baseline_failure.csv")
