"""Extract simulation validation data from the baseline 10-run.

From result/baseline/simulation_{1..10}/passenger_marker.json:
- status 1 = success: timestamp = [request time, pickup time] (min) → waiting time = end - start
- status 0 = failure: timestamp = [request time, failure-decision time]

Outputs (data/verification/):
1. sim_baseline_waiting.csv : run, hour (by request time), waiting_time (min)
2. sim_baseline_failure.csv : request/failure counts per run x hour
   - failure_hour_request : failures counted by request time
   - failure_hour_fail    : counted by failure-decision time (method of the original paper figure)
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
print(f"runs: {counts['run'].nunique()}, avg requests/run {n_req.mean():.0f}, avg failure rate {(n_fail/n_req).mean()*100:.2f}%")
print(f"Time to failure decision (min): min {fd.min():.1f} / median {fd.median():.1f} / max {fd.max():.1f}")
print(f"Waiting time of served requests (min): mean {waiting['waiting_time'].mean():.1f} / median {waiting['waiting_time'].median():.1f}")
print(f"Saved: {OUT_DIR}/sim_baseline_waiting.csv, sim_baseline_failure.csv")
