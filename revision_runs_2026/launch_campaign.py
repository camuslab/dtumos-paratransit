# -*- coding: utf-8 -*-
"""병렬 캠페인 런처: jobs 파일(한 줄에 'SCENARIO REALIZATION')을 읽어
N개 동시 실행, OSRM 포트(5001/5002/5003) 라운드로빈 배정.
usage: launch_campaign.py --jobs jobs_phase1.txt --concurrency 6
"""
import argparse, subprocess, os, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "..", "venv312", "bin", "python")
PORTS = ["5001", "5002", "5003"]

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)
ap.add_argument("--concurrency", type=int, default=6)
args = ap.parse_args()

jobs = []
with open(args.jobs) as f:
    for ln in f:
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            parts = ln.split()
            jobs.append((parts[0], int(parts[1]), parts[2:]))

print(f"총 {len(jobs)}개 작업, 동시 {args.concurrency}개, 포트 {PORTS}")
t0 = time.time()

def run(idx_job):
    idx, (scen, r, extra) = idx_job
    env = dict(os.environ, OSRM_PORT=PORTS[idx % len(PORTS)])
    suffix = "".join(t.replace("--", "_").replace(".", "p") for t in extra)
    log = os.path.join(HERE, "logs", f"{scen}_r{r}{suffix}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as lf:
        rc = subprocess.call([PY, os.path.join(HERE, "run_scenario.py"),
                              "--scenario", scen, "--realization", str(r)] + extra,
                             stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    print(f"[{time.time()-t0:7.0f}s] {scen} r{r}: {status}", flush=True)
    return rc

with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
    rcs = list(ex.map(run, enumerate(jobs)))

fails = sum(1 for rc in rcs if rc != 0)
print(f"완료: {len(jobs)-fails}/{len(jobs)} 성공, 총 {(time.time()-t0)/3600:.1f}시간")
sys.exit(1 if fails else 0)
