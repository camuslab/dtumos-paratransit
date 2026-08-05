# -*- coding: utf-8 -*-
"""Parallel campaign launcher: reads a jobs file (one 'SCENARIO REALIZATION' per line),
runs N jobs concurrently, assigns OSRM ports (5001/5002/5003) round-robin.
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

print(f"{len(jobs)} jobs total, {args.concurrency} concurrent, ports {PORTS}")
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
print(f"Done: {len(jobs)-fails}/{len(jobs)} succeeded, {(time.time()-t0)/3600:.1f} h total")
sys.exit(1 if fails else 0)
