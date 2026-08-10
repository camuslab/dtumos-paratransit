# -*- coding: utf-8 -*-
"""Campaign scenario runner v2: paper-version inputs + ETA ON.
scenario codes:
  base/B/C/D/E/BC/BD/CD  — 2^3 combinations (B=optimization, C=all-day operation, D=wheelchair-only)
  greedy                 — batched nearest-greedy dispatch (waiting time descending)
  F<n>                   — fleet grid (n vehicles; fleet_files/vehicle_<n>.csv)
  Clite<h>               — reassign vehicles with work_end<=h to h~24h (naive)
  Cbudget                — preserve Base total driver-hours, demand-shaped reassignment of start times only
"""
import os, sys, time, argparse, json, fcntl
import multiprocess
multiprocess.set_start_method("fork", force=True)
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True)
ap.add_argument("--realization", type=int, required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--fail_time", type=int, default=30)
ap.add_argument("--dwell", type=int, default=10)
ap.add_argument("--supply", type=float, default=1.0)
ap.add_argument("--demand", default="1.0")  # 0.9 / 1.1 / 1.2 / surge / od
ap.add_argument("--uptake", type=float, default=1.0)   # D voucher uptake rate
ap.add_argument("--dwell_sigma", type=float, default=0.0)
ap.add_argument("--year", type=int, default=2023)
ap.add_argument("--fleet_year", type=int, default=0)  # 0 = same as year; for the 2x2 decomposition
ap.add_argument("--sched_file", default="")  # schedule_opt CSV for Cs codes ([B]Cs[D])
ap.add_argument("--timing", action="store_true")
args = ap.parse_args()
scen = args.scenario
i = args.realization

import pandas as pd
import numpy as np
import joblib
from module.simulator import Simulator
from module.simulator_helper import get_preprocessed_seoul_data, base_configs

ARCH = "/Users/jihoyeo/research/dtumos-paratransit/_simulation_archive_20260730"
if args.year == 2024:
    passengers = pd.read_csv(f"{HERE}/passenger24/passenger24_{i}.csv")
else:
    passengers = pd.read_csv(f"{ARCH}/05_demand_generator/passenger_output_2025_paper/passenger_{i}.csv")

if args.demand != "1.0":
    donor_path = (f"{HERE}/passenger24/passenger24_{(i+1) % 10}.csv" if args.year == 2024
                  else f"{ARCH}/05_demand_generator/passenger_output_2025_paper/passenger_{(i+1) % 10}.csv")
    donor = pd.read_csv(donor_path)
    if args.demand == "od":  # bootstrap of observed daily totals (factors exclude holidays)
        _factors = json.load(open(f"{HERE}/od_factors.json"))
        _od_mode = True
        args.demand = str(_factors[i % len(_factors)])
    if args.demand == "surge":  # 14-15h demand ×1.5
        pool = donor[(donor.ride_time >= 840) & (donor.ride_time < 900)]
        extra = pool.sample(frac=0.5, random_state=3000 + i)
    else:
        f = float(args.demand)
        if f < 1.0:  # thinning
            passengers = passengers.sample(frac=f, random_state=2000 + i)
            extra = passengers.iloc[0:0]
        else:  # overlay an independent realization (Poisson superposition)
            extra = donor.sample(frac=f - 1.0, random_state=3000 + i)
    if len(extra):
        extra = extra.copy()
        extra["ID"] = range(int(passengers["ID"].max()) + 1, int(passengers["ID"].max()) + 1 + len(extra))
        passengers = pd.concat([passengers, extra], ignore_index=True)
    passengers = passengers.sort_values("ride_time").reset_index(drop=True)

COMBO = {"base": (0,0,0), "B": (1,0,0), "C": (0,1,0), "D": (0,0,1),
         "BC": (1,1,0), "BD": (1,0,1), "CD": (0,1,1), "E": (1,1,1)}
use_B, use_C, use_D = COMBO.get(scen, (0,0,0))

# C-targeted combination codes: [B]Ct<K>[D]  (e.g. BCt100, Ct100D, BCt100D)
import re
_m_ct = re.fullmatch(r"(B?)Ct(\d+)(D?)", scen)
K_ct = None
if _m_ct:
    use_B = bool(_m_ct.group(1))
    K_ct = int(_m_ct.group(2))
    use_D = bool(_m_ct.group(3))

# Cs codes: optimized schedule from schedule_opt/ ([B]Cs[D], vehicles from --sched_file)
_m_cs = re.fullmatch(r"(B?)Cs(D?)", scen)
if _m_cs:
    use_B = bool(_m_cs.group(1))
    use_D = bool(_m_cs.group(2))
    assert args.sched_file, "Cs scenario requires --sched_file"

dispatch_mode = "optimization" if use_B else "in_order"

_fy = args.fleet_year or args.year
_vprefix = "vehicle24" if _fy == 2024 else "vehicle"
if scen.startswith("F"):
    n = int(scen[1:])
    vehicles = pd.read_csv(f"{HERE}/fleet_files/{_vprefix}_{n}.csv")
elif _fy == 2024:
    vehicles = pd.read_csv(f"{HERE}/fleet_files/vehicle24_636.csv")
else:
    vehicles = pd.read_csv(f"{ARCH}/02_vehicle_inputs/paper_2025__613_782_870/Baseline_613__vehicle_0.csv")

if scen == "greedy":
    dispatch_mode = "batched_greedy"

if use_D:
    t1 = passengers[passengers["type"] == 1]
    t0 = passengers[passengers["type"] == 0]
    keep0 = t0.sample(frac=1 - args.uptake, random_state=4000 + i) if args.uptake < 1.0 else t0.iloc[0:0]
    passengers = pd.concat([t1, keep0]).sort_values("ride_time").reset_index(drop=True)
if use_C:
    vehicles["work_start"] = 0
    vehicles["work_end"] = 24

if scen.startswith("Clite"):
    start_h = int(scen.replace("Clite", "") or 14)
    mask = vehicles["work_end"] <= start_h
    vehicles.loc[mask, "work_start"] = start_h
    vehicles.loc[mask, "work_end"] = 24
    print(f"Clite{start_h}: {mask.sum()} vehicles reassigned")

if K_ct is not None:
    # C-targeted: add K extra 9-hour shifts in the observed bottleneck window
    # (07–16h; covers the 7 am and 14–16h failure peaks). Assign first to vehicles
    # whose own shift does not overlap 07–16h (idle); the excess is flagged as
    # requiring reserve vehicles. Added driver-hours = K×9.
    K = K_ct
    ws0, we0 = vehicles["work_start"], vehicles["work_end"]
    we_adj = we0.where(we0 > ws0, we0 + 24)
    no_ovl = vehicles[(we_adj <= 7) | (ws0 >= 16)]
    take1 = no_ovl.sample(n=min(K, len(no_ovl)), replace=False, random_state=K)
    rest = K - len(take1)
    pool = vehicles.drop(no_ovl.index)
    take2 = pool.sample(n=rest, replace=False, random_state=K + 1) if rest > 0 else pool.iloc[0:0]
    add = pd.concat([take1, take2]).copy()
    mx = int(vehicles["vehicle_id"].max())
    add["vehicle_id"] = range(mx + 1, mx + 1 + len(add))
    add["work_start"] = 7
    add["work_end"] = 16
    vehicles = pd.concat([vehicles, add], ignore_index=True)
    print(f"Ct{K}: {K} extra shifts (07-16h), {len(take1)} idle vehicles / {rest} reserve vehicles needed, +{K*9} veh-h")

if scen == "Cbudget":
    # Target Scenario C's hourly on-duty profile (10-run mean); keep each vehicle's
    # shift length and reassign only the start time (total driver-hours budget-neutral; deterministic greedy fill)
    TGT = {6:56, 7:432, 8:515, 9:273, 10:306, 11:376, 12:480, 13:390,
           14:572, 15:612, 16:502, 17:244, 18:111, 19:83, 20:74, 21:48, 22:22, 23:12}
    ws = vehicles["work_start"].clip(lower=6)
    we = vehicles["work_end"].where(vehicles["work_end"] > vehicles["work_start"], vehicles["work_end"] + 24).clip(upper=24)
    dur = (we - ws).clip(lower=1).astype(int)
    planned = {h: 0 for h in range(6, 24)}
    order = dur.sort_values(ascending=False, kind="stable").index
    for idx in order:
        d = int(dur[idx])
        best_s, best_score = 6, -1e18
        for s in range(6, max(7, 25 - d)):
            span = range(s, min(s + d, 24))
            score = sum(TGT.get(h, 0) - planned[h] for h in span)
            if score > best_score:
                best_score, best_s = score, s
        for h in range(best_s, min(best_s + d, 24)):
            planned[h] += 1
        vehicles.loc[idx, "work_start"] = best_s
        vehicles.loc[idx, "work_end"] = min(best_s + int(dur[idx]), 24)
    print(f"Cbudget: reassigned preserving {int(dur.sum())} veh-h total (same in-window total as Base)")

if _m_cs:
    vehicles = pd.read_csv(args.sched_file)
    _sh = int((vehicles.work_end - vehicles.work_start).sum())
    _pk = int(max(((vehicles.work_start <= h) & (vehicles.work_end > h)).sum() for h in range(6, 24)))
    assert _pk <= 613, f"schedule exceeds fleet: peak {_pk} > 613"
    print(f"Cs: {os.path.basename(args.sched_file)} — {len(vehicles)} shifts, {_sh} veh-h, peak on-duty {_pk}")

if args.supply < 1.0:
    vehicles = vehicles.sample(frac=args.supply, random_state=1000 + i).reset_index(drop=True)

passengers, vehicles = get_preprocessed_seoul_data(passengers, vehicles)
eta_model = joblib.load(f"{ARCH}/01_ETA_model/seoul_eta_model.pkl")

_variants = []
if args.fail_time != 30: _variants.append(f"T{args.fail_time}")
if args.dwell != 10: _variants.append(f"dw{args.dwell}")
if args.supply != 1.0: _variants.append(f"sup{args.supply}")
if globals().get("_od_mode"): _variants.append("dmod")
elif args.demand != "1.0": _variants.append(f"dm{args.demand}")
if args.uptake != 1.0: _variants.append(f"up{args.uptake}")
if args.dwell_sigma > 0: _variants.append(f"ds{args.dwell_sigma:g}")
if args.year != 2023: _variants.append("y24")
if args.fleet_year and args.fleet_year != args.year: _variants.append(f"f{args.fleet_year % 100}")
if args.timing: _variants.append("tm")
if _m_cs: _variants.append(os.path.splitext(os.path.basename(args.sched_file))[0].replace("sched_", ""))
label = scen + ("_" + "-".join(_variants) if _variants else "")

simul_configs = dict(base_configs)
simul_configs["fail_time"] = args.fail_time
simul_configs["add_board_time"] = args.dwell
simul_configs["add_disembark_time"] = args.dwell
if args.dwell_sigma > 0:
    simul_configs["dwell_sigma"] = args.dwell_sigma
    np.random.seed(9000 + i)
simul_configs["additional_path"] = f"{label}_r{i}{('_' + args.tag) if args.tag else ''}"
simul_configs["dispatch_mode"] = dispatch_mode
simul_configs["matrix_mode"] = "ETA"
simul_configs["eta_model"] = eta_model

print(f"[{scen} r{i}] passengers {len(passengers)}, vehicles {len(vehicles)}, dispatch={dispatch_mode}, OSRM_PORT={os.environ.get('OSRM_PORT','5001')}", flush=True)
if args.timing:
    os.environ["TIMING_FILE"] = os.path.join(HERE, f"solve_times_{label}_r{i}.csv")
t0 = time.time()
sim = Simulator(passengers=passengers, vehicles=vehicles, configs=simul_configs)
sim.run()
elapsed = time.time() - t0
from module.simulator_helper import flush_json_buffers
flush_json_buffers()

save_path = simul_configs["save_path"]
with open(os.path.join(save_path, "passenger_marker.json")) as f:
    m = pd.DataFrame(json.load(f))
wait = m["timestamp"].str[-1] - m["timestamp"].str[0]
served = m["status"] == 1
kpi = {
    "scenario": label, "realization": i, "save_path": save_path,
    "n": len(m),
    "request_failure_pct": round(100 * (~served).sum() / len(m), 3),
    "service_delay_pct": round(100 * ((served) & (wait >= 30)).sum() / max(served.sum(), 1), 3),
    "avg_wait_min": round(float(wait[served].mean()), 3),
    "elapsed_min": round(elapsed / 60, 1),
}
print(json.dumps(kpi, ensure_ascii=False), flush=True)
with open(os.path.join(HERE, "campaign_results.csv"), "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    if f.tell() == 0:
        f.write(",".join(kpi.keys()) + "\n")
    f.write(",".join(str(v) for v in kpi.values()) + "\n")
    fcntl.flock(f, fcntl.LOCK_UN)
