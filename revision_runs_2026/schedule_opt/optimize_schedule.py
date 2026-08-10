# -*- coding: utf-8 -*-
"""Scenario C redesign: no-fleet-expansion driver shift scheduling.

Pipeline (simulation-based staffing & scheduling; Feldman et al. 2008;
Atlason et al. 2008; Cezik & L'Ecuyer 2008; review: Defraeye & Van
Nieuwenhuyse 2016):
  1. Hourly requirements R_h initialized from the busy-vehicle profile of
     all-day-staffing runs (unconstrained-supply workload), buffered.
  2. Min-cost / budget-neutral shift covering IP over the operator's legal
     shift catalog: 9-h regular (staggered starts) + 5-h part-time.
  3. Schedules are validated in the simulator; hours with residual failures
     get coverage cuts (R_h += step) and the IP is re-solved (run separately
     via --bump).

Variants:
  C1  budget-neutral: total in-window staffed hours <= Base (5,116 h),
      minimize demand-weighted coverage shortfall vs R_h.
  C2  target-driven: cover R_h fully, minimize labor cost.

usage:
  python optimize_schedule.py --variant C1 --iter 0
  python optimize_schedule.py --variant C2 --iter 1 --bump "7:20,14:15"
"""
import argparse, glob, json, os
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp

HERE = os.path.dirname(os.path.abspath(__file__))
RR = os.path.dirname(HERE)                      # revision_runs_2026
ROOT = os.path.dirname(RR)                      # dtumos-paratransit
ARCH = f"{ROOT}/_simulation_archive_20260730"
BASE_VEH = f"{ARCH}/02_vehicle_inputs/paper_2025__613_782_870/Baseline_613__vehicle_0.csv"
PASS_GLOB = f"{ARCH}/05_demand_generator/passenger_output_2025_paper/passenger_*.csv"

FLEET = 613
H0, H1 = 6, 24                                  # simulation horizon (hours)
HOURS = range(H0, H1)
WAGE = 12_600                                   # won per staffed hour (2023 formula, loaded)
PT_PREMIUM = 0.125                              # formula-based; sensitivity: 0 / 0.30
                                                # (analysis/05_cost/README.md 단시간 단가 절)

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["C1", "C2"], required=True)
ap.add_argument("--iter", type=int, default=0)
ap.add_argument("--buffer", type=float, default=1.15)
ap.add_argument("--bump", default="", help="cut list 'hour:delta,hour:delta' added to R_h")
ap.add_argument("--pt_premium", type=float, default=PT_PREMIUM)
ap.add_argument("--pt_cap", type=int, default=175,
                help="max part-time drivers (operator's 2026 program scale)")
ap.add_argument("--return_h", type=float, default=0.5,
                help="end-of-shift depot return + handover time (h); paid but not in service")
ap.add_argument("--floor_f", type=float, default=0.9,
                help="C1 service floor as fraction of min(R_h, base coverage)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

# ---------------------------------------------------------------- inputs
# workload B_h: busy vehicles under all-day staffing (C = all-day, BC = +opt)
recs = sorted(glob.glob(f"{RR}/campaign_fast/simul_result/C_r[0-9]/simulation_1/record.csv")) \
     + sorted(glob.glob(f"{RR}/campaign_fast/simul_result/BC_r[0-9]/simulation_1/record.csv"))
prof = []
for f in recs:
    r = pd.read_csv(f)
    r["hour"] = r.time // 60
    prof.append(r.groupby("hour").driving_vehicle_cnt.mean())
B = pd.concat(prof, axis=1).mean(axis=1)
print(f"workload profile from {len(recs)} all-day runs")

# demand weights
lam = None
for f in sorted(glob.glob(PASS_GLOB)):
    p = pd.read_csv(f)
    l = (p.ride_time // 60).value_counts().sort_index()
    lam = l if lam is None else lam.add(l, fill_value=0)
lam = (lam / len(glob.glob(PASS_GLOB))).reindex(range(24), fill_value=0.0)

# base in-window staffed hours (budget for C1)
bv = pd.read_csv(BASE_VEH)
ov = (bv.work_end.clip(upper=H1) - bv.work_start.clip(lower=H0)).clip(lower=0)
BUDGET = int(ov.sum())
print(f"Base in-window staffed hours: {BUDGET}")

# requirements
R = {h: min(FLEET, int(np.ceil(args.buffer * B.get(h, 0.0)))) for h in HOURS}
for kv in filter(None, args.bump.split(",")):
    h, d = map(int, kv.split(":"))
    R[h] = min(FLEET, R[h] + d)
print("R_h:", {h: R[h] for h in HOURS})

# ---------------------------------------------------------------- catalog
# official structure only: 9h regular (8h work + 1h break, staggered starts),
# 5h part-time. Spans inside [06,24].
shifts = []
for s in range(6, 16):                          # 9h: 06-15 → ends 15-24
    shifts.append(dict(kind="REG9", start=s, end=s + 9, cost=9 * WAGE))
# 5h part-time: ONLY the operator's actual program windows (2026 posting) —
# morning 07-12, evening 16-21 / 17-22. Free-floating starts are not
# operationally realistic (fixed rosters, depot handover blocks).
for s in (7, 16, 17):
    shifts.append(dict(kind="PT5", start=s, end=s + 5,
                       cost=int(5 * WAGE * (1 + args.pt_premium))))
# depot return charge: the last return_h of each shift is paid but out of
# service (drive back to depot, handover, breathalyzer check)
for sh in shifts:
    sh["svc_end"] = sh["end"] - args.return_h

def covw(sh, h):
    """coverage weight of shift sh in hour bucket h"""
    if h < sh["start"] or h >= sh["end"]:
        return 0.0
    return max(0.0, min(sh["svc_end"] - h, 1.0))

# ---------------------------------------------------------------- IP
solver = pywraplp.Solver.CreateSolver("SCIP")
x = {i: solver.IntVar(0, FLEET, f"x{i}") for i in range(len(shifts))}
# N = service coverage (return_h discounted); P = physical vehicle presence
# (paid span — a returning vehicle still occupies its vehicle)
N = {h: solver.Sum(x[i] * covw(sh, h) for i, sh in enumerate(shifts)
                   if covw(sh, h) > 0) for h in HOURS}
P = {h: solver.Sum(x[i] for i, sh in enumerate(shifts)
                   if sh["start"] <= h < sh["end"]) for h in HOURS}
base_on_h = {h: int(((pd.read_csv(BASE_VEH).work_start <= h)
                     & (pd.read_csv(BASE_VEH).work_end > h)).sum()) for h in HOURS}
FLOOR_F = args.floor_f if args.variant == "C1" else 1.0  # C1: return charge makes exact
for h in HOURS:                                 # base-matching infeasible
    solver.Add(P[h] <= FLEET)                   # no fleet expansion (physical)
    # service floor: never (much) below Base coverage, nor above requirement
    solver.Add(N[h] >= FLOOR_F * min(R[h], base_on_h[h]))
# part-time program scale cap (operator's 2026 hiring: 175 drivers)
solver.Add(solver.Sum(x[i] for i, sh in enumerate(shifts)
                      if sh["kind"] == "PT5") <= args.pt_cap)

if args.variant == "C1":
    solver.Add(solver.Sum(x[i] * (shifts[i]["end"] - shifts[i]["start"])
                          for i in x) <= BUDGET)
    short = {h: solver.NumVar(0, FLEET, f"s{h}") for h in HOURS}
    for h in HOURS:
        solver.Add(short[h] >= R[h] - N[h])
    solver.Minimize(solver.Sum(short[h] * float(lam[h] + 1.0) for h in HOURS)
                    + 1e-6 * solver.Sum(x[i] * shifts[i]["cost"] for i in x))
else:  # C2 — soft coverage: report shortfall instead of infeasibility
    # (the 15:00 handover wall can make full coverage physically impossible
    # within 613 vehicles; the shortfall itself is a finding)
    short = {h: solver.NumVar(0, FLEET, f"s{h}") for h in HOURS}
    for h in HOURS:
        solver.Add(short[h] >= R[h] - N[h])
    BIGM = 10_000_000                            # >> any daily labor cost delta
    cost_expr = solver.Sum(x[i] * shifts[i]["cost"] for i in x)
    solver.Minimize(cost_expr + BIGM * solver.Sum(short[h] for h in HOURS))

st = solver.Solve()
assert st in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE), "no solution"

if args.variant == "C2":
    # stage 2 (lexicographic): among min-cost optima, place surplus coverage
    # where demand is — stabilizes degenerate solutions across pt_cap values
    best_cost = sum(int(x[i].solution_value()) * shifts[i]["cost"] for i in x)
    best_short = sum(short[h].solution_value() for h in HOURS)
    solver.Add(cost_expr <= best_cost)
    solver.Add(solver.Sum(short[h] for h in HOURS) <= best_short + 1e-6)
    solver.Maximize(solver.Sum(N[h] * float(lam[h]) for h in HOURS))
    st = solver.Solve()
    assert st in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE), "stage2 failed"
    tot_short = sum(short[h].solution_value() for h in HOURS)
    if tot_short > 0.01:
        print(f"NOTE: coverage shortfall {tot_short:.1f} veh-h "
              f"(physically unreachable within 613 vehicles + handover)")

# ---------------------------------------------------------------- report
sol = [(sh, int(x[i].solution_value())) for i, sh in enumerate(shifts)
       if x[i].solution_value() > 0.5]
tot_h = sum((sh["end"] - sh["start"]) * n for sh, n in sol)
tot_cost = sum(sh["cost"] * n for sh, n in sol)
n_reg = sum(n for sh, n in sol if sh["kind"] == "REG9")
n_pt = sum(n for sh, n in sol if sh["kind"] == "PT5")
print(f"\n=== {args.variant} iter{args.iter}: {n_reg} REG9 + {n_pt} PT5, "
      f"{tot_h} veh-h/day (Base {BUDGET}), labor {tot_cost/1e6:.1f}M won/day ===")
for sh, n in sorted(sol, key=lambda t: (t[0]["kind"], t[0]["start"])):
    print(f"  {sh['kind']:4s} {sh['start']:02d}:00-{sh['end']:02d}:00  x{n}")
cov = {h: sum(n * covw(sh, h) for sh, n in sol) for h in HOURS}
print("hour  need  planned  base   (planned = service coverage after return charge)")
base_on = {h: int(((bv.work_start <= h) & (bv.work_end > h)).sum()) for h in HOURS}
for h in HOURS:
    flag = " *SHORT*" if cov[h] < R[h] else ""
    print(f"{h:4d} {R[h]:5d} {cov[h]:8.1f} {base_on[h]:5d}{flag}")

# ---------------------------------------------------------------- export
rng = np.random.default_rng(args.seed)
rows = []
for sh, n in sol:
    # sample base rows jointly (garage position + cartype) to preserve the
    # fleet's wheelchair-accessible mix (559 cartype-1 / 54 cartype-0)
    picks = bv.sample(n=n, replace=True, random_state=int(rng.integers(1e9)))
    for _, b in picks.iterrows():
        # sim gets the service span (svc_end); the paid span (end) stays in
        # the cost accounting — the last return_h is depot return + handover
        rows.append(dict(vehicle_id=len(rows), cartype=int(b.cartype),
                         work_start=sh["start"], work_end=sh["svc_end"],
                         temporary_stopTime=0, lat=b.lat, lon=b.lon))
out = pd.DataFrame(rows)
# self-check: physical presence (paid spans) within fleet at every hour
_pk = max(sum(n for sh, n in sol if sh["start"] <= h < sh["end"]) for h in HOURS)
assert _pk <= FLEET, f"export violates fleet cap: peak {_pk}"
# repair: concurrent cartype-1 must not exceed the physical accessible fleet
CAP1 = int((bv.cartype == 1).sum())
for h in HOURS:
    on = out[(out.work_start <= h) & (out.work_end > h)]
    over = int((on.cartype == 1).sum()) - CAP1
    if over > 0:
        flip = on[on.cartype == 1].sample(n=over, random_state=args.seed + h).index
        out.loc[flip, "cartype"] = 0
print("cartype mix:", out.cartype.value_counts().to_dict(),
      f"(base {CAP1}/{len(bv)-CAP1})")
tag = f"{args.variant}i{args.iter}"
path = f"{HERE}/sched_{tag}.csv"
out.to_csv(path, index=False)
json.dump(dict(variant=args.variant, iter=args.iter, R=R, buffer=args.buffer,
               bump=args.bump, pt_premium=args.pt_premium, n_reg=n_reg,
               n_pt=n_pt, tot_h=tot_h, tot_cost_day=tot_cost),
          open(f"{HERE}/sched_{tag}.json", "w"), indent=1)
print(f"\nexported {path} ({len(out)} shift-vehicles)")
