# -*- coding: utf-8 -*-
"""Cut generator for the schedule optimization loop.

Reads passenger_marker.json of one or more Cs runs, reports hourly request
failures (by request hour), and prints a --bump string raising R_h where the
failure rate exceeds the target.

usage: python cuts.py <save_path>... [--target 9.22] [--step 15]
"""
import argparse, json, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--target", type=float, default=9.22, help="overall target failure %")
ap.add_argument("--step", type=int, default=15, help="R_h increment per cut")
args = ap.parse_args()

frames = []
for sp in args.paths:
    f = os.path.join(sp, "passenger_marker.json")
    p = pd.DataFrame(json.load(open(f)))
    p["start"] = p["timestamp"].str[0]
    p["hour"] = (p["start"] // 60).astype(int)
    frames.append(p)
p = pd.concat(frames)

tot_fail = 100 * (p.status == 0).sum() / len(p)
g = p.groupby("hour").agg(req=("status", "size"),
                          fail=("status", lambda s: (s == 0).sum()))
g["rate"] = 100 * g.fail / g.req
print(f"overall failure: {tot_fail:.2f}%  (target {args.target}%)  n={len(p)} over {len(args.paths)} run(s)")
print(g.round(1).to_string())

# cut rule: hours whose failure rate exceeds the overall target get a bump,
# scaled by how far above target they sit (1x or 2x step)
bumps = []
for h, row in g.iterrows():
    if row.rate > args.target and row.fail >= 5:
        mult = 2 if row.rate > 3 * args.target else 1
        bumps.append(f"{h}:{args.step * mult}")
print("\nbump:", ",".join(bumps) if bumps else "(none — target met everywhere)")
