# -*- coding: utf-8 -*-
"""Fig 11 (hourly failure: Base/A782/C) and Fig 12 (factorial waterfall)."""
import json, glob, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RR = os.path.dirname(HERE)
OUT = ('/Users/jihoyeo/Library/CloudStorage/OneDrive-개인/research/'
       '장애인콜택시/paper/TR_A/revision/drafts/figures')

def hourly(pattern):
    rates = []
    for sp in sorted(glob.glob(f"{RR}/simul_result/{pattern}/simulation_*")):
        if not os.path.exists(f"{sp}/passenger_marker.json"):
            continue  # crashed/partial run dirs
        p = pd.DataFrame(json.load(open(f"{sp}/passenger_marker.json")))
        p["hour"] = (p["timestamp"].str[0] // 60).astype(int)
        g = p.groupby("hour").agg(req=("status", "size"),
                                  fail=("status", lambda s: (s == 0).sum()))
        rates.append(100 * g.fail / g.req)
    return pd.concat(rates, axis=1).mean(axis=1)

base = hourly("base_r*")
a782 = hourly("F782_r*")
c = hourly("Cs_C2i23_r*")

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(base.index, base.values, "o-", color="#555555", label="Base (613 vehicles)")
ax.plot(a782.index, a782.values, "s--", color="#C00000", label="A (782): fleet expansion")
ax.plot(c.index, c.values, "^-", color="#1f5fbf", label="C: optimized schedule (613 vehicles)")
ax.set_xlabel("Hour of day (request time)")
ax.set_ylabel("Request failure rate (%)")
ax.set_xticks(range(6, 24, 2))
ax.grid(alpha=.3)
ax.legend(frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig_matched_staffing_hourly.{ext}", dpi=300)
print("fig11 saved")

# Fig 12: sequential waterfall Base -> +C -> +D -> +B (order by Shapley share)
steps = [("Base", 24.26), ("+ C (schedule)", 6.66), ("+ D (voucher)", 1.58), ("+ B (dispatch)", 1.13)]
fig2, ax2 = plt.subplots(figsize=(6.8, 4.0))
xs = range(len(steps))
vals = [v for _, v in steps]
cols = ["#555555", "#1f5fbf", "#1f5fbf", "#1f5fbf"]
bars = ax2.bar(xs, vals, color=cols, width=.6)
for i, (lab, v) in enumerate(steps):
    ax2.text(i, v + .4, f"{v:.2f}%", ha="center", fontsize=10)
    if i: ax2.annotate("", xy=(i, vals[i]), xytext=(i - 1, vals[i - 1]),
                       arrowprops=dict(arrowstyle="->", color="#999999", lw=1))
ax2.set_xticks(list(xs))
ax2.set_xticklabels([l for l, _ in steps])
ax2.set_ylabel("Request failure rate (%)")
ax2.text(.98, .95, "Shapley shares: C 57%, D 34%, B 10%",
         transform=ax2.transAxes, ha="right", va="top", fontsize=9)
ax2.grid(axis="y", alpha=.3)
fig2.tight_layout()
for ext in ("png", "pdf"):
    fig2.savefig(f"{OUT}/fig_factorial_waterfall.{ext}", dpi=300)
print("fig12 saved")
