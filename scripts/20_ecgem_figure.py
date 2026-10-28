# -*- coding: utf-8 -*-
"""Step 20: figure for the enzyme-constrained dry validation (script 19)."""
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.6,
    "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
    "axes.labelsize": 6.8, "legend.fontsize": 5.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

pred = pd.read_csv("results/ecgem_predictions.tsv", sep="\t")
cap = pd.read_csv("results/ecgem_capacity.tsv", sep="\t")
tgt = pd.read_csv("results/ecgem_targets.tsv", sep="\t")
env = pd.read_csv("results/ecgem_envelope.csv")
sm = json.load(open("results/ecgem_summary.json"))

# literature kcat for the same enzyme-substrate pairs (results/case_study.tsv)
LIT = {"CrtE": 0.800, "CrtB": 6.000, "CrtI": 0.100}
LIT_RANGE = {"CrtB": (6.0, 14.0)}

C_MAIN, C_ALT, C_MEAS, C_GREY = "#2f5d8a", "#c8632c", "#1a7a4c", "#8a8a8a"

fig = plt.figure(figsize=(7.2, 2.45))
gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.075, right=0.985,
                      top=0.90, bottom=0.20)

# ---------------------------------------------------------------- panel A
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(pred))
ax.bar(x, pred["kcat_s-1"], width=0.52, color=C_MAIN, alpha=0.85,
       label="CarotKcat predicted")
lit = [LIT[e] for e in pred.enzyme]
ax.scatter(x, lit, s=22, marker="D", facecolor="white", edgecolor=C_MEAS,
           linewidth=1.0, zorder=5, label="literature")
for i, e in enumerate(pred.enzyme):
    if e in LIT_RANGE:
        lo, hi_ = LIT_RANGE[e]
        ax.plot([i, i], [lo, hi_], color=C_MEAS, lw=1.6, alpha=0.5, zorder=4)
    f = pred["kcat_s-1"].iloc[i] / LIT[e]
    lab = f"{f:.1f}\u00d7 over" if f >= 1 else f"{1/f:.1f}\u00d7 under"
    ax.annotate(lab, (i, max(lit[i], pred["kcat_s-1"].iloc[i]) * 1.5),
                ha="center", fontsize=5.2, color=C_MEAS)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([f"{e}\n({s})" for e, s in zip(pred.enzyme, pred.substrate)],
                   fontsize=5.6)
ax.set_ylabel(r"$k_{cat}$ (s$^{-1}$)")
ax.set_ylim(0.05, 60)
ax.set_title("A  Module $k_{cat}$: prediction vs literature", loc="left",
             fontsize=6.8, fontweight="bold")
ax.legend(frameon=False, loc="upper left", handletextpad=0.4, borderpad=0.1)

# ---------------------------------------------------------------- panel B
ax = fig.add_subplot(gs[0, 1])
hi = tgt.flux_umol_gDCW_h.max()
sel = cap[np.isclose(cap.target_flux, hi)].copy()
sel = sel.set_index("enzyme").loc[pred.enzyme].reset_index()
pct = sel["required_frac_of_protein"].values * 100
y = np.arange(len(sel))[::-1]
ax.axvspan(0.1, 5.0, color=C_GREY, alpha=0.16, lw=0)
ax.text(0.70, 2.18, "achievable expression", fontsize=5.2,
        color="#555555", ha="center", va="center")
ax.barh(y, pct, height=0.5, color=C_ALT, alpha=0.88)
for yy, p in zip(y, pct):
    ax.text(p * 1.35, yy, f"{p:.3f}%", va="center", fontsize=5.6, color=C_ALT)
ax.set_xscale("log")
ax.set_xlim(1e-3, 20)
ax.set_yticks(y)
ax.set_yticklabels(sel.enzyme, fontsize=6.0)
ax.set_xlabel("enzyme needed to sustain 3.12\n$\\mu$mol gDCW$^{-1}$ h$^{-1}$ (% of protein)")
ax.set_title("B  Required vs achievable expression", loc="left",
             fontsize=6.8, fontweight="bold")

# ---------------------------------------------------------------- panel C
ax = fig.add_subplot(gs[0, 2])
e = env.sort_values("lycopene_umol_gDCW_h")
ax.plot(e.lycopene_umol_gDCW_h, e.mu_per_h, color=C_MAIN, lw=1.4,
        label="iML1515 + CrtEBI")
ax.axvline(hi, color=C_MEAS, lw=1.0, ls="--", label=f"measured {hi:.1f}")
ax.axvline(2.30, color=C_GREY, lw=0.9, ls=":", label="native MEP flux 2.3")
ax.set_xscale("log")
ax.set_xlim(1.6, 1200)
ax.set_ylim(0, 0.92)
ax.set_xlabel(r"lycopene flux ($\mu$mol gDCW$^{-1}$ h$^{-1}$)")
ax.set_ylabel(r"growth rate $\mu$ (h$^{-1}$)")
ax.set_title("C  Growth\u2013production envelope", loc="left",
             fontsize=6.8, fontweight="bold")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.02),
          handletextpad=0.4, borderpad=0.1, labelspacing=0.25)

fig.savefig("results/fig_ecgem_validation.png", dpi=600)
print("saved results/fig_ecgem_validation.png")
print(f"headroom at 1%: " + ", ".join(
    f"{r.enzyme} {r.headroom_at_1pct:.0f}x" for _, r in sel.iterrows()))
