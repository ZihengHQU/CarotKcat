# -*- coding: utf-8 -*-
"""Step 23: metrics and figure for the curated terpenoid benchmark (script 22).

Two generalisation regimes are reported separately because they answer
different questions:

* ``seen EC family``   -- a new enzyme from a family represented in training
                          (DXP synthase, HMG-CoA synthase, mevalonate kinase,
                          isoprenyl diphosphate synthases / isomerase).
* ``unseen EC family`` -- the carotenoid and terpene-synthase families, which
                          are entirely absent from the 17,051 training records
                          (BRENDA holds no turnover numbers at all for phytoene
                          synthase, phytoene desaturase or lycopene cyclase).

The response slope (predicted regressed on measured) is reported alongside the
correlation because a model that predicts a near-constant value can still reach
a respectable r over a wide measured range.
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.6,
    "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
    "axes.labelsize": 6.8, "legend.fontsize": 5.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

C_GREY = "#8a8a8a"
GROUP_COLOR = {
    "carotenoid": "#c8632c",
    "terpene synthase": "#2f5d8a",
    "isoprenoid supply": "#7b5aa6",
    "precursor": "#1a7a4c",
}
GROUP_ORDER = ["precursor", "isoprenoid supply", "terpene synthase", "carotenoid"]

d = pd.read_csv("results/terpenoid_bench.tsv", sep="\t")
d["group"] = d["class"].map(
    {"carotenoid": "carotenoid", "terpene synthase": "terpene synthase",
     "precursor": "precursor"}).fillna("isoprenoid supply")


def metrics(g):
    x, y = g["log10_kcat"].values, g["pred_log10"].values
    n = len(g)
    flat = np.std(x) < 1e-9
    r = np.nan if flat or n < 3 else stats.pearsonr(x, y)[0]
    rho = np.nan if flat or n < 3 else stats.spearmanr(x, y)[0]
    r2 = np.nan if flat or n < 3 else 1 - ((y - x) ** 2).sum() / ((x - x.mean()) ** 2).sum()
    fold = 10 ** np.abs(y - x)
    slope = np.nan if flat or n < 3 else np.polyfit(x, y, 1)[0]
    return pd.Series({
        "n": n, "Pearson": r, "Spearman": rho, "R2": r2,
        "RMSE_log10": float(np.sqrt(((y - x) ** 2).mean())),
        "median_fold": float(np.median(fold)),
        "within_3x": float((fold <= 3).mean()),
        "within_10x": float((fold <= 10).mean()),
        "bias_log10": float((y - x).mean()),
        "slope_pred_on_meas": slope,
        "meas_IQR_log10": float(np.subtract(*np.percentile(x, [75, 25]))),
        "pred_IQR_log10": float(np.subtract(*np.percentile(y, [75, 25]))),
    })


rows = {"overall": metrics(d)}
for reg, g in d.groupby("regime"):
    rows[reg] = metrics(g)
for cls, g in d.groupby("group"):
    rows[cls] = metrics(g)
rows["zero-shot (unseen family, novel enzyme)"] = metrics(
    d[d["zero_shot"] & (d["regime"] == "unseen EC family")])
M = pd.DataFrame(rows).T
M.index.name = "subset"
M.to_csv("results/terpenoid_bench_metrics.csv")
print(M.round(3).to_string())

# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(7.2, 2.45))
gs = fig.add_gridspec(1, 3, wspace=0.40, left=0.085, right=0.985,
                      bottom=0.20, top=0.90)

# ---- A: predicted vs measured -------------------------------------------- #
ax = fig.add_subplot(gs[0, 0])
lo, hi = -3.4, 1.9
ax.plot([lo, hi], [lo, hi], color="#333333", lw=0.7, ls="--", zorder=1)
ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color=C_GREY,
                alpha=0.13, lw=0, zorder=0)
for grp in GROUP_ORDER:
    g = d[d["group"] == grp]
    if g.empty:
        continue
    ax.scatter(g["log10_kcat"], g["pred_log10"], s=17, lw=0.3,
               edgecolor="white", color=GROUP_COLOR[grp],
               label=f"{grp} (n={len(g)})", zorder=3)
ax.set_xlim(lo, hi)
ax.set_ylim(-2.05, 1.4)
ax.set_xlabel("measured $\\log_{10}k_{cat}$ [s$^{-1}$]")
ax.set_ylabel("predicted $\\log_{10}k_{cat}$")
ax.text(0.03, 0.96, "A", transform=ax.transAxes, fontsize=8, fontweight="bold",
        va="top")
ax.legend(frameon=False, loc="lower right", handletextpad=0.3, borderpad=0.15,
          labelspacing=0.22, fontsize=5.2)
ax.text(0.03, 0.19, "$r$ = %.2f\nmedian %.1f$\\times$"
        % (M.loc["overall", "Pearson"], M.loc["overall", "median_fold"]),
        transform=ax.transAxes, fontsize=5.4, color="#444444")

# ---- B: signed error vs measured ----------------------------------------- #
ax = fig.add_subplot(gs[0, 1])
ax.axhline(0, color="#333333", lw=0.7)
for yv in (-1, 1):
    ax.axhline(yv, color=C_GREY, lw=0.6, ls=":")
for grp in GROUP_ORDER:
    g = d[d["group"] == grp]
    if g.empty:
        continue
    ax.scatter(g["log10_kcat"], g["abs_log10_error"], s=17, lw=0.3,
               edgecolor="white", color=GROUP_COLOR[grp], zorder=3)
x = d["log10_kcat"].values
y = d["abs_log10_error"].values
b, a = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 20)
ax.plot(xs, a + b * xs, color="#333333", lw=0.9, ls="--", zorder=2)
ax.set_xlabel("measured $\\log_{10}k_{cat}$ [s$^{-1}$]")
ax.set_ylabel("predicted $-$ measured")
ax.set_ylim(-3.0, 3.4)
ax.text(0.03, 0.96, "B", transform=ax.transAxes, fontsize=8, fontweight="bold",
        va="top")
ax.text(0.97, 0.93, "slope %.2f" % b, transform=ax.transAxes, fontsize=5.4,
        color="#444444", ha="right")

# ---- C: response slope per group ----------------------------------------- #
ax = fig.add_subplot(gs[0, 2])
vals = [M.loc[k, "slope_pred_on_meas"] for k in GROUP_ORDER]
ns = [int(M.loc[k, "n"]) for k in GROUP_ORDER]
yp = np.arange(len(GROUP_ORDER))[::-1]
ax.axvline(1.0, color="#333333", lw=0.8, ls="--")
ax.axvline(0.0, color=C_GREY, lw=0.7)
ax.barh(yp, vals, height=0.46, color=[GROUP_COLOR[k] for k in GROUP_ORDER], lw=0)
for y0, v in zip(yp, vals):
    if v > 0.55:                       # keep the label clear of the 1.0 guide
        ax.text(v - 0.035, y0, f"{v:+.2f}", va="center", ha="right",
                fontsize=5.6, color="white")
    else:
        ax.text(v + 0.035, y0, f"{v:+.2f}", va="center", ha="left", fontsize=5.6)
ax.set_yticks(yp)
ax.set_yticklabels([f"{k}\n(n={n})" for k, n in zip(GROUP_ORDER, ns)], fontsize=5.6)
ax.set_xlim(-0.45, 1.30)
ax.set_xlabel("response slope (1.0 = perfect, 0 = none)")
ax.text(0.03, 0.96, "C", transform=ax.transAxes, fontsize=8, fontweight="bold",
        va="top")

fig.savefig("results/fig_terpenoid_benchmark.png", dpi=420,
            bbox_inches="tight", facecolor="white")
print("\nsaved results/fig_terpenoid_benchmark.png")
