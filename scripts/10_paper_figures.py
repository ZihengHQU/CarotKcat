"""Generate publication figures for the CarotKcat manuscript.
Fig 1: model architecture schematic.
Fig 2: benchmark comparison (grouped bars, random vs enzyme-group split).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

# ---------------- Figure 1: architecture ----------------
fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=300)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

def box(x, y, w, h, text, fc, ec="#333333", fs=8.5, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.3, color="#333333"))

C_SEQ = "#dbeafe"; C_SUB = "#dcfce7"; C_FUSE = "#fef3c7"; C_OUT = "#fee2e2"

# inputs
box(0.2, 7.6, 2.0, 1.1, "Enzyme\namino-acid\nsequence", C_SEQ)
box(0.2, 2.2, 2.0, 1.1, "Substrate\nSMILES", C_SUB)
# encoders
box(2.9, 7.5, 2.2, 1.3, "ESM-2 (650M)\nfrozen\nmean pooling", C_SEQ, fs=8)
box(2.9, 2.1, 2.2, 1.3, "Morgan fingerprint\nradius 2\n2048 bits", C_SUB, fs=8)
# towers
box(5.8, 7.5, 1.7, 1.3, "MLP tower\n1280-512-256", C_SEQ, fs=8)
box(5.8, 2.1, 1.7, 1.3, "MLP tower\n2048-512-256", C_SUB, fs=8)
# fusion
box(8.0, 4.6, 1.7, 1.3, "Concat 512\nFusion MLP\n512-256-64-1", C_FUSE, fs=8, bold=True)
# output
box(8.0, 0.6, 1.7, 1.0, "log$_{10}$ k$_{cat}$\n(s$^{-1}$)", C_OUT, fs=9, bold=True)

arrow(2.25, 8.15, 2.85, 8.15)
arrow(2.25, 2.75, 2.85, 2.75)
arrow(5.15, 8.15, 5.75, 8.15)
arrow(5.15, 2.75, 5.75, 2.75)
arrow(6.65, 7.45, 8.55, 5.95)
arrow(6.65, 3.45, 8.55, 4.55)
arrow(8.85, 4.55, 8.85, 1.7)

ax.text(5.0, 9.55, "CarotKcat dual-tower architecture", ha="center",
        fontsize=11, fontweight="bold")
ax.text(0.2, 0.9, "Ensemble: prediction averaged with XGBoost (ESM+FP) in log space",
        fontsize=7.5, style="italic", color="#444444")
fig.tight_layout()
fig.savefig("results/fig_architecture.png", bbox_inches="tight")
plt.close(fig)
print("saved results/fig_architecture.png")

# ---------------- Figure 2: benchmark ----------------
df = pd.read_csv("results/benchmark_table.csv")
order = ["CarotKcat-ensemble", "CarotKcat (dual-tower)",
         "CarotKcat-GNN (ESM+GNN substrate)",
         "DLKcat-style (GNN+CNN, reimplemented)",
         "TurNuP-style (ESM+FP, XGBoost)", "UniKP-style (ESM+FP, ExtraTrees)",
         "ESM only (MLP)", "Substrate-FP only (MLP)"]
labels = ["CarotKcat\nensemble", "CarotKcat\ndual-tower", "CarotKcat\nGNN",
          "DLKcat-style\n(GNN+CNN)", "TurNuP-style\n(XGBoost)",
          "UniKP-style\n(ExtraTrees)", "Sequence\nonly", "Substrate\nonly"]
r_vals, g_vals, r_err, g_err = [], [], [], []
for m in order:
    for proto, vals, errs in (("random", r_vals, r_err), ("group", g_vals, g_err)):
        row = df[(df.model == m) & (df.protocol == proto)].iloc[0]
        vals.append(row["Pearson"])
        errs.append([row["Pearson"] - row["Pearson_lo"], row["Pearson_hi"] - row["Pearson"]])

fig, ax = plt.subplots(figsize=(8.8, 3.8), dpi=300)
x = range(len(order)); w = 0.38
b1 = ax.bar([i - w/2 for i in x], r_vals, w, label="Random split", color="#93c5fd",
            ec="#1e40af", lw=0.7,
            yerr=np.array(r_err).T, ecolor="#1e40af", capsize=2.5,
            error_kw={"lw": 0.8, "capthick": 0.8})
b2 = ax.bar([i + w/2 for i in x], g_vals, w, label="Enzyme-group split", color="#fca5a5",
            ec="#991b1b", lw=0.7,
            yerr=np.array(g_err).T, ecolor="#991b1b", capsize=2.5,
            error_kw={"lw": 0.8, "capthick": 0.8})
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.030,
                f"{b.get_height():.2f}", ha="center", fontsize=6.3)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=6.8)
ax.set_ylabel("Pearson r (test set, 95% bootstrap CI)")
ax.set_ylim(0, 0.86)
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("results/fig_benchmark.png", bbox_inches="tight")
plt.close(fig)
print("saved results/fig_benchmark.png")
