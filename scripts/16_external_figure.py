# -*- coding: utf-8 -*-
"""Step 16: figure for external validation + mutation sensitivity."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), dpi=250)

# ---- A: KcatTIE temporal holdout ----
tie = pd.read_csv("results/external_KcatTIE.csv")
ax = axes[0]
x = tie["measured_log10_kcat"].values
y = tie["pred_ENS_log10_kcat"].values
ax.scatter(x, y, s=14, alpha=0.55, c="#2b6cb0", edgecolors="none")
lo, hi = min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3
ax.plot([lo, hi], [lo, hi], "r--", lw=1)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
r = pearsonr(x, y)[0]; rho = spearmanr(x, y)[0]
ax.set_title(f"KcatTIE (2023-2025 temporal holdout)\nn={len(x)}, r={r:.2f}, ρ={rho:.2f}", fontsize=9)
ax.set_xlabel("Measured log$_{10}$(k$_{cat}$)")
ax.set_ylabel("Predicted log$_{10}$(k$_{cat}$)")

# ---- B: ADK orthologs ----
adk = pd.read_csv("results/external_ADK.csv")
ax = axes[1]
x = adk["measured_log10_kcat"].values
y = adk["pred_ENS_log10_kcat"].values
ax.scatter(x, y, s=14, alpha=0.55, c="#c53030", edgecolors="none")
lo, hi = min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3
ax.plot([lo, hi], [lo, hi], "k--", lw=1)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
r = pearsonr(x, y)[0]; rho = spearmanr(x, y)[0]
ax.set_title(f"ADK orthologs (high-k$_{{cat}}$ family)\nn={len(x)}, r={r:.2f}, ρ={rho:.2f}", fontsize=9)
ax.set_xlabel("Measured log$_{10}$(k$_{cat}$)")
ax.set_ylabel("Predicted log$_{10}$(k$_{cat}$)")

# ---- C: mutation sensitivity ----
allm = pd.read_csv("results/mutant_sensitivity_all.tsv", sep="\t")
m = allm[allm["substrate"] == "15-cis-phytoene"].copy()
m = m[m["mutant"] != "WT"].sort_values("fold_vs_WT", ascending=False).reset_index(drop=True)
# Anonymise mutant identities: the sequences are confidential and must not be
# published.  Labels are rank-only (M1 = largest predicted fold-change).
m["anon"] = [f"M{i+1}" for i in range(len(m))]
ax = axes[2]
colors = {"within noise": "#9ca3af", "weak signal": "#60a5fa",
          "candidate": "#f59e0b", "strong signal": "#dc2626"}
ax.bar(m["anon"], m["fold_vs_WT"], color=[colors[v] for v in m["verdict"]],
       edgecolor="#374151", lw=0.5)
ax.axhline(1.0, color="#111827", lw=1)
for yy, lab in [(3, "3x noise floor"), (10, "10x candidate"), (30, "30x strong")]:
    ax.axhline(yy, color="#6b7280", lw=0.8, ls="--")
    ax.text(len(m) - 0.5, yy * 1.05, lab, fontsize=6.5, color="#4b5563", ha="right")
ax.set_yscale("log")
ax.set_ylim(0.6, 40)
ax.set_ylabel("Predicted fold-change vs WT")
ax.set_xlabel("single point mutants (ranked, anonymised)")
ax.set_title("17 single point mutants (substrate: phytoene)\nall changes < 2x, below model noise", fontsize=9)
ax.tick_params(axis="x", labelsize=6.5)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("results/fig_external_validation.png", bbox_inches="tight")
print("saved results/fig_external_validation.png")
