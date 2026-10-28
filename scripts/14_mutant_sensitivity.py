# -*- coding: utf-8 -*-
"""Step 14: mutant sensitivity analysis for a user-supplied wild type.

IMPORTANT: this is a *sensitivity / ranking* analysis, NOT validation.  There
are no experimental kcat values for these mutants, so nothing here proves that a
mutation is beneficial.  The output only ranks mutants by predicted fold-change
relative to the wild type and flags which ones exceed the model's own error band.

Decision rule (derived from the model's measured error: median fold error ~6.5x,
85% of case-study points within +/-10x):
    fold < 3        -> within noise
    3 <= fold < 10  -> weak signal
    10 <= fold < 30 -> candidate (worth experimental testing)
    fold >= 30      -> strong computational signal

Run:  python scripts/14_mutant_sensitivity.py
Reads mutants.fasta from the project root.  Outputs stay local (not published)
because the sequences are confidential.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
# localhost / model loading must not go through the system proxy
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = "127.0.0.1,localhost,::1"

import esm  # noqa: E402

DEV = "cuda"

# --------------------------------------------------------------------------- #
# sequences
# --------------------------------------------------------------------------- #
def parse_fasta(path):
    recs, name, buf = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name:
                recs.append((name, "".join(buf)))
            name, buf = line[1:].strip(), []
        else:
            buf.append(re.sub(r"[^A-Z]", "", line.upper()))
    if name:
        recs.append((name, "".join(buf)))
    return recs


recs = parse_fasta("mutants.fasta")
print(f"loaded {len(recs)} sequences (WT + mutants)", flush=True)
wt_name, wt_seq = recs[0]
assert wt_name.upper().startswith("WT"), recs[0][0]

# --------------------------------------------------------------------------- #
# ESM-2 embeddings
# --------------------------------------------------------------------------- #
print("loading ESM-2 650M ...", flush=True)
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.to(DEV).half().eval()
bc = alphabet.get_batch_converter()


def embed(seq):
    _, _, toks = bc([("q", seq[:1022])])
    with torch.no_grad():
        out = esm_model(toks.to(DEV), repr_layers=[33])["representations"][33]
    L = min(len(seq), 1022)
    return out[0, 1:L + 1].float().mean(0).cpu().numpy()


embs = np.stack([embed(s) for _, s in recs]).astype(np.float32)
print(f"embedded: {embs.shape}", flush=True)

# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class DualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(1280, 512), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(512, 256), nn.GELU())
        self.sub = nn.Sequential(nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(512, 256), nn.GELU())
        self.head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(torch.cat([self.seq(x[:, :1280]), self.sub(x[:, 1280:])], 1)).squeeze(-1)


nn_model = DualNet().to(DEV)
nn_model.load_state_dict(torch.load("models/carotkcat_ens_nn.pt", weights_only=False)["state_dict"])
nn_model.eval()

import pickle
xgb = pickle.load(open("models/carotkcat_ens_xgb.pkl", "rb"))

# --------------------------------------------------------------------------- #
# substrates
# --------------------------------------------------------------------------- #
subs = {}
for line in open("data/substrates_smiles.tsv", encoding="utf-8"):
    if "=>" in line:
        n, s = line.strip().split("=>")
        subs[n.strip()] = s.strip()

TARGETS = [t for t in ["15-cis-phytoene", "geranylgeranyl diphosphate", "beta-carotene"]
           if t in subs]


def verdict(fold):
    if fold < 3:
        return "within noise"
    if fold < 10:
        return "weak signal"
    if fold < 30:
        return "candidate"
    return "strong signal"


all_rows = []
for sub_name in TARGETS:
    mol = Chem.MolFromSmiles(subs[sub_name])
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
                  dtype=np.float32)
    X = np.concatenate([embs, np.repeat(fp[None, :], len(embs), 0)], 1)
    with torch.no_grad():
        p_nn = nn_model(torch.tensor(X, device=DEV)).cpu().numpy()
    p_xgb = xgb.predict(X)
    p = (p_nn + p_xgb) / 2

    wt_log = p[0]
    rows = []
    for i, (name, seq) in enumerate(recs):
        fold = 10 ** (p[i] - wt_log)
        rows.append({
            "mutant": name,
            "len": len(seq),
            "log10_kcat_NN": p_nn[i],
            "log10_kcat_XGB": p_xgb[i],
            "log10_kcat_ensemble": p[i],
            "kcat_s-1": 10 ** p[i],
            "fold_vs_WT": fold,
            "NN_XGB_gap_log10": abs(p_nn[i] - p_xgb[i]),
            "verdict": "WT" if i == 0 else verdict(fold),
        })
    df = pd.DataFrame(rows).sort_values("fold_vs_WT", ascending=False)
    df.insert(0, "substrate", sub_name)
    df.to_csv(f"results/mutant_sensitivity_{sub_name.split()[0]}.tsv", index=False, sep="\t")
    all_rows.append(df)

    print(f"\n=== {sub_name} ===")
    print(df[["mutant", "kcat_s-1", "fold_vs_WT", "NN_XGB_gap_log10", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # figure
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=200)
    colors = {"WT": "#111827", "within noise": "#9ca3af", "weak signal": "#60a5fa",
              "candidate": "#f59e0b", "strong signal": "#dc2626"}
    plot_df = df[df.mutant != "WT"]
    bars = ax.bar(plot_df["mutant"], plot_df["fold_vs_WT"],
                  color=[colors[v] for v in plot_df["verdict"]], edgecolor="#374151", lw=0.6)
    ax.axhline(1.0, color="#111827", lw=1, ls="-")
    for y, lab in [(3, "3x noise floor"), (10, "10x candidate"), (30, "30x strong")]:
        ax.axhline(y, color="#6b7280", lw=0.8, ls="--")
        ax.text(len(plot_df) - 0.4, y * 1.04, lab, fontsize=7, color="#4b5563", ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Predicted fold-change vs wild type")
    ax.set_title(f"Mutant sensitivity — substrate: {sub_name}")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"results/fig_mutants_{sub_name.split()[0]}.png")
    plt.close(fig)

combined = pd.concat(all_rows, ignore_index=True)
combined.to_csv("results/mutant_sensitivity_all.tsv", sep="\t", index=False)
print("\nsaved results/mutant_sensitivity_all.tsv and per-substrate figures")
print("NOTE: sensitivity ranking only — no experimental ground truth for these mutants.")
