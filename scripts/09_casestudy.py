# -*- coding: utf-8 -*-
"""Step 9: Literature case study for carotenoid-pathway enzymes.
All case enzymes are ABSENT from the training set (verified) -> true held-out validation.
Literature kcat sources:
  [1] Lavrijsen/Rousseau et al., Sci Rep 2021 (s41598-020-77876-4), Suppl. Table S4 (COPASI fit)
  [2] Berviller-Magnan et al., Mol Syst Biol 2024 (PMC11535393), in vivo kcat^cell
  [3] Iwata-Reuyl et al., Biochemistry 2003 (in vitro, P. agglomerans CrtB)
  [4] Kavanagh et al., JBC 2006 (human GGPS1, kcat 0.204 s-1)
  [5] Chang et al., JBC 2006 (yeast BTS1, kcat 0.025 s-1)
  [6] BRENDA-curated Pantoea CrtE 0.8 s-1
  [7] Fraser et al., JBC 1997 (CrtZ specific activities, pmol/h/mg)
"""
import json, pickle
import numpy as np, pandas as pd, torch, torch.nn as nn, esm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RDLogger.DisableLog("rdApp.*")

SMI = {
    "phytoene":       "CC(=CCCC(=CCCC(=CCCC(=CC=CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C)C)C)C)C",
    "phytofluene":    "CC(=CCCC(=CCCC(=CCCC(=CC=CC=C(C)C=CC=C(C)CCC=C(C)CCC=C(C)C)C)C)C)C",
    "zeta-carotene":  "CC(=CCCC(=CCCC(=CC=CC(=CC=CC=C(C)C=CC=C(C)CCC=C(C)CCC=C(C)C)C)C)C)C",
    "neurosporene":   "CC(=CCCC(=CCCC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC=C(C)CCC=C(C)C)C)C)C)C",
    "GGPP":           "CC(=CCCC(=CCCC(=CCCC(=CCOP(=O)(O)OP(=O)(O)O)C)C)C)C",
    "FPP":            "CC(=CCCC(=CCCC(=CCOP(=O)(O)OP(=O)(O)O)C)C)C",
    "beta-carotene":  "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(CCCC2(C)C)C)C)C",
    "canthaxanthin":  "CC1=C(C(CCC1=O)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(C(=O)CCC2(C)C)C)C)C",
}

# name, accession, substrate, lit kcat (s-1), panel, source
CASES = [
    ("CrtI R.capsulatus",    "P17054",      "phytoene", 0.003,  "A", "[1]"),
    ("CrtI M.aurum",         "Q9K566",      "phytoene", 0.1,    "A", "[1]"),
    ("CrtI B.trispora",      "Q67GI0",      "phytoene", 0.1,    "A", "[1]"),
    ("CrtI P.ananatis",      "P21685",      "phytoene", 0.1,    "A", "[1]"),
    ("CrtI E.agglomerans",   "E9LFG2",      "phytoene", 0.1,    "A", "[1]"),
    ("CrtB P.agglomerans (in vitro)", "D5KXJ0", "GGPP", 14.0,   "B", "[3]"),
    ("CrtB P.ananatis (in vivo)",     "P21683", "GGPP", 6.0,    "B", "[2]"),
    ("CrtE P.ananatis",      "P21684",      "FPP",      0.8,    "B", "[6]"),
    ("CrtI P.ananatis",      "P21685",      "phytoene", 0.1,    "B", "[1]"),
    ("CrtI X.dendrorhous (in vivo)", "A0A0P0KMF9", "phytoene", 0.030, "B", "[2]"),
    ("GGPS1 human",          "O95749",      "FPP",      0.204,  "C", "[4]"),
    ("BTS1 yeast",           "Q12051",      "FPP",      0.025,  "C", "[5]"),
    ("CrtE P.ananatis",      "P21684",      "FPP",      0.8,    "C", "[6]"),
    # Panel D: substrate discrimination, relative specific activity (pmol/h/mg) [7]
    ("CrtZ P.ananatis",      "P21688",      "beta-carotene", 37.0, "D", "[7]"),
    ("CrtZ P.ananatis",      "P21688",      "canthaxanthin", 6.0,  "D", "[7]"),
    # Panel E: CrtI R.capsulatus per-step substrates [1]
    ("CrtI Rc step1",        "P17054",      "phytoene",      0.003,  "E", "[1]"),
    ("CrtI Rc step2",        "P17054",      "phytofluene",   0.001,  "E", "[1]"),
    ("CrtI Rc step3",        "P17054",      "zeta-carotene", 0.04,   "E", "[1]"),
    ("CrtI Rc step4",        "P17054",      "neurosporene",  0.00024,"E", "[1]"),
]

def load_seq(acc):
    if acc == "E9LFG2":
        return "".join(open("data/case_E9LFG2.fasta").read().splitlines()[1:])
    j = json.load(open(f"data/case_{acc}.json"))
    return j["sequence"]["value"]

# ESM-2 embeddings for unique enzymes
accs = sorted({c[1] for c in CASES})
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.cuda().half().eval()
bc = alphabet.get_batch_converter()
emb = {}
for a in accs:
    s = load_seq(a)
    _, _, toks = bc([(a, s[:1022])])
    with torch.no_grad():
        out = esm_model(toks.cuda(), repr_layers=[33])["representations"][33]
    L = min(len(s), 1022)
    emb[a] = out[0, 1:L+1].float().mean(0).cpu().numpy()
    print("embedded", a, len(s), flush=True)

def fp(smi):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048), dtype=np.float32)

X = np.stack([np.concatenate([emb[a], fp(SMI[sub])]) for _, a, sub, *_ in CASES]).astype(np.float32)

class DualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(1280, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.GELU())
        self.sub = nn.Sequential(nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.GELU())
        self.head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.head(torch.cat([self.seq(x[:, :1280]), self.sub(x[:, 1280:])], 1)).squeeze(-1)

net = DualNet().cuda()
net.load_state_dict(torch.load("models/carotkcat_ens_nn.pt", weights_only=False)["state_dict"]); net.eval()
xgb = pickle.load(open("models/carotkcat_ens_xgb.pkl", "rb"))
Xt = torch.tensor(X, device="cuda")
with torch.no_grad():
    p_nn = net(Xt).cpu().numpy()
p_xgb = xgb.predict(X)
p_ens = (p_nn + p_xgb) / 2

df = pd.DataFrame(CASES, columns=["enzyme", "accession", "substrate", "lit_kcat", "panel", "source"])
df["pred_log10_nn"] = p_nn; df["pred_log10_xgb"] = p_xgb
df["pred_kcat"] = 10 ** p_ens
df["lit_log10"] = np.log10(df["lit_kcat"])
df.to_csv("results/case_study.tsv", sep="\t", index=False)

# ---- statistics on quantitative panels A+B+C (kcat in s-1) ----
q = df[df["panel"].isin(["A", "B", "C"])]
sp_all = spearmanr(q["lit_kcat"], q["pred_kcat"])[0]
pe_all = pearsonr(q["lit_log10"], p_ens[df["panel"].isin(["A", "B", "C"])])[0]
sp_A = spearmanr(df[df.panel=="A"]["lit_kcat"], df[df.panel=="A"]["pred_kcat"])[0]
sp_B = spearmanr(df[df.panel=="B"]["lit_kcat"], df[df.panel=="B"]["pred_kcat"])[0]
sp_C = spearmanr(df[df.panel=="C"]["lit_kcat"], df[df.panel=="C"]["pred_kcat"])[0]
sp_E = spearmanr(df[df.panel=="E"]["lit_kcat"], df[df.panel=="E"]["pred_kcat"])[0]
fold = np.abs(np.log10(q["pred_kcat"] / q["lit_kcat"]))
d = df[df.panel=="D"]
d_ratio_lit = d.iloc[0]["lit_kcat"] / d.iloc[1]["lit_kcat"]
d_ratio_pred = d.iloc[0]["pred_kcat"] / d.iloc[1]["pred_kcat"]
stats = {"spearman_ABC": float(sp_all), "pearson_log10_ABC": float(pe_all),
         "spearman_A_crtI": float(sp_A), "spearman_B_pathway": float(sp_B),
         "spearman_C_ggpps": float(sp_C),
         "spearman_E_crtI_steps": float(sp_E),
         "median_fold_error_ABC": float(10 ** fold.median()),
         "max_fold_error_ABC": float(10 ** fold.max()),
         "frac_within_10fold_ABC": float((fold <= 1).mean()),
         "note_E": "phytofluene/zeta-carotene/neurosporene share identical Morgan-r2-2048 fingerprints (verified) -> model gives identical predictions; limitation of fingerprint substrate encoding",
         "note_A": "literature kcat of the four I4 CrtI enzymes (0.1 s-1) sits at the upper constraint bound of the COPASI fit in [1], i.e. only a lower bound",
         "D_crtZ_lit_ratio_bcarotene_vs_canthaxanthin": float(d_ratio_lit),
         "D_crtZ_pred_ratio": float(d_ratio_pred)}
print(json.dumps(stats, indent=2))
json.dump(stats, open("results/case_study_stats.json", "w"), indent=2)

# ---- figure ----
fig, axes = plt.subplots(2, 2, figsize=(10, 8.4), dpi=200)
for ax, panel, title in zip(axes.flat[:3], ["A", "B", "C"],
                            ["A  CrtI orthologs (phytoene desaturation)",
                             "B  Pathway-level enzymes",
                             "C  GGPP synthase family"]):
    sub = df[df.panel == panel]
    ax.scatter(sub["lit_kcat"], sub["pred_kcat"], s=55, c="#d62728", zorder=3)
    lo = min(sub["lit_kcat"].min(), sub["pred_kcat"].min()) / 5
    hi = max(sub["lit_kcat"].max(), sub["pred_kcat"].max()) * 5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    ax.fill_between([lo, hi], [lo/10, hi/10], [lo*10, hi*10], color="gray", alpha=0.12,
                    label="±10-fold")
    for _, r in sub.iterrows():
        ax.annotate(r["enzyme"].split(" ", 1)[1] if " " in r["enzyme"] else r["enzyme"],
                    (r["lit_kcat"], r["pred_kcat"]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Literature kcat (s$^{-1}$)"); ax.set_ylabel("CarotKcat predicted kcat (s$^{-1}$)")
    ax.set_title(title, fontsize=10)
axd = axes.flat[3]
sub = df[df.panel == "D"]
xpos = np.arange(len(sub))
w = 0.38
axd.bar(xpos - w/2, sub["lit_kcat"], w, label="Literature (rel. activity)", color="#1f77b4")
axd.bar(xpos + w/2, sub["pred_kcat"] * (sub["lit_kcat"].max() / sub["pred_kcat"].max()), w,
        label="CarotKcat (rescaled)", color="#d62728")
axd.set_xticks(xpos); axd.set_xticklabels(sub["substrate"], fontsize=8)
axd.set_ylabel("relative activity"); axd.legend(fontsize=7)
axd.set_title("D  CrtZ substrate discrimination (P.ananatis)", fontsize=10)
fig.tight_layout()
fig.savefig("results/fig_case_study.png")
print("saved results/case_study.tsv, results/fig_case_study.png")
