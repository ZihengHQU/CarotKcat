# -*- coding: utf-8 -*-
"""Step 13: GNN substrate branch (CarotKcat-GNN) + polyene discrimination test.

Motivation
----------
CarotKcat encodes substrates with Morgan fingerprints, which are *identical* for
phytoene / zeta-carotene / neurosporene (documented limitation in the paper).
Here we replace the fingerprint tower with a molecular-graph GNN and ask two
questions:
  (1) does test accuracy change?
  (2) does the GNN distinguish the closely related polyene substrates?

Outputs
  results/gnn_eval.csv
  results/polyene_discrimination.csv
  models/carotkcat_gnn_{random,group}.pt
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

RDLogger.DisableLog("rdApp.*")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
kcat = pd.read_csv("data/kcat_clean.tsv", sep="\t")
seqs = (
    pd.read_csv("data/sequences.tsv", sep="\t")
    .drop_duplicates("accession").set_index("accession")["sequence"]
)
kcat["sequence"] = kcat["UniprotID"].map(seqs)
kcat = kcat[kcat["sequence"].notna() & kcat["Smiles"].notna()]
kcat["Smiles"] = kcat["Smiles"].astype(str).str.strip()
kcat = kcat[kcat["Smiles"].str.len() > 2]

fp_map = {}
for s in kcat["Smiles"].unique():
    m = Chem.MolFromSmiles(s)
    if m is not None:
        fp_map[s] = np.array(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), dtype=np.float32
        )
kcat = kcat[kcat["Smiles"].isin(fp_map)].reset_index(drop=True)

agg = pd.read_csv("data/dataset_sequence_level.tsv", sep="\t")
emb_all = np.load("models/embeddings.npz")["embs"]
acc2emb = {a: emb_all[i] for i, a in enumerate(agg["UniprotID"])}
kcat = kcat[kcat["UniprotID"].isin(acc2emb)].reset_index(drop=True)

X_seq = np.stack(kcat["UniprotID"].map(acc2emb).values).astype(np.float32)
y = kcat["log10_kcat"].values.astype(np.float32)
groups = kcat["UniprotID"].values
print(f"records: {len(y)}", flush=True)

# --------------------------------------------------------------------------- #
# molecular graphs
# --------------------------------------------------------------------------- #
ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"]
HYBRID = [
    Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
]
MAX_ATOMS = 128


def atom_features(a):
    f = [0.0] * (len(ELEMENTS) + 1)
    sym = a.GetSymbol()
    f[ELEMENTS.index(sym) if sym in ELEMENTS else -1] = 1.0
    deg = [0.0] * 7; deg[min(a.GetDegree(), 6)] = 1.0; f += deg
    chg = [0.0] * 5; chg[min(max(a.GetFormalCharge() + 2, 0), 4)] = 1.0; f += chg
    f.append(1.0 if a.GetIsAromatic() else 0.0)
    hs = [0.0] * 5; hs[min(a.GetTotalNumHs(), 4)] = 1.0; f += hs
    hy = [0.0] * (len(HYBRID) + 1)
    hy[HYBRID.index(a.GetHybridization()) if a.GetHybridization() in HYBRID else -1] = 1.0
    return f + hy


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    n = mol.GetNumAtoms()
    feats = np.stack([atom_features(a) for a in mol.GetAtoms()]).astype(np.float32)
    adj = np.zeros((n, n), dtype=np.float32)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        w = np.sqrt(b.GetBondTypeAsDouble() / 3.0)
        adj[i, j] = adj[j, i] = w
    if n > MAX_ATOMS:
        feats, adj = feats[:MAX_ATOMS], adj[:MAX_ATOMS, :MAX_ATOMS]
    return feats, adj


GRAPH = {s: mol_to_graph(s) for s in kcat["Smiles"].unique()}


def batch_graphs(smiles_list):
    B = len(smiles_list)
    x = np.zeros((B, MAX_ATOMS, ATOM_DIM), np.float32)
    a = np.zeros((B, MAX_ATOMS, MAX_ATOMS), np.float32)
    m = np.zeros((B, MAX_ATOMS), np.float32)
    for k, s in enumerate(smiles_list):
        if s not in GRAPH:
            GRAPH[s] = mol_to_graph(s)
        fe, ad = GRAPH[s]
        n = fe.shape[0]
        x[k, :n], m[k, :n] = fe, 1.0
        ad = ad.copy()
        ad[np.arange(n), np.arange(n)] = 1.0
        d = ad.sum(-1, keepdims=True); d[d == 0] = 1.0
        a[k, :n, :n] = ad / d
    return (torch.tensor(x, device=DEV), torch.tensor(a, device=DEV),
            torch.tensor(m, device=DEV))


class GraphConvGRU(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.msg = nn.Linear(h, h)
        self.gru = nn.Linear(2 * h, 3 * h)

    def forward(self, h, adj):
        m = torch.bmm(adj, self.msg(h))
        z, r, n = torch.chunk(self.gru(torch.cat([h, m], -1)), 3, -1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        return (1 - z) * torch.tanh(n) + z * h


class DualNetGNN(nn.Module):
    def __init__(self, h=128, dropout=0.2):
        super().__init__()
        self.atom_in = nn.Linear(ATOM_DIM, h)
        self.gnn = nn.ModuleList([GraphConvGRU(h) for _ in range(3)])
        self.sub = nn.Sequential(nn.Linear(2 * h, 256), nn.GELU(), nn.Dropout(dropout))
        self.seq = nn.Sequential(
            nn.Linear(1280, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, esm, atoms, adj, amask):
        h = self.atom_in(atoms)
        for layer in self.gnn:
            h = layer(h, adj)
        h = h * amask.unsqueeze(-1)
        cnt = amask.sum(-1, keepdim=True).clamp(min=1.0)
        mean = h.sum(1) / cnt
        mx = h.masked_fill(amask.unsqueeze(-1) == 0, -1e9).max(1).values
        sub = self.sub(torch.cat([mean, mx], -1))
        return self.head(torch.cat([self.seq(esm), sub], -1)).squeeze(-1)


def train(esm_tr, g_tr, y_tr, esm_val, g_val, y_val, esm_te, g_te,
          epochs=200, patience=30):
    m = DualNetGNN().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.SmoothL1Loss()
    Et = torch.tensor(esm_tr, device=DEV); yt = torch.tensor(y_tr, device=DEV)
    Ev = torch.tensor(esm_val, device=DEV); yv = torch.tensor(y_val, device=DEV)
    Ee = torch.tensor(esm_te, device=DEV)
    best, pat, bs = 1e9, 0, None
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(yt), device=DEV)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            p = m(Et[b], g_tr[0][b], g_tr[1][b], g_tr[2][b])
            loss = lossf(p, yt[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        sched.step(); m.eval()
        with torch.no_grad():
            vr = torch.sqrt(((m(Ev, g_val[0], g_val[1], g_val[2]) - yv) ** 2).mean()).item()
        if vr < best:
            best, pat = vr, 0
            bs = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            pat += 1
        if ep % 20 == 0:
            print(f"  ep{ep:3d} val {vr:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if pat > patience:
            break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        return m(Ee, g_te[0], g_te[1], g_te[2]).cpu().numpy(), m, best


rows = []
for proto in ["random", "group"]:
    if proto == "random":
        tr_full, te = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42)
        tr, val = train_test_split(tr_full, test_size=0.10, random_state=7)
    else:
        tr_full, te = next(GroupShuffleSplit(1, test_size=0.15, random_state=42).split(
            X_seq, y, groups))
        # second element is the held-out side -> validation set
        _, vrel = next(GroupShuffleSplit(1, test_size=0.10, random_state=7).split(
            tr_full, y[tr_full], groups[tr_full]))
        val = tr_full[vrel]
        tr = np.setdiff1d(tr_full, val)

    print(f"--- {proto}: {len(tr)}/{len(val)}/{len(te)}", flush=True)
    smis = kcat["Smiles"].values
    g_tr = batch_graphs(smis[tr])
    g_val = batch_graphs(smis[val])
    g_te = batch_graphs(smis[te])

    p, model, bv = train(X_seq[tr], g_tr, y[tr], X_seq[val], g_val, y[val],
                         X_seq[te], g_te)
    torch.save({"state_dict": model.state_dict()}, f"models/carotkcat_gnn_{proto}.pt")
    m = {"Pearson": float(pearsonr(y[te], p)[0]), "R2": float(r2_score(y[te], p)),
         "RMSE": float(np.sqrt(mean_squared_error(y[te], p)))}
    print(f"[{proto}] CarotKcat-GNN {m}", flush=True)
    rows.append({"protocol": proto, "model": "CarotKcat-GNN (ESM+GNN)", **m})
    np.save(f"results/gnn_pred_{proto}.npy", p)

pd.DataFrame(rows).to_csv("results/gnn_eval.csv", index=False)

# --------------------------------------------------------------------------- #
# polyene discrimination: same enzyme, five related substrates
# --------------------------------------------------------------------------- #
subs = {}
for line in open("data/substrates_smiles.tsv", encoding="utf-8"):
    if "=>" in line:
        name, smi = line.strip().split("=>")
        subs[name.strip()] = smi.strip()

polyenes = ["15-cis-phytoene", "phytofluene", "zeta-carotene",
            "neurosporene", "lycopene"]
polyenes = [p for p in polyenes if p in subs]

# use an enzyme already embedded in the corpus (no extra ESM pass needed)
ref_acc = kcat["UniprotID"].iloc[0]
esm_vec = acc2emb[ref_acc][None, :]

g = batch_graphs([subs[p] for p in polyenes])
fps = np.stack([
    np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(subs[p]), 2, nBits=2048),
             dtype=np.float32) for p in polyenes
])

# FP-based dual tower (the shipped CarotKcat) — reuse clean checkpoint if present
class DualNetFP(nn.Module):
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


fp_model = DualNetFP().to(DEV)
fp_model.load_state_dict(torch.load("models/kcat_dual_randomsplit.pt",
                                    weights_only=False)["state_dict"])
fp_model.eval()
Xp = torch.tensor(np.concatenate([np.repeat(esm_vec, len(polyenes), 0), fps], 1),
                  device=DEV)
with torch.no_grad():
    fp_pred = fp_model(Xp).cpu().numpy()

gnn_model = DualNetGNN().to(DEV)
gnn_model.load_state_dict(torch.load("models/carotkcat_gnn_random.pt",
                                     weights_only=False)["state_dict"])
gnn_model.eval()
with torch.no_grad():
    gnn_pred = gnn_model(torch.tensor(np.repeat(esm_vec, len(polyenes), 0), device=DEV),
                         g[0], g[1], g[2]).cpu().numpy()

out = pd.DataFrame({
    "substrate": polyenes,
    "log10kcat_MorganFP": fp_pred,
    "log10kcat_GNN": gnn_pred,
    "kcat_MorganFP": 10 ** fp_pred,
    "kcat_GNN": 10 ** gnn_pred,
})
out["unique_FP"] = out["log10kcat_MorganFP"].round(6).nunique() == len(polyenes)
out.to_csv("results/polyene_discrimination.csv", index=False)
print(out.to_string(index=False))
print(f"FP distinct values: {out['log10kcat_MorganFP'].round(6).nunique()}/{len(polyenes)}")
print(f"GNN distinct values: {out['log10kcat_GNN'].round(6).nunique()}/{len(polyenes)}")
print("saved results/gnn_eval.csv, results/polyene_discrimination.csv")
