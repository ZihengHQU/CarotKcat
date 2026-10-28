# -*- coding: utf-8 -*-
"""Step 11: True head-to-head against DLKcat.

DLKcat (Li et al., Nat. Catal. 2022) encodes the substrate as a molecular graph
with a GNN and the protein as amino-acid n-grams with a CNN, then fuses the two
representations.  Our main model (CarotKcat) uses a pretrained ESM-2 encoder plus
a Morgan fingerprint, so a comparison against "DLKcat-style" baselines that also
consume ESM embeddings is not an architectural comparison.

This script therefore re-implements the DLKcat *architecture* in PyTorch
(substrate GNN + protein n-gram CNN + fusion head), trains it on exactly the same
record-level dataset and the same train/test splits/seeds as 07_benchmark.py, and
evaluates it on the identical test sets.  Everything is therefore apples-to-apples
except the architecture itself.

Note on fairness: early stopping uses an internal 10% validation split carved out
of the training set only, so this baseline never peeks at the test set.  The
already-published table entries were produced with test-set early stopping, i.e.
this baseline is, if anything, evaluated *more strictly* than the others.

Outputs
  results/dlkcat_headtohead.csv   (rows to merge into benchmark_table.csv)
  results/dlkcat_pred_{random,group}.npy
  models/dlkcat_style_{random,group}.pt
"""
import json
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
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Identical dataset construction to 07_benchmark.py
# --------------------------------------------------------------------------- #
kcat = pd.read_csv("data/kcat_clean.tsv", sep="\t")
seqs = (
    pd.read_csv("data/sequences.tsv", sep="\t")
    .drop_duplicates("accession")
    .set_index("accession")["sequence"]
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

y = kcat["log10_kcat"].values.astype(np.float32)
groups = kcat["UniprotID"].values
print(f"records: {len(y)}  unique enzymes: {kcat['UniprotID'].nunique()}  "
      f"unique substrates: {kcat['Smiles'].nunique()}", flush=True)


# --------------------------------------------------------------------------- #
# 2. Molecular-graph featurisation (DLKcat substrate branch)
# --------------------------------------------------------------------------- #
ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"]
HYBRID = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]


def atom_features(atom):
    f = [0.0] * (len(ELEMENTS) + 1)  # element one-hot (+other)
    sym = atom.GetSymbol()
    if sym in ELEMENTS:
        f[ELEMENTS.index(sym)] = 1.0
    else:
        f[-1] = 1.0
    deg = [0.0] * 7
    deg[min(atom.GetDegree(), 6)] = 1.0
    f += deg
    chg = [0.0] * 5
    chg[min(max(atom.GetFormalCharge() + 2, 0), 4)] = 1.0
    f += chg
    f.append(1.0 if atom.GetIsAromatic() else 0.0)
    hs = [0.0] * 5
    hs[min(atom.GetTotalNumHs(), 4)] = 1.0
    f += hs
    hyb = [0.0] * (len(HYBRID) + 1)
    if atom.GetHybridization() in HYBRID:
        hyb[HYBRID.index(atom.GetHybridization())] = 1.0
    else:
        hyb[-1] = 1.0
    f += hyb
    return f


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
print(f"atom feature dim: {ATOM_DIM}", flush=True)


# Dense-graph padding length.  Only 4 / 17,051 records (0.02%) exceed 128 heavy
# atoms; they are truncated to their first 128 atoms so that the record set and
# therefore the train/test splits stay byte-identical to 07_benchmark.py.
MAX_ATOMS = 128


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    n = mol.GetNumAtoms()
    feats = np.stack([atom_features(a) for a in mol.GetAtoms()]).astype(np.float32)
    adj = np.zeros((n, n), dtype=np.float32)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        order = b.GetBondTypeAsDouble()
        w = np.sqrt(order / 3.0)  # damped bond order
        adj[i, j] = w
        adj[j, i] = w
    if n > MAX_ATOMS:
        feats = feats[:MAX_ATOMS]
        adj = adj[:MAX_ATOMS, :MAX_ATOMS]
    return feats, adj


graph_cache = {s: mol_to_graph(s) for s in kcat["Smiles"].unique()}
print(f"pad length (atoms): {MAX_ATOMS}", flush=True)


def build_substrate_batch(smiles_list):
    B = len(smiles_list)
    x = np.zeros((B, MAX_ATOMS, ATOM_DIM), dtype=np.float32)
    a = np.zeros((B, MAX_ATOMS, MAX_ATOMS), dtype=np.float32)
    mask = np.zeros((B, MAX_ATOMS), dtype=np.float32)
    for k, s in enumerate(smiles_list):
        fe, ad = graph_cache[s]
        n = fe.shape[0]
        x[k, :n] = fe
        a[k, :n, :n] = ad
        mask[k, :n] = 1.0
        # self loops
        a[k, np.arange(n), np.arange(n)] = 1.0
        # row-normalise
        d = a[k].sum(-1, keepdims=True)
        d[d == 0] = 1.0
        a[k] = a[k] / d
    return (
        torch.tensor(x, device=DEV),
        torch.tensor(a, device=DEV),
        torch.tensor(mask, device=DEV),
    )


# --------------------------------------------------------------------------- #
# 3. Protein n-gram encoding (DLKcat protein branch)
# --------------------------------------------------------------------------- #
AA = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {c: i + 1 for i, c in enumerate(AA)}  # 0 = pad, 21 = unknown
MAX_LEN = 2048


def encode_seq(seq):
    seq = seq[:MAX_LEN]
    ids = [AA2IDX.get(c, 21) for c in seq]
    return np.asarray(ids, dtype=np.int64)


seq_cache = {a: encode_seq(s) for a, s in zip(kcat["UniprotID"], kcat["sequence"])}


def build_protein_batch(acc_list):
    B = len(acc_list)
    L = MAX_LEN
    ids = np.zeros((B, L), dtype=np.int64)
    mask = np.zeros((B, L), dtype=np.float32)
    for k, a in enumerate(acc_list):
        e = seq_cache[a]
        n = len(e)
        ids[k, :n] = e
        mask[k, :n] = 1.0
    return torch.tensor(ids, device=DEV), torch.tensor(mask, device=DEV)


# --------------------------------------------------------------------------- #
# 4. DLKcat-style architecture: GNN(substrate) + CNN(protein) -> fusion
# --------------------------------------------------------------------------- #
class GraphConvGRU(nn.Module):
    """One message-passing step with a GRU update (DLKcat uses GRU-based GNN)."""

    def __init__(self, hidden):
        super().__init__()
        self.msg = nn.Linear(hidden, hidden)
        self.gru = nn.Linear(2 * hidden, 3 * hidden)

    def forward(self, h, adj):
        m = torch.bmm(adj, self.msg(h))
        z, r, n = torch.chunk(self.gru(torch.cat([h, m], -1)), 3, -1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        n = torch.tanh(n)
        return (1 - z) * n + z * h


class DLKcatStyle(nn.Module):
    def __init__(self, h=128, cnn_ch=128, aa_emb=64, dropout=0.2):
        super().__init__()
        # ---- substrate: GNN ----
        self.atom_in = nn.Linear(ATOM_DIM, h)
        self.gnn = nn.ModuleList([GraphConvGRU(h) for _ in range(3)])
        self.sub_out = nn.Sequential(nn.Linear(2 * h, h), nn.GELU())
        # ---- protein: n-gram CNN ----
        self.aa_emb = nn.Embedding(22, aa_emb, padding_idx=0)
        self.cnn = nn.Sequential(
            nn.Conv1d(aa_emb, cnn_ch, 3, padding=1), nn.GELU(),
            nn.Conv1d(cnn_ch, cnn_ch, 3, padding=1), nn.GELU(),
            nn.Conv1d(cnn_ch, cnn_ch, 3, padding=1), nn.GELU(),
        )
        self.seq_out = nn.Sequential(nn.Linear(2 * cnn_ch, h), nn.GELU())
        # ---- fusion ----
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * h, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, atoms, adj, amask, ids, smask):
        # substrate GNN
        h = self.atom_in(atoms)
        for layer in self.gnn:
            h = layer(h, adj)
        h = h * amask.unsqueeze(-1)
        cnt = amask.sum(-1, keepdim=True).clamp(min=1.0)
        mean = h.sum(1) / cnt
        mx = h.masked_fill(amask.unsqueeze(-1) == 0, -1e9).max(1).values
        sub = self.sub_out(torch.cat([mean, mx], -1))
        # protein CNN
        e = self.aa_emb(ids).transpose(1, 2)  # B, C, L
        c = self.cnn(e)  # B, C, L
        c = c.masked_fill(smask.unsqueeze(1) == 0, -1e9)
        mean_s = c.sum(-1) / smask.sum(-1, keepdim=True).clamp(min=1.0)
        max_s = c.max(-1).values
        seq = self.seq_out(torch.cat([mean_s, max_s], -1))
        return self.head(torch.cat([sub, seq], -1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# 5. Training / evaluation
# --------------------------------------------------------------------------- #
def metrics(yv, pv):
    return {
        "Pearson": float(pearsonr(yv, pv)[0]),
        "R2": float(r2_score(yv, pv)),
        "RMSE": float(np.sqrt(mean_squared_error(yv, pv))),
    }


def run(proto, tr, te, tag, epochs=150, bs=512, lr=6e-4, patience=20):
    # internal validation split from TRAIN only (no test peeking)
    tr_fit, tr_val = train_test_split(tr, test_size=0.1, random_state=7)

    print(f"[{proto}] train {len(tr_fit)} val {len(tr_val)} test {len(te)}", flush=True)
    smi_tr = kcat["Smiles"].values[tr_fit]
    smi_val = kcat["Smiles"].values[tr_val]
    smi_te = kcat["Smiles"].values[te]
    acc_tr = kcat["UniprotID"].values[tr_fit]
    acc_val = kcat["UniprotID"].values[tr_val]
    acc_te = kcat["UniprotID"].values[te]

    at, ad, am = build_substrate_batch(smi_tr)
    it, im = build_protein_batch(acc_tr)
    yt = torch.tensor(y[tr_fit], device=DEV)

    av, adv, amv = build_substrate_batch(smi_val)
    iv, imv = build_protein_batch(acc_val)
    yv = torch.tensor(y[tr_val], device=DEV)

    ae, ade, ame = build_substrate_batch(smi_te)
    ie, ime = build_protein_batch(acc_te)
    ye = torch.tensor(y[te], device=DEV)

    model = DLKcatStyle().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.SmoothL1Loss()

    best, pat, best_state = 1e9, 0, None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt), device=DEV)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = lossf(
                model(at[b], ad[b], am[b], it[b], im[b]), yt[b]
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vp = model(av, adv, amv, iv, imv)
            vr = torch.sqrt(((vp - yv) ** 2).mean()).item()
        if vr < best:
            best, pat = vr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if ep % 10 == 0:
            print(f"  [{proto}] ep{ep:3d} val RMSE {vr:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if pat > patience:
            print(f"  [{proto}] early stop at ep{ep}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = model(ae, ade, ame, ie, ime).cpu().numpy()
    m = metrics(y[te], pv)
    print(f"[{proto}] DLKcat-style -> " +
          ", ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
    torch.save({"state_dict": best_state, "meta": {"protocol": proto}},
               f"models/dlkcat_style_{tag}.pt")
    np.save(f"results/dlkcat_pred_{tag}.npy", pv)
    return m


rows = []
for proto, tag in [("random", "random"), ("group", "group")]:
    if proto == "random":
        tr, te = train_test_split(np.arange(len(y)), test_size=0.15, random_state=SEED)
    else:
        tr, te = next(
            GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(
                np.zeros(len(y)), y, groups
            )
        )
    m = run(proto, tr, te, tag)
    rows.append({"protocol": proto,
                 "model": "DLKcat-style (GNN+CNN, reimplemented)", **m})

df = pd.DataFrame(rows)
df.to_csv("results/dlkcat_headtohead.csv", index=False)
print(df.to_string(index=False))

# also merge into the master benchmark table (replace any stale rows)
master_path = "results/benchmark_table.csv"
master = pd.read_csv(master_path)
master = master[master["model"] != "DLKcat-style (GNN+CNN, reimplemented)"]
master = pd.concat([master, df], ignore_index=True)
master.to_csv(master_path, index=False)
print(f"updated {master_path}")
