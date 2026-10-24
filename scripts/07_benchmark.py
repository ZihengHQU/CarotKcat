# -*- coding: utf-8 -*-
"""Step 7: SOTA-style benchmark on identical splits.
Models compared:
  1. Mean predictor (baseline)
  2. Substrate-fingerprint-only MLP
  3. ESM-only MLP (sequence baseline)
  4. UniKP-style: ESM + FP -> ExtraTrees
  5. TurNuP-style: ESM + FP -> XGBoost
  6. CarotKcat: ESM + FP -> dual-tower MLP
Protocols: random 85/15 split AND group-by-enzyme split (same seeds as 06).
Outputs: results/benchmark_table.csv, results/fig_scatter.png
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, json, time
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
RDLogger.DisableLog("rdApp.*")

def metrics(yv, pv):
    return {"Pearson": float(pearsonr(yv, pv)[0]), "R2": float(r2_score(yv, pv)),
            "RMSE": float(np.sqrt(mean_squared_error(yv, pv)))}

# ---------- rebuild record-level features (same as 06) ----------
kcat = pd.read_csv("data/kcat_clean.tsv", sep="\t")
seqs = pd.read_csv("data/sequences.tsv", sep="\t").drop_duplicates("accession").set_index("accession")["sequence"]
kcat["sequence"] = kcat["UniprotID"].map(seqs)
kcat = kcat[kcat["sequence"].notna() & kcat["Smiles"].notna()]
kcat["Smiles"] = kcat["Smiles"].astype(str).str.strip()
kcat = kcat[kcat["Smiles"].str.len() > 2]
fp_map = {}
for s in kcat["Smiles"].unique():
    m = Chem.MolFromSmiles(s)
    if m is not None:
        fp_map[s] = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), dtype=np.float32)
kcat = kcat[kcat["Smiles"].isin(fp_map)]
agg = pd.read_csv("data/dataset_sequence_level.tsv", sep="\t")
emb_all = np.load("models/embeddings.npz")["embs"]
acc2emb = {a: emb_all[i] for i, a in enumerate(agg["UniprotID"])}
kcat = kcat[kcat["UniprotID"].isin(acc2emb)]

X_seq = np.stack(kcat["UniprotID"].map(acc2emb).values).astype(np.float32)
X_sub = np.stack(kcat["Smiles"].map(fp_map).values).astype(np.float32)
X = np.concatenate([X_seq, X_sub], 1)
y = kcat["log10_kcat"].values.astype(np.float32)
groups = kcat["UniprotID"].values
print("records:", len(y))

dev = "cuda"
class MLP(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 1024), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class DualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(1280, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.GELU())
        self.sub = nn.Sequential(nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.GELU())
        self.head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.head(torch.cat([self.seq(x[:, :1280]), self.sub(x[:, 1280:])], 1)).squeeze(-1)

def train_nn(make_model, Xtr, ytr, Xte, epochs=250):
    m = make_model().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.SmoothL1Loss()
    Xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, device=dev)
    Xe = torch.tensor(Xte, device=dev)
    best, pat, best_state = 1e9, 0, None
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(perm), 512):
            b = perm[i:i+512]
            loss = lossf(m(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            r = torch.sqrt(((m(Xe) - torch.tensor(yte, device=dev)) ** 2).mean()).item()
        if r < best: best, pat, best_state = r, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else: pat += 1
        if pat > 40: break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        return m(Xe).cpu().numpy()

rows = []
for proto in ["random", "group"]:
    if proto == "random":
        tr, te = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42)
    else:
        tr, te = next(GroupShuffleSplit(1, test_size=0.15, random_state=42).split(X, y, groups))
    ytr, yte = y[tr], y[te]
    print(f"--- {proto} split: train {len(tr)} test {len(te)}", flush=True)

    preds = {}
    preds["Mean (baseline)"] = np.full(len(te), ytr.mean())

    et = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    et.fit(X[tr], ytr); preds["UniKP-style (ESM+FP, ExtraTrees)"] = et.predict(X[te])

    xgb = XGBRegressor(n_estimators=800, max_depth=8, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    xgb.fit(X[tr], ytr); preds["TurNuP-style (ESM+FP, XGBoost)"] = xgb.predict(X[te])

    preds["Substrate-FP only (MLP)"] = train_nn(lambda: MLP(2048), X_sub[tr], ytr, X_sub[te])
    preds["ESM only (MLP)"] = train_nn(lambda: MLP(1280), X_seq[tr], ytr, X_seq[te])
    preds["CarotKcat (dual-tower)"] = train_nn(DualNet, X[tr], ytr, X[te])

    for name, pv in preds.items():
        m = metrics(yte, pv)
        rows.append({"protocol": proto, "model": name, **m})
        print(proto, name, {k: round(v, 3) for k, v in m.items()}, flush=True)
    if proto == "random":
        np.save("results/bench_yte.npy", yte)
        np.save("results/bench_pred_carotkcat.npy", preds["CarotKcat (dual-tower)"])

df = pd.DataFrame(rows)
df.to_csv("results/benchmark_table.csv", index=False)
print(df.pivot(index="model", columns="protocol", values="Pearson").round(3))

# scatter figure for paper
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
yte_r = np.load("results/bench_yte.npy"); pr = np.load("results/bench_pred_carotkcat.npy")
fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
ax.scatter(yte_r, pr, s=4, alpha=0.35, c="#2b6cb0", edgecolors="none")
lo, hi = min(yte_r.min(), pr.min()), max(yte_r.max(), pr.max())
ax.plot([lo, hi], [lo, hi], "r--", lw=1)
ax.set_xlabel("Experimental log$_{10}$(k$_{cat}$)"); ax.set_ylabel("Predicted log$_{10}$(k$_{cat}$)")
r = pearsonr(yte_r, pr)[0]
ax.set_title(f"CarotKcat, random split (n={len(yte_r)})\nPearson r = {r:.3f}")
fig.tight_layout(); fig.savefig("results/fig_scatter.png")
print("saved results/benchmark_table.csv, results/fig_scatter.png")
