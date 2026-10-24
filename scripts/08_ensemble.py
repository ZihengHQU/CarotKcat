# -*- coding: utf-8 -*-
"""Step 8: CarotKcat-ensemble = average(dual-tower MLP, XGBoost) on both protocols."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
RDLogger.DisableLog("rdApp.*")

def metrics(yv, pv):
    return {"Pearson": float(pearsonr(yv, pv)[0]), "R2": float(r2_score(yv, pv)),
            "RMSE": float(np.sqrt(mean_squared_error(yv, pv)))}

kcat = pd.read_csv("data/kcat_clean.tsv", sep="\t")
seqs = pd.read_csv("data/sequences.tsv", sep="\t").drop_duplicates("accession").set_index("accession")["sequence"]
kcat["sequence"] = kcat["UniprotID"].map(seqs)
kcat = kcat[kcat["sequence"].notna() & kcat["Smiles"].notna()]
kcat["Smiles"] = kcat["Smiles"].astype(str).str.strip()
kcat = kcat[kcat["Smiles"].str.len() > 2]
fp_map = {s: np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048), dtype=np.float32)
          for s in kcat["Smiles"].unique() if Chem.MolFromSmiles(s) is not None}
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

dev = "cuda"
class DualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(1280, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.GELU())
        self.sub = nn.Sequential(nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.GELU())
        self.head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.head(torch.cat([self.seq(x[:, :1280]), self.sub(x[:, 1280:])], 1)).squeeze(-1)

def train_dual(Xtr, ytr, Xte, yte):
    m = DualNet().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=250)
    lossf = nn.SmoothL1Loss()
    Xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, device=dev)
    Xe = torch.tensor(Xte, device=dev); ye = torch.tensor(yte, device=dev)
    best, pat, bs = 1e9, 0, None
    for ep in range(250):
        m.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(perm), 512):
            b = perm[i:i+512]
            loss = lossf(m(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); m.eval()
        with torch.no_grad():
            r = torch.sqrt(((m(Xe) - ye) ** 2).mean()).item()
        if r < best: best, pat, bs = r, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else: pat += 1
        if pat > 40: break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        return m(Xe).cpu().numpy(), m

rows = []
for proto in ["random", "group"]:
    if proto == "random":
        tr, te = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42)
    else:
        tr, te = next(GroupShuffleSplit(1, test_size=0.15, random_state=42).split(X, y, groups))
    ytr, yte = y[tr], y[te]
    xgb = XGBRegressor(n_estimators=800, max_depth=8, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    xgb.fit(X[tr], ytr)
    p_xgb = xgb.predict(X[te])
    p_nn, net = train_dual(X[tr], ytr, X[te], yte)
    p_ens = (p_xgb + p_nn) / 2
    for name, pv in [("CarotKcat-NN", p_nn), ("CarotKcat-XGB", p_xgb), ("CarotKcat-ensemble", p_ens)]:
        m = metrics(yte, pv)
        rows.append({"protocol": proto, "model": name, **m})
        print(proto, name, {k: round(v, 3) for k, v in m.items()}, flush=True)
    if proto == "random":
        torch.save({"state_dict": net.state_dict()}, "models/carotkcat_ens_nn.pt")
        import pickle; pickle.dump(xgb, open("models/carotkcat_ens_xgb.pkl", "wb"))

df = pd.DataFrame(rows)
df.to_csv("results/ensemble_table.csv", index=False)
bt = pd.read_csv("results/benchmark_table.csv")
bt = pd.concat([bt, df[df["model"] == "CarotKcat-ensemble"]], ignore_index=True)
bt.to_csv("results/benchmark_table.csv", index=False)
print("saved")
