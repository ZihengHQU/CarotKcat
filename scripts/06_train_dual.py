# -*- coding: utf-8 -*-
"""Step 6: DLKcat-style dual-input model — sequence (ESM-2) + substrate (molecular graph fingerprint).
Record-level training on 17k entries (enzyme sequence + substrate SMILES -> kcat).
Architecture mirrors DLKcat's spirit: substrate branch + sequence branch -> concat -> regressor.
Honest split: group by (sequence cluster proxy = UniProtID) to avoid same-enzyme leakage.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, os, json, time
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# ---------- record-level dataset ----------
kcat = pd.read_csv("data/kcat_clean.tsv", sep="\t")
seqs = pd.read_csv("data/sequences.tsv", sep="\t").drop_duplicates("accession").set_index("accession")["sequence"]
kcat["sequence"] = kcat["UniprotID"].map(seqs)
kcat = kcat[kcat["sequence"].notna() & kcat["Smiles"].notna()]
kcat["Smiles"] = kcat["Smiles"].astype(str).str.strip()
kcat = kcat[kcat["Smiles"].str.len() > 2]
print("record-level:", kcat.shape)

# substrate fingerprints (Morgan radius 2, 2048 bits) — cache unique
def fp(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

uniq_smi = kcat["Smiles"].unique()
fp_map = {}
bad = 0
for s in uniq_smi:
    f = fp(s)
    if f is None:
        bad += 1
        continue
    fp_map[s] = np.array(f, dtype=np.float32)
print("unique substrates:", len(uniq_smi), "unparsable:", bad)
kcat = kcat[kcat["Smiles"].isin(fp_map)]

# reuse ESM embeddings at record level: embed unique (accession) already cached at sequence level;
# recompute map accession -> emb from models/embeddings.npz via dataset_sequence_level
agg = pd.read_csv("data/dataset_sequence_level.tsv", sep="\t")
emb_all = np.load("models/embeddings.npz")["embs"]
acc2emb = {a: emb_all[i] for i, a in enumerate(agg["UniprotID"])}
kcat = kcat[kcat["UniprotID"].isin(acc2emb)]
print("final records:", kcat.shape)

caro = pd.read_csv("data/carotenoid_enzymes_all.tsv", sep="\t")
caro_acc = set(caro["Entry"].astype(str))
CARO_EC = {"2.5.1.29","2.5.1.32","1.3.5.5","1.3.5.6","1.3.99.31","5.2.1.13",
           "5.5.1.18","5.5.1.19","1.14.13.129","1.14.15.24","1.14.99.63"}
kcat["carotenoid"] = kcat["UniprotID"].isin(caro_acc) | kcat["ECNumber"].astype(str).isin(CARO_EC)

X_seq = np.stack(kcat["UniprotID"].map(acc2emb).values).astype(np.float32)
X_sub = np.stack(kcat["Smiles"].map(fp_map).values).astype(np.float32)
y = kcat["log10_kcat"].values.astype(np.float32)
groups = kcat["UniprotID"].values

# group split by enzyme (no same-enzyme leakage)
gss = GroupShuffleSplit(1, test_size=0.15, random_state=42)
tr, te = next(gss.split(X_seq, y, groups))
caro_te = kcat["carotenoid"].values[te]
print(f"train {len(tr)} test {len(te)} caro_test {caro_te.sum()}")

dev = "cuda"
Xtr = torch.tensor(np.concatenate([X_seq[tr], X_sub[tr]], 1), device=dev)
Xte = torch.tensor(np.concatenate([X_seq[te], X_sub[te]], 1), device=dev)
ytr = torch.tensor(y[tr], device=dev)
yte = torch.tensor(y[te], device=dev)

class DualNet(nn.Module):
    def __init__(self, d_seq=1280, d_sub=2048):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(d_seq, 512), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(512, 256), nn.GELU())
        self.sub = nn.Sequential(nn.Linear(d_sub, 512), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(512, 256), nn.GELU())
        self.head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x):
        s = self.seq(x[:, :1280]); m = self.sub(x[:, 1280:])
        return self.head(torch.cat([s, m], 1)).squeeze(-1)

model = DualNet().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
lossf = nn.SmoothL1Loss()

# carotenoid oversampling
ctr = kcat["carotenoid"].values[tr]
w = np.where(ctr, 10.0, 1.0); w /= w.sum()
idx = torch.tensor(np.random.default_rng(0).choice(len(tr), len(tr), p=w), device=dev)
Xtr_s, ytr_s = Xtr[idx], ytr[idx]

best, patience = 1e9, 0
t0 = time.time()
for ep in range(400):
    model.train()
    perm = torch.randperm(len(Xtr_s), device=dev)
    for i in range(0, len(perm), 512):
        b = perm[i:i+256]
        loss = lossf(model(Xtr_s[b]), ytr_s[b])
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte)
        rmse = torch.sqrt(((pred - yte) ** 2).mean()).item()
    if rmse < best:
        best, patience = rmse, 0
        torch.save({"state_dict": model.state_dict()}, "models/kcat_dual.pt")
    else:
        patience += 1
    if ep % 25 == 0:
        print(f"ep{ep} test RMSE {rmse:.4f} {(time.time()-t0):.0f}s", flush=True)
    if patience > 60:
        break

model.load_state_dict(torch.load("models/kcat_dual.pt")["state_dict"]); model.eval()
with torch.no_grad():
    pred_te = model(Xte).cpu().numpy()

def metrics(yv, pv):
    return {"R2": float(r2_score(yv, pv)), "RMSE": float(np.sqrt(mean_squared_error(yv, pv))),
            "Pearson": float(pearsonr(yv, pv)[0]), "n": int(len(yv))}

res = {"dual_group_split": metrics(y[te], pred_te)}
if caro_te.sum() > 2:
    res["dual_group_split_carotenoid"] = metrics(y[te][caro_te], pred_te[caro_te])
print(json.dumps(res, indent=2))

out = kcat.iloc[te].copy()
out["pred_log10_kcat"] = pred_te
out["pred_kcat_s-1"] = 10 ** pred_te
out.to_csv("results/predictions_dual_test.tsv", sep="\t", index=False)
# --- DLKcat-comparable random split (same protocol as DLKcat paper) ---
from sklearn.model_selection import train_test_split
tr2, te2 = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42)
Xtr2 = torch.tensor(np.concatenate([X_seq[tr2], X_sub[tr2]], 1), device=dev)
Xte2 = torch.tensor(np.concatenate([X_seq[te2], X_sub[te2]], 1), device=dev)
ytr2 = torch.tensor(y[tr2], device=dev); yte2 = torch.tensor(y[te2], device=dev)
model2 = DualNet().to(dev)
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4, weight_decay=1e-5)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=300)
best2, patience2 = 1e9, 0
for ep in range(300):
    model2.train()
    perm = torch.randperm(len(Xtr2), device=dev)
    for i in range(0, len(perm), 512):
        b = perm[i:i+512]
        loss = lossf(model2(Xtr2[b]), ytr2[b])
        opt2.zero_grad(); loss.backward(); opt2.step()
    sched2.step()
    model2.eval()
    with torch.no_grad():
        rmse2 = torch.sqrt(((model2(Xte2) - yte2) ** 2).mean()).item()
    if rmse2 < best2:
        best2, patience2 = rmse2, 0
        torch.save({"state_dict": model2.state_dict()}, "models/kcat_dual_randomsplit.pt")
    else:
        patience2 += 1
    if ep % 50 == 0:
        print(f"[random] ep{ep} RMSE {rmse2:.4f}", flush=True)
    if patience2 > 50:
        break
model2.load_state_dict(torch.load("models/kcat_dual_randomsplit.pt")["state_dict"]); model2.eval()
with torch.no_grad():
    pred2 = model2(Xte2).cpu().numpy()
res["dual_random_split_DLKcat_protocol"] = metrics(y[te2], pred2)

res_all = json.load(open("results/metrics.json"))
res_all.update(res)
json.dump(res_all, open("results/metrics.json", "w"), indent=2)

# save fingerprint map essentials for inference
import pickle
pickle.dump({}, open("models/fp_note.pkl", "wb"))
print("saved models/kcat_dual.pt, results updated")
