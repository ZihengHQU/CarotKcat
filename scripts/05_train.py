# -*- coding: utf-8 -*-
"""Step 5: train MLP regressor on ESM embeddings -> log10(kcat).
Carotenoid entries oversampled x10. Reports overall + carotenoid subset metrics.
Saves model to models/kcat_mlp.pt and predictions to results/.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, os, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

dev = "cuda"
df = pd.read_csv("data/dataset_sequence_level.tsv", sep="\t")
X = np.load("models/embeddings.npz")["embs"]
y = df["log10_kcat"].values.astype(np.float32)
caro = df["carotenoid"].values.astype(bool)

Xtr, Xte, ytr, yte, ctr, cte = train_test_split(
    X, y, caro, test_size=0.15, random_state=42)
print(f"train {len(Xtr)}  test {len(Xte)}  caro_test {cte.sum()}")

# oversample carotenoid in training
w = np.where(ctr, 10.0, 1.0)
idx = np.random.default_rng(0).choice(len(Xtr), size=len(Xtr), replace=True, p=w/w.sum())
Xtr_s, ytr_s = torch.tensor(Xtr[idx], device=dev), torch.tensor(ytr[idx], device=dev)

model = nn.Sequential(
    nn.Linear(X.shape[1], 1024), nn.GELU(), nn.Dropout(0.2),
    nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.2),
    nn.Linear(512, 128), nn.GELU(),
    nn.Linear(128, 1),
).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
lossf = nn.SmoothL1Loss()

Xte_t, yte_t = torch.tensor(Xte, device=dev), torch.tensor(yte, device=dev)
best = 1e9; patience = 0
for ep in range(400):
    model.train()
    perm = torch.randperm(len(Xtr_s), device=dev)
    tot = 0
    for i in range(0, len(perm), 256):
        b = perm[i:i+256]
        loss = lossf(model(Xtr_s[b]).squeeze(-1), ytr_s[b])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
    sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).squeeze(-1)
        rmse = torch.sqrt(((pred - yte_t) ** 2).mean()).item()
    if rmse < best:
        best = rmse; patience = 0
        torch.save({"state_dict": model.state_dict(), "in_dim": X.shape[1]}, "models/kcat_mlp.pt")
    else:
        patience += 1
    if ep % 20 == 0:
        print(f"ep{ep} train {tot/len(perm):.4f} test RMSE {rmse:.4f}", flush=True)
    if patience > 40:
        break

# final eval
ckpt = torch.load("models/kcat_mlp.pt")
model.load_state_dict(ckpt["state_dict"]); model.eval()
with torch.no_grad():
    pred_all = model(torch.tensor(X, device=dev)).squeeze(-1).cpu().numpy()
    pred_te = model(Xte_t).squeeze(-1).cpu().numpy()

def metrics(yv, pv):
    return {"R2": float(r2_score(yv, pv)),
            "RMSE": float(np.sqrt(mean_squared_error(yv, pv))),
            "Pearson": float(pearsonr(yv, pv)[0]), "n": int(len(yv))}

res = {"test_all": metrics(yte, pred_te)}
if cte.sum() > 2:
    res["test_carotenoid"] = metrics(yte[cte], pred_te[cte])
print(json.dumps(res, indent=2))

df["pred_log10_kcat"] = pred_all
df["pred_kcat_s-1"] = 10 ** pred_all
os.makedirs("results", exist_ok=True)
df.to_csv("results/predictions_all.tsv", sep="\t", index=False)
json.dump(res, open("results/metrics.json", "w"), indent=2)
print("saved results/")
