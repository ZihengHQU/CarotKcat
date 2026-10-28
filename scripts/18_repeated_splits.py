# -*- coding: utf-8 -*-
"""Step 18: 5-fold repeated random / group splits -> mean +- SD stability.

Motivation
----------
Every number reported so far (scripts/12, 17) comes from a *single* train/test
split.  A reviewer's first question is whether that split happened to be lucky.
Here the whole leak-free pipeline is repeated over 5 independent splits
(5 different random seeds) for both evaluation protocols, and the spread is
reported as mean +- SD.

Same rules as script 12:
  * early stopping on an internal validation split carved out of train only,
  * the test set is scored exactly once per fold,
  * identical hyper-parameters across folds (no per-fold tuning).

Omitted on purpose
------------------
The DLKcat-style reimplementation (scripts/11) is not re-run here: one fold of
it costs ~40 min on CPU-side graph batching, so 10 folds is impractical. Its
single-split value is reported separately in results/dlkcat_headtohead.csv and
labelled as such in the manuscript.

Outputs
  results/repeated_splits.csv          one row per (protocol, fold, model)
  results/repeated_splits_summary.csv  mean / SD / min / max across folds
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBRegressor

RDLogger.DisableLog("rdApp.*")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

SEEDS = [101, 202, 303, 404, 505]

# --------------------------------------------------------------------------- #
# data (identical construction to 06/07/08/12)
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

X_seq = np.stack(kcat["UniprotID"].map(acc2emb).values).astype(np.float32)
X_sub = np.stack(kcat["Smiles"].map(fp_map).values).astype(np.float32)
X = np.concatenate([X_seq, X_sub], 1)
y = kcat["log10_kcat"].values.astype(np.float32)
groups = kcat["UniprotID"].values
print(f"records: {len(y)}  unique enzymes: {len(np.unique(groups))}", flush=True)


# --------------------------------------------------------------------------- #
# model / trainer (identical to script 12)
# --------------------------------------------------------------------------- #
class DualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1280, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(),
        )
        self.sub = nn.Sequential(
            nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(
            torch.cat([self.seq(x[:, :1280]), self.sub(x[:, 1280:])], 1)
        ).squeeze(-1)


def train_nn(make_model, Xtr, ytr, Xval, yval, Xte, epochs=200, patience=30):
    m = make_model().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.SmoothL1Loss()
    Xt = torch.tensor(Xtr, device=DEV)
    yt = torch.tensor(ytr, device=DEV)
    Xv = torch.tensor(Xval, device=DEV)
    yv = torch.tensor(yval, device=DEV)
    Xe = torch.tensor(Xte, device=DEV)
    best, pat, best_state = 1e9, 0, None
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = lossf(m(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            vr = torch.sqrt(((m(Xv) - yv) ** 2).mean()).item()
        if vr < best:
            best, pat = vr, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            pat += 1
        if pat > patience:
            break
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return m(Xe).cpu().numpy()


def metrics(yv, pv):
    r = float(pearsonr(yv, pv)[0]) if np.std(pv) > 1e-12 else float("nan")
    return {
        "Pearson": r,
        "R2": float(r2_score(yv, pv)),
        "RMSE": float(np.sqrt(mean_squared_error(yv, pv))),
    }


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
rows = []
t_all = time.time()

for proto in ["random", "group"]:
    for k, seed in enumerate(SEEDS, 1):
        t0 = time.time()
        if proto == "random":
            tr_full, te = train_test_split(
                np.arange(len(y)), test_size=0.15, random_state=seed
            )
            tr, val = train_test_split(tr_full, test_size=0.10, random_state=seed + 1)
        else:
            tr_full, te = next(
                GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(
                    X, y, groups
                )
            )
            _, vrel = next(
                GroupShuffleSplit(1, test_size=0.10, random_state=seed + 1).split(
                    tr_full, y[tr_full], groups[tr_full]
                )
            )
            val = tr_full[vrel]
            tr = np.setdiff1d(tr_full, val)

        # sanity: zero enzyme overlap between test and everything else
        # (only meaningful for the group protocol; the random split shares
        #  enzymes by design)
        if proto == "group":
            assert not (set(groups[te]) & set(groups[tr_full])), "group leakage"
        print(f"--- {proto} fold {k} seed {seed}: "
              f"train {len(tr)} val {len(val)} test {len(te)}", flush=True)
        yte = y[te]

        et = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=seed)
        et.fit(X[tr_full], y[tr_full])
        p_et = et.predict(X[te])

        xgb = XGBRegressor(
            n_estimators=800, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed,
        )
        xgb.fit(X[tr_full], y[tr_full])
        p_xgb = xgb.predict(X[te])
        print(f"  trees {time.time()-t0:.0f}s", flush=True)

        p_dual = train_nn(DualNet, X[tr], y[tr], X[val], y[val], X[te])
        p_ens = (p_dual + p_xgb) / 2

        for name, pv in [
            ("UniKP-style (ESM+FP, ExtraTrees)", p_et),
            ("TurNuP-style (ESM+FP, XGBoost)", p_xgb),
            ("CarotKcat (dual-tower)", p_dual),
            ("CarotKcat-ensemble", p_ens),
        ]:
            rows.append({"protocol": proto, "fold": k, "seed": seed,
                         "model": name, **metrics(yte, pv)})
            print(f"    {name:38s} r={rows[-1]['Pearson']:.3f}", flush=True)
        print(f"  fold {k} done in {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/repeated_splits.csv", index=False)

g = df.groupby(["protocol", "model"], sort=False)
summ = g.agg(
    n_folds=("Pearson", "size"),
    Pearson_mean=("Pearson", "mean"), Pearson_sd=("Pearson", "std"),
    Pearson_min=("Pearson", "min"), Pearson_max=("Pearson", "max"),
    R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
    RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
).reset_index()
summ.to_csv("results/repeated_splits_summary.csv", index=False)

print("\n==== repeated-split stability (5 folds) ====")
print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\ntotal wall time {time.time()-t_all:.0f}s")
print("saved results/repeated_splits.csv, results/repeated_splits_summary.csv")
