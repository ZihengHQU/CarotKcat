# -*- coding: utf-8 -*-
"""Step 12: Leak-free re-evaluation + bootstrap confidence intervals.

Problem being fixed
------------------
06/07/08 selected the best checkpoint by *test-set* RMSE (patience-based early
stopping on the evaluation split).  That is a mild form of test-set leakage and
inflates the reported numbers.  Here every neural model early-stops on an
internal validation split carved out of the training data only; the test set is
scored exactly once.

The DLKcat-style baseline (scripts/11) already used validation-based early
stopping, so its saved test predictions are reused as-is.

Outputs
  results/clean_eval.csv        metrics + bootstrap 95% CI for every model
  results/paired_tests.csv      paired bootstrap tests vs CarotKcat-ensemble
  results/clean_preds_{proto}.npz  y_test and all model predictions
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

# --------------------------------------------------------------------------- #
# data (identical construction to 06/07/08)
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
print(f"records: {len(y)}", flush=True)

# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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
    """Early stopping on an internal validation split; test touched once."""
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
    for ep in range(epochs):
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
        return m(Xe).cpu().numpy(), m, best


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def point_metrics(yv, pv):
    r = float(pearsonr(yv, pv)[0]) if np.std(pv) > 1e-12 else float("nan")
    return {
        "Pearson": r,
        "R2": float(r2_score(yv, pv)),
        "RMSE": float(np.sqrt(mean_squared_error(yv, pv))),
    }


def bootstrap_ci(yv, pv, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    N = len(yv)
    vals = {"Pearson": [], "R2": [], "RMSE": []}
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(idx)) < 10:
            continue
        a, b = yv[idx], pv[idx]
        if np.std(b) < 1e-12:
            continue
        vals["Pearson"].append(pearsonr(a, b)[0])
        vals["R2"].append(r2_score(a, b))
        vals["RMSE"].append(np.sqrt(mean_squared_error(a, b)))
    if not vals["Pearson"]:
        return {k: (float("nan"), float("nan")) for k in vals}
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in vals.items()}


def paired_delta(yv, p_a, p_b, metric="Pearson", n=2000, seed=2):
    """Paired bootstrap on the same resampled test indices: model A - model B."""
    rng = np.random.default_rng(seed)
    N = len(yv)
    d = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(idx)) < 10:
            continue
        if np.std(p_a[idx]) < 1e-12 or np.std(p_b[idx]) < 1e-12:
            continue
        ya = yv[idx]
        if metric == "Pearson":
            va = pearsonr(ya, p_a[idx])[0]
            vb = pearsonr(ya, p_b[idx])[0]
        elif metric == "RMSE":
            va = np.sqrt(mean_squared_error(ya, p_a[idx]))
            vb = np.sqrt(mean_squared_error(ya, p_b[idx]))
        d.append(va - vb)
    d = np.asarray(d)
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    # two-sided bootstrap p-value for delta = 0
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(max(p, 0.0), 1.0))


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
summary, paired_rows, all_metrics = [], [], {}

for proto in ["random", "group"]:
    if proto == "random":
        tr_full, te = train_test_split(
            np.arange(len(y)), test_size=0.15, random_state=42
        )
        tr, val = train_test_split(tr_full, test_size=0.10, random_state=7)
    else:
        tr_full, te = next(
            GroupShuffleSplit(1, test_size=0.15, random_state=42).split(X, y, groups)
        )
        # GroupShuffleSplit.split returns (train_idx, test_idx); the validation
        # set is the *test* side, i.e. the second element.
        _, vrel = next(
            GroupShuffleSplit(1, test_size=0.10, random_state=7).split(
                tr_full, y[tr_full], groups[tr_full]
            )
        )
        val = tr_full[vrel]
        tr = np.setdiff1d(tr_full, val)

    print(f"--- {proto}: train {len(tr)} val {len(val)} test {len(te)}", flush=True)
    yte = y[te]
    preds = {"Mean (baseline)": np.full(len(te), y[tr].mean())}

    # trees on all non-test data (no early stopping involved)
    et = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    et.fit(X[tr_full], y[tr_full])
    preds["UniKP-style (ESM+FP, ExtraTrees)"] = et.predict(X[te])

    xgb = XGBRegressor(
        n_estimators=800, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
    )
    xgb.fit(X[tr_full], y[tr_full])
    preds["TurNuP-style (ESM+FP, XGBoost)"] = xgb.predict(X[te])

    # neural models: validation-based early stopping
    t0 = time.time()
    p_sub, _, _ = train_nn(lambda: MLP(2048), X_sub[tr], y[tr],
                          X_sub[val], y[val], X_sub[te])
    preds["Substrate-FP only (MLP)"] = p_sub
    print(f"  substrate MLP done {time.time()-t0:.0f}s", flush=True)

    p_seq, _, _ = train_nn(lambda: MLP(1280), X_seq[tr], y[tr],
                           X_seq[val], y[val], X_seq[te])
    preds["ESM only (MLP)"] = p_seq
    print(f"  ESM MLP done {time.time()-t0:.0f}s", flush=True)

    p_dual, net, best_val = train_nn(DualNet, X[tr], y[tr], X[val], y[val], X[te])
    preds["CarotKcat (dual-tower)"] = p_dual
    print(f"  dual tower done {time.time()-t0:.0f}s (val RMSE {best_val:.3f})", flush=True)

    preds["CarotKcat-ensemble"] = (p_dual + preds["TurNuP-style (ESM+FP, XGBoost)"]) / 2

    # reuse the already leak-free DLKcat-style predictions
    dlk = np.load(f"results/dlkcat_pred_{proto}.npy")
    assert len(dlk) == len(te), (len(dlk), len(te))
    preds["DLKcat-style (GNN+CNN, reimplemented)"] = dlk

    all_metrics[proto] = {"y": yte, "preds": preds}
    np.savez(f"results/clean_preds_{proto}.npz", y=yte, **preds)

    for name, pv in preds.items():
        m = point_metrics(yte, pv)
        ci = bootstrap_ci(yte, pv)
        row = {"protocol": proto, "model": name, **m}
        for k in ("Pearson", "R2", "RMSE"):
            row[f"{k}_lo"], row[f"{k}_hi"] = ci[k]
        summary.append(row)
        print(f"  {name:42s} r={m['Pearson']:.3f} "
              f"[{ci['Pearson'][0]:.3f},{ci['Pearson'][1]:.3f}]", flush=True)

    # paired tests vs ensemble
    ref = preds["CarotKcat-ensemble"]
    for name, pv in preds.items():
        if name in ("CarotKcat-ensemble", "Mean (baseline)"):
            continue
        if np.std(pv) < 1e-12:
            continue
        d, lo, hi, p = paired_delta(yte, ref, pv, "Pearson")
        dr, rlo, rhi, _ = paired_delta(yte, ref, pv, "RMSE")
        paired_rows.append({
            "protocol": proto, "comparison": f"CarotKcat-ensemble vs {name}",
            "delta_Pearson": d, "dP_lo": lo, "dP_hi": hi, "p_value": p,
            "delta_RMSE": dr, "dR_lo": rlo, "dR_hi": rhi,
        })

df = pd.DataFrame(summary)
df.to_csv("results/clean_eval.csv", index=False)
pd.DataFrame(paired_rows).to_csv("results/paired_tests.csv", index=False)
print("\n==== clean_eval ====")
print(df[["protocol", "model", "Pearson", "R2", "RMSE"]].to_string(index=False))
print("\n==== paired (Pearson diff) ====")
pt = pd.DataFrame(paired_rows)
print(pt[["protocol", "comparison", "delta_Pearson", "p_value"]].to_string(index=False))
print("\nsaved results/clean_eval.csv, results/paired_tests.csv")
