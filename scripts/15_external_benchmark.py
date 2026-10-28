# -*- coding: utf-8 -*-
"""Step 15: external validation on two independent benchmark sets.

Datasets (from github.com/zelixirSH/kcat_benchmark_2025):
  ADK     - 177 adenylate-kinase orthologs with measured kcat (Muir et al. 2025).
            Tests ranking of *enzyme variants* that the model never saw.
  KcatTIE - 152 kcat values curated from 2023-2025 literature.  A temporal
            holdout: the training corpus predates 2023.

Leakage is checked by exact sequence match against the training set; any hit is
reported and excluded from the metrics.

Prediction uses the shipped CarotKcat-ensemble (dual-tower NN + XGBoost).  The
XGBoost component involves no early stopping at all, so it is reported
separately as a leakage-free reference.

Outputs
  results/external_ADK.csv
  results/external_KcatTIE.csv
  results/external_summary.csv
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = "127.0.0.1,localhost,::1"
import esm  # noqa: E402
import pickle  # noqa: E402

DEV = "cuda"

# --------------------------------------------------------------------------- #
# training sequences for the leakage check
# --------------------------------------------------------------------------- #
train_seqs = set()
k = pd.read_csv("data/kcat_clean.tsv", sep="\t")
sq = pd.read_csv("data/sequences.tsv", sep="\t").drop_duplicates("accession").set_index("accession")["sequence"]
k["sequence"] = k["UniprotID"].map(sq)
train_seqs = set(k["sequence"].dropna().astype(str))
print(f"training sequences: {len(train_seqs)}", flush=True)

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
xgb = pickle.load(open("models/carotkcat_ens_xgb.pkl", "rb"))

print("loading ESM-2 ...", flush=True)
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.to(DEV).half().eval()
bc = alphabet.get_batch_converter()


def embed(seq):
    _, _, toks = bc([("q", seq[:1022])])
    with torch.no_grad():
        out = esm_model(toks.to(DEV), repr_layers=[33])["representations"][33]
    L = min(len(seq), 1022)
    return out[0, 1:L + 1].float().mean(0).cpu().numpy()


def predict_frame(seqs, smiles_list):
    embs = np.stack([embed(s) for s in seqs]).astype(np.float32)
    fps, keep = [], []
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(str(s)) if isinstance(s, str) and s else None
        if m is None:
            keep.append(False)
            continue
        keep.append(True)
        fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048),
                            dtype=np.float32))
    keep = np.array(keep)
    if keep.sum() == 0:
        return None, keep
    X = np.concatenate([embs[keep], np.stack(fps)], 1)
    with torch.no_grad():
        p_nn = nn_model(torch.tensor(X, device=DEV)).cpu().numpy()
    p_xgb = xgb.predict(X)
    return {"NN": p_nn, "XGB": p_xgb, "ENS": (p_nn + p_xgb) / 2, "X": X}, keep


def stats(y_log, p):
    e = np.abs(y_log - p)
    return {
        "n": int(len(y_log)),
        "Pearson": float(pearsonr(y_log, p)[0]),
        "Spearman": float(spearmanr(y_log, p)[0]),
        "R2": float(r2_score(y_log, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y_log, p))),
        "MAE": float(mean_absolute_error(y_log, p)),
        "median_fold_error": float(np.median(10 ** e)),
        "within_10x": float((e <= 1).mean()),
    }


def top_overlap(y_log, p, frac=0.2):
    n = len(y_log)
    m = max(int(round(n * frac)), 1)
    top_true = set(np.argsort(-y_log)[:m])
    top_pred = set(np.argsort(-p)[:m])
    return len(top_true & top_pred) / m


summary = []

# --------------------------------------------------------------------------- #
# ADK
# --------------------------------------------------------------------------- #
adk = pd.read_csv("data/external/ADK_experiment_data.csv", encoding="utf-8-sig")
adk.columns = [c.strip() for c in adk.columns]
adk["kcat_val"] = pd.to_numeric(adk["kcat(1/s)"], errors="coerce")
adk = adk[adk["kcat_val"].notna() & (adk["kcat_val"] > 0)].reset_index(drop=True)
adk["is_train"] = adk["sequence"].astype(str).str.strip().isin(
    {s.strip() for s in train_seqs})
print(f"\nADK: {len(adk)} variants, {adk['is_train'].sum()} exact-match training sequences", flush=True)
adk = adk[~adk["is_train"]].reset_index(drop=True)
res, keep = predict_frame(adk["sequence"].tolist(), adk["substrates"].tolist())
adk = adk[keep].reset_index(drop=True)
y_log = np.log10(adk["kcat_val"].values)
adk["measured_log10_kcat"] = y_log
adk["pred_ENS_log10_kcat"] = res["ENS"]
adk["pred_NN_log10_kcat"] = res["NN"]
adk["pred_XGB_log10_kcat"] = res["XGB"]
adk.to_csv("results/external_ADK.csv", index=False)
for tag, col in [("CarotKcat-ensemble", "pred_ENS_log10_kcat"),
                 ("CarotKcat-XGB (leakage-free)", "pred_XGB_log10_kcat")]:
    st = stats(y_log, adk[col].values)
    st.update({"dataset": "ADK (177 variants)", "model": tag,
               "top20_overlap": top_overlap(y_log, adk[col].values)})
    summary.append(st)
    print(f"  {tag:32s} r={st['Pearson']:.3f} rho={st['Spearman']:.3f} "
          f"top20={st['top20_overlap']:.2f}", flush=True)

# --------------------------------------------------------------------------- #
# KcatTIE (temporal holdout)
# --------------------------------------------------------------------------- #
tie = pd.read_csv("data/external/KcatTIE_experiment_data.csv", encoding="utf-8-sig")
tie.columns = [c.strip() for c in tie.columns]
tie["kcat_val"] = pd.to_numeric(tie["kcat"], errors="coerce")
tie = tie[tie["kcat_val"].notna() & (tie["kcat_val"] > 0)].reset_index(drop=True)
tie["is_train"] = tie["sequence"].astype(str).str.strip().isin(
    {s.strip() for s in train_seqs})
print(f"\nKcatTIE: {len(tie)} entries, {tie['is_train'].sum()} exact-match training sequences", flush=True)
tie = tie[~tie["is_train"]].reset_index(drop=True)
res, keep = predict_frame(tie["sequence"].tolist(), tie["substrate_smiles"].tolist())
tie = tie[keep].reset_index(drop=True)
y_log = np.log10(tie["kcat_val"].values)
tie["measured_log10_kcat"] = y_log
tie["pred_ENS_log10_kcat"] = res["ENS"]
tie["pred_NN_log10_kcat"] = res["NN"]
tie["pred_XGB_log10_kcat"] = res["XGB"]
tie.to_csv("results/external_KcatTIE.csv", index=False)
for tag, col in [("CarotKcat-ensemble", "pred_ENS_log10_kcat"),
                 ("CarotKcat-XGB (leakage-free)", "pred_XGB_log10_kcat")]:
    st = stats(y_log, tie[col].values)
    st.update({"dataset": "KcatTIE (2023-2025)", "model": tag,
               "top20_overlap": top_overlap(y_log, tie[col].values)})
    summary.append(st)
    print(f"  {tag:32s} r={st['Pearson']:.3f} rho={st['Spearman']:.3f} "
          f"R2={st['R2']:.3f} RMSE={st['RMSE']:.3f}", flush=True)

pd.DataFrame(summary).to_csv("results/external_summary.csv", index=False)
print("\n==== external summary ====")
print(pd.DataFrame(summary)[["dataset", "model", "n", "Pearson", "Spearman",
                             "R2", "RMSE", "median_fold_error"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("saved results/external_*.csv")
