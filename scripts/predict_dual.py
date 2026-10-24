# -*- coding: utf-8 -*-
"""Predict kcat from enzyme sequence + substrate SMILES (DLKcat-style dual model).
Usage:
  python scripts/predict_dual.py --seq "MKKVL..." --smiles "CC(C)=CCC..."
  python scripts/predict_dual.py --tsv input.tsv   # columns: id, sequence, smiles
Output: TSV with id, log10_kcat, kcat(s^-1)
"""
import argparse
import numpy as np, torch, torch.nn as nn, esm, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

p = argparse.ArgumentParser()
p.add_argument("--seq"); p.add_argument("--smiles")
p.add_argument("--tsv")
p.add_argument("--out", default="results/predictions_query.tsv")
p.add_argument("--model", default="models/kcat_dual_randomsplit.pt")
args = p.parse_args()

if args.tsv:
    df_in = pd.read_csv(args.tsv, sep="\t")
    recs = [(str(r["id"]), str(r["sequence"]), str(r["smiles"])) for _, r in df_in.iterrows()]
else:
    recs = [(args.id if hasattr(args, "id") else "query", args.seq, args.smiles)]

esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.cuda().half().eval()
bc = alphabet.get_batch_converter()

def seq_emb(seq):
    _, _, toks = bc([("q", seq[:1022])])
    with torch.no_grad():
        out = esm_model(toks.cuda(), repr_layers=[33])["representations"][33]
    L = min(len(seq), 1022)
    return out[0, 1:L+1].float().mean(0).cpu().numpy()

def sub_fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError(f"invalid SMILES: {smi}")
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), dtype=np.float32)

X = np.stack([np.concatenate([seq_emb(s), sub_fp(m)]) for _, s, m in recs]).astype(np.float32)

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
        return self.head(torch.cat([self.seq(x[:, :1280]), self.sub(x[:, 1280:])], 1)).squeeze(-1)

mlp = DualNet().cuda()
mlp.load_state_dict(torch.load(args.model, weights_only=False)["state_dict"]); mlp.eval()
with torch.no_grad():
    logk = mlp(torch.tensor(X, device="cuda")).cpu().numpy()

df = pd.DataFrame({"id": [n for n, _, _ in recs], "log10_kcat": logk, "kcat_s-1": 10**logk})
df.to_csv(args.out, sep="\t", index=False)
print(df.to_string(index=False) if len(df) <= 20 else df.describe().to_string())
print("saved", args.out)
