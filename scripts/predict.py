# -*- coding: utf-8 -*-
"""Predict kcat from a FASTA or single sequence using ESM-2 + trained MLP.
Usage:
  python scripts/predict.py --fasta data/carotenoid_enzymes_all.fasta
  python scripts/predict.py --seq "MKKVL..."
Output: TSV with accession/id, log10_kcat, kcat(s^-1)
"""
import argparse, io
import numpy as np, torch, torch.nn as nn, esm, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--fasta")
p.add_argument("--seq")
p.add_argument("--id", default="query")
p.add_argument("--out", default="results/predictions_query.tsv")
args = p.parse_args()

def read_fasta(path):
    recs, name, seq = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith(">"):
            if name: recs.append((name, "".join(seq)))
            name, seq = line[1:], []
        elif line:
            seq.append(line)
    if name: recs.append((name, "".join(seq)))
    return recs

if args.fasta:
    recs = read_fasta(args.fasta)
else:
    recs = [(args.id, args.seq)]

esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.cuda().half().eval()
bc = alphabet.get_batch_converter()

embs = []
with torch.no_grad():
    for i in range(0, len(recs), 4):
        batch = [(n, s[:1022]) for n, s in recs[i:i+4]]
        _, _, toks = bc(batch)
        out = esm_model(toks.cuda(), repr_layers=[33])["representations"][33]
        for k, (_, s) in enumerate(batch):
            L = min(len(s), 1022)
            embs.append(out[k, 1:L+1].float().mean(0).cpu().numpy())
X = np.stack(embs)

ckpt = torch.load("models/kcat_mlp.pt", weights_only=False)
mlp = nn.Sequential(
    nn.Linear(ckpt["in_dim"], 1024), nn.GELU(), nn.Dropout(0.2),
    nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.2),
    nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1)).cuda()
mlp.load_state_dict(ckpt["state_dict"]); mlp.eval()
with torch.no_grad():
    logk = mlp(torch.tensor(X, dtype=torch.float32, device="cuda")).squeeze(-1).cpu().numpy()

df = pd.DataFrame({"id": [n for n, _ in recs], "log10_kcat": logk, "kcat_s-1": 10**logk})
df.to_csv(args.out, sep="\t", index=False)
print(df.to_string(index=False) if len(df) <= 20 else df.describe().to_string())
print("saved", args.out)
