# -*- coding: utf-8 -*-
"""Step 4: ESM-2 650M mean-pooled embeddings via fair-esm (weights from dl.fbaipublicfiles.com).
Cached to models/embeddings.npz. Restartable via partial npz."""
import numpy as np, pandas as pd, torch, time, os, esm

df = pd.read_csv("data/dataset_sequence_level.tsv", sep="\t")
seqs = [s[:1022] for s in df["sequence"].tolist()]
n = len(seqs)
print("sequences:", n, flush=True)

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.cuda().half().eval()
bc = alphabet.get_batch_converter()
H = 1280

cache = "models/embeddings_partial.npz"
done = 0
if os.path.exists(cache):
    z = np.load(cache)
    embs, done = z["embs"], int(z["done"])
else:
    embs = np.zeros((n, H), dtype=np.float32)

B = 6
t0 = time.time()
with torch.no_grad():
    for i in range(done, n, B):
        batch = [(f"p{j}", seqs[j]) for j in range(i, min(i + B, n))]
        _, _, toks = bc(batch)
        toks = toks.cuda()
        out = model(toks, repr_layers=[33])["representations"][33]
        for k, (_, s) in enumerate(batch):
            L = len(s)
            embs[i + k] = out[k, 1:L + 1].float().mean(0).cpu().numpy()
        if (i // B) % 50 == 0:
            np.savez_compressed(cache, embs=embs, done=i)
            print(f"{i}/{n} {(time.time()-t0):.0f}s", flush=True)

np.savez_compressed("models/embeddings.npz", embs=embs)
print("saved models/embeddings.npz", embs.shape, flush=True)
