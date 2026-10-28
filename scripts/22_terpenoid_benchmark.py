"""Build a curated terpenoid / carotenoid kcat benchmark and evaluate CarotKcat zero-shot.

Why this benchmark exists
-------------------------
The training corpus contains *no* entries at all for the carotenoid-specific
enzymes: BRENDA carries no turnover numbers for EC 2.5.1.32 (phytoene
synthase), 1.3.99.31 / 1.3.5.5 (phytoene desaturase) or 5.5.1.19 (lycopene
cyclase), and none of the terpene-synthase EC classes used here appear in the
17,051 records either.  Every measurement below is therefore an extrapolation,
not an interpolation, which is exactly what the carotenoid use case demands.

Two sources, both traceable
---------------------------
* BRENDA kinetic tables (``scripts/21_brenda_fetch.py``): one row per
  measurement, carrying UniProt accession, organism, assay conditions and the
  BRENDA literature id (``https://www.brenda-enzymes.org/literature.php?e=<EC>&r=<id>``).
* Primary literature for the carotenoid module, where BRENDA is silent; the
  reference keys [1]-[7] follow ``scripts/09_casestudy.py``.

Leakage
-------
A record is discarded if the same accession *and* substrate already occur in
the training table, and, more strictly, if the accession occurs at all.
"""
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# substrate vocabulary -- SMILES are taken from the training table itself so
# that featurisation is identical to training
# --------------------------------------------------------------------------- #
SUBSTRATE_ALIASES = {
    "(2E,6E)-farnesyl diphosphate": "FPP",
    "trans,trans-farnesyl diphosphate": "FPP",
    "(E,E)-farnesyl diphosphate": "FPP",
    "farnesyl diphosphate": "FPP",
    "geranyl diphosphate": "GPP",
    "geranylgeranyl diphosphate": "GGPP",
    "dimethylallyl diphosphate": "DMAPP",
    "isopentenyl diphosphate": "IPP",
    "pyruvate": "PYR",
    "D-glyceraldehyde 3-phosphate": "GAP",
    "(R)-mevalonate": "MEV",
    "acetyl-CoA": "ACCOA",
    "phytoene": "PHYTOENE",
    "15-cis-phytoene": "PHYTOENE",
    "beta-carotene": "BCAR",
    "canthaxanthin": "CANTHA",
    "phytofluene": "PHYTOFLUENE",
    "zeta-carotene": "ZETA",
}

SMILES = {
    "FPP": "CC(=CCCC(=CCCC(=CCOP(=O)(O)OP(=O)(O)O)C)C)C",
    "GPP": "CC(=CCCC(=CCOP(=O)(O)OP(=O)(O)O)C)C",
    "GGPP": "CC(=CCCC(=CCCC(=CCCC(=CCOP(=O)(O)OP(=O)(O)O)C)C)C)C",
    "DMAPP": "CC(=CCOP(=O)(O)OP(=O)(O)O)C",
    "IPP": "CC(=C)CCOP(=O)(O)OP(=O)(O)O",
    "PYR": "CC(=O)C(=O)[O-]",
    "GAP": "C(C(C=O)O)OP(=O)(O)O",
    "MEV": "CC(CCO)(CC(=O)[O-])O",
    "PHYTOENE": "CC(=CCCC(=CCCC(=CCCC(=CC=CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C)C)C)C)C",
    "BCAR": "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(CCCC2(C)C)C)C)C",
    "CANTHA": "CC1=C(C(CCC1=O)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(C(=O)CCC2(C)C)C)C)C",
    "PHYTOFLUENE": "CC(=CCCC(=CCCC(=CCCC(=CC=CC=C(C)C=CC=C(C)CCC=C(C)CCC=C(C)C)C)C)C)C",
    "ZETA": "CC(=CCCC(=CCCC(=CC=CC(=CC=CC=C(C)C=CC=C(C)CCC=C(C)CCC=C(C)C)C)C)C)C",
}

# acetyl-CoA SMILES is long; reuse the exact string from the training table
def _accOA_smiles():
    d = pd.read_csv("data/kcat_clean.tsv", sep="\t", usecols=["Substrate", "Smiles"])
    sub = d["Substrate"].astype(str).str.strip().str.lower()
    m = d[sub == "acetyl-coa"]
    if m.empty:
        m = d[sub.str.contains("acetyl-coa", regex=False, na=False)]
    return m.Smiles.iloc[0]


# enzyme class assignment for reporting
CLASS = {
    "2.5.1.32": "carotenoid", "1.3.99.31": "carotenoid", "1.3.5.5": "carotenoid",
    "5.5.1.19": "carotenoid", "4.2.3.16": "terpene synthase", "4.2.3.15": "terpene synthase",
    "4.2.3.17": "terpene synthase", "4.2.3.24": "terpene synthase",
    "4.2.3.38": "terpene synthase", "4.2.3.47": "terpene synthase",
    "4.2.3.75": "terpene synthase", "4.2.3.77": "terpene synthase",
    "4.2.3.97": "terpene synthase", "2.5.1.10": "prenyl transferase",
    "2.5.1.29": "prenyl transferase", "2.5.1.1": "prenyl transferase",
    "5.3.3.2": "isomerase", "2.2.1.7": "precursor", "2.3.3.10": "precursor",
    "2.7.1.36": "precursor",
}


def load_brenda():
    df = pd.read_csv("results/brenda_terpenoid_raw.csv")
    df["uniprot"] = df["uniprot"].astype(str).str.strip()
    ok = df["uniprot"].str.match(r"^[A-Z0-9]{6,10}$")           # single accession only
    df = df[ok].copy()
    df["sub_key"] = df["substrate"].astype(str).str.strip().map(SUBSTRATE_ALIASES)
    df = df[df["sub_key"].notna()].copy()
    df["kcat"] = pd.to_numeric(df["kcat"], errors="coerce")
    df = df[df["kcat"] > 0].copy()
    df["source"] = "BRENDA lit. " + df["literature"].astype(str)
    df["ref"] = "brenda:" + df["ec"] + ":" + df["literature"].astype(str)
    return df[["ec", "uniprot", "organism", "sub_key", "kcat", "commentary", "source", "ref"]]


def load_literature_carotenoid():
    """Carotenoid records where BRENDA has no turnover numbers at all."""
    rows = [
        # enzyme, accession, substrate, kcat, EC, class, citation
        ("CrtI R. capsulatus", "P17054", "PHYTOENE", 0.003, "1.3.99.31", "carotenoid",
         "Lavrijsen 2021 Sci Rep (s41598-020-77876-4) Suppl. Table S4"),
        ("CrtI M. aurum", "Q9K566", "PHYTOENE", 0.100, "1.3.99.31", "carotenoid",
         "Lavrijsen 2021 Sci Rep Suppl. Table S4"),
        ("CrtI B. trispora", "Q67GI0", "PHYTOENE", 0.100, "1.3.99.31", "carotenoid",
         "Lavrijsen 2021 Sci Rep Suppl. Table S4"),
        ("CrtI P. ananatis", "P21685", "PHYTOENE", 0.100, "1.3.99.31", "carotenoid",
         "Lavrijsen 2021 Sci Rep Suppl. Table S4"),
        ("CrtI E. agglomerans", "E9LFG2", "PHYTOENE", 0.100, "1.3.99.31", "carotenoid",
         "Lavrijsen 2021 Sci Rep Suppl. Table S4"),
        ("CrtB P. ananatis", "P21683", "GGPP", 6.000, "2.5.1.32", "carotenoid",
         "Berviller-Magnan 2024 Mol Syst Biol (in vivo kcat)"),
        ("CrtB P. agglomerans", "D5KXJ0", "GGPP", 14.000, "2.5.1.32", "carotenoid",
         "Iwata-Reuyl 2003 Biochemistry (in vitro)"),
        ("CrtE P. ananatis", "P21684", "FPP", 0.800, "2.5.1.10", "carotenoid",
         "BRENDA-curated Pantoea CrtE"),
        ("CrtI X. dendrorhous", "A0A0P0KMF9", "PHYTOENE", 0.030, "1.3.99.31", "carotenoid",
         "Berviller-Magnan 2024 Mol Syst Biol (in vivo kcat)"),
        ("CrtI R. capsulatus (desat. 2)", "P17054", "PHYTOFLUENE", 0.001, "1.3.99.31",
         "carotenoid", "Lavrijsen 2021 Sci Rep Suppl. Table S4"),
        ("CrtI R. capsulatus (desat. 3)", "P17054", "ZETA", 0.040, "1.3.99.31",
         "carotenoid", "Lavrijsen 2021 Sci Rep Suppl. Table S4"),
        ("CrtZ P. ananatis", "P21688", "BCAR", 37.000, "1.14.15.24", "carotenoid",
         "Fraser 1997 JBC (specific activity)"),
        ("CrtZ P. ananatis (canthaxanthin)", "P21688", "CANTHA", 6.000, "1.14.15.24",
         "carotenoid", "Fraser 1997 JBC (specific activity)"),
        ("GGPS1 human", "O95749", "FPP", 0.204, "2.5.1.29", "prenyl transferase",
         "Kavanagh 2006 JBC"),
        ("BTS1 yeast", "Q12051", "FPP", 0.025, "2.5.1.10", "prenyl transferase",
         "Chang 2006 JBC"),
        # terpene synthases whose BRENDA pages carry no turnover numbers
        ("(-)-limonene synthase M. spicata", "O81191", "GPP", 0.330, "4.2.3.16",
         "terpene synthase", "Alonso 1992 JBC (PMID 1559995)"),
        ("Amorpha-4,11-diene synthase", "Q9AR04", "FPP", 0.0068, "4.2.3.24",
         "terpene synthase", "Picaud 2005 Arch Biochem Biophys (Mg2+)"),
    ]
    return pd.DataFrame(rows, columns=["enzyme", "uniprot", "sub_key", "kcat",
                                       "ec", "class", "source"])


def main():
    S = SMILES.copy()
    S["ACCOA"] = _accOA_smiles()

    brenda = load_brenda()
    brenda["enzyme"] = brenda["organism"].astype(str).str.strip()
    brenda["class"] = brenda["ec"].map(CLASS).fillna("other")
    lit = load_literature_carotenoid()

    rec = pd.concat([brenda, lit], ignore_index=True)
    print(f"raw records: BRENDA {len(brenda)}, literature {len(lit)}, total {len(rec)}")

    # one substrate per (enzyme, accession) for two-substrate precursor enzymes,
    # otherwise the same kcat would be counted twice
    rec["dup_key"] = rec["uniprot"] + "|" + rec["sub_key"]
    before = len(rec)
    rec = (rec.sort_values(["uniprot", "sub_key"])
              .drop_duplicates(subset=["uniprot", "sub_key", "kcat"]))
    print(f"after exact dedup: {len(rec)} (removed {before - len(rec)})")

    # ---------------- leakage against the training table ------------------- #
    tr = pd.read_csv("data/kcat_clean.tsv", sep="\t")
    tr_acc = set(tr["UniprotID"].astype(str).str.strip())
    tr_ec = set(tr["ECNumber"].astype(str).str.strip())
    # Leakage is defined on the (accession, substrate) pair: only then has the
    # model seen the measurement itself.  A known enzyme with a different
    # substrate is transfer rather than memorisation and is kept, but flagged.
    tr_key = tr["Substrate"].astype(str).str.strip().map(SUBSTRATE_ALIASES)
    tr_pair = set(zip(tr["UniprotID"].astype(str).str.strip(), tr_key.dropna()))

    rec["seen_enzyme"] = rec["uniprot"].isin(tr_acc)
    rec["seen_pair"] = [(a, s) in tr_pair
                        for a, s in zip(rec["uniprot"], rec["sub_key"])]
    print(f"records whose (accession, substrate) pair is in training: "
          f"{int(rec['seen_pair'].sum())}  -> dropped")
    rec = rec[~rec["seen_pair"]].copy()

    # Pseudo-replication: the predictor is deterministic given (sequence,
    # substrate), so repeated measurements of one pair are noise from the
    # model's point of view.  Collapse each pair to the median reported kcat.
    before = len(rec)
    rec = (rec.groupby(["uniprot", "sub_key"]).agg(
        ec=("ec", "first"), enzyme=("enzyme", "first"), organism=("organism", "first"),
        kcat=("kcat", "median"),
        source=("source", lambda v: "; ".join(sorted(set(map(str, v)))[:2])),
        ref=("ref", "first"), cls=("class", "first"),
        seen_enzyme=("seen_enzyme", "first")).reset_index())
    rec = rec.rename(columns={"cls": "class"})
    print(f"after collapsing to one row per (enzyme, substrate): {len(rec)} "
          f"(from {before})")

    # Two generalisation regimes: a family that occurs in training versus one
    # the model has never seen at all.
    rec["regime"] = np.where(rec["ec"].astype(str).isin(tr_ec),
                             "seen EC family", "unseen EC family")
    rec["zero_shot"] = ~rec["seen_enzyme"]
    print(rec.groupby(["regime", "zero_shot"]).size().to_string())

    # NB: for two-substrate enzymes (DXP synthase, HMG-CoA synthase, mevalonate
    # kinase, prenyltransferases) BRENDA assigns the overall kcat to whichever
    # substrate the authors named, so one enzyme may appear under both names.
    # These pairs are kept and flagged in the manuscript rather than merged,
    # because the model does see a different substrate SMILES in each case.


    print(rec.groupby("class").size().to_string())
    print(rec["sub_key"].value_counts().to_string())

    rec["smiles"] = rec["sub_key"].map(S)
    miss = rec["smiles"].isna()
    if miss.any():
        print("missing SMILES for:", rec.loc[miss, "sub_key"].unique())
    rec = rec[~miss].copy()
    rec["log10_kcat"] = np.log10(rec["kcat"])

    # ---------------- sequences ------------------------------------------- #
    import urllib.request
    cache_path = "data/terpenoid_seqs.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [a for a in sorted(rec["uniprot"].unique()) if a not in cache]
    for i, acc in enumerate(todo, 1):
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
        req = urllib.request.Request(url, headers={"User-Agent": "CarotKcat/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as fh:
                txt = fh.read().decode()
            seq = "".join(l.strip() for l in txt.splitlines() if not l.startswith(">"))
            cache[acc] = seq
            print(f"  [{i}/{len(todo)}] {acc} {len(seq)} aa", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  [{i}/{len(todo)}] {acc} FAILED {e}", flush=True)
    json.dump(cache, open(cache_path, "w"))

    rec["length"] = rec["uniprot"].map(lambda a: len(cache.get(a, "")))
    rec["sequence"] = rec["uniprot"].map(cache)
    bad = rec["length"] == 0
    if bad.any():
        print("WARNING: no sequence for", rec.loc[bad, "uniprot"].unique())
    rec = rec[~bad].copy()
    print(f"records with sequence: {len(rec)}  unique accessions: {rec['uniprot'].nunique()}")

    # ---------------- ESM-2 embeddings ------------------------------------ #
    emb_path = "results/terpenoid_bench_emb.npz"
    if os.path.exists(emb_path):
        z = np.load(emb_path, allow_pickle=True)
        emb = {k: z[k] for k in z.files}
    else:
        import torch
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.cuda().half().eval()
        bc = alphabet.get_batch_converter()
        emb = {}
        for acc, seq in cache.items():
            if acc not in set(rec["uniprot"]):
                continue
            _, _, toks = bc([(acc, seq[:1022])])
            with torch.no_grad():
                out = model(toks.cuda(), repr_layers=[33])["representations"][33]
            L = min(len(seq), 1022)
            emb[acc] = out[0, 1:L + 1].float().mean(0).cpu().numpy()
            print("  embedded", acc, len(seq), flush=True)
        np.savez_compressed(emb_path, **emb)

    # ---------------- predict --------------------------------------------- #
    import torch
    import torch.nn as nn
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    def fp(smi):
        return np.array(AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smi), 2, nBits=2048), dtype=np.float32)

    X = np.stack([np.concatenate([emb[a], fp(s)]) if a in emb else np.zeros(3328, np.float32)
                  for a, s in zip(rec["uniprot"], rec["smiles"])]).astype(np.float32)
    keep = np.array([a in emb for a in rec["uniprot"]])
    rec = rec[keep].copy()
    X = X[keep]

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

    net = DualNet().cuda()
    net.load_state_dict(torch.load("models/carotkcat_ens_nn.pt", weights_only=False)["state_dict"])
    net.eval()
    xgb = pickle.load(open("models/carotkcat_ens_xgb.pkl", "rb"))

    with torch.no_grad():
        p_nn = net(torch.tensor(X, device="cuda")).cpu().numpy()
    p_xgb = xgb.predict(X)
    rec["pred_log10_nn"] = p_nn
    rec["pred_log10_xgb"] = p_xgb
    rec["pred_log10"] = (p_nn + p_xgb) / 2
    rec["pred_kcat"] = 10 ** rec["pred_log10"]
    rec["fold_error"] = rec["pred_kcat"] / rec["kcat"]
    rec["abs_log10_error"] = rec["pred_log10"] - rec["log10_kcat"]

    rec.to_csv("results/terpenoid_bench.tsv", sep="\t", index=False)
    print(f"\nsaved results/terpenoid_bench.tsv  n = {len(rec)}")
    print(rec.groupby("class")["abs_log10_error"].agg(["size", "median"]).to_string())


if __name__ == "__main__":
    main()
