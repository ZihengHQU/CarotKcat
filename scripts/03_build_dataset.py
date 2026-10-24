# -*- coding: utf-8 -*-
"""Step 3: build model-ready dataset.
- join kcat_clean with sequences
- flag carotenoid-related entries (accession in UniProt carotenoid set OR curated EC)
- group duplicates by accession (median kcat) -> sequence-level dataset
"""
import numpy as np, pandas as pd

kcat = pd.read_csv("data/kcat_clean.tsv", sep="\t")
seqs = pd.read_csv("data/sequences.tsv", sep="\t").drop_duplicates("accession").set_index("accession")["sequence"]
kcat["sequence"] = kcat["UniprotID"].map(seqs)
kcat = kcat[kcat["sequence"].notna() & (kcat["sequence"].str.len() > 30) & (kcat["sequence"].str.len() < 2000)]
print("with sequence:", kcat.shape)

# carotenoid flag
caro = pd.read_csv("data/carotenoid_enzymes_all.tsv", sep="\t")
caro_acc = set(caro["Entry"].astype(str))
CARO_EC = {"2.5.1.29","2.5.1.32","1.3.5.5","1.3.5.6","1.3.99.31","5.2.1.13",
           "5.5.1.18","5.5.1.19","1.14.13.129","1.14.15.24","1.14.99.63",
           "1.13.11.71","1.14.13.176","2.4.1.330","2.1.1.271"}
kcat["EC"] = kcat["ECNumber"].astype(str).str.strip()
kcat["carotenoid"] = kcat["UniprotID"].isin(caro_acc) | kcat["EC"].isin(CARO_EC)
print("carotenoid rows:", kcat["carotenoid"].sum())

# sequence-level aggregation (median log kcat)
agg = kcat.groupby("UniprotID").agg(
    sequence=("sequence", "first"),
    log10_kcat=("log10_kcat", "median"),
    n_measure=("log10_kcat", "size"),
    carotenoid=("carotenoid", "max"),
    EC=("EC", "first"),
).reset_index()
print("sequence-level:", agg.shape, "| carotenoid enzymes:", agg["carotenoid"].sum())

agg.to_csv("data/dataset_sequence_level.tsv", sep="\t", index=False)
print("saved data/dataset_sequence_level.tsv")
