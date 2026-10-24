# -*- coding: utf-8 -*-
"""Step 1: clean DLKcat combined dataset -> kcat(s^-1) entries with UniProt ID.
Output: data/kcat_clean.tsv
"""
import pandas as pd

df = pd.read_csv("data/Kcat_combination_41559.tsv", sep="\t")
print("raw:", df.shape)
print(df["Unit"].value_counts().head(10))

# keep kcat entries (per second)
df = df[df["Unit"].astype(str).str.contains("s", case=False, na=False)]
df = df[~df["Unit"].astype(str).str.contains("mol|M|min|h", case=False, na=False)]
df = df[df["UniprotID"].notna() & (df["UniprotID"].astype(str).str.strip() != "-")]
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df[df["Value"].notna() & (df["Value"] > 0)]
print("kcat-per-sec with UniProtID:", df.shape)

# expand multi-accession cells
df["UniprotID"] = df["UniprotID"].astype(str).str.split(",")
df = df.explode("UniprotID")
df["UniprotID"] = df["UniprotID"].str.strip()

import numpy as np
df["log10_kcat"] = np.log10(df["Value"])
df.to_csv("data/kcat_clean.tsv", sep="\t", index=False)
ids = sorted(df["UniprotID"].unique())
pd.Series(ids).to_csv("data/uniprot_ids.txt", index=False, header=False)
print("unique UniProt IDs:", len(ids))
