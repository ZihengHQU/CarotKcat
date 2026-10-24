# -*- coding: utf-8 -*-
"""Step 2: fetch sequences from UniProt for all unique IDs. Chunked stream queries."""
import time, sys, requests

ids = [l.strip() for l in open("data/uniprot_ids.txt") if l.strip()]
print("ids:", len(ids))

have = {}
try:
    with open("data/sequences.tsv") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                have[p[0]] = p[1]
except FileNotFoundError:
    pass
todo = [i for i in ids if i not in have]
print("already have:", len(have), "todo:", len(todo))

out = open("data/sequences.tsv", "a")
if not have:
    out.write("accession\tsequence\n")

CH = 150
for s in range(0, len(todo), CH):
    chunk = todo[s:s+CH]
    q = " OR ".join(f"accession:{a}" for a in chunk)
    url = f"https://rest.uniprot.org/uniprotkb/stream?query=({q})&format=tsv&fields=accession,sequence"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                lines = r.text.strip().split("\n")
                for line in lines[1:]:
                    p = line.split("\t")
                    if len(p) >= 2:
                        out.write(f"{p[0]}\t{p[1]}\n")
                break
            time.sleep(3 * (attempt + 1))
        except Exception as e:
            print("err", e); time.sleep(5)
    out.flush()
    if (s // CH) % 10 == 0:
        print(f"{s}/{len(todo)}")
print("DONE")
