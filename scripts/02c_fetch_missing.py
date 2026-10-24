# -*- coding: utf-8 -*-
"""Fetch sequences via UniProt ID mapping API. Restartable."""
import time, requests, io, csv

ids = [l.strip() for l in open("data/missing_ids.txt") if l.strip()]
have = {}
try:
    with open("data/sequences.tsv", encoding="utf-8") as f:
        rdr = csv.reader(f, delimiter="\t")
        next(rdr, None)
        for row in rdr:
            if len(row) >= 2:
                have[row[0]] = row[1]
except FileNotFoundError:
    pass
todo = [i for i in ids if i not in have]
print("total", len(ids), "have", len(have), "todo", len(todo), flush=True)

out = open("data/sequences.tsv", "a", encoding="utf-8", newline="")
w = csv.writer(out, delimiter="\t")
if not have:
    w.writerow(["accession", "sequence"])

B = 50
for s in range(0, len(todo), B):
    chunk = todo[s:s+B]
    try:
        r = requests.post("https://rest.uniprot.org/idmapping/run",
                          data={"from": "UniProtKB_AC-ID", "to": "UniProtKB",
                                "ids": ",".join(chunk)}, timeout=60)
        job = r.json()["jobId"]
        while True:
            st = requests.get(f"https://rest.uniprot.org/idmapping/status/{job}", timeout=60).json()
            if "results" in st or st.get("jobStatus") == "FINISHED":
                break
            if st.get("jobStatus") in ("ERROR",):
                raise RuntimeError(st)
            time.sleep(3)
        url = f"https://rest.uniprot.org/idmapping/uniprotkb/results/stream/{job}?format=tsv&fields=accession,sequence"
        t = requests.get(url, timeout=180)
        for row in csv.reader(io.StringIO(t.text), delimiter="\t"):
            if len(row) >= 2 and row[0] != "Entry":
                w.writerow([row[0], row[1]])
        out.flush()
        print(f"{s}/{len(todo)}", flush=True)
    except Exception as e:
        print("chunk failed", s, repr(e)[:200], flush=True)
        time.sleep(10)
print("DONE", flush=True)
