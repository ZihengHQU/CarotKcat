"""Step 1 of the terpenoid benchmark: fetch BRENDA EC pages and inventory records.

BRENDA exposes each kinetic table as a set of collapsed ``div`` rows.  Every
detail row carries value / substrate / organism / UniProt / conditions /
literature id, which is exactly the provenance needed for a citable benchmark.

This script only downloads and parses; selection and evaluation happen in
``22_terpenoid_benchmark.py``.
"""
import os
import re
import time
import html
import urllib.request

EC_LIST = [
    # --- carotenoid backbone ------------------------------------------------
    "2.5.1.10",    # GGPP synthase (CrtE)
    "2.5.1.32",    # phytoene synthase (CrtB)
    "1.3.99.31",   # phytoene desaturase, 4-step (CrtI)
    "1.3.5.5",     # phytoene desaturase, 3-step
    "5.5.1.19",    # lycopene beta-cyclase (CrtY)
    # --- isoprenoid precursors ---------------------------------------------
    "2.2.1.7",     # DXP synthase (DXS)
    "2.5.1.1",     # DMAPP:IPP dimethylallyltransferase
    "2.5.1.29",    # GGPP synthase (plant type)
    "5.3.3.2",     # IPP isomerase
    "2.3.3.10",    # HMG-CoA synthase
    "2.7.1.36",    # mevalonate kinase
    # --- terpene synthases --------------------------------------------------
    "4.2.3.16",    # (-)-limonene synthase
    "4.2.3.15",    # myrcene synthase
    "4.2.3.17",    # taxadiene synthase
    "4.2.3.24",    # amorpha-4,11-diene synthase
    "4.2.3.38",    # alpha-bisabolene synthase
    "4.2.3.47",    # (E)-beta-farnesene synthase
    "4.2.3.75",    # vetispiradiene / sesquiterpene
    "4.2.3.77",    # (+)-germacrene D synthase
    "4.2.3.97",    # (-)-delta-cadinene synthase
]

CACHE = os.path.join("data", "brenda")
UA = {"User-Agent": "Mozilla/5.0 (compatible; CarotKcat benchmark builder)"}


def fetch(ec):
    """Read the cached BRENDA page; download only if absent.

    Large EC pages (HMG-CoA reductase, GGPP synthase) exceed 2 MB, and
    urllib occasionally raises IncompleteRead on them, so pages are normally
    pre-fetched with curl.  A partial file is rejected by the size check.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{ec}.html")
    if os.path.exists(path) and os.path.getsize(path) > 50_000:
        return open(path, encoding="utf-8", errors="ignore").read()
    req = urllib.request.Request(
        f"https://www.brenda-enzymes.org/enzyme.php?ecno={ec}", headers=UA)
    with urllib.request.urlopen(req, timeout=180) as fh:
        chunks = []
        while True:
            b = fh.read(65536)
            if not b:
                break
            chunks.append(b)
        text = b"".join(chunks).decode("utf-8", errors="ignore")
    open(path, "w", encoding="utf-8").write(text)
    time.sleep(1.5)
    return text


def clean(s):
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).replace("\n", " ").strip()


def parse_turnover(text):
    """Return list of dicts for the TURNOVER NUMBER table."""
    anchor = text.find('name="TURNOVER NUMBER [1/s]"')
    if anchor < 0:
        return []
    m = re.search(r'<div id="tab(\d+)" class="equal"', text[anchor:])
    if not m:
        return []
    tab = m.group(1)
    # the table body ends at the closing of its container
    body_start = text.find(f'<div id="tab{tab}" class="equal"', anchor)
    nxt = re.search(rf'<div id="tab(?!{tab})\d+" class="equal"', text[body_start + 10:])
    body = text[body_start: body_start + 10 + (nxt.start() if nxt else len(text))]

    # locate every detail-row header, then slice the text between consecutive ones
    marks = list(re.finditer(rf'<div id="tab{tab}r(\d+)sr(\d+)"', body))
    out = []
    for i, mk in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        chunk = body[mk.start():end]
        r, sr = mk.group(1), mk.group(2)
        pat = r'<div id="tab' + tab + r'r' + r + r'sr' + sr + r'c(\d+)"[^>]*>(.*?)</div>'
        cells = re.findall(pat, chunk, re.S)
        out.append({int(c): clean(v) for c, v in cells})
    return out


def main():
    all_rows = []
    for ec in EC_LIST:
        text = fetch(ec)
        rows = parse_turnover(text)
        print(f"{ec:10s} rows={len(rows):4d}")
        for d in rows:
            all_rows.append({
                "ec": ec,
                "kcat": d.get(0, ""),
                "substrate": d.get(1, ""),
                "organism": d.get(2, ""),
                "uniprot": d.get(3, ""),
                "commentary": d.get(4, ""),
                "literature": d.get(5, ""),
            })
    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_csv("results/brenda_terpenoid_raw.csv", index=False)
    print("\ntotal rows:", len(df))
    sub = df["substrate"].value_counts()
    print("\nunique substrates:", len(sub))
    print(sub.head(40).to_string())


if __name__ == "__main__":
    main()
