# -*- coding: utf-8 -*-
"""Step 19: dry (in silico) validation of CarotKcat in an enzyme-constrained context.

No wet lab is available, so the model is validated against *published physiology*
rather than against new experiments.  The logic is:

  A. Predict kcat for the four heterologous lycopene-pathway enzymes
     (CrtE / CrtB / CrtI / CrtY of Pantoea ananatis) with CarotKcat-ensemble.
  B. Convert three independently published lycopene productivities into a
     specific flux (µmol gDCW^-1 h^-1).
  C. Ask what enzyme expression level (as % of total protein) is needed to
     sustain that flux:  capacity = kcat * [E] * 3600.
  D. Embed the pathway in iML1515 and compute (i) the stoichiometric maximum
     lycopene flux, (ii) the native isoprenoid (GGPP) flux at µ_max, and
     (iii) the growth-production envelope.

If the predictions are even roughly right they must show that the heterologous
enzymes are NOT the bottleneck at the measured productivity -- because three
independent engineering studies all had to amplify the *upstream* MEP pathway
(dxs / idi / ispA) instead.  That is a falsifiable prediction and it is what we
test here.

Assumptions that are stated in the manuscript rather than hidden:
  * total protein 0.50 g per gDCW (E. coli, glucose aerobic, fast growth);
  * molecular weight = 110 Da per residue;
  * kcat is a per-substrate quantity, so CrtE is evaluated with FPP, matching
    the case study (scripts/09).

Outputs (results/)
  ecgem_predictions.tsv     predicted kcat for the four module enzymes
  ecgem_targets.tsv         published productivities -> required flux
  ecgem_capacity.tsv        required enzyme fraction per enzyme and target
  ecgem_envelope.csv        growth vs max lycopene flux (production envelope)
  ecgem_summary.json        headline numbers used in the manuscript
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
import joblib

RDLogger.DisableLog("rdApp.*")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "results"

# --------------------------------------------------------------------------- #
# A. CarotKcat predictions for the lycopene module
# --------------------------------------------------------------------------- #
SMILES = {
    "FPP": "CC(=CCCC(=CCCC(=CCOP(=O)(O)OP(=O)(O)O)C)C)C",
    "GGPP": "CC(=CCCC(=CCCC(=CCCC(=CCOP(=O)(O)OP(=O)(O)O)C)C)C)C",
    "phytoene": "CC(=CCCC(=CCCC(=CCCC(=CC=CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C)C)C)C)C",
}

# All three published lycopene strains used for the comparison carry crtE + crtB
# + crtI only (no crtY, i.e. lycopene rather than beta-carotene is the product),
# so the module is restricted to those three enzymes.  Lycopene itself therefore
# only needs its elemental composition (C40H56) for the flux conversion.
LYC_MW = 536.87   # g/mol, C40H56

MODULE = [
    # name, UniProt, gene, substrate key, reaction in the model
    ("CrtE", "P21684", "crtE", "FPP",      "frdp_c + ipdp_c -> ggpp_c + ppi_c"),
    ("CrtB", "P21683", "crtB", "GGPP",     "2 ggpp_c -> phytoene_c + 2 ppi_c"),
    ("CrtI", "P21685", "crtI", "phytoene", "phytoene_c + 4 fad_c -> lycopene_c + 4 fadh2_c"),
]

for k, s in SMILES.items():
    m = Chem.MolFromSmiles(s)
    assert m is not None, f"bad SMILES for {k}"
    print(f"  {k:9s} {Chem.rdMolDescriptors.CalcMolFormula(m):12s} "
          f"MW {Descriptors.MolWt(m):7.2f}", flush=True)


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048),
                    dtype=np.float32)


class DualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1280, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU())
        self.sub = nn.Sequential(
            nn.Linear(2048, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(torch.cat([self.seq(x[:, :1280]),
                                    self.sub(x[:, 1280:])], 1)).squeeze(-1)


seqs = pd.read_csv("data/sequences.tsv", sep="\t").drop_duplicates("accession").set_index("accession")["sequence"]
need = [u for _, u, _, _, _ in MODULE if u not in seqs.index]
if need:
    print("fetching sequences from UniProt:", need, flush=True)
    import urllib.request
    q = "%20OR%20".join(f"accession:{a}" for a in need)
    url = ("https://rest.uniprot.org/uniprotkb/search?query=" + q +
           "&fields=accession,sequence&format=tsv")
    with urllib.request.urlopen(url, timeout=120) as f:
        txt = f.read().decode()
    got = {}
    for line in txt.strip().split("\n")[1:]:
        a, s = line.split("\t")
        got[a] = s
    for a, s in got.items():
        seqs[a] = s
    print("  fetched", len(got), flush=True)

# ESM-2 embeddings for the module
import esm
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.to(DEV).half().eval()
bc = alphabet.get_batch_converter()


def embed(seq):
    _, _, toks = bc([("q", seq[:1022])])
    with torch.no_grad():
        out = esm_model(toks.to(DEV), repr_layers=[33])["representations"][33]
    L = min(len(seq), 1022)
    return out[0, 1:L + 1].float().mean(0).cpu().numpy()


net = DualNet().to(DEV)
net.load_state_dict(torch.load("models/carotkcat_ens_nn.pt", map_location=DEV,
                               weights_only=False)["state_dict"])
net.eval()
xgb = joblib.load("models/carotkcat_ens_xgb.pkl")

rows = []
for name, acc, gene, subkey, rxn in MODULE:
    s = seqs[acc]
    e = embed(s)
    x = np.concatenate([e, fp(SMILES[subkey])])[None, :].astype(np.float32)
    with torch.no_grad():
        p_nn = float(net(torch.tensor(x, device=DEV)))
    p_xgb = float(xgb.predict(x)[0])
    log10_k = (p_nn + p_xgb) / 2
    rows.append({"enzyme": name, "accession": acc, "gene": gene,
                 "substrate": subkey, "length": len(s),
                 "MW_kDa": round(len(s) * 110 / 1000, 2),
                 "pred_log10_nn": p_nn, "pred_log10_xgb": p_xgb,
                 "pred_log10": log10_k, "kcat_s-1": 10 ** log10_k,
                 "disagreement_log10": abs(p_nn - p_xgb), "reaction": rxn})
    print(f"  {name} ({gene}, {len(s)} aa) + {subkey:9s} -> "
          f"kcat = {10 ** log10_k:7.3f} s-1  (NN {p_nn:+.2f} / XGB {p_xgb:+.2f})",
          flush=True)

pred = pd.DataFrame(rows)
pred.to_csv(f"{OUT}/ecgem_predictions.tsv", sep="\t", index=False)

# --------------------------------------------------------------------------- #
# B. published productivities -> required specific flux
# --------------------------------------------------------------------------- #
PUBLISHED = [
    # label, mg per gDCW, hours of production, source
    ("Xu 2018 (Bioprocess Biosyst Eng 41:489) content/40 h", 67.0, 40.0,
     "Xu et al. 2018, 67 mg/gDCW at 40 h"),
    ("POSTECH JYJ0 base strain (crtEBI only)", 0.38, 24.0,
     "Kim thesis 2016, 0.38 mg/gDCW"),
    ("POSTECH JYJ3 (dxs/idi/ispA amplified)", 4.0, 24.0,
     "Kim thesis 2016, 4.0 mg/gDCW"),
]
tgt = []
for label, mg_per_gdcw, hours, src in PUBLISHED:
    umol_per_g_h = (mg_per_gdcw * 1e-3 / LYC_MW) * 1e6 / hours
    tgt.append({"case": label, "mg_per_gDCW": mg_per_gdcw, "hours": hours,
                "flux_umol_gDCW_h": umol_per_g_h, "source": src})
tgt = pd.DataFrame(tgt)
tgt["flux_note"] = ["peak-value case (upper bound)" if "Xu" in c else
                    ("unevolved pathway" if "base" in c else "debottlenecked pathway")
                    for c in tgt["case"]]
tgt.to_csv(f"{OUT}/ecgem_targets.tsv", sep="\t", index=False)
print("\n--- required lycopene flux ---")
print(tgt[["case", "flux_umol_gDCW_h"]].to_string(index=False), flush=True)

# --------------------------------------------------------------------------- #
# C. capacity analysis
# --------------------------------------------------------------------------- #
PROTEIN_FRACTION = 0.50      # g total protein per gDCW
RESIDUE_DA = 110.0           # g/mol per residue


def capacity(kcat, length, frac):
    """µmol product gDCW^-1 h^-1 for an enzyme at `frac` of total protein."""
    mw = length * RESIDUE_DA                       # g/mol
    e_umol = frac * PROTEIN_FRACTION / mw * 1e6    # µmol enzyme / gDCW
    return kcat * e_umol * 3600.0


def required_fraction(kcat, length, flux):
    """Fraction of total protein needed to sustain `flux` (µmol gDCW^-1 h^-1)."""
    mw = length * RESIDUE_DA
    return flux * mw / (PROTEIN_FRACTION * 1e6 * kcat * 3600.0)


REF_FRACTIONS = [0.001, 0.005, 0.01, 0.05]     # 0.1 %, 0.5 %, 1 %, 5 % of proteome
cap_rows = []
for _, r in pred.iterrows():
    for _, t in tgt.iterrows():
        cap_rows.append({
            "enzyme": r.enzyme, "substrate": r.substrate, "kcat_s-1": r["kcat_s-1"],
            "case": t.case, "target_flux": t.flux_umol_gDCW_h,
            "required_frac_of_protein": required_fraction(r["kcat_s-1"], r.length,
                                                          t.flux_umol_gDCW_h),
            **{f"capacity_at_{f*100:g}pct": capacity(r["kcat_s-1"], r.length, f)
               for f in REF_FRACTIONS},
        })
cap = pd.DataFrame(cap_rows)
for f in REF_FRACTIONS:
    cap[f"headroom_at_{f*100:g}pct"] = cap[f"capacity_at_{f*100:g}pct"] / cap["target_flux"]
cap.to_csv(f"{OUT}/ecgem_capacity.tsv", sep="\t", index=False)

print("\n--- enzyme fraction needed to sustain measured productivity ---")
hi = tgt.flux_umol_gDCW_h.max()
sel = cap[np.isclose(cap.target_flux, hi)]
print(sel[["enzyme", "kcat_s-1", "required_frac_of_protein"]]
      .assign(required_pct=lambda d: d.required_frac_of_protein * 100)
      .to_string(index=False), flush=True)

# --------------------------------------------------------------------------- #
# D. iML1515: stoichiometric ceiling and production envelope
# --------------------------------------------------------------------------- #
import cobra
from cobra import Metabolite, Reaction

# iML1515 is a third-party model and is therefore not redistributed in this
# repository; it is fetched from BiGG on first use.
MODEL_PATH = "data/ecgem/iML1515.xml"
if not os.path.exists(MODEL_PATH):
    os.makedirs("data/ecgem", exist_ok=True)
    import urllib.request
    print("downloading iML1515 from BiGG ...", flush=True)
    urllib.request.urlretrieve(
        "https://bigg.ucsd.edu/static/models/iML1515.xml", MODEL_PATH)

model = cobra.io.read_sbml_model(MODEL_PATH)
frdp = model.metabolites.get_by_id("frdp_c")
print(f"\nfrdp_c formula {frdp.formula} charge {frdp.charge}", flush=True)


def get_or_add(mid, formula, name, charge):
    """Return an existing metabolite or create and register a new one."""
    if mid in model.metabolites:
        return model.metabolites.get_by_id(mid)
    m = Metabolite(mid, formula=formula, name=name, charge=charge,
                   compartment="c")
    model.add_metabolites([m])
    return m


ggpp = get_or_add("ggpp_c", "C20H33O7P2", "geranylgeranyl diphosphate", -3)
phy = get_or_add("phytoene_c", "C40H64", "15-cis-phytoene", 0)
lyc = get_or_add("lycopene_c", "C40H56", "lycopene", 0)

NEW = [
    ("CRTE", "CrtE geranylgeranyl diphosphate synthase",
     {"frdp_c": -1.0, "ipdp_c": -1.0, "ggpp_c": 1.0, "ppi_c": 1.0}),
    ("CRTB", "CrtB phytoene synthase",
     {"ggpp_c": -2.0, "phytoene_c": 1.0, "ppi_c": 2.0}),
    ("CRTI", "CrtI phytoene desaturase",
     {"phytoene_c": -1.0, "fad_c": -4.0, "lycopene_c": 1.0, "fadh2_c": 4.0}),
]
for rid, name, stoich in NEW:
    if rid in [r.id for r in model.reactions]:
        model.remove_reactions([model.reactions.get_by_id(rid)])
    r = Reaction(rid)
    r.name = name
    r.lower_bound, r.upper_bound = -1000, 1000
    model.add_reactions([r])
    r.add_metabolites({model.metabolites.get_by_id(k): v for k, v in stoich.items()})
    bal = r.check_mass_balance()
    print(f"  {rid}: {'balanced' if not bal else bal}", flush=True)

if "DM_lycopene_c" in [r.id for r in model.reactions]:
    model.remove_reactions([model.reactions.get_by_id("DM_lycopene_c")])
dm = Reaction("DM_lycopene_c")
dm.name = "lycopene demand"
dm.lower_bound, dm.upper_bound = 0, 1000
model.add_reactions([dm])
dm.add_metabolites({lyc: -1.0})

ENV = dict(glucose=10.0, oxygen=20.0)
model.reactions.get_by_id("EX_glc__D_e").lower_bound = -ENV["glucose"]
model.reactions.get_by_id("EX_o2_e").lower_bound = -ENV["oxygen"]

mu_max = model.slim_optimize()
print(f"\nmu_max (iML1515, glucose 10 mmol/gDCW/h, aerobic) = {mu_max:.4f} /h", flush=True)

# native isoprenoid flux at the biomass optimum
sol = model.optimize()
for rid in ["DXPS", "IPDDI", "OCTDPS", "UDCPDPS", "GRTT", "DMATT"]:
    if rid in [r.id for r in model.reactions]:
        print(f"  {rid:9s} {sol.fluxes[rid]:8.4f} mmol/gDCW/h", flush=True)

# stoichiometric ceiling for lycopene
model.objective = "DM_lycopene_c"
lyc_max = model.slim_optimize()
print(f"stoichiometric max lycopene = {lyc_max:.3f} mmol/gDCW/h "
      f"= {lyc_max*1000:.0f} umol/gDCW/h", flush=True)

# growth-production envelope
model.objective = model.reactions.get_by_id("BIOMASS_Ec_iML1515_core_75p37M")
env_rows = []
for mu in np.linspace(0, mu_max * 0.99, 25):
    with model:
        model.reactions.get_by_id("BIOMASS_Ec_iML1515_core_75p37M").lower_bound = mu
        model.objective = "DM_lycopene_c"
        v = model.slim_optimize()
    env_rows.append({"mu_per_h": mu, "lycopene_umol_gDCW_h": v * 1000 if v else 0.0})
env = pd.DataFrame(env_rows)
env.to_csv(f"{OUT}/ecgem_envelope.csv", index=False)
print("\n--- production envelope (selected) ---")
print(env.iloc[::6].to_string(index=False), flush=True)

# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
summary = {
    "kcat_predictions": pred[["enzyme", "substrate", "kcat_s-1",
                              "disagreement_log10"]].to_dict("records"),
    "published_targets_umol_gDCW_h": tgt[["case", "flux_umol_gDCW_h"]].to_dict("records"),
    "assumptions": {
        "protein_fraction_g_per_gDCW": PROTEIN_FRACTION,
        "dalton_per_residue": RESIDUE_DA,
        "lycopene_MW": LYC_MW,
        "glucose_uptake_mmol_gDCW_h": ENV["glucose"],
    },
    "mu_max_per_h": mu_max,
    "stoich_max_lycopene_umol_gDCW_h": lyc_max * 1000,
    "measured_flux_range_umol_gDCW_h": [float(tgt.flux_umol_gDCW_h.min()),
                                        float(tgt.flux_umol_gDCW_h.max())],
    "headroom_at_1pct_protein": {r.enzyme: float(r.headroom_at_1pct)
                                 for _, r in cap[np.isclose(cap.target_flux, hi)].iterrows()},
    "required_frac_at_max_target": {r.enzyme: float(r.required_frac_of_protein)
                                    for _, r in cap[np.isclose(cap.target_flux, hi)].iterrows()},
    "native_isoprenoid_flux_at_mumax": {
        k: float(sol.fluxes[k]) for k in
        ["DXPS", "OCTDPS", "UDCPDPS"] if k in [r.id for r in model.reactions]},
}
with open(f"{OUT}/ecgem_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nsaved results/ecgem_{predictions,targets,capacity}.tsv, "
      "ecgem_envelope.csv, ecgem_summary.json")
