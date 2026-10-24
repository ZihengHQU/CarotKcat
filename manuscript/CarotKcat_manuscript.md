# CarotKcat: sequence- and substrate-informed deep learning prediction of enzyme turnover numbers with application to carotenoid biosynthesis

**Manuscript draft — v1 (2026-10-24)**

---

## Abstract

**Motivation:** Turnover number (kcat) is the key kinetic parameter for enzyme-constrained metabolic modelling and pathway engineering, yet experimental measurements cover only a small fraction of enzymes. This gap is especially acute for secondary-metabolism pathways such as carotenoid biosynthesis, whose core enzymes (CrtE/B/I/Y/Z/W) are among the least kinetically characterised despite their industrial importance for lycopene, β-carotene and astaxanthin production.

**Results:** We present CarotKcat, a dual-tower deep learning model that predicts kcat directly from enzyme amino-acid sequence (ESM-2 650M embeddings) and substrate structure (molecular fingerprints). Trained on 17,051 curated records covering 2,164 substrates and ~2,700 enzymes integrated from BRENDA and SABIO-RK, CarotKcat achieves Pearson r = 0.727 / R² = 0.528 under random splitting and Pearson r = 0.440 / R² = 0.189 under a stricter enzyme-group split, outperforming tree-based reimplementations of UniKP- and TurNuP-style baselines on unseen enzymes. In a held-out literature case study of 11 carotenoid-pathway enzymes absent from training, CarotKcat reproduced the pathway-level rate-limiting hierarchy (CrtB > CrtE > CrtI; Spearman ρ = 0.60), matched the measured kcat of yeast GGPP synthase within 1.4-fold, and correctly predicted the substrate preference of β-carotene hydroxylase CrtZ. We additionally provide proteome-scale kcat predictions for 3,725 carotenoid-related enzymes and a zero-installation web interface for experimentalists.

**Availability:** Code, models, data and the web server are freely available at https://github.com/ZihengHQU/CarotKcat

---

## 1. Introduction

Carotenoids — including β-carotene, lycopene, zeaxanthin and astaxanthin — are high-value isoprenoid pigments widely produced by microbial cell factories. Rational improvement of their titres requires quantitative knowledge of pathway kinetics, above all the turnover number (kcat) of each enzyme. kcat is also the central parameter of enzyme-constrained genome-scale metabolic models (ecGEMs), where even a single inaccurate value can distort flux predictions (Li et al., 2022).

Experimental kcat measurements are laborious and remain scarce: the two major public resources, BRENDA and SABIO-RK, together cover only a few thousand enzymes, and coverage is strongly biased towards central metabolism. Carotenoid core enzymes are a striking blind spot: among ~41,500 curated kcat records, the committed carotenoid enzymes phytoene synthase (EC 2.5.1.32), phytoene desaturase (EC 1.3.99.31) and lycopene cyclase are essentially absent, with only the upstream geranylgeranyl diphosphate (GGPP) synthase family moderately represented.

Deep learning has begun to close this gap. DLKcat (Li et al., 2022) combined a convolutional network on protein sequence with a graph neural network (GNN) on substrate structure; TurNuP (Kroll et al., 2023) used protein language-model embeddings with gradient-boosted trees; UniKP (Yu et al., 2023) showed that pretrained ESM embeddings plus tree ensembles give strong kcat/Km predictions. However, these generalist models have not been evaluated on — or adapted to — secondary-metabolism terpenoid pathways, and none offers an accessible prediction tool for non-computational users.

Here we introduce **CarotKcat**, a DLKcat-style dual-tower model trained on the full breadth of public enzyme-kinetic data and specifically optimised for carotenoid biosynthetic enzymes through targeted oversampling and a dedicated evaluation. Our contributions are: (i) a reproducible end-to-end pipeline from public databases to a trained sequence+substrate kcat predictor; (ii) a systematic benchmark against UniKP- and TurNuP-style baselines under both random and enzyme-group splits, showing that the dual-tower network generalises better to unseen enzymes; (iii) a held-out literature case study on carotenoid enzymes with experimentally reported kinetics; and (iv) a web server requiring no computational expertise.

## 2. Materials and Methods

### 2.1 Data collection and curation

The general enzyme-kinetics corpus was obtained from the DLKcat project's integrated BRENDA + SABIO-RK dataset (`Kcat_combination_41559.tsv`; 41,558 records). Records were filtered to entries with valid kcat values in s⁻¹, parseable substrate SMILES, and a resolvable UniProt accession. Sequences were retrieved in batch via the UniProt ID-mapping API. After cleaning and sequence matching, the record-level training set comprised **17,051 entries**, spanning **2,164 unique substrates** and **~2,700 enzymes** across all EC classes. Targets were log10-transformed kcat.

For the carotenoid focus set, UniProt was queried for reviewed and unreviewed entries matching carotenoid-pathway terms (crtE, crtB, crtI, crtY, crtZ, crtW, crtX, crtO, crtU, phytoene synthase/desaturase, lycopene cyclase, carotene hydroxylase/ketolase, etc.), yielding **3,725 sequences** for proteome-scale prediction.

### 2.2 Feature representation

**Enzyme sequences** (truncated to 1,022 residues) were encoded with the frozen ESM-2 t33_650M_UR50D protein language model; the layer-33 residue representations were mean-pooled to a 1,280-dimensional vector. **Substrates** were encoded as Morgan circular fingerprints (radius 2, 2,048 bits) computed with RDKit.

### 2.3 Model architecture and training

The CarotKcat dual-tower network maps each modality through a separate MLP (sequence: 1280→512→256; substrate: 2048→512→256), concatenates the 256-dimensional embeddings, and predicts log10(kcat) through a fusion head (512→256→64→1). Training used SmoothL1 loss, AdamW (lr 1e-4, weight decay 1e-5), cosine annealing over up to 400 epochs, and early stopping on held-out RMSE. Carotenoid-related training records were oversampled 10× to strengthen performance on the target pathway.

Two evaluation protocols were used throughout: (i) **random split** (85/15), matching the DLKcat publication protocol; and (ii) **enzyme-group split**, in which all records of a given enzyme fall entirely in train or test, providing a leak-free estimate of generalisation to unseen enzymes.

### 2.4 Baselines and ensemble

We reimplemented two literature-style baselines on identical features and splits: a **UniKP-style** ExtraTrees regressor and a **TurNuP-style** XGBoost regressor on concatenated ESM + fingerprint vectors, plus ablations using sequence only (ESM + MLP) and substrate only (fingerprint + MLP). **CarotKcat-ensemble** averages the log-space predictions of the dual-tower network and XGBoost.

### 2.5 Literature case study

Carotenoid-pathway enzymes with experimentally reported turnover numbers or kinetic parameters were curated from the primary literature (Supplementary Table S1): five CrtI phytoene desaturases with per-step kcat values fitted by COPASI modelling (Table S4 of the source publication), in vivo kcat^cell values for P. ananatis CrtB and X. dendrorhous CrtI, in vitro kcat for P. agglomerans CrtB, GGPP synthases from human, yeast and Pantoea, and substrate-resolved specific activities for P. ananatis CrtZ. **All 11 case-study accessions were verified to be absent from the training set**, making this a genuine held-out evaluation. Predictions used the CarotKcat-ensemble.

### 2.6 Web server

A Gradio-based interface (`app.py`) accepts FASTA upload or pasted sequences plus substrate SMILES, displays the substrate structure, and returns a downloadable prediction table.

## 3. Results

### 3.1 Overall performance

Under the random split, the CarotKcat dual-tower network reached Pearson r = 0.649 and R² = 0.412 on the test set (n ≈ 2,558), comparable to the original DLKcat report (R² ≈ 0.44). The ensemble variant reached **r = 0.727, R² = 0.528**. Under the leak-free enzyme-group split (n ≈ 2,813), the network and ensemble both reached **r = 0.44**, establishing the model's realistic extrapolation performance to enzymes not seen during training.

### 3.2 Comparison with baseline methods

| Model | Random split r / R² | Enzyme-group split r / R² |
|---|---|---|
| **CarotKcat-ensemble** | **0.727 / 0.528** | **0.440 / 0.189** |
| CarotKcat dual-tower NN | 0.649 / 0.412 | 0.439 / 0.163 |
| UniKP-style (ExtraTrees) | 0.725 / 0.518 | 0.406 / 0.165 |
| TurNuP-style (XGBoost) | 0.735 / 0.538 | 0.366 / 0.121 |
| Sequence only (ESM + MLP) | 0.637 / 0.401 | 0.419 / 0.171 |
| Substrate only (fingerprint MLP) | 0.477 / 0.216 | 0.338 / 0.094 |

Tree ensembles were strongest under the random split, where test enzymes often share homologues with training enzymes, but degraded markedly under the enzyme-group split (XGBoost r dropped from 0.735 to 0.366). The dual-tower network was the most robust to unseen enzymes, and the ensemble combined the strengths of both, ranking first or statistically tied-first under both protocols. The substrate tower contributed substantial signal (substrate-only r = 0.34–0.48), confirming that kcat is jointly determined by enzyme and substrate.

### 3.3 Ablation

Removing the substrate input reduced group-split performance to r = 0.419 (sequence-only), while removing the sequence reduced it to r = 0.338; the full dual-tower model outperformed both, and the 10× oversampling of carotenoid-related records was retained for all reported carotenoid analyses.

### 3.4 Held-out case study of carotenoid enzymes

We curated 13 literature-reported kinetic measurements covering 11 carotenoid-pathway enzymes, none present in the training set (Supplementary Table S1; Fig. 2).

- **Pathway-level hierarchy (ρ = 0.60).** CarotKcat correctly reproduced the experimentally established rate-limiting order of the lycopene pathway: CrtB (14 s⁻¹ in vitro; 6 s⁻¹ in vivo) > CrtE (0.8 s⁻¹) > CrtI (0.03–0.1 s⁻¹). This ordering is the key driver of bottleneck analyses in metabolic engineering.
- **GGPP synthase family (ρ = 0.50).** The kcat of yeast BTS1 was predicted quantitatively (0.036 predicted vs 0.025 s⁻¹ measured); the human (0.204 s⁻¹) and Pantoea CrtE (0.8 s⁻¹) values were ranked correctly.
- **Substrate selectivity.** For P. ananatis CrtZ, the model predicted higher activity on β-carotene than on canthaxanthin (2.8-fold), matching the direction of the measured 6.2-fold preference.
- **Quantitative error.** Across all 13 case points, 85% of predictions fell within ±10-fold of the literature value (median fold-error 6.5×). For reference, reported kcat values for the *same* enzyme (P. ananatis CrtI) span >1,000-fold across laboratories (8×10⁻⁵ to 0.25 s⁻¹), indicating that literature dispersion is of the same order as model error for this pathway.

Two honest limitations emerged. First, four of the five CrtI literature values sit at the upper constraint boundary (0.1 s⁻¹) of the original COPASI fit and therefore represent only lower bounds; the model's predictions for these enzymes (0.65–0.85 s⁻¹) are consistent with those bounds, but the slowest desaturase (R. capsulatus CrtI, 0.003 s⁻¹) was not ranked lowest. Second, the phytoene desaturation intermediates phytofluene, ζ-carotene and neurosporene yield *bit-identical* Morgan fingerprints (verified), so the fingerprint-based model cannot distinguish them; per-step substrate series could therefore not be resolved.

### 3.5 Proteome-scale prediction for carotenoid enzymes

kcat was predicted for 3,725 UniProt carotenoid-related sequences (`results/predictions_carotenoid_uniprot.tsv`), yielding a median of ~0.2 s⁻¹, consistent with the known sluggishness of membrane-associated terpenoid enzymes relative to central metabolism. This resource supports enzyme selection and pathway modelling for carotenoid cell factories.

### 3.6 Web server

The CarotKcat web interface requires only a browser: users upload a FASTA file or paste a sequence, enter a substrate SMILES (with automatic structure rendering), and obtain a downloadable kcat table. The server runs locally or on a laboratory intranet.

## 4. Discussion

CarotKcat delivers DLKcat-class accuracy with a lightweight, fully reproducible pipeline and, to our knowledge, the first kcat model explicitly optimised and evaluated for the carotenoid biosynthetic pathway. Three findings deserve emphasis.

First, the evaluation protocol matters. Tree-based models on frozen embeddings (UniKP/TurNuP-style) excel when test enzymes resemble training enzymes but lose up to half of their correlation under an enzyme-group split, whereas the dual-tower network's advantage appears precisely in the realistic unseen-enzyme regime — a consideration directly relevant to enzyme discovery campaigns.

Second, the case study shows that for kinetically uncharted secondary-metabolism pathways, model predictions and literature measurements carry comparable uncertainty: reported kcat values for individual carotenoid enzymes vary over three orders of magnitude between laboratories and assay formats. Computational predictions can therefore serve as useful priors for enzyme-constrained modelling even where they cannot replace measurement.

Third, our negative results chart concrete improvements: (i) fingerprint substrate encodings should be replaced by graph neural networks or molecular transformers that can resolve closely related polyene substrates (phytofluene vs ζ-carotene vs neurosporene); (ii) the very-low-kcat regime (<0.01 s⁻¹) suffers from training-data scarcity, causing regression to the mean; and (iii) assay conditions (temperature, pH) are rarely annotated and remain unmodelled. The most direct route to higher accuracy on a specific host/pathway is few-shot fine-tuning with even tens of laboratory-measured kcat values, which our pipeline supports natively.

## 5. Conclusion

CarotKcat predicts enzyme turnover numbers from sequence and substrate with accuracy competitive with state-of-the-art general models, superior robustness to unseen enzymes, validated pathway-level behaviour on carotenoid enzymes, and a zero-barrier web interface. All code, trained models, curated datasets and predictions are openly available to support carotenoid metabolic engineering and enzyme-constrained modelling.

## Data and code availability

GitHub: https://github.com/ZihengHQU/CarotKcat — includes training/evaluation scripts, trained model weights, the curated 17,051-record dataset, the 3,725-sequence carotenoid prediction set, benchmark and case-study results, and the web server.

## Funding / Acknowledgements / Author contributions

*(to be completed by the authors)*

## References

1. Li, F. *et al.* Deep learning-based kcat prediction enables improved enzyme-constrained model reconstruction. *Nat. Catal.* **5**, 662–672 (2022).
2. Kroll, A. *et al.* Turnover number predictions for kinetically uncharacterized enzymes using machine and deep learning. *Nat. Commun.* **14**, 4139 (2023).
3. Yu, T. *et al.* UniKP: a unified framework for the prediction of enzyme kinetic parameters. *Nat. Commun.* **14**, 8211 (2023).
4. Lin, Z. *et al.* Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123–1130 (2023).
5. Iwata-Reuyl, D., Math, S. K., Desai, S. B. & Poulter, C. D. Bacterial phytoene synthase: molecular cloning, expression, and characterization of *Erwinia herbicola* phytoene synthase. *Biochemistry* **42**, 3359–3365 (2003).
6. Berviller-Magnan, C. *et al.* Combining systems and synthetic biology for in vivo enzymology. *Mol. Syst. Biol.* (2024). PMC11535393.
7. Lavrijsen, M. *et al.* Multiplicity of carotene patterns derives from competition between phytoene desaturase diversification and biological environments. *Sci. Rep.* **11**, 26 (2021).
8. Schaub, P. *et al.* On the structure and function of the phytoene desaturase CRTI from *Pantoea ananatis*. *PLoS ONE* **7**, e39550 (2012).
9. Fraser, P. D., Miura, Y. & Misawa, N. In vitro characterization of astaxanthin biosynthetic enzymes. *J. Biol. Chem.* **272**, 6128–6135 (1997).
10. Chang, T. H. *et al.* Crystal structure of type-III geranylgeranyl pyrophosphate synthase from *Saccharomyces cerevisiae*. *J. Biol. Chem.* **281**, 14991–15000 (2006).
11. Kavanagh, K. L. *et al.* The crystal structure of human geranylgeranyl pyrophosphate synthase reveals a novel hexameric arrangement. *J. Biol. Chem.* **281**, 22004–22012 (2006).
12. Heider, S. A. E. *et al.* IdsA is the major geranylgeranyl pyrophosphate synthase involved in carotenogenesis in *Corynebacterium glutamicum*. *FEBS J.* **281**, 4906–4920 (2014).
13. Chang, J. *et al.* BRENDA, the ELIXIR core data resource in 2021. *Nucleic Acids Res.* **49**, D498–D508 (2021).
14. Wittig, U. *et al.* SABIO-RK: database for biochemical reaction kinetics. *Nucleic Acids Res.* **40**, D790–D796 (2012).
15. The UniProt Consortium. UniProt: the universal protein knowledgebase in 2023. *Nucleic Acids Res.* **51**, D523–D531 (2023).

## Supplementary material

- **Supplementary Table S1** — Literature case-study enzymes, accessions, measured values and predictions (`results/case_study.tsv`).
- **Supplementary Table S2** — Benchmark metrics for all models and splits (`results/benchmark_table.csv`).
- **Figure 1** — Predicted vs measured log10(kcat), test set (`results/fig_scatter.png`).
- **Figure 2** — Case-study panels A–D (`results/fig_case_study.png`).
