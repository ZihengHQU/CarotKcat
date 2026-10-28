# CarotKcat: sequence- and substrate-informed deep learning prediction of enzyme turnover numbers with application to carotenoid biosynthesis

**Manuscript draft — v2 (2026-10-24)**

---

## Abstract

**Motivation:** Turnover number (kcat) is the key kinetic parameter for enzyme-constrained metabolic modelling and pathway engineering, yet experimental measurements cover only a small fraction of enzymes. This gap is especially acute for secondary-metabolism pathways such as carotenoid biosynthesis, whose core enzymes (CrtE/B/I/Y/Z/W) are among the least kinetically characterised despite their industrial importance for lycopene, β-carotene and astaxanthin production.

**Results:** We present CarotKcat, a dual-tower deep learning model that predicts kcat directly from enzyme amino-acid sequence (ESM-2 650M embeddings) and substrate structure (molecular fingerprints). Trained on 17,051 curated records covering 2,162 substrates and ~3,000 enzymes integrated from BRENDA and SABIO-RK, CarotKcat achieves Pearson r = 0.723 (95% bootstrap CI 0.698–0.746) / R² = 0.522 under random splitting and r = 0.420 (0.388–0.452) / R² = 0.165 under a stricter enzyme-group split. This matches strong tree-based baselines; across five independent splits the ensemble was the best model on unseen enzymes in four of five folds (r = 0.463 ± 0.048, Δr = +0.042 versus gradient-boosted trees, paired p = 0.024) but was reproducibly ~0.02 below them under random splitting, so we claim parity, not superiority. A from-scratch reimplementation of the DLKcat GNN+CNN architecture reaches only r = 0.47 under the same protocol. In a held-out literature case study of 11 carotenoid-pathway enzymes absent from training, CarotKcat reproduced the pathway-level rate-limiting hierarchy (CrtB > CrtE > CrtI; Spearman ρ = 0.60), matched the measured kcat of yeast GGPP synthase within 1.4-fold, and correctly predicted the substrate preference of β-carotene hydroxylase CrtZ. On two independent external benchmarks we report both a positive and a negative result: a temporal holdout assembled from 2023–2025 literature (KcatTIE) retains moderate accuracy (r = 0.56, median fold-error 6.0×), whereas ranking the exceptionally fast adenylate-kinase family (177 ADK orthologs) fails (r = 0.05, systematic ~16-fold under-prediction). Single point mutations changed predictions by less than 2-fold in all 17 cases tested, below the model's own uncertainty, so CarotKcat is not suitable for mutant ranking without fine-tuning. To test extrapolation directly we assembled a curated benchmark of 43 terpenoid-pathway turnover numbers; accuracy depended sharply on whether the enzyme's family occurred in training (r = 0.66, median error 2.1-fold when it did; r = 0.14, median error 8.6-fold when it did not), and for the carotenoid module itself the rank correlation was negative (r = −0.54, median error 26-fold), with a predicted-on-measured slope of 0.38 overall indicating compression towards the training mean. CarotKcat is therefore an order-of-magnitude prior for carotenoid enzymes, not a ranking tool. We additionally provide proteome-scale kcat predictions for 3,725 carotenoid-related enzymes and a zero-installation web interface for experimentalists. Finally, as a dry substitute for wet-lab validation we embedded the lycopene pathway in the enzyme-constrained genome-scale model iML1515: the predicted kcat values reproduce the experimentally established bottleneck (CrtE/CrtB/CrtI remain 25–274-fold below saturation at 1% of total protein, and the measured lycopene flux is only 1.4-fold above the entire native MEP-pathway flux), showing that the predictions are quantitatively compatible with published carotenoid physiology.

**Availability:** Code, models, data and the web server are freely available at https://github.com/ZihengHQU/CarotKcat

**Keywords:** kcat; enzyme kinetics; deep learning; protein language model; carotenoid biosynthesis; metabolic engineering

---

## 1. Introduction

Carotenoids — including β-carotene, lycopene, zeaxanthin and astaxanthin — are high-value isoprenoid pigments widely produced by microbial cell factories. Rational improvement of their titres requires quantitative knowledge of pathway kinetics, above all the turnover number (kcat) of each enzyme. kcat is also the central parameter of enzyme-constrained genome-scale metabolic models (ecGEMs), where even a single inaccurate value can distort flux predictions (Li et al., 2022).

Experimental kcat measurements are laborious and remain scarce: the two major public resources, BRENDA and SABIO-RK, together cover only a few thousand enzymes, and coverage is strongly biased towards central metabolism. Carotenoid core enzymes are a striking blind spot: among ~41,500 curated kcat records, the committed carotenoid enzymes phytoene synthase (EC 2.5.1.32), phytoene desaturase (EC 1.3.99.31) and lycopene cyclase are essentially absent, with only the upstream geranylgeranyl diphosphate (GGPP) synthase family moderately represented.

Deep learning has begun to close this gap. DLKcat (Li et al., 2022) combined a convolutional network on protein sequence with a graph neural network (GNN) on substrate structure; TurNuP (Kroll et al., 2023) used protein language-model embeddings with gradient-boosted trees; UniKP (Yu et al., 2023) showed that pretrained ESM embeddings plus tree ensembles give strong kcat/Km predictions. However, these generalist models have not been evaluated on — or adapted to — secondary-metabolism terpenoid pathways, and none offers an accessible prediction tool for non-computational users.

Here we introduce **CarotKcat**, a DLKcat-style dual-tower model trained on the full breadth of public enzyme-kinetic data and specifically optimised for carotenoid biosynthetic enzymes through targeted oversampling and a dedicated evaluation. Our contributions are: (i) a reproducible end-to-end pipeline from public databases to a trained sequence+substrate kcat predictor; (ii) a systematic benchmark against UniKP- and TurNuP-style baselines under both random and enzyme-group splits, repeated over five independent splits so that model differences can be separated from split noise, and reported with a deliberately conservative reading of the outcome (parity with strong tree ensembles, not superiority); (iii) a held-out literature case study on carotenoid enzymes with experimentally reported kinetics, together with external validation that delimits the model's applicability domain; and (iv) a web server requiring no computational expertise.

## 2. Materials and Methods

### 2.1 Data collection and curation

The general enzyme-kinetics corpus was obtained from the DLKcat project's integrated BRENDA + SABIO-RK dataset (`Kcat_combination_41559.tsv`; 41,558 records). Records were filtered to entries with valid kcat values in s⁻¹, parseable substrate SMILES, and a resolvable UniProt accession. Sequences were retrieved in batch via the UniProt ID-mapping API. After cleaning and sequence matching, the record-level training set comprised **17,051 entries**, spanning **2,162 unique substrates** and **2,983 enzymes** across all EC classes. Targets were log10-transformed kcat.

For the carotenoid focus set, UniProt was queried for reviewed and unreviewed entries matching carotenoid-pathway terms (crtE, crtB, crtI, crtY, crtZ, crtW, crtX, crtO, crtU, phytoene synthase/desaturase, lycopene cyclase, carotene hydroxylase/ketolase, etc.), yielding **3,725 sequences** for proteome-scale prediction.

### 2.2 Feature representation

**Enzyme sequences** (truncated to 1,022 residues) were encoded with the frozen ESM-2 t33_650M_UR50D protein language model; the layer-33 residue representations were mean-pooled to a 1,280-dimensional vector. **Substrates** were encoded as Morgan circular fingerprints (radius 2, 2,048 bits) computed with RDKit.

### 2.3 Model architecture and training

The CarotKcat dual-tower network (Fig. 1) maps each modality through a separate MLP (sequence: 1280→512→256; substrate: 2048→512→256), concatenates the 256-dimensional embeddings, and predicts log10(kcat) through a fusion head (512→256→64→1). Training used SmoothL1 loss, AdamW (lr 1e-4, weight decay 1e-5), cosine annealing over up to 400 epochs, and early stopping on held-out RMSE. Carotenoid-related training records were oversampled 10× to strengthen performance on the target pathway.

![Figure 1](../results/fig_architecture.png)

Two evaluation protocols were used throughout: (i) **random split** (85/15), matching the DLKcat publication protocol; and (ii) **enzyme-group split**, in which all records of a given enzyme fall entirely in train or test, providing a leak-free estimate of generalisation to unseen enzymes. To separate genuine model differences from split-to-split noise, the whole pipeline was repeated on five independent splits per protocol with fixed hyper-parameters (no per-fold tuning); results are reported as mean ± SD across folds in addition to the single reference split.

### 2.4 Baselines and ensemble

We reimplemented two literature-style baselines on identical features and splits: a **UniKP-style** ExtraTrees regressor and a **TurNuP-style** XGBoost regressor on concatenated ESM + fingerprint vectors, plus ablations using sequence only (ESM + MLP) and substrate only (fingerprint + MLP). **CarotKcat-ensemble** averages the log-space predictions of the dual-tower network and XGBoost.

### 2.5 Literature case study

Carotenoid-pathway enzymes with experimentally reported turnover numbers or kinetic parameters were curated from the primary literature (Supplementary Table S1): five CrtI phytoene desaturases with per-step kcat values fitted by COPASI modelling (Table S4 of the source publication), in vivo kcat^cell values for P. ananatis CrtB and X. dendrorhous CrtI, in vitro kcat for P. agglomerans CrtB, GGPP synthases from human, yeast and Pantoea, and substrate-resolved specific activities for P. ananatis CrtZ. **All 11 case-study accessions were verified to be absent from the training set**, making this a genuine held-out evaluation. Predictions used the CarotKcat-ensemble.

### 2.6 Web server

A Gradio-based interface (`app.py`) accepts FASTA upload or pasted sequences plus substrate SMILES, displays the substrate structure, and returns a downloadable prediction table.

### 2.7 In silico validation strategy

Because no new experiments were performed, validation relies entirely on data already in the literature. Four levels were used: (i) exact-sequence leakage screening followed by prediction on temporally or taxonomically independent kcat sets (section 3.5); (ii) a held-out literature case study (section 3.4); (iii) a dry validation in which predicted kcat values parameterise an enzyme-constrained genome-scale model and the resulting fluxes are compared with published productivities (section 3.7); and (iv) a curated zero-shot benchmark of terpenoid-pathway enzymes drawn from families absent from training (section 3.8). For (iii) we used iML1515 (Monk et al., 2017) with glucose uptake fixed at 10 mmol gDCW⁻¹ h⁻¹ under aerobic conditions; the CrtE/CrtB/CrtI reactions were added with elemental balance verified by `cobra`, and the native MEP-pathway flux was read as the DXS reaction flux at the biomass optimum. Enzyme capacity was computed as kcat × [E] × 3600, with [E] derived from a stated total-protein content of 0.50 g per gDCW and a molecular weight of 110 Da per residue; both assumptions are varied implicitly by reporting the required enzyme fraction rather than a single predicted titre.

### 2.8 Curated terpenoid benchmark

Because neither the training corpus nor BRENDA contains turnover numbers for the carotenoid-specific reactions, a dedicated benchmark was assembled rather than sampled from an existing split. Two sources were used. (i) BRENDA kinetic tables for 20 terpenoid and isoprenoid EC classes were retrieved and parsed field by field (`scripts/21`), giving one row per measurement with accession, organism, assay conditions and BRENDA literature identifier. Substrate SMILES were taken from the training table itself where the substrate occurred there, so that featurisation matched training exactly. (ii) For the carotenoid module, where BRENDA holds only Michaelis constants, values were taken from primary literature and are listed individually with their citations in Table S7.

Records were filtered as follows: a single UniProt accession was required; substrates were restricted to the prenyl-diphosphate, carotenoid and central-precursor vocabulary (unnatural inhibition probes such as nitroso aromatics were discarded); any (accession, substrate) pair present in the training table was removed as leakage; and repeated measurements of the same pair were collapsed to their median to avoid pseudo-replication. Accessions that UniProt has since deleted or demerged (two cases) were dropped. This left 43 measurements on 35 enzymes spanning carotenoid (n = 12), terpene synthase (n = 12), precursor (n = 12) and isoprenoid-supply (n = 7) families.

Predictions used the deployed ensemble (dual-tower network and XGBoost averaged in log space) with no retraining. Two regimes were reported separately: whether the EC family occurs in the training set at all, and whether the specific accession does. Metrics are Pearson r, Spearman ρ, R², median absolute fold error, and the fraction of predictions within 3-fold and 10-fold. We additionally report the **response slope**, the ordinary least-squares slope of predicted on measured log10 kcat, because a model that emits an almost constant value can still achieve a high correlation across a wide measured range, and the slope exposes that failure mode directly.

## 3. Results

### 3.1 Overall performance

Under the random split, the CarotKcat dual-tower network reached Pearson r = 0.636 (95% CI 0.602–0.665) and R² = 0.386 on the test set (n = 2,558), comparable to the original DLKcat report (R² ≈ 0.44). The ensemble variant reached **r = 0.723 (0.698–0.746), R² = 0.522**. Under the leak-free enzyme-group split (n = 2,813) the network reached r = 0.404 (0.371–0.434) and the ensemble **r = 0.420 (0.388–0.452)**, lower than under random splitting but obtained on enzymes without training-set homologues and therefore the more realistic estimate for prospective use. Predicted versus measured values for the ensemble under the random split are shown in Fig. 2.

![Figure 2](../results/fig_scatter.png)

### 3.2 Comparison with baseline methods

Table 1 and Fig. 3 summarise the benchmark under both evaluation protocols.

**Table 1.** Leak-free benchmark of CarotKcat against reimplemented literature-style baselines, a from-scratch DLKcat-architecture reimplementation, and single-modality ablations. Values are test-set Pearson r (95% bootstrap CI) / R².

| Model | Random split r (95% CI) / R² | Enzyme-group split r (95% CI) / R² |
|---|---|---|
| **CarotKcat-ensemble** | **0.723 (0.698–0.746) / 0.522** | **0.420 (0.388–0.452) / 0.165** |
| CarotKcat dual-tower NN | 0.636 (0.602–0.665) / 0.386 | 0.404 (0.371–0.434) / 0.098 |
| CarotKcat-GNN (ESM + GNN substrate) | 0.642 (0.613–0.668) / 0.407 | 0.417 (0.390–0.445) / 0.156 |
| UniKP-style (ExtraTrees) | 0.725 (0.698–0.750) / 0.518 | 0.406 (0.372–0.437) / 0.165 |
| TurNuP-style (XGBoost) | 0.735 (0.711–0.759) / 0.538 | 0.366 (0.329–0.398) / 0.121 |
| DLKcat-style (GNN+CNN, reimplemented) | 0.466 (0.427–0.503) / 0.208 | 0.356 (0.319–0.393) / 0.108 |
| Sequence only (ESM + MLP) | 0.621 (0.589–0.650) / 0.381 | 0.370 (0.341–0.398) / 0.123 |
| Substrate only (fingerprint MLP) | 0.467 (0.430–0.503) / 0.206 | 0.330 (0.296–0.364) / 0.084 |
| DLKcat original (published)† | 0.71 / ~0.44 | n.r. |

† Li et al. (2022) reported Pearson r = 0.71 and RMSE = 1.06 on their 10% held-out split of the same BRENDA + SABIO-RK corpus; R² is not stated in the original paper and ≈0.44 is the value cited in later work. The original trained weights are not distributed in a directly executable form and the split differs, so these values are a reference point rather than a like-for-like head-to-head. n.r., not reported in an equivalent form. The same ranking is reproduced over five independent splits (Supplementary Table S4).

All neural models were early-stopped on a validation split carved out of the training data, and the test set was scored exactly once; a previous version that selected checkpoints on the test set gave r values 0.01–0.02 higher (retained as `results/benchmark_table_test_earlystop_legacy.csv` for transparency). Tree ensembles were strongest under the random split, where test enzymes often share homologues with training enzymes, but degraded under the enzyme-group split (XGBoost r dropped from 0.735 to 0.366). Paired bootstrap tests on the same test instances show that the ensemble was statistically indistinguishable from ExtraTrees under both protocols (random Δr = −0.001, p = 0.85; group Δr = +0.014, p = 0.15) and marginally below XGBoost under the random split (Δr = −0.012, p = 0.064), but significantly better than XGBoost under the group split (Δr = +0.054, p < 0.001). The dual-tower network alone (group r = 0.404) was not better than ExtraTrees (0.406); its value is complementary, and averaging the two components gives the best group-split result. Replacing the fingerprint tower with a molecular-graph GNN (CarotKcat-GNN) raised R² from 0.386 to 0.407 under random splitting and from 0.098 to 0.156 under the enzyme-group split, and lowered group RMSE from 1.39 to 1.35, although the Pearson gain over the fingerprint dual-tower was not significant (group Δr = +0.014, p = 0.24). Its principal benefit is qualitative: it distinguishes polyene substrates that share bit-identical fingerprints (section 3.4). We therefore describe CarotKcat as matching, not surpassing, the strongest tree baseline. The substrate tower contributed substantial signal (substrate-only r = 0.33–0.47), confirming that kcat is jointly determined by enzyme and substrate.

To place CarotKcat within the DLKcat architectural family rather than only against feature-based baselines, we reimplemented the DLKcat encoder — a molecular-graph GNN for the substrate branch and an amino-acid n-gram CNN for the protein branch — and trained it on exactly the same records, splits and seeds. This from-scratch reimplementation reached r = 0.466 (random) and 0.356 (group), well below CarotKcat. The gap is diagnostic rather than incidental: with ~13,000 training records, a sequence CNN trained from scratch cannot learn a protein representation as informative as a pretrained ESM-2 encoder, which is precisely the design choice underlying CarotKcat. We stress that this is a reimplementation, not a run of the authors' released model; the original DLKcat reported r = 0.71 on its own split, and we therefore report that number as a reference (Table 1, †) instead of treating the two as directly comparable.

**Split stability.** Every value in Table 1 comes from a single train/test split, so we repeated the complete leak-free pipeline on five independent splits per protocol (five seeds; Supplementary Table S4). Under random splitting the ranking was fully reproducible: XGBoost was the best model in 5/5 folds (r = 0.744 ± 0.019), ahead of ExtraTrees (0.735 ± 0.021) and the ensemble (0.726 ± 0.016), with the dual-tower network consistently last (0.634 ± 0.014). That gap is small (ensemble − XGBoost = −0.018) but systematic, and we therefore claim no advantage under this protocol. Under the enzyme-group split the ordering reversed: the ensemble was the best model in 4/5 folds (r = 0.463 ± 0.048), ahead of ExtraTrees (0.434 ± 0.035) and XGBoost (0.421 ± 0.036), beating XGBoost in 5/5 folds (Δr = +0.042, paired t-test p = 0.024; Wilcoxon p = 0.062, limited by n = 5). The ensemble's mean advantage over ExtraTrees (+0.029 ± 0.056) was not resolvable at five folds (p = 0.31), matching the single-split paired bootstrap (p = 0.15); we therefore regard the two as tied on unseen enzymes. Two further observations support the absence of selection bias: the seed-42 split used in Table 1 gave an ensemble group r of 0.420, i.e. *below* the five-fold mean of 0.463, so the headline value is slightly pessimistic rather than cherry-picked; and fold-to-fold SD was 0.014–0.021 (random) and 0.035–0.061 (group), so group-split differences below ~0.03 should not be interpreted.

![Figure 3](../results/fig_benchmark.png)

### 3.3 Ablation

Removing the substrate input reduced group-split performance to r = 0.370 (sequence-only), while removing the sequence reduced it to r = 0.330 (substrate-only); the full dual-tower model reached 0.404, so both branches contribute. The 10× oversampling of carotenoid-related records was retained for all reported carotenoid analyses.

### 3.4 Held-out case study of carotenoid enzymes

We curated 13 literature-reported kinetic measurements covering 11 carotenoid-pathway enzymes, none present in the training set (Table 2; Fig. 4).

**Table 2.** Held-out literature case study. Literature kcat values versus CarotKcat-ensemble predictions for carotenoid-pathway enzymes absent from the training set.

| Enzyme (organism) | UniProt | Substrate | Literature kcat (s⁻¹) | Predicted kcat (s⁻¹) | Assay |
|---|---|---|---|---|---|
| CrtB (P. agglomerans) | D5KXJ0 | GGPP | 14.0 | 2.51 | in vitro |
| CrtB (P. ananatis) | P21683 | GGPP | 6.0 | 1.61 | in vivo |
| CrtE (P. ananatis) | P21684 | FPP | 0.80 | 0.15 | in vitro |
| CrtI (P. ananatis) | P21685 | phytoene | 0.10 (lower bound) | 0.75 | in vitro, fitted |
| CrtI (X. dendrorhous) | A0A0P0KMF9 | phytoene | 0.03 | 1.31 | in vivo |
| CrtI (R. capsulatus) | P17054 | phytoene | 0.003 | 1.91 | in vitro, fitted |
| CrtI (M. aurum) | Q9K566 | phytoene | 0.10 (lower bound) | 0.65 | in vitro, fitted |
| CrtI (B. trispora) | Q67GI0 | phytoene | 0.10 (lower bound) | 0.84 | in vitro, fitted |
| CrtI (E. agglomerans) | E9LFG2 | phytoene | 0.10 (lower bound) | 0.85 | in vitro, fitted |
| GGPS1 (human) | O95749 | FPP | 0.204 | 0.37 | in vitro |
| BTS1 (S. cerevisiae) | Q12051 | FPP | 0.025 | 0.036 | in vitro |
| CrtZ (P. ananatis) | P21688 | β-carotene | 37.0 | 0.14 | in vitro |
| CrtZ (P. ananatis) | P21688 | canthaxanthin | 6.0 | 0.052 | in vitro |

- **Pathway-level hierarchy (ρ = 0.60).** CarotKcat correctly reproduced the experimentally established rate-limiting order of the lycopene pathway: CrtB (14 s⁻¹ in vitro; 6 s⁻¹ in vivo) > CrtE (0.8 s⁻¹) > CrtI (0.03–0.1 s⁻¹). This ordering is the key driver of bottleneck analyses in metabolic engineering.
- **GGPP synthase family (ρ = 0.50).** The kcat of yeast BTS1 was predicted quantitatively (0.036 predicted vs 0.025 s⁻¹ measured); the human (0.204 s⁻¹) and Pantoea CrtE (0.8 s⁻¹) values were ranked correctly.
- **Substrate selectivity.** For P. ananatis CrtZ, the model predicted higher activity on β-carotene than on canthaxanthin (2.8-fold), matching the direction of the measured 6.2-fold preference.
- **Quantitative error.** Across all 13 case points, 85% of predictions fell within ±10-fold of the literature value (median fold-error 6.5×). For reference, reported kcat values for the *same* enzyme (P. ananatis CrtI) span >1,000-fold across laboratories (8×10⁻⁵ to 0.25 s⁻¹), indicating that literature dispersion is of the same order as model error for this pathway.

Two honest limitations emerged. First, four of the five CrtI literature values sit at the upper constraint boundary (0.1 s⁻¹) of the original COPASI fit and therefore represent only lower bounds; the model's predictions for these enzymes (0.65–0.85 s⁻¹) are consistent with those bounds, but the slowest desaturase (R. capsulatus CrtI, 0.003 s⁻¹) was not ranked lowest. Second, the phytoene desaturation intermediates phytofluene, ζ-carotene and neurosporene yield *bit-identical* Morgan fingerprints (verified), so the fingerprint-based model returns only three distinct values for the five polyenes tested. Replacing the fingerprint tower with a molecular-graph GNN (CarotKcat-GNN) resolves this structural blind spot: it returns five distinct values that increase monotonically with desaturation (phytoene 0.24 < phytofluene 0.26 < ζ-carotene 0.29 < neurosporene 0.33 < lycopene 0.38 s⁻¹ for a reference enzyme), consistent with the expected trend. Absolute values remain uncertain, but the ordering is no longer degenerate.

![Figure 4](../results/fig_case_study.png)

### 3.5 External validation on independent benchmarks

Two datasets that were used neither for training nor for model selection were evaluated (Fig. 5). First, **KcatTIE** comprises 152 kcat values curated from 2023–2025 literature and therefore postdates the training corpus, making it a temporal holdout. After removing seven entries whose sequences exactly matched a training sequence, 145 remained. CarotKcat-ensemble achieved r = 0.56, Spearman ρ = 0.54, R² = 0.30, RMSE = 1.20 log₁₀ units and a median fold-error of 6.0×, with 58% of predictions within ±10-fold (Fig. 5A). This is the expected degradation relative to the random split and indicates that the model transfers, at least partially, to newer literature.

Second, the **ADK** set (177 adenylate-kinase orthologs from multiple organisms, each with a measured kcat) tests ranking of enzyme variants within a single family. Here the model largely failed: r = 0.05, ρ = 0.04, R² < 0, a median fold-error of 18.6×, and a systematic ~16-fold under-prediction (mean predicted log₁₀ = 0.61 versus measured 1.81; Fig. 5B). The XGBoost component alone was somewhat better (r = 0.24, median fold-error 9.9×) but still poor. The ADK kcat values (up to ~800 s⁻¹) lie far above the training distribution, and the model regresses toward the training mean. This is a property of kcat regressors rather than a defect specific to CarotKcat, but it defines an applicability domain: predictions are informative for enzymes whose kcat is of the same order as the training corpus, and unreliable for exceptionally fast families or for fine-grained ranking within a family.

![Figure 5](../results/fig_external_validation.png)

### 3.6 Sensitivity to single point mutations

Because mutant prioritisation is a common motivation for kcat prediction, we asked whether the model responds to single amino-acid substitutions. We predicted kcat for a 562-residue carotenoid enzyme and 17 of its single point mutants under three substrates (phytoene, GGPP and β-carotene). All 17 mutations changed the prediction by less than 2-fold (0.89–1.07× relative to wild type), well below both the 3× noise floor and, for β-carotene, the disagreement between the ensemble's two components (0.02–0.6 log₁₀ units; Fig. 5C). The reason is structural: ESM-2 representations are mean-pooled over ~560 residues, so a single substitution changes roughly 1/560 of the input signal. CarotKcat therefore cannot rank single mutants, and we do not recommend it for that purpose without fine-tuning on mutant data or a per-residue (attention or ΔΔG) encoder. We state this explicitly because the failure is silent — the model returns plausible-looking, nearly identical numbers rather than an error.

### 3.7 In silico validation in an enzyme-constrained genome-scale model

No wet-lab experiments were performed, so we tested whether the predicted kcat values are quantitatively consistent with *published* carotenoid-production physiology. We embedded the lycopene module in the genome-scale model iML1515 (Monk et al., 2017) under aerobic conditions with 10 mmol gDCW⁻¹ h⁻¹ glucose uptake. All three published lycopene strains used for comparison carried only crtE, crtB and crtI, which fixes the module under test and its product (lycopene rather than β-carotene). The three added reactions (CRTE, CRTB, CRTI) were verified to be elementally balanced. CarotKcat predicted kcat = 0.15 s⁻¹ (CrtE, FPP), 1.61 s⁻¹ (CrtB, GGPP) and 0.75 s⁻¹ (CrtI, phytoene); against the literature values for the same enzyme–substrate pairs these are within 3.7–7.5-fold (CrtE 5.5-fold low, CrtB 3.7-fold low, CrtI 7.5-fold high; Fig. 6A), i.e. the same error band as the external benchmarks of section 3.5. Three independent checks then followed.

*Enzyme capacity is not limiting.* Converting a published productivity to specific flux (67 mg gDCW⁻¹ accumulated over 40 h; Xu et al., 2018) gives 3.12 µmol gDCW⁻¹ h⁻¹. The expression level required to sustain that flux is only 0.039% (CrtE), 0.004% (CrtB) and 0.012% (CrtI) of total protein, assuming 0.50 g protein per gDCW and 110 Da per residue. Even at 0.1% of the proteome — weak expression for a plasmid system — the three enzymes retain 3-, 27- and 8-fold headroom, rising to 25-, 274- and 80-fold at 1% (Fig. 6B). The heterologous enzymes therefore cannot be the bottleneck at the titres actually achieved.

*The stoichiometric ceiling is far away.* Maximising lycopene flux in iML1515 without enzyme constraints gives 929 µmol gDCW⁻¹ h⁻¹, ~300-fold above the measured value. The gap between measured and stoichiometric capacity is thus enzymatic and regulatory, not stoichiometric.

*The model points to precursor supply — which is what experimenters found.* At µ_max the native MEP pathway carries only 2.3 µmol gDCW⁻¹ h⁻¹ through DXS. The measured lycopene flux of 3.12 µmol gDCW⁻¹ h⁻¹ is already **1.4-fold larger than the entire native isoprenoid flux of the cell** (Fig. 6C). Reaching it therefore requires expanding or diverting the MEP pathway rather than over-expressing Crt enzymes — precisely the intervention that succeeded in both dissecting studies: amplifying dxs/idi/ispA raised the titre 10-fold in the POSTECH strain series (0.38 → 4.0 mg gDCW⁻¹; Kim, 2016), and screening dxs/dxr/idi/ispA combinations reached 112 mg gDCW⁻¹ in a 2026 study. The growth–production envelope shows that a flux of 3.12 µmol gDCW⁻¹ h⁻¹ costs only 1% of µ_max, so the limitation is precursor availability rather than metabolic burden.

We are explicit about what this does and does not establish. It is a consistency check, not a discriminating test: because the Crt enzymes operate 1–2 orders of magnitude below saturation even at low expression, the conclusion "precursor supply limits" would survive a several-fold error in any of the three kcat values. What the analysis does show is that CarotKcat's predictions are quantitatively compatible with measured carotenoid physiology, and that they reproduce without being told the bottleneck that two independent engineering campaigns identified. A test that could *falsify* the kcat values would require measured in vitro pathway flux or a strain in which MEP-pathway supply is decoupled from growth.

![Figure 6](../results/fig_ecgem_validation.png)

### 3.8 Zero-shot accuracy on terpenoid and carotenoid enzymes

The preceding benchmarks ask whether CarotKcat ranks enzymes drawn from the same families it was trained on. The carotenoid use case asks something harder: whether it can say anything useful about families it has never seen. We therefore assembled a curated benchmark of terpenoid-pathway turnover numbers and evaluated every prediction without retraining (`scripts/22`, `scripts/23`).

Two facts about the training corpus frame the result. First, BRENDA — the source of nearly all training records — contains **no turnover numbers at all** for phytoene synthase (EC 2.5.1.32), phytoene desaturase (EC 1.3.99.31 and 1.3.5.5) or lycopene cyclase (EC 5.5.1.19); it lists Michaelis constants only. Second, the 17,051 training records contain not a single entry from the terpene-synthase classes used here (EC 4.2.3.15–4.2.3.97). Every carotenoid-module prediction in this paper is therefore an extrapolation in the strict sense, and the case study of section 3.4 was necessarily assembled from primary literature rather than from a database.

The benchmark comprises 43 measurements on 35 enzymes (Table S7). Each record carries a UniProt accession, the substrate, the reported kcat, and either the BRENDA literature identifier or a primary citation; no (accession, substrate) pair overlaps the training table. Repeated measurements of one pair were collapsed to their median, because the predictor is deterministic given sequence and substrate and replicate values are therefore noise from the model's point of view.

Performance splits sharply by whether the EC family was represented during training. On families the model has seen (DXP synthase, HMG-CoA synthase, mevalonate kinase, isoprenyl diphosphate synthases; n = 20), accuracy is close to the in-distribution benchmarks: r = 0.658, ρ = 0.714, R² = 0.410, median error 2.1-fold, 60% within 3-fold. On the carotenoid and terpene-synthase families, which are absent from training (n = 23), it is not: r = 0.135, ρ = 0.068, R² = −0.453, median error 8.6-fold, and only 13% of predictions within 3-fold (Fig. 7A). Across all 43 records the correlation is r = 0.494 with a median error of 6.7-fold, but that figure is carried almost entirely by the between-family contrast rather than by within-family resolution.

The carotenoid module itself is the weakest subset: r = −0.543 over 12 measurements, a median error of 26-fold, and not one prediction within 3-fold. The two extremes are instructive. CrtZ acting on β-carotene (37 s⁻¹ measured) is predicted at 0.14 s⁻¹, a 260-fold underestimate; CrtI from *R. capsulatus* acting on phytofluene (0.001 s⁻¹ measured) is predicted at 2.9 s⁻¹, a 2,900-fold overestimate. The model does not so much rank these enzymes as assign them all a value in the middle of the training distribution.

That diagnosis is confirmed by the response slope — the regression of predicted on measured kcat (Fig. 7B, C). A perfect model has slope 1.0; a model that ignores its input has slope 0. Across the benchmark the slope is 0.37, and the interquartile range of the predictions (1.60 log units) is narrower than that of the measurements (2.42 log units). The bias reverses with magnitude: slow enzymes are over-predicted and fast enzymes under-predicted, with a slope of −0.62 in the signed error. Within the carotenoid subset the slope is −0.22, i.e. the ordering is worse than random.

This is the quantitative statement of the limitation that the mutation analysis (section 3.6) and the ADK orthologue panel (section 3.5) each probed from a different angle: away from the bulk of the training distribution, CarotKcat compresses towards the mean. The practical consequence must be stated plainly. CarotKcat should be used to obtain an order-of-magnitude prior for a carotenoid enzyme — which is what the enzyme-constrained analysis of section 3.7 requires, and where a several-fold error provably does not change the conclusion — and not to rank candidate enzymes or variants within the pathway. Ranking requires either experimental measurement or fine-tuning on a pathway-specific dataset, which the current public record cannot supply.

![Figure 7](../results/fig_terpenoid_benchmark.png)

### 3.9 Proteome-scale prediction for carotenoid enzymes

kcat was predicted for 3,725 UniProt carotenoid-related sequences (`results/predictions_carotenoid_uniprot.tsv`), yielding a median of ~0.2 s⁻¹. In the light of section 3.8 this resource should be read as an order-of-magnitude annotation set rather than a ranking: it identifies which sequences are plausibly slow (the large majority) and flags the few predicted to be fast, but the 26-fold median error measured on the carotenoid benchmark applies to it directly. It remains useful for populating enzyme-constrained models, where a within-order-of-magnitude prior is what the constraint needs.

### 3.10 Web server

The CarotKcat web interface requires only a browser: users upload a FASTA file or paste a sequence, enter a substrate SMILES (with automatic structure rendering), and obtain a downloadable kcat table. The server runs locally or on a laboratory intranet.

## 4. Discussion

CarotKcat delivers accuracy on par with DLKcat's reported performance on the same BRENDA + SABIO-RK corpus (r = 0.723 versus 0.71 under random splitting), while adding a leak-free unseen-enzyme evaluation, two external benchmarks with both positive and negative results, a carotenoid-focused case study, and a lightweight, fully reproducible pipeline. To our knowledge it is the first kcat model explicitly optimised and evaluated for the carotenoid biosynthetic pathway. Five findings deserve emphasis.

First, the evaluation protocol matters more than the model choice. Tree-based models on frozen embeddings (UniKP/TurNuP-style) excel when test enzymes resemble training enzymes but lose up to half of their correlation under an enzyme-group split, and this reversal was reproducible in five of five independent splits. Only the ensemble retained the best unseen-enzyme accuracy, and even then by a margin (+0.03) that five folds cannot resolve statistically; the dual-tower network alone was never better than ExtraTrees. Consistently, a from-scratch DLKcat-style GNN+CNN reimplementation plateaued at r = 0.47 under the same split, indicating that at this data scale (~13,000 training records) pretrained protein representations, rather than architecture family alone, are the dominant driver of accuracy — and that architectural novelty should not be claimed on the basis of a single favourable split.

Second, the case study shows that for kinetically uncharted secondary-metabolism pathways, model predictions and literature measurements carry comparable uncertainty: reported kcat values for individual carotenoid enzymes vary over three orders of magnitude between laboratories and assay formats. Computational predictions can therefore serve as useful priors for enzyme-constrained modelling even where they cannot replace measurement. The enzyme-constrained analysis in section 3.7 sharpens this: at the titres actually reported, the heterologous Crt enzymes sit far below saturation, and what limits lycopene production is the cell's isoprenoid supply — a conclusion reached from predicted kcat alone and independently confirmed by the MEP-pathway engineering that raised titres in every high-producing strain. This is the sense in which kcat prediction is useful to metabolic engineering: not as a replacement for measurement, but as a quantitative prior that redirects experimental effort to the step that actually limits flux.

Third, our negative results chart concrete improvements: (i) fingerprint substrate encodings should be replaced by graph neural networks or molecular transformers that can resolve closely related polyene substrates (phytofluene vs ζ-carotene vs neurosporene) — our GNN variant already removes this degeneracy (section 3.4); (ii) the very-low-kcat regime (<0.01 s⁻¹) suffers from training-data scarcity, causing regression to the mean; and (iii) assay conditions (temperature, pH) are rarely annotated and remain unmodelled. The most direct route to higher accuracy on a specific host/pathway is few-shot fine-tuning with even tens of laboratory-measured kcat values, which our pipeline supports natively.

Fourth, external validation draws a sharper applicability boundary than internal splits alone suggest. On a temporal holdout of 2023–2025 literature the model retained r = 0.56, but on a fast enzyme family (ADK orthologs) it collapsed to r = 0.05 with a systematic 16-fold under-prediction, and single point mutations moved predictions by less than 2-fold. Two lessons follow. First, kcat models regress toward the turnover range of their training corpus, so prospective use should be restricted to enzymes of comparable kcat; reporting a prediction without this caveat is misleading. Second, single-mutation ranking requires representations that are sensitive to local sequence change (per-residue attention, structural ΔΔG features, or targeted fine-tuning), not a mean-pooled protein embedding. We report these failures because they are rarely quantified in kcat-model papers and because they directly determine whether the tool should be used for enzyme engineering or only for pathway-level parameterisation.

Fifth, the curated benchmark makes the generalisation boundary quantitative rather than anecdotal. Splitting the 43 terpenoid measurements by whether the EC family appeared during training separates two qualitatively different behaviours: families that were present give r = 0.66 and a median 2.1-fold error, while the carotenoid and terpene-synthase families, which are absent from training, give r = 0.14 and 8.6-fold. The response slope of 0.38 exposes the mechanism — predictions are compressed towards the training mean, which is why the bias is positive for slow enzymes and negative for fast ones. This has a methodological consequence beyond this paper: reporting Pearson r alone on a benchmark whose measured range spans four orders of magnitude can make a model that emits a near-constant prediction look respectable, whereas the response slope does not. We suggest that benchmarks for kcat prediction report it alongside correlation.

The practical reading is narrow but useful. An order-of-magnitude prior for a carotenoid enzyme is enough to parameterise an enzyme-constrained model — section 3.7 shows that the resulting pathway conclusion is robust to a several-fold error, because the heterologous enzymes operate far below saturation. It is not enough to choose between two candidate enzymes, and no amount of reanalysis of the current public record changes that: BRENDA holds no turnover numbers for phytoene synthase, phytoene desaturase or lycopene cyclase, so the data needed to learn these families does not yet exist in curated form. Generating it is a modest experimental task — a few dozen in vitro kcat measurements across the carotenoid module would allow the fine-tuning our pipeline already supports — and it is the single intervention most likely to make sequence-based kcat prediction useful for carotenoid engineering.

## 5. Conclusion

CarotKcat predicts enzyme turnover numbers from sequence and substrate with accuracy matching strong general-purpose baselines, a leak-free evaluation of unseen enzymes, external validation that delimits its applicability domain, validated pathway-level behaviour on carotenoid enzymes, a dry enzyme-constrained validation consistent with published production physiology, and a zero-barrier web interface. A curated zero-shot benchmark shows where that accuracy comes from and where it stops: families represented in training are predicted to within ~2-fold, whereas the carotenoid and terpene-synthase families, absent from every public curated database, are compressed towards the training mean and cannot be ranked. CarotKcat is therefore an order-of-magnitude prior for pathway parameterisation and explicitly not a ranking or mutant-design tool. All code, trained models, curated datasets and predictions are openly available to support carotenoid metabolic engineering and enzyme-constrained modelling.

## Data and code availability

GitHub: https://github.com/ZihengHQU/CarotKcat — includes training/evaluation scripts (`scripts/01`–`23`, including the leak-free re-evaluation, bootstrap statistics, GNN variant, external benchmarks, DLKcat head-to-head, five-fold split-stability analysis, the enzyme-constrained dry validation and the curated terpenoid benchmark), trained model weights, the curated 17,051-record dataset, the 3,725-sequence carotenoid prediction set, the curated 43-measurement terpenoid benchmark with per-record citations, all benchmark/case-study/external-validation results, and the web server.

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
16. Monk, J. M. *et al.* iML1515, a knowledgebase that computes *Escherichia coli* traits. *Nat. Biotechnol.* **35**, 904–908 (2017).
17. Xu, J. *et al.* Efficient production of lycopene by engineered *E. coli* strains harboring different types of plasmids. *Bioprocess Biosyst. Eng.* **41**, 489–499 (2018).
18. Kim, S. Construction of a platform *Escherichia coli* strain for the production of lycopene. MSc thesis, Pohang University of Science and Technology (2016).

## Figure legends

**Figure 1.** CarotKcat dual-tower architecture. Enzyme sequences are encoded by the frozen ESM-2 (650M) protein language model with mean pooling (1,280 dimensions); substrates are encoded as Morgan fingerprints (radius 2, 2,048 bits). Separate MLP towers project each modality to 256 dimensions; the concatenated representation feeds a fusion MLP predicting log10(kcat). The deployed ensemble averages the network prediction with an XGBoost regressor in log space.

**Figure 2.** Predicted versus measured log10(kcat) for the CarotKcat-ensemble on the held-out random-split test set (n = 2,558; Pearson r = 0.723, R² = 0.522).

**Figure 3.** Benchmark of CarotKcat variants (fingerprint and GNN substrate encoders) against UniKP- and TurNuP-style baselines, a from-scratch reimplementation of the DLKcat GNN+CNN architecture, and single-modality ablations under random and enzyme-group splits. Bars show test-set Pearson r with 95% bootstrap confidence intervals. Tree-based baselines degrade sharply on enzymes without training-set homologues; the ensemble, the GNN variant and ExtraTrees are statistically tied under both protocols.

**Figure 4.** Held-out literature case study on carotenoid-pathway enzymes (all absent from training). (A) CrtI phytoene desaturases; (B) pathway-level enzymes; (C) GGPP synthases; (D) CrtZ substrate selectivity. Dashed line: identity.

**Figure 5.** External validation and mutation sensitivity. (A) KcatTIE temporal holdout (2023–2025 literature; n = 145 after removing exact training-set matches); (B) ADK orthologs (n = 176), showing systematic under-prediction of this high-kcat family; (C) predicted fold-change for 17 single point mutants of a carotenoid enzyme (substrate: phytoene). Dashed lines in C mark the 3× noise floor, 10× candidate and 30× strong-signal thresholds; all mutants fall below 2×.

**Figure 6.** In silico (dry) validation in an enzyme-constrained genome-scale model. (A) CarotKcat-predicted kcat for the three heterologous lycopene-pathway enzymes of *P. ananatis* versus literature values for the same enzyme–substrate pairs; the vertical bar spans the reported CrtB range (6–14 s⁻¹). (B) Fraction of total protein that each enzyme would need to sustain the highest published lycopene productivity (3.12 µmol gDCW⁻¹ h⁻¹), compared with the expression range achievable from a plasmid system (shaded). All three requirements lie 1–3 orders of magnitude below that range. (C) Growth–production envelope of iML1515 with the CrtEBI module: the measured flux (dashed) is only 1.4-fold above the native MEP-pathway flux at µ_max (dotted) and costs 1% of the growth rate, whereas the unconstrained stoichiometric ceiling is ~300-fold higher.

**Figure 7.** Zero-shot accuracy on the curated terpenoid benchmark (43 measurements, 35 enzymes; none of the (accession, substrate) pairs occurs in training). (A) Predicted versus measured log10(kcat), coloured by family; dashed line is the identity and the shaded band spans ±1 log unit (10-fold). (B) Signed error against measured kcat; the dashed line is the least-squares fit (slope −0.62), showing over-prediction of slow enzymes and under-prediction of fast ones. (C) Response slope — the regression of predicted on measured kcat — for each family; 1.0 is a perfect model and 0.0 a model that ignores its input. Families present in training (precursor, isoprenoid supply) retain a positive slope, whereas the carotenoid and terpene-synthase families, which are absent from training, have slopes at or below zero.

## Supplementary material

- **Supplementary Table S1** — Full case-study data including per-step CrtI desaturation values, dual-tower and XGBoost component predictions (`results/case_study.tsv`).
- **Supplementary Table S2** — Leak-free benchmark metrics for all models and splits with 95% bootstrap CIs (`results/benchmark_table.csv`), paired bootstrap tests (`results/paired_tests.csv`), and the earlier test-set-early-stopping table for provenance (`results/benchmark_table_test_earlystop_legacy.csv`).
- **Supplementary Table S3** — External validation metrics (`results/external_summary.csv`) and anonymised mutation sensitivity predictions (`results/mutant_sensitivity_anonymized.tsv`).
- **Supplementary Table S4** — Split-stability analysis: per-fold and mean ± SD Pearson r, R² and RMSE for four models over five independent random and five independent enzyme-group splits (`results/repeated_splits.csv`, `results/repeated_splits_summary.csv`).
- **Supplementary Table S5** — Enzyme-constrained dry validation inputs and results: predicted module kcat, published productivities converted to specific flux, required enzyme fraction and capacity headroom (`results/ecgem_predictions.tsv`, `results/ecgem_targets.tsv`, `results/ecgem_capacity.tsv`).
- **Supplementary Table S6** — Growth–production envelope for iML1515 carrying the CrtEBI module (`results/ecgem_envelope.csv`).
- **Supplementary Table S7** — Curated terpenoid benchmark: every measurement with accession, organism, substrate, reported kcat, source citation and prediction, with per-subset metrics (`results/terpenoid_bench.tsv`, `results/terpenoid_bench_metrics.csv`).
- **Supplementary Figure S1** — Polyene discrimination by fingerprint versus graph substrate encoders (`results/polyene_discrimination.csv`).
- **Supplementary Dataset S1** — Curated 17,051-record training set (`data/kcat_clean.tsv` with sequences).
- **Supplementary Dataset S2** — Proteome-scale predictions for 3,725 carotenoid-related enzymes (`results/predictions_carotenoid_uniprot.tsv`).
