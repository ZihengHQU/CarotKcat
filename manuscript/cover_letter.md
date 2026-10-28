# Cover letter (draft)

**To the Editor,**

We submit for your consideration the manuscript **"CarotKcat: a sequence- and substrate-aware deep learning model for kcat prediction in carotenoid biosynthesis"**.

**What the paper reports.** CarotKcat is a dual-tower model that predicts enzyme turnover numbers (kcat) from an amino-acid sequence and a substrate SMILES. It is trained on 17,051 curated records derived from the BRENDA + SABIO-RK corpus and is evaluated with an explicit enzyme-group (leak-free) split rather than only the usual random split. On the same test instances it reaches r = 0.723 (95% CI 0.698–0.746) under random splitting and r = 0.420 (0.388–0.452) on enzymes without training-set homologues (five-split mean 0.463 ± 0.048). We also provide a held-out literature case study on 11 carotenoid-pathway enzymes, a molecular-graph variant that resolves polyene substrates with identical fingerprints, proteome-scale predictions for 3,725 carotenoid-related sequences, and a zero-installation web interface.

**Why we believe it is suitable for [JOURNAL].** Most kcat models are benchmarked only under random splits, where homologous enzymes leak between training and test sets. Our contribution is as much methodological as predictive: we report a leak-free estimate, 95% bootstrap confidence intervals, paired bootstrap tests against every baseline, five-fold repeated splits (so that model differences can be separated from split noise), and two external benchmarks. We are deliberately conservative about the headline claim — across five splits CarotKcat ties the strongest tree ensemble on unseen enzymes and is reproducibly ~0.02 below gradient-boosted trees under random splitting, which we state explicitly rather than frame as an improvement. We also report *negative* results that are usually omitted — the model fails on an exceptionally fast enzyme family (ADK orthologs, r = 0.05) and is insensitive to single point mutations (<2-fold predicted change).

**A benchmark for the pathway the model was built for.** Because BRENDA contains no turnover numbers at all for phytoene synthase, phytoene desaturase or lycopene cyclase, we assembled a curated benchmark of 43 terpenoid-pathway measurements with per-record provenance and used it to measure extrapolation directly. Accuracy depends entirely on whether the enzyme's family was represented in training: r = 0.66 with a median 2.1-fold error for families that were, r = 0.14 with 8.6-fold for the carotenoid and terpene-synthase families that were not. We report the response slope (predicted regressed on measured, 0.38 overall and negative within the carotenoid subset) because it exposes the failure mode — compression towards the training mean — which a correlation coefficient alone conceals. The conclusion we draw is deliberately narrow: CarotKcat is an order-of-magnitude prior for pathway parameterisation, not a ranking tool, and closing that gap requires new measurements rather than further modelling of the existing record.

**No wet-lab validation.** We are aware that this is the main limitation of any purely computational kcat paper, and we have addressed it as directly as the data allow. In addition to a held-out literature case study and two external benchmarks, we embedded the predicted kcat values in the enzyme-constrained genome-scale model iML1515 and compared the resulting fluxes with published lycopene productivities. The predictions place the three heterologous Crt enzymes 25–274-fold below saturation at 1% of total protein, show that the measured lycopene flux is only 1.4-fold above the entire native MEP-pathway flux, and therefore identify precursor supply — rather than the Crt enzymes — as the limiting step. This is exactly the bottleneck that the two engineering studies we compare against had to relieve experimentally. We state plainly in the manuscript that this is a consistency check rather than a falsifying test, and specify what such a test would require.

**Novelty relative to DLKcat.** We reimplemented the DLKcat GNN+CNN architecture and trained it on exactly our data and splits; it reaches r = 0.47, below CarotKcat (0.72). The comparison isolates the contribution of pretrained protein representations from architecture, and is reported with the caveat that the original DLKcat weights are not distributed in a directly runnable form, so the published DLKcat number is cited as a footnoted reference rather than a like-for-like head-to-head.

**Availability.** All code, trained weights, curated data, predictions and the web server are openly available at https://github.com/ZihengHQU/CarotKcat.

**Declarations.** The manuscript is original, has not been published elsewhere and is not under consideration by another journal. All authors have approved the submission and declare no competing interests. No wet-lab experiments were performed; all analyses are computational, and the manuscript says so explicitly rather than implying experimental support.

We hope the manuscript is suitable for review and look forward to your response.

Sincerely,

**[CORRESPONDING AUTHOR NAME]**
[Affiliation]
[Email]
[ORCID]
on behalf of all authors

---

*Suggested reviewers (optional — fill in names and affiliations; do not list anyone with a conflict of interest):*

1. [Name, Affiliation, Email]
2. [Name, Affiliation, Email]
3. [Name, Affiliation, Email]
