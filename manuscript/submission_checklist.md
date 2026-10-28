# Submission checklist

Status legend: [x] done, [ ] to do by the author, [!] cannot be completed by the AI / requires external input.

## Manuscript
- [x] Abstract, introduction, methods, results, discussion, conclusion
- [x] 5 main figures embedded (architecture, scatter, benchmark, case study, external validation + mutation sensitivity)
- [x] 2 main tables (benchmark, case study) + supplementary tables
- [x] Figure legends and supplementary material list
- [x] Leak-free re-evaluation; bootstrap CIs; paired bootstrap tests
- [x] External validation (KcatTIE, ADK) with both positive and negative results
- [x] Explicit limitation statement: not a mutant-ranking tool
- [ ] Author names, affiliations, emails, ORCID (placeholders in `manuscript/*.md`)
- [ ] Funding statement
- [ ] Acknowledgements
- [ ] Author contributions (CRediT)
- [ ] Competing-interests statement
- [ ] Reference style converted to the target journal's format
- [ ] Word count / figure count limits checked against the target journal

## Data and code
- [x] Public GitHub repository with scripts, weights, curated data and results
- [x] Reproducible pipeline (`scripts/01`–`17`)
- [x] Raw-data archive (`CarotKcat_raw_data.zip`)
- [x] External benchmark data documented (`data/external/`, source: github.com/zelixirSH/kcat_benchmark_2025)
- [ ] Archive deposited to Zenodo and DOI obtained (recommended for the Data Availability statement)
- [!] DLKcat official trained weights: not distributed in a runnable form; comparison uses a same-split reimplementation plus published numbers as a footnoted reference

## Analyses
- [x] Random split and enzyme-group split
- [x] 95% bootstrap CIs and paired significance tests
- [x] DLKcat-style GNN+CNN head-to-head on identical splits
- [x] GNN substrate variant; polyene discrimination resolved (3/5 → 5/5 distinct values)
- [x] Temporal holdout (KcatTIE) and cross-family variant set (ADK)
- [x] Mutation sensitivity analysis
- [!] No wet-lab validation (by design; stated in the manuscript)
- [ ] Optionally: 5-fold repeated random splits to report mean ± SD (not yet run)

## Journal-specific
- [ ] Choose target journal (candidates: Briefings in Bioinformatics; Bioinformatics; Synthetic and Systems Biotechnology; BMC Bioinformatics)
- [ ] Format references and headings to the journal template
- [ ] Prepare graphical abstract (if required)
- [ ] Prepare highlights / lay summary (if required)
- [ ] Cover letter personalised to the editor (`manuscript/cover_letter.md`)
- [ ] Suggest 3 reviewers without conflicts

## Integrity
- [x] No test-set early stopping; legacy table retained for provenance
- [x] Exact-sequence leakage check against external sets (1 ADK and 7 KcatTIE matches removed)
- [x] Negative results reported (ADK failure, mutation insensitivity)
- [!] Confidential mutant sequences (`mutants.fasta`, `results/mutant_sensitivity_*.tsv`) are **not** in the public repository; do not upload them unless the sequences are cleared for release
