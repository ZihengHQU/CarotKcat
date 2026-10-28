# kcat_benchmark_2025
Bechmarking the deep learning kcat prediction models. This repository also hosts curated datasets for benchmarking computational models predicting enzyme catalytic turnover numbers ($k_{\text{cat}}$). It supports the research presented in the paper:  
**"Deep learning for enzyme $k_{\text{cat}}$ prediction: what works, what doesn't, and why?"**  
*Zhang, Z., Wang, Z., Li, D., et al. (2026). [Journal Name], DOI: [To be updated]*


![Overview of Benchmark framework design](https://github.com/zelixirSH/kcat_benchmark_2025/blob/main/images/fig1.png)

---

## 📦 Datasets Overview
All datasets were processed using task-specific curation protocols (detailed in the [Supplementary Information](supplementary/Supplementary_Information.pdf)).  
Data files are stored in `/data`.

| Dataset             | Source & Description                                                                 | Key Metrics                                                                 | File Path                     |
|---------------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|-------------------------------|
| **KcatTIE**         | Literature-derived (2023–2025) via zERExtractor                                      | 152 $k_{\text{cat}}$ values for 55 enzymes                               | `/data/KcatTIE/KcatTIE_experiment_data.csv`      |
| **ADK**             | High-throughput assays (Muir et al., 2025)                                           | $k_{\text{cat}}$ for 178 adenylate kinase variants                       | `/data/ADK/ADK_experiment_data.csv`          |
| **PETase**          | Engineered PET hydrolases (Norton-Baker et al., 2025)                                | Relative activity scores for 214 enzymes                                   | `/data/PET/PETase_experiment_data.csv`   |
| **DMS**             | ProteinGym database                                                                  | Fitness scores for single-point mutants (8 enzyme systems)                 | `/data/DMS/`|
| **Enzyme-Inhibitor**| BindingDB & ChEMBL (2026)                                                            | 2899 entries for 28 enzymes                                      | `/data/inhibitor/Inhibitor_data.csv`|
| **P450**            | P450 kinetics data from Brenda and SABIO-RK and via zERExtractor                     | 215 enzyme-catalyzed entries with kinetics data                            | `/data/P450/P450_experiment_data.csv`|
---

## ⚙️ Installation & Usage
### Data Access
Clone the repository:

```
# bash
git clone https://github.com/zelixirSH/kcat_benchmark_2025.git
```

Data files are in `/data`. Column descriptions (e.g., units for `kcat_value` in s⁻¹) are in file headers.

---
## 🔗 Model Implementations
Benchmarked models with GitHub repositories:
| Model              | Model Description & Key Features                                                                | Repository Link                                      |
|--------------------|-----------------------------------------------------------------------------|------------------------------------------------------|
| **DLKcat**         | As the first deep learning model for $k_{\text{cat}}$ prediction, DLKcat was developed to parameterize enzyme-constrained genome-scale metabolic models. It employs a convolutional neural network (CNN) to process protein sequences, a graph neural network (GNN) to encode substrate structures, and an attention mechanism for the final prediction.      | [SysBioChalmers/DLKcat](https://github.com/SysBioChalmers/DLKcat) |
| **UniKP**          | UniKP is a multi-task model capable of predicting $k_{\text{cat}}$, $K_{\text{m}}$, and $k_{\text{cat}}$/ $K_{\text{m}}$. It utilizes ProtT5 for protein embeddings, a Transformer for substrate SMILES strings, and an Extra Trees regressor for output. Its extensible design allows integration of environmental factors (e.g., pH, temperature) in the EF-UniKP variant, supporting enzyme discovery and directed evolution. | [Luo-SynBioLab/UniKP](https://github.com/Luo-SynBioLab/UniKP) |
| **CataPro**        | CataPro aims to enhance $k_{\text{cat}}$/ $K_{\text{m}}$ (catalytic efficiency) prediction. It integrates multi-source features (ProtT5, MoT5, RDKit fingerprints), uses a feature attention layer, and incorporates a correction term mechanism. A strict sequence-similarity-based data split ensures generalization. | [zchwang/CataPro](https://github.com/zchwang/CataPro) |
| **EITLEM-Kinetics**| EK is designed for mutant enzyme prediction via iterative transfer learning. It uses ESM-1b for protein embeddings, MACCS Keys for substrate fingerprints, and employs ProMolAtt and AttentionAgg modules for feature enhancement before final prediction with a fully connected network. | [XvesS/EITLEM-Kinetics](https://github.com/XvesS/EITLEM-Kinetics) |
| **TurNuP**         | TurNuP predicts in vitro $k_{\text{cat}}$ for wild-type enzymes in natural reactions. It combines a Transformer for protein sequence embeddings with Differential Reaction Fingerprints (DRFP) to encode full reaction context (substrates \& products), followed by an XGBoost regressor.  | [AlexanderKroll/kcat_prediction](https://github.com/AlexanderKroll/kcat_prediction) |

---


## 📄 Supplementary Information
Full curation protocols (filtering criteria, sequence similarity analysis, etc.) are in:  
`/supplementary/Supplementary_Information.pdf`  
Key sections:
- Sequence deduplication (CD-HIT, sequence identity <0.4)
- Substrate modification protocol (random functional group replacement)
- Benchmark framework design (Fig. 1)

---

## 📊 Experimental Validation
Results for 98 *PjxA* xylanase mutants (Paper Fig. 6):
- **Top-20 mutant screening**: DLKcat/UniKP/CataPro show <10% positive rate (activity increase >20%)
- **Global correlation**: Pearson/Spearman coefficients <0.3 (UniKP highest: ~0.2)

---

## 📜 License & Citation
### License
The datasets and supplementary materials in this repository are licensed under the [MIT License](LICENSE).  
Key permissions under this license include:
- ✅ Free use for academic and commercial purposes
- ✅ Freedom to modify and distribute
- ✅ Permission to sublicense
- ✅ Use in proprietary software

**Requirements**:
- Include the original copyright notice in all copies
- Maintain the license text in all substantial portions

**Disclaimer**:  
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. See full [LICENSE](LICENSE) for details.

### How to Cite
bibtex

@article{Zhang2026kcat,

title={Deep learning for enzyme k
cat
	​

prediction: what works, what doesn't, and why?},

author={Zhang, Zhongyang and Wang, Zhihao and Li, D. and et al.},

journal={[Journal Name]},

volume={},

pages={},

year={2026},

doi={[To be updated]}

}

**Always cite original data sources** (e.g., BRENDA, ProteinGym).

---

## ❓ Support
- **Issues**: [GitHub Issues](https://github.com/zelixirSH/kcat_benchmark_2025/issues)
- **Contact**:  
  Zhongyang Zhang: `zhangzhongyang@ynnu.edu.cn`  

---

## 📂 Repository Structure

kcat_benchmark_2025/

├── data/

├── images/

├── LICENSE

└── README.md

