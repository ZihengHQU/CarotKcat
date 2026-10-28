# CarotKcat：酶 kcat / 酶活预测（序列 + 底物双输入，类胡萝卜素酶重点优化）

> 输入**酶氨基酸序列 + 底物 SMILES**，预测催化常数 kcat（s⁻¹）。
> 以**类胡萝卜素（Carotenoid）合成通路酶**为应用重点（crtE/crtB/crtI/crtY/crtZ/crtW 等），底座为覆盖全部 EC 类的通用酶活大数据训练。

## 🖥️ 零门槛网页界面（推荐非计算机用户使用）

```bash
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

页面支持：上传 FASTA 文件 / 直接粘贴序列 + 底物 SMILES → 一键预测 → 表格结果 + CSV 下载 + 底物结构可视化。

## 模型架构（DLKcat 式双塔）

```
酶序列 (≤1022 aa) ──► ESM-2 650M 第33层均值池化 (1280d) ──► 序列塔 MLP (256d) ──┐
                                                                              ├──► 融合回归头 ──► log10(kcat)
底物 SMILES ──► Morgan 指纹 (r=2, 2048bit) ──► 底物塔 MLP (256d) ────────────┘
```

- 序列编码器：ESM-2 t33_650M_UR50D（冻结权重）
- 底物编码：RDKit Morgan 指纹（DLKcat 用 GNN，指纹版本更轻量、无额外依赖）
- 融合头：512 → 256 → 64 → 1，SmoothL1 损失，AdamW + 余弦退火
- **类胡萝卜素相关酶训练时 10× 过采样**，强化目标通路的表现

## 数据

| 数据集 | 规模 | 来源 |
|---|---|---|
| 通用酶 kcat 原始记录 | 41,558 条 | DLKcat 项目（BRENDA + SABIO-RK 整合，覆盖全部 EC 类） |
| 记录级训练集（序列+底物齐全） | **17,051 条** / 2,164 种底物 / ~2,700 种酶 | 清洗 + UniProt 序列匹配 |
| 类胡萝卜素酶预测集 | 3,725 条序列 | UniProt（crt 家族 + phytoene/lycopene/carotene 相关酶） |

## 性能

### 与基线/同类方法的系统对比（同一测试集，n≈2,558/2,813，**无测试集泄漏**）

所有神经模型均在训练集内部的验证集上早停，测试集只评估一次；括号内为 95% bootstrap 置信区间。旧版（用测试集早停）结果保留在 `results/benchmark_table_test_earlystop_legacy.csv` 供溯源，差异约 0.01–0.02。

| 模型 | 随机划分 Pearson（95% CI）/ R² | 按酶分组划分 Pearson（95% CI）/ R² |
|---|---|---|
| CarotKcat-ensemble（双塔+XGBoost 平均） | **0.723 (0.698–0.746) / 0.522** | **0.420 (0.388–0.452) / 0.165** |
| CarotKcat 双塔神经网络（指纹） | 0.636 (0.602–0.665) / 0.386 | 0.404 (0.371–0.434) / 0.098 |
| CarotKcat-GNN（ESM + 分子图 GNN 底物） | 0.642 (0.613–0.668) / 0.407 | 0.417 (0.390–0.445) / 0.156 |
| UniKP 式（ESM+指纹, ExtraTrees） | 0.725 (0.698–0.750) / 0.518 | 0.406 (0.372–0.437) / 0.165 |
| TurNuP 式（ESM+指纹, XGBoost） | 0.735 (0.711–0.759) / 0.538 | 0.366 (0.329–0.398) / 0.121 |
| DLKcat 式（底物 GNN + 蛋白 n-gram CNN，同划分复现） | 0.466 (0.427–0.503) / 0.208 | 0.356 (0.319–0.393) / 0.108 |
| 仅序列（ESM+MLP） | 0.621 (0.589–0.650) / 0.381 | 0.370 (0.341–0.398) / 0.123 |
| 仅底物指纹（MLP） | 0.467 (0.430–0.503) / 0.206 | 0.330 (0.296–0.364) / 0.084 |

> DLKcat 原论文（Li et al., *Nat. Catal.* 2022）在其自身 10% 随机留出集上报告 Pearson r = 0.71、RMSE = 1.06（同一 BRENDA+SABIO-RK 语料）。该权重未以可直接执行的形式公开，且划分不同，故仅作参考，不作同口径头对头。上表的 "DLKcat 式" 行是我们按同一架构（GNN+CNN）在同一数据与划分上从零训练的复现结果（`scripts/11_dlkcat_headtohead.py`），用于隔离"预训练表征 vs 架构"的贡献。

关键发现（配对 bootstrap 检验，`results/paired_tests.csv`）：

- 随机划分下，集成与 ExtraTrees 统计上无差异（Δr = −0.001, p = 0.85），略低于 XGBoost（Δr = −0.012, p = 0.064）；
- 分组划分下，集成最高（0.420），显著优于 XGBoost（Δr = +0.054, p < 0.001），但与 ExtraTrees 的差异不显著（Δr = +0.014, p = 0.15）；
- 因此准确表述是**"与最强树模型持平"，而非"全面超越"**。同划分复现的 DLKcat 式 GNN+CNN 仅 0.47，说明在 ~1.3 万条训练记录下，**预训练蛋白表征（ESM-2）比架构本身更决定精度**。

![benchmark](results/fig_benchmark.png)

## 外部验证（独立数据，未参与训练/调参）

| 数据集 | 说明 | 结果 |
|---|---|---|
| **KcatTIE**（2023–2025 文献，n=145） | 时间留出集，晚于训练语料；已剔除 7 条与训练集完全相同的序列 | r = 0.56，ρ = 0.54，R² = 0.30，中位倍数误差 6.0×，58% 在 ±10× 内 ✅ |
| **ADK**（177 个腺苷酸激酶同源变体，n=176） | 单一高活性酶家族的变体排序 | r = 0.05，ρ = 0.04，R² < 0，**系统性低估约 16×**；XGB 单模型 r = 0.24 仍差 ❌ |

**结论**：模型对 kcat 量级与训练语料相近的酶可信；对 kcat 极高（如 ADK 可达 ~800 s⁻¹）的家族会向训练均值回归，**不能用于跨家族变体排序**。脚本 `scripts/15_external_benchmark.py`。

## 突变体敏感性（重要局限）

对一条 562 aa 类胡萝卜素酶及其 17 个单点突变体，在 phytoene / GGPP / β-carotene 三种底物下预测：**所有突变体相对野生型的变化均 < 2×**（0.89–1.07×），远低于 3× 噪声阈，也小于集成两分量的分歧。原因是 ESM-2 表征对 ~560 个残基做均值池化，单点替换只改变约 1/560 的信号。

**因此本模型不能用于单点突变体排序**，需微调或改用逐残基注意力 / ΔΔG 特征。脚本 `scripts/14_mutant_sensitivity.py`；公开版结果见 `results/mutant_sensitivity_anonymized.tsv`（突变体身份已匿名化为 M1–M17，原始序列仅本地保留）。

## 多烯底物区分（Morgan 指纹的已知盲点，已由 GNN 修复）

| 编码方式 | phytoene / phytofluene / ζ-carotene / neurosporene / lycopene | 不同取值数 |
|---|---|---|
| Morgan 指纹 | 1.04 / 4.87 / 4.87 / 4.87 / 3.11 s⁻¹（后三者逐位相同） | 3 / 5 |
| GNN 底物编码 | 0.24 / 0.26 / 0.29 / 0.33 / 0.38 s⁻¹，随去饱和单调递增 | **5 / 5** |

GNN 变体同时把分组划分 R² 从 0.098 提升到 0.156（RMSE 1.39→1.35），Pearson 提升不显著（p = 0.24）。脚本 `scripts/13_gnn_polyene.py`。

![scatter](results/fig_scatter.png)

## 文献案例验证（held-out，11 个案例酶均不在训练集中）

从文献挖掘有实测动力学数据的类胡萝卜素通路酶，与模型预测对比（`scripts/09_casestudy.py`）：

| 案例面板 | 内容 | 结果 |
|---|---|---|
| B 通路级酶（CrtB/CrtE/CrtI，含体内外值 14/6/0.8/0.1/0.03 s⁻¹） | **Spearman = 0.60**，正确复现 CrtB > CrtE > CrtI 的限速排序 | ✅ |
| C GGPP 合酶家族（人 0.204 / 酵母 0.025 / Pantoea 0.8 s⁻¹） | Spearman = 0.50；酵母 BTS1 定量吻合（预测 0.036 vs 实测 0.025 s⁻¹） | ✅ |
| D CrtZ 底物选择性（β-胡萝卜素 vs 角黄素，文献活性比 6.2×） | 预测方向正确（2.8×） | ✅ |
| A CrtI 同源酶排序 | 未通过：文献值 4 个并列 0.1 s⁻¹（原文拟合上限封顶，仅为下限值），模型对四者预测 0.65–0.85 s⁻¹ 与该下限一致，但最慢的 R. capsulatus CrtI（0.003）方向排反 | ⚠️ |
| E CrtI 逐步底物系列 | 八氢番茄红素/ζ-胡萝卜素/神经孢子烯的 Morgan 指纹**逐位相同**，指纹版无法区分（3/5）；**GNN 变体已修复（5/5）**，见上文多烯表 | ⚠️→✅ |

**定量误差**：A+B+C 全部 13 个点中，85% 预测落在文献值 ±10 倍以内，中位倍数误差 6.5×。作为参照，同一酶（PaCrtI）不同实验室报道的 kcat 本身相差 >1000 倍（8×10⁻⁵ ~ 0.25 s⁻¹，Schaub 2012 vs BRENDA），文献测定离散度已与模型误差同量级。

![case](results/fig_case_study.png)

**局限启示**：① 底物编码应升级为 GNN/图 Transformer 以区分高度相似的多烯底物；② 极低 kcat（<0.01 s⁻¹）区域训练数据稀缺，预测向训练分布中值回归。

## 快速使用

```bash
pip install -r requirements.txt   # GPU 需 CUDA 版 torch

# 序列 + 底物（主模型）
python scripts/predict_dual.py --seq "MKKVL..." --smiles "CC(C)=CCC/C(C)=C/..."
# 批量：TSV 含 id / sequence / smiles 三列
python scripts/predict_dual.py --tsv input.tsv --out results/out.tsv

# 仅序列（兜底模型，无底物信息时用）
python scripts/predict.py --fasta enzymes.fasta
```

首次运行自动下载 ESM-2 650M 权重（约 2.5 GB，Meta 官方源 dl.fbaipublicfiles.com）。

## 复现训练

```bash
python scripts/01_prepare_data.py            # 清洗 DLKcat 41.5k 数据集
python scripts/02b_fetch_sequences_idmap.py  # UniProt 批量抓序列
python scripts/03_build_dataset.py           # 序列级聚合（纯序列模型用）
python scripts/04_embed.py                   # ESM-2 embedding（GPU，~90s/3000条 @RTX4090）
python scripts/05_train.py                   # 纯序列基线模型
python scripts/06_train_dual.py              # 序列+底物双塔主模型（随机+分组两种划分）
python scripts/07_benchmark.py               # 与 UniKP/TurNuP 式基线对比
python scripts/08_ensemble.py                # CarotKcat-ensemble（双塔+XGBoost）
python scripts/09_casestudy.py               # 文献案例验证（图+统计）
python scripts/10_paper_figures.py           # 论文图（架构图 + 基准对比图）
python scripts/11_dlkcat_headtohead.py       # DLKcat 式 GNN+CNN 同划分头对头复现
python scripts/12_clean_eval.py              # 无泄漏重评 + bootstrap CI + 配对检验
python scripts/13_gnn_polyene.py             # GNN 底物分支 + 多烯区分检验
python scripts/14_mutant_sensitivity.py      # 突变体敏感性分析（需本地 mutants.fasta）
python scripts/15_external_benchmark.py      # ADK / KcatTIE 外部验证
python scripts/16_external_figure.py         # 外部验证组合图
python scripts/17_gnn_ci.py                  # GNN 变体 CI 与配对检验并入主表
```

## 类胡萝卜素通路覆盖与局限

- 公共数据库中类胡萝卜素核心酶（八氢番茄红素合酶 EC 2.5.1.32、脱氢酶 EC 1.3.5.5/1.3.5.6、番茄红素环化酶 EC 5.5.1.18/19 等）的 kcat 实测记录极少（4.1 万条中仅 GGPP 合酶 EC 2.5.1.29 有 53 条）。当前模型对这些酶的预测基于**通用酶大数据 + 序列/底物相似性外推 + 过采样强化**。
- 进一步提升精度的最直接途径：用实验室实测的 crt 酶动力学数据做 **few-shot 微调**（在 `scripts/06_train_dual.py` 数据集中追加记录即可重训）。
- 测定条件（温度/pH）未建模（原始数据多缺失）。

## 环境

Windows 11 / RTX 4090 (48GB) / torch 2.14.1+cu126 / Python 3.13 / RDKit / fair-esm
