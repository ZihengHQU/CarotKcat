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

### 划分稳健性：5 折重复划分（mean ± SD）

单次划分的数值可能是运气，因此把整套无泄漏流程在 **5 个独立划分**上各重跑一遍（共 10 次完整重训，`scripts/18_repeated_splits.py`，约 62 分钟 / RTX 4090）。

| 模型 | 随机划分 Pearson（mean ± SD） | 分组划分 Pearson（mean ± SD） |
|---|---|---|
| CarotKcat-ensemble | 0.726 ± 0.016 | **0.463 ± 0.048** |
| UniKP 式（ExtraTrees） | 0.735 ± 0.021 | 0.434 ± 0.035 |
| TurNuP 式（XGBoost） | **0.744 ± 0.019** | 0.421 ± 0.036 |
| CarotKcat 双塔（指纹） | 0.634 ± 0.014 | 0.415 ± 0.061 |

- **随机划分**：XGBoost 在 **5/5 折**都最好，集成稳定低 0.018（小但系统）→ 该协议下不主张优势；
- **分组划分**：集成在 **4/5 折**最好，对 XGBoost 5/5 折胜（Δr = +0.042，配对 t 检验 p = 0.024；Wilcoxon p = 0.062，n=5 受限）；对 ExtraTrees 的 +0.029 在 5 折下无法分辨（p = 0.31）→ 仍表述为**持平**；
- 单次划分（seed 42）的集成分组 r = 0.420，**低于** 5 折均值 0.463 → 主表数字是偏保守而非"挑好看的"；
- 折间 SD：随机 0.014–0.021，分组 0.035–0.061 → 分组 r 上 <0.03 的差异不应过度解读。

明细：`results/repeated_splits.csv`、`results/repeated_splits_summary.csv`。

## 外部验证（独立数据，未参与训练/调参）

| 数据集 | 说明 | 结果 |
|---|---|---|
| **KcatTIE**（2023–2025 文献，n=145） | 时间留出集，晚于训练语料；已剔除 7 条与训练集完全相同的序列 | r = 0.56，ρ = 0.54，R² = 0.30，中位倍数误差 6.0×，58% 在 ±10× 内 ✅ |
| **ADK**（177 个腺苷酸激酶同源变体，n=176） | 单一高活性酶家族的变体排序 | r = 0.05，ρ = 0.04，R² < 0，**系统性低估约 16×**；XGB 单模型 r = 0.24 仍差 ❌ |

**结论**：模型对 kcat 量级与训练语料相近的酶可信；对 kcat 极高（如 ADK 可达 ~800 s⁻¹）的家族会向训练均值回归，**不能用于跨家族变体排序**。脚本 `scripts/15_external_benchmark.py`。

## 干验证：酶约束基因组尺度模型（iML1515，替代湿实验的定量检验）

无湿实验时，用**已发表的番茄红素生产数据**反过来检验预测值是否自洽。三篇来源研究都只转入 `crtE/crtB/crtI`（产物为番茄红素，非 β-胡萝卜素），因此模块确定为这三个酶。

| CrtE (FPP) | CrtB (GGPP) | CrtI (phytoene) |
|---|---|---|
| 预测 0.146 s⁻¹ / 文献 0.80（低 5.5×） | 预测 1.61 s⁻¹ / 文献 6–14（低 3.7×） | 预测 0.752 s⁻¹ / 文献 0.10（高 7.5×） |

把预测 kcat 代入 iML1515（葡萄糖 10 mmol/gDCW/h、好氧，三条新增反应均通过元素配平）后得到三个结论：

1. **异源酶不是瓶颈**：维持已报道最高产率 3.12 µmol/gDCW/h 只需 CrtE 0.039%、CrtB 0.004%、CrtI 0.012% 的总蛋白；即使按 0.1% 弱表达算，仍有 3×/27×/8× 余量，1% 表达下余量 25×/274×/80×。
2. **化学计量上限很远**：无酶约束时 iML1515 最大番茄红素通量 929 µmol/gDCW/h，是实测值的 ~300 倍，说明差距来自酶学/调控而非计量。
3. **瓶颈指向前体供给**：µ_max 下天然 MEP 途径（DXS）仅 2.3 µmol/gDCW/h，而实测番茄红素通量 3.12 已是其 **1.4 倍**——必须扩增/分流 MEP 途径才能达到，这正是两篇拆解研究实际做的（dxs/idi/ispA 扩增使效价提升 10×，0.38 → 4.0 mg/gDCW）。生长-产量包络显示该通量只损失 1% 的 µ_max，故限制来自前体而非代谢负担。

**诚实边界**：这是**一致性检验，不是判别性检验**——因为三个酶离饱和有 1–2 个数量级，"前体限制"这一结论在 kcat 有数倍误差时依然成立。它证明的是预测值与实测生理**定量相容**，且**未经提示**复现了两项独立工程研究认定的瓶颈。真正能**证伪** kcat 的检验需要体外通路通量实测，或 MEP 供给与生长解耦的菌株。

脚本 `scripts/19_ecgem_dry_validation.py`、`scripts/20_ecgem_figure.py`；图 `results/fig_ecgem_validation.png`。iML1515 为第三方模型，首次运行自动从 BiGG 下载（不入库）。

## 零样本基准：萜类 / 类胡萝卜素酶（43 条实测，35 个酶）

因为 **BRENDA 对 CrtB（2.5.1.32）、CrtI（1.3.99.31/1.3.5.5）、CrtY（5.5.1.19）根本没有周转数条目**（只有 Km），训练集里这些 EC 也一条都没有，所以类胡萝卜素模块的预测**全是外推**。为此自建了基准集：BRENDA 动力学表（逐条带 UniProt + 文献编号）+ 类胡萝卜素模块的原始文献值。

| 子集 | n | r | ρ | R² | 中位倍数误差 | 3× 内 | 响应斜率 |
|---|---|---|---|---|---|---|---|
| 总体 | 43 | 0.494 | 0.505 | 0.134 | 6.7× | 35% | 0.38 |
| 训练集见过的 EC 家族 | 20 | 0.658 | 0.714 | 0.410 | 2.1× | 60% | 0.47 |
| **训练集没有的家族** | 23 | 0.135 | 0.068 | −0.453 | **8.6×** | 13% | 0.10 |
| 其中：类胡萝卜素 | 12 | −0.543 | −0.544 | −0.711 | **26×** | 0% | −0.22 |
| 其中：萜类合酶 | 12 | −0.135 | −0.280 | −0.893 | 8.4× | 25% | −0.11 |
| 前体（DXS 等，见过） | 12 | 0.756 | 0.844 | 0.415 | 2.1× | 67% | 0.29 |
| 异戊二烯供应（见过） | 7 | 0.670 | 0.571 | −0.023 | 1.9× | 57% | 0.82 |

**结论**：模型对"见过的家族、没见过的酶"能给出 ~2× 的先验，对**没见过的家族则基本没有排序能力**（类胡萝卜素子集 r 为负）。**响应斜率**（预测对实测回归的斜率，1.0 为完美、0 为忽略输入）为 0.38，说明预测被压缩到训练均值附近；预测的四分位距 1.60 log 单位小于实测的 2.42。误差随量级反向：慢酶高估、快酶低估（斜率 −0.62）。

**用途边界**：可作为通量建模的**数量级先验**（干验证已证明数倍误差不改变通路结论），**不能用于在通路内挑选/排序候选酶或突变体**。

脚本 `scripts/21_brenda_fetch.py`（抓取解析）、`scripts/22_terpenoid_benchmark.py`（构建 + 预测）、`scripts/23_terpenoid_figure.py`（指标 + 图）；数据 `results/terpenoid_bench.tsv`、`results/terpenoid_bench_metrics.csv`，图 `results/fig_terpenoid_benchmark.png`。表中每条记录都带 UniProt 编号与文献出处，无 (酶, 底物) 组合与训练集重叠。

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
python scripts/18_repeated_splits.py         # 5 折重复划分稳健性（mean ± SD，~62 min）
python scripts/19_ecgem_dry_validation.py    # 酶约束模型干验证（iML1515，自动下载模型）
python scripts/20_ecgem_figure.py            # 干验证三联图
python scripts/21_brenda_fetch.py            # 抓取并解析 BRENDA 萜类 EC 动力学表
python scripts/22_terpenoid_benchmark.py     # 自建零样本基准 + 预测（含 ESM 编码）
python scripts/23_terpenoid_figure.py        # 基准指标与三联图
```

## 类胡萝卜素通路覆盖与局限

- 公共数据库中类胡萝卜素核心酶（八氢番茄红素合酶 EC 2.5.1.32、脱氢酶 EC 1.3.99.31/1.3.5.5、番茄红素环化酶 EC 5.5.1.19）**在 BRENDA 中没有任何周转数条目**，只有 Km；训练集里这些 EC 也一条都没有。因此模型对这些酶属于**完全外推**，基准实测中位倍数误差 **26×**、相关系数 **r = −0.54**（见上文"零样本基准"）。
- 同一批基准里，训练集**见过**的家族（DXS、异戊二烯供应酶）中位误差只有 **2.1×**、r = 0.66–0.76。两者对比说明瓶颈是**数据缺失**，不是架构。
- 因此当前定位是：**数量级先验**（可用于酶约束建模），**不可用于通路内候选酶或突变体的排序**。
- 唯一有效的提升途径：用实验室实测的 crt 酶动力学数据做 **few-shot 微调**——几十条体外 kcat 即可（在 `scripts/06_train_dual.py` 数据集中追加记录后重训）。
- 测定条件（温度/pH）未建模（原始数据多缺失）。

## 环境

Windows 11 / RTX 4090 (48GB) / torch 2.14.1+cu126 / Python 3.13 / RDKit / fair-esm
