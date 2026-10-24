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

### 与基线/同类方法的系统对比（同一测试集，n≈2,558/2,813）

| 模型 | 随机划分 Pearson / R² | 按酶分组划分 Pearson / R² |
|---|---|---|
| CarotKcat-ensemble（双塔+XGBoost 平均） | **0.727 / 0.528** | **0.440 / 0.189** |
| CarotKcat 双塔神经网络 | 0.649 / 0.412 | 0.439 / 0.163 |
| UniKP 式（ESM+指纹, ExtraTrees） | 0.725 / 0.518 | 0.406 / 0.165 |
| TurNuP 式（ESM+指纹, XGBoost） | **0.735 / 0.538** | 0.366 / 0.121 |
| 仅序列（ESM+MLP） | 0.637 / 0.401 | 0.419 / 0.171 |
| 仅底物指纹（MLP） | 0.477 / 0.216 | 0.338 / 0.094 |

关键发现：**树模型在随机划分下最强，但面对未见过的酶（分组划分）时 CarotKcat 双塔/集成明显更优**——深度双塔对全新酶的外推能力更强，集成版在两种协议下都保持最佳或近最佳。

![scatter](results/fig_scatter.png)

## 文献案例验证（held-out，11 个案例酶均不在训练集中）

从文献挖掘有实测动力学数据的类胡萝卜素通路酶，与模型预测对比（`scripts/09_casestudy.py`）：

| 案例面板 | 内容 | 结果 |
|---|---|---|
| B 通路级酶（CrtB/CrtE/CrtI，含体内外值 14/6/0.8/0.1/0.03 s⁻¹） | **Spearman = 0.60**，正确复现 CrtB > CrtE > CrtI 的限速排序 | ✅ |
| C GGPP 合酶家族（人 0.204 / 酵母 0.025 / Pantoea 0.8 s⁻¹） | Spearman = 0.50；酵母 BTS1 定量吻合（预测 0.036 vs 实测 0.025 s⁻¹） | ✅ |
| D CrtZ 底物选择性（β-胡萝卜素 vs 角黄素，文献活性比 6.2×） | 预测方向正确（2.8×） | ✅ |
| A CrtI 同源酶排序 | 未通过：文献值 4 个并列 0.1 s⁻¹（原文拟合上限封顶，仅为下限值），模型对四者预测 0.65–0.85 s⁻¹ 与该下限一致，但最慢的 R. capsulatus CrtI（0.003）方向排反 | ⚠️ |
| E CrtI 逐步底物系列 | 八氢番茄红素/ζ-胡萝卜素/神经孢子烯的 Morgan 指纹**逐位相同**（已验证），模型无法区分——指纹编码的固有局限 | ⚠️ |

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
```

## 类胡萝卜素通路覆盖与局限

- 公共数据库中类胡萝卜素核心酶（八氢番茄红素合酶 EC 2.5.1.32、脱氢酶 EC 1.3.5.5/1.3.5.6、番茄红素环化酶 EC 5.5.1.18/19 等）的 kcat 实测记录极少（4.1 万条中仅 GGPP 合酶 EC 2.5.1.29 有 53 条）。当前模型对这些酶的预测基于**通用酶大数据 + 序列/底物相似性外推 + 过采样强化**。
- 进一步提升精度的最直接途径：用实验室实测的 crt 酶动力学数据做 **few-shot 微调**（在 `scripts/06_train_dual.py` 数据集中追加记录即可重训）。
- 测定条件（温度/pH）未建模（原始数据多缺失）。

## 环境

Windows 11 / RTX 4090 (48GB) / torch 2.14.1+cu126 / Python 3.13 / RDKit / fair-esm
