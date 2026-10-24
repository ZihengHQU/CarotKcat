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

| 协议 | Pearson r | R² | RMSE (log10) | 说明 |
|---|---|---|---|---|
| 随机划分（DLKcat 论文协议） | **0.653** | **0.413** | 1.16 | 与 DLKcat 原论文（R²≈0.44）相当 |
| 按酶分组划分（无泄漏，更严格） | 0.404 | 0.127 | 1.37 | 测试集酶完全未见过的真实外推能力 |
| 纯序列模型（无底物输入） | 0.445 | 0.168 | 1.28 | 基线对照 |

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
```

## 类胡萝卜素通路覆盖与局限

- 公共数据库中类胡萝卜素核心酶（八氢番茄红素合酶 EC 2.5.1.32、脱氢酶 EC 1.3.5.5/1.3.5.6、番茄红素环化酶 EC 5.5.1.18/19 等）的 kcat 实测记录极少（4.1 万条中仅 GGPP 合酶 EC 2.5.1.29 有 53 条）。当前模型对这些酶的预测基于**通用酶大数据 + 序列/底物相似性外推 + 过采样强化**。
- 进一步提升精度的最直接途径：用实验室实测的 crt 酶动力学数据做 **few-shot 微调**（在 `scripts/06_train_dual.py` 数据集中追加记录即可重训）。
- 测定条件（温度/pH）未建模（原始数据多缺失）。

## 环境

Windows 11 / RTX 4090 (48GB) / torch 2.14.1+cu126 / Python 3.13 / RDKit / fair-esm
