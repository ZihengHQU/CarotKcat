# Kcat-Predictor：基于酶氨基酸序列的 Kcat / 酶活预测模型

> 输入一条酶的氨基酸序列，直接预测其催化常数 kcat（s⁻¹）。
> 面向类胡萝卜素（Carotenoid）合成通路酶的定向优化场景，底座为通用酶活大数据训练。

## 模型概览

```
氨基酸序列 ──► ESM-2 (650M, mean pooling, 1280维) ──► MLP 回归头 ──► log10(kcat) ──► kcat (s⁻¹)
```

- **序列编码器**：ESM-2 t33_650M_UR50D（Meta 蛋白质语言模型，冻结权重，取第 33 层表示做序列均值池化）
- **回归头**：1280 → 1024 → 512 → 128 → 1 的 MLP（GELU + Dropout 0.2，SmoothL1 损失）
- **训练策略**：AdamW + 余弦退火，类胡萝卜素相关酶训练时 10× 过采样加权

## 数据

| 数据集 | 规模 | 来源 |
|---|---|---|
| 通用酶 kcat 原始记录 | 41,558 条 | DLKcat 项目（BRENDA + SABIO-RK 整合） |
| 有效 kcat 记录（含 UniProt ID） | 18,616 条 | 过滤单位 s⁻¹、正值 |
| 序列级训练集 | 3,010 个酶 | 同一 UniProt ID 的多次测量取 log10 中位数聚合 |
| 类胡萝卜素酶预测集 | 3,725 条序列 | UniProt（crtE/B/I/Y/Z/W/X/O/U、phytoene synthase/desaturase、lycopene cyclase、carotene hydroxylase/ketolase 等关键词 + 相关 EC） |

## 性能（序列级 85/15 划分，测试集 n=452）

| 指标 | 数值 |
|---|---|
| Pearson r | ≈ 0.45 |
| R² | ≈ 0.17 |
| RMSE | ≈ 1.28（log10 kcat） |

说明：纯序列（不含底物信息）预测 kcat 存在理论上限——同一酶对不同底物的 kcat 差异天然被聚合掉。该水平与已发表的 sequence-only 基线相当。底物特异性预测可在此基础上叠加底物指纹/GNN（DLKcat 式结构）。

## 目录结构

```
data/      原始与清洗后数据（DLKcat 数据集、UniProt 类胡萝卜素酶、序列库）
scripts/   全流程脚本（01 数据清洗 → 02 抓序列 → 03 建数据集 → 04 ESM embedding → 05 训练；predict.py 推理）
models/    训练好的模型 kcat_mlp.pt、embedding 缓存
results/   测试集指标、全数据预测、3725 条类胡萝卜素酶 kcat 预测
```

## 快速使用

```bash
pip install -r requirements.txt   # 需 CUDA 版 torch 以获得 GPU 加速

# 预测单条序列
python scripts/predict.py --seq "MKKVL..." 

# 批量预测 FASTA
python scripts/predict.py --fasta your_enzymes.fasta --out results/my_predictions.tsv
```

首次运行会自动下载 ESM-2 650M 权重（约 2.5 GB，来自 dl.fbaipublicfiles.com 官方源）。

## 复现训练

```bash
python scripts/01_prepare_data.py        # 清洗 DLKcat 41.5k 数据集
python scripts/02b_fetch_sequences_idmap.py  # UniProt ID mapping 批量抓序列
python scripts/03_build_dataset.py       # 序列级聚合 + 类胡萝卜素标记
python scripts/04_embed.py               # ESM-2 embedding（GPU，~90s/3000条 @RTX4090）
python scripts/05_train.py               # 训练 + 评估
```

## 已知局限与后续路线

1. **类胡萝卜素酶 kcat 实验数据全球稀缺**：DLKcat 41.5k 数据集中仅 GGPP 合酶（EC 2.5.1.29）等有少量记录，核心酶（八氢番茄红素合酶/脱氢酶、番茄红素环化酶）几乎无公开 kcat。当前模型对这类酶的预测依赖序列相似性外推，建议后续用实验室实测数据做 few-shot 微调（脚本已支持追加数据重训）。
2. 未建模温度/pH/底物浓度等测定条件（原始数据多缺失）。
3. 计划：叠加底物 SMILES 图特征（DLKcat/TurNuP 思路）→ 序列+底物双塔模型。

## 环境

- Windows 11 原生 / RTX 4090 (48GB) / torch 2.14.1+cu126 / Python 3.13
- 依赖见 `requirements.txt`；ESM 权重缓存在 `~/.cache/torch/hub/`
