---
date: 2026-05-15
author: ExpDesignAgent
input_from:
  - "04_Model_Design/01_model_architecture.md"
  - "03_Innovation/01_innovation_blueprint.md"
  - "02_Literature/03_dataset_analysis/dataset_analysis_report.md"
output_to: "05_Exp_Design/01_experiment_plan.md"
status: draft
---

# 实验设计方案

## 一、数据划分

### 1.1 DFID (200 张, 4-class balanced)

| Split | 数量 | 每类 | 策略 |
|-------|------|------|------|
| Train | 120 (60%) | 30/class | 分层随机 |
| Val | 20 (10%) | 5/class | 分层随机, 用于超参选择 |
| Test | 60 (30%) | 15/class | 固定, 与 MLTrMR 论文 test split 对齐 |

**对齐策略**: MLTrMR 使用 140/60 split。我们调整为其子集 (140 = 120 + 20 val)，保持 test 60 的划分一致，确保可比性。

### 1.2 SFXRay (80 张, imbalanced)

| Split | 数量 | 分布 | 策略 |
|-------|------|------|------|
| Train | 48 (60%) | N13/M20/Mo8/Se7 | 分层, 与 Mwinc-Mamba 论文 split 对齐 |
| Val | 8 (10%) | N2/M4/Mo1/Se1 | 分层随机 |
| Test | 24 (30%) | N6/M10/Mo4/Se4 | 固定, 与 Mwinc-Mamba test split 完全对齐 |

**对齐策略**: 从原 GT.xlsx 的 train (56) 中分出 8 张作 val，test (24) 完全不动。

---

## 二、Baseline 复现矩阵

### 2.1 DF Baselines

| # | Model | Loss | 说明 | 数据来源 |
|---|-------|------|------|---------|
| B1 | ResNet-50 | CE | 通用 baseline | own run |
| B2 | ViT-Base | CE | Backbone baseline (no EDL/ORCU) | own run |
| B3 | MLTrMR | CE | Reported SOTA | 论文数值 (80.19%) |

### 2.2 SF Baselines

| # | Model | Loss | 说明 | 数据来源 |
|---|-------|------|------|---------|
| B4 | ResNet-50 | CE | 通用 baseline | own run |
| B5 | Mwinc-Mamba | CE | Reported SOTA (Mamba backbone) | 论文数值 (66.67%) |
| B6 | 5 Radiologists avg | — | 人类基线 (仅 SF) | 论文数值 (70.00%) |

### 2.3 Our Methods

| # | Model | Loss | 说明 |
|---|-------|------|------|
| O1 | ViT-B + EvidenceHead | EDL only | EDL 单独效果 (DF) |
| O2 | ViT-B | ORCU only | ORCU 单独效果 (DF) |
| O3 | ViT-B + EvidenceHead | EDL + ORCU (ours) | 完整方法 (DF) |
| O4 | ResNet-50 + EvidenceHead | EDL only | EDL 单独效果 (SF) |
| O5 | ResNet-50 | ORCU only | ORCU 单独效果 (SF) |
| O6 | ResNet-50 + EvidenceHead | EDL + ORCU (ours) | 完整方法 (SF) |
| O7 | ResNet-50 + EvidenceHead | EDL + ORCU + MR | + Multi-Rater soft labels (SF only) |

---

## 三、消融实验矩阵

| ID | 实验 | 控制变量 | 假设 |
|----|------|---------|------|
| A1 | CE vs EDL vs EDL+ORCU | Loss function | ORCU 在 EDL 基础上进一步提升校准和有序性 |
| A2 | One-hot vs SORD soft label | Label encoding | SORD 软标签对有序分类优于 one-hot |
| A3 | λ_orcu ∈ {0.1, 0.3, 0.5, 0.7, 1.0} | Loss weight | 存在最优 λ 平衡 EDL 和 ORCU |
| A4 | t ∈ {1.0, 3.0, 5.0, 10.0} | ORCU temperature | t=3.0 (论文默认) 是否在氟中毒数据上最优 |
| A5 | KL annealing vs no annealing | Training stability | KL 退火稳定 EDL 训练 |
| A6 | With vs without multi-rater soft target | Label type | 多标注者 soft label 优于 majority-vote one-hot |

---

## 四、不确定性分析实验

| ID | 实验 | 预期发现 |
|----|------|---------|
| U1 | Evidence per class: α_k 分布 | 哪个 Dean/fluorosis 等级 evidence 最低 → 最难学 |
| U2 | Uncertainty vs misclassification rate | 高 u 样本应对应高错误率 → EDL 的不确定性是否 informative |
| U3 | OOD detection: 非氟区域牙齿照片 | EDL 应给 OOD 样本高 u → 验证 EDL 的 OOD 检测能力 |
| U4 | SF uncertainty by body part | 前臂 vs 小腿 vs 骨盆 — 哪个部位最难 |
| U5 | Radiologist agreement vs model uncertainty | 医生间不一致的样本 → 模型是否也给高 u |
| U6 | RecAcc & SafePred across θ ∈ {0.1, 0.2, 0.3, 0.4, 0.5} | 验证不确定性阈值对自动分诊效果的影响，确定最优 θ |

### U5 是关键创新分析

5 位医生对每张 SF 图像标注的一致性 (可用方差或 entropy) vs 模型对该图像的 uncertainty (u = 4/Σα)。期望看到正相关 — 这直接验证 EDL 捕捉了 ground truth 不确定性。

---

## 五、评价协议

### 5.1 指标汇总

| 维度 | 指标 | 越好方向 |
|------|------|:------:|
| 分类性能 | Acc, macro-F1 | ↑ |
| 有序一致性 | QWK (Quadratic Weighted Kappa) | ↑ |
| 校准度 | ECE (Expected Calibration Error, 15 bins) | ↓ |
| 类级校准 | SCE (Static Calibration Error) | ↓ |
| 单峰性 | %Unimodal predictions | ↑ |
| 不确定性质量 | U-ECE (Uncertainty-ECE) | ↓ |
| 不确定性鉴别力 | AUROC (u vs error) | ↑ |
| 自动分诊准确率 | RecAcc (Recommendation Accuracy) | ↑ |
| 安全预测率 | SafePred (Safe Prediction Rate) | ↑ |

RecAcc 和 SafePred 的定义 (θ=0.30):
- **RecAcc = 1 - Acc(high-u)** — 高不确定性样本中，模型推荐人工复核的比例中实际误诊的占比（越高越好，说明 u 有效识别了困难样本）
- **SafePred = Acc(low-u)** — 低不确定性样本的准确率（越高越好，说明模型对"有把握"的样本确实可靠）

这两个指标直接支撑自动诊断报告（03_Innovation/02_idea_proposals/auto_diagnosis_report.md）中的临床分诊场景。

### 5.2 统计检验

- 5-fold cross-validation (DF: 5×40 test; SF: 5×5 test with grouped body-part strata)
- Paired t-test (α=0.05) for method comparison
- 报告 mean ± std over 5 folds

### 5.3 显著性标记

论文表格中使用上标标记:
- \*: p < 0.05 vs best CE baseline
- †: p < 0.05 vs EDL only
- ‡: p < 0.05 vs ORCU only

---

## 六、对 MLTrMR / Mwinc-Mamba 论文实验的复现范围

| 论文 | 需完全复现 | 可引用数值 | 备注 |
|------|:--------:|:--------:|------|
| MLTrMR (DF) | — | Acc 80.19%, F1 75.79% | ViT-B+CE 作为我们复现的 baseline proxy |
| Mwinc-Mamba (SF) | — | Acc 66.67%, Radiologist 70.00% | Mwinc-SSM+CE 作为我们的 baseline proxy |

**复现策略**: 不试图完全复现 MLTrMR 的 Masked Latent Transformer (太复杂)，而是以 ViT-B+CE 作为合理的 Transformer baseline。如果我们的 ViT-B+CE 接近 80.19%，则我们的 O3 (ViT+EDL+ORCU) 超越它是公平对比。同理，SF 以 ResNet-50+CE 为 baseline，与 Mwinc-Mamba (66.67%) 的对比在相同 ResNet-50 backbone 下进行 loss-function 消融。论文中明确区分 backbone 差异和 loss 贡献。
