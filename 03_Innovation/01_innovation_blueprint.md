---
date: 2026-05-15
author: LitAgent + InnoAgent
input_from:
  - "02_Literature/research_gap_analysis.md"
  - "02_Literature/literature_comparison_table.md"
  - "02_Literature/02_deep_read/06_ORCU_Ordinal_Calibration.md"
  - "02_Literature/02_deep_read/16_Mwinc-Mamba_SF_Diagnosis_BSPC2025.md"
  - "02_Literature/02_deep_read/17_MLTrMR_DF_Diagnosis_JCVI2025.md"
output_to: "03_Innovation/01_innovation_blueprint.md"
status: draft
---

# 创新蓝图：EDL + 有序回归用于氟中毒自动化诊断

## 一、核心创新命题

> **命题**: 将 Evidential Deep Learning (EDL) 与 Ordinal Regression for Calibration and Unimodality (ORCU) 引入氟中毒 DL 诊断，首次为氟斑牙 (DF) 和氟骨症 (SF) 提供带有校准不确定性的有序分级模型。

### 1.1 创新三组件

| 组件 | 来源 | 解决的问题 | 创新层级 |
|------|------|-----------|:--------:|
| **EDL Evidence** | MEDL (TMI 2023) | 替代 CE softmax 点估计，输出 Dirichlet evidence + uncertainty | 领域首次应用 |
| **ORCU Loss** | ORCU (MedIA 2025) | 强制有序约束、概率校准、预测单峰性 | 首次医学分类应用 |
| **Multi-Rater Soft Label** | — | 用 5 位医生标注分布替代 majority-vote one-hot，标注不确定性入训 | 首次利用 SFXRay 完整标注 |

### 1.2 论文题目（草案）

> **Evidential Deep Learning with Ordinal Calibration for Automated Diagnosis of Endemic Fluorosis**

备选:
- "Uncertainty-Aware Ordinal Classification of Dental and Skeletal Fluorosis via Evidential Deep Learning"
- "From Point Estimates to Evidence: Ordinal Evidential Diagnosis of Endemic Fluorosis"

---

## 二、方法设计概要

### 2.1 整体架构

```
Input Image (DF 512×256 / SF 512×1024)
        │
        ▼
┌───────────────────────┐
│  Feature Extractor    │  ← ResNet/ViT/Mamba backbone (task-specific)
│  (CNN/Transformer/SSM)│
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Evidence Head        │  ← 替换 classifier head
│  FC → Softplus → α    │  ← 输出 Dirichlet concentration params α = [α₀,α₁,α₂,α₃]
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Dirichlet Distribution│
│  p ~ Dir(α)           │  ← 预测 = argmax(α-1); uncertainty = K/∑α
└───────────────────────┘
```

### 2.2 Loss Function 设计

```
L_total = L_EDL + λ_orcu · L_ORCU
```

**L_EDL** (Evidential loss):
- Type II Maximum Likelihood: 对 Dirichlet 分布求负对数似然
- 包含 evidence 的贝叶斯风险最小化项
- KL 正则项: 当 evidence 不支持正确类别时施加惩罚

**L_ORCU** (Ordinal calibration loss):
- 有序回归约束: f(x;θ) 输出 monotonic 分数 → softmax → 与 GT 的 CE
- 校准约束: ECE-based 正则，强制置信度与准确率对齐
- 单峰性约束: 对非单峰预测分布施加惩罚

**Multi-Rater Target**:
- 不直接用 one-hot majority vote
- 从 5 位医生的标注分布构建 soft probability target
- 例: [CBQ=1, CF=2, XR=2, ZY=2, ZYC=3] → target = [0, 0.2, 0.6, 0.2]

### 2.3 Backbone 选型矩阵

| Backbone | DF 适用性 | SF 适用性 | 理由 |
|----------|:---------:|:---------:|------|
| ResNet-50 | ★★★ | ★★★ | 通用 baseline, 轻量 |
| ViT-Base | ★★★★ | ★★★ | 全局注意力, DF 已验证 (MLTrMR) |
| Swin-T | ★★★★ | ★★★ | 层次化窗口注意力 |
| Mamba/Vim | ★★★ | ★★★★ | 线性复杂度, SF 已验证 (Mwinc-Mamba) |

建议: DF 用 ViT/Swin, SF 用 Mamba/CNN hybrid — 每个任务选最优 backbone，EDL+ORCU 是 backbone-agnostic 的 loss 创新。

---

## 三、实验设计概要

### 3.1 主实验

| 实验 | 数据集 | 指标 | 目标 |
|------|--------|------|------|
| DF 4-grade | DFID 200 (140/60) | Acc, F1, QWK, ECE, U-ECE | > MLTrMR 80.19% Acc |
| SF 4-grade | SFXRay 80 (56/24) | Acc, F1, QWK, ECE, U-ECE | > Mwinc-Mamba 66.67% Acc, ≥ 70% (radiologist avg) |

### 3.2 消融实验

| # | 内容 | 目的 |
|---|------|------|
| A1 | CE vs EDL vs EDL+ORCU | 隔离每个 loss 组件的贡献 |
| A2 | One-hot vs soft multi-rater target | 验证多标注者建模收益 |
| A3 | Fixed backbone: ViT vs Swin vs Mamba | Backbone 对 EDL 的影响 |
| A4 | λ_orcu sweep (0.1, 0.5, 1.0, 2.0) | 损失权重敏感性 |

### 3.3 不确定性分析

| # | 内容 |
|---|------|
| U1 | Evidence distribution per class — 哪个等级最难学？ |
| U2 | Uncertainty vs accuracy — 高不确定性样本是否确实是难例？ |
| U3 | Uncertainty by body part (SF forearm vs calf vs pelvis) |
| U4 | OOD detection: 用非氟中毒牙齿/SF X 光测试 EDL 的拒绝能力 |

---

## 四、创新贡献声明

1. **首次将 EDL 引入氟中毒 DL 诊断**，提供预测的不确定性估计，解决放射科医生间一致性差 (54.1%) 带来的 ground truth 模糊问题。

2. **首次将 ORCU 有序回归 loss 应用于医学图像分类**，用有序约束 + 校准 + 单峰性三重机制替换传统的名义分类 CE loss。

3. **首次利用 SFXRay 的 5 位放射科医生完整标注进行多标注者建模**，用 soft label distribution 替代 majority-vote one-hot。

4. **在小样本医学分类场景下系统评估 EDL 的不确定性校准** (SFXRay 训练仅 56 张)，为 EDL 在数据稀缺医学影像场景的应用提供实证。

---

## 五、风险与缓解

| 风险 | 概率 | 影响 | 缓解策略 |
|------|:----:|:----:|----------|
| EDL 在小样本下不确定性不可靠 | 中 | 高 | 先在 DF (200张) 验证 EDL 有效性，再迁移到 SF; 用 ECE 硬指标衡量 |
| ORCU + EDL 联合训练不稳定 | 中 | 中 | 逐步训练: 先 CE warmup → 加 EDL → 加 ORCU; 学习率 warmup |
| Acc 未超越 SOTA baseline | 低-中 | 高 | EDL 的主要贡献是不确定性而非 Acc，即使 Acc 持平，不确定性+校准也是足够的新贡献 |
| SF 56 张训练太少的固有上限 | 高 | 中 | 用 heavy augmentation + transfer learning 缓解; 在 Discussion 中坦诚讨论 |

---

## 六、与竞品的关键差异

| 维度 | 吴云组 (Xu et al. 2025) | 我们的方法 |
|------|------------------------|-----------|
| 预测形式 | 点估计 (softmax prob) | Dirichlet evidence + 不确定性 |
| Loss | Cross-Entropy | EDL + ORCU (有序+校准+单峰) |
| 标注处理 | Majority vote one-hot | 5 医生 soft distribution |
| 不确定性 | 无 | 核心产出 |
| 有序建模 | 无 (名义分类) | ORCU 三重约束 |
| 校准 | 无 | ECE 正则化 |
