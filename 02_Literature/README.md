---
date: 2026-05-15
author: LitAgent
output_to: "02_Literature/README.md"
status: final
---

# 02_Literature — 文献调研

## 目录结构

```
02_Literature/
├── README.md                          ← 本文件
├── literature_comparison_table.md     ← 核心文献对比表 (19 papers)
├── research_gap_analysis.md           ← 研究空白分析与创新机会
├── 01_initial_search/
│   ├── literature_list.md             ← 初始检索记录
│   ├── Reference_PDF_Index.md         ← 134 篇 PDF 索引 (A/B/C 分类)
│   ├── References/                    ← 134 PDF 原文
│   └── References_extracted/          ← OCR 文本提取
├── 02_deep_read/                      ← 19 篇精读笔记
└── 03_dataset_analysis/
    └── dataset_analysis_report.md      ← 数据集分析报告
```

## 精读笔记索引

### A1 — 氟斑牙 DL 诊断 (DF)

| # | 文件 | 核心内容 |
|---|------|---------|
| 17 | `17_MLTrMR_DF_Diagnosis_JCVI2025.md` | **SOTA DF**: Masked Latent Transformer, DFID 200, 80.19% Acc |
| 09 | `09_DF_Quantum_Fuzzy_Segmentation.md` | 传统 ML: Fuzzy C-means + K-means 分割 |
| 02 | `02_DF_Web_Based_Detection.md` | Web 诊断系统 |
| 03 | `03_Differential_Diagnosis_DF.md` | 临床鉴别诊断 |

### A2 — 氟骨症 DL/ML 诊断 (SF)

| # | 文件 | 核心内容 |
|---|------|---------|
| 16 | `16_Mwinc-Mamba_SF_Diagnosis_BSPC2025.md` | **SOTA SF**: CNN+SSM, SFXRay 80, 66.67% Acc, 最接近的竞品 |
| 01 | `01_SF_Integrated_Learning.md` | 早期 CNN ensemble (Liu 2021), private data |
| 10 | `10_SF_ML_Prediction_s41598.md` | ML 临床预测模型 |
| 18 | `18_Quadri2016_SF_Spinal_MRI_CaseReport.md` | SF MRI 病例报告, pseudomyeloma |

### A3 — 流行病学/综述

| # | 文件 | 核心内容 |
|---|------|---------|
| 19 | `19_Gu2024_Zunyi_Fluorosis_Survey.md` | 遵义 2009 vs 2023 DF 调查, 吴云课题组 |
| 04 | `04_Fejerskov_Nature_Mechanisms_DF.md` | DF 病理机制 Nature Review |
| 14 | `14_DL_Dental_Survey.md` | 牙科 DL 综述 |

### B — EDL/不确定性/有序回归 (方法论核心)

| # | 文件 | 核心内容 | 评级 |
|---|------|---------|:----:|
| 06 | `06_ORCU_Ordinal_Calibration.md` | Ordinal Regression + Calibration + Unimodality | ★★★★★ |
| 05 | `05_MEDL_Mutual_Evidential_Deep_Learning.md` | Mutual EDL for multi-view | ★★★★☆ |
| 07 | `07_EviVLM_Evidential_VLM_Segmentation.md` | EDL + Vision Language Model | ★★★★☆ |
| 08 | `08_REDNet_Evidential_Discounting.md` | Evidential discounting for noisy labels | ★★★★☆ |
| 11 | `11_Uncertainty_Evidential_Fusion_SemiSeg.md` | Evidential fusion, semi-supervised | ★★★☆☆ |
| 12 | `12_DuEDL_Scribble_Segmentation.md` | EDL from scribble annotations | ★★★☆☆ |

### C — 相关方法参考

| # | 文件 | 核心内容 |
|---|------|---------|
| 13 | `13_HyperDenseNet_Multimodal.md` | 多模态医学图像融合 |
| 15 | `15_FewShot_Osteopenia_Xie2024.md` | Few-shot 骨疾病分类 |

## 关键发现

1. **唯一竞品**: 贵州大学吴云组 (Xu Hao et al.) — 两篇 SOTA 论文覆盖 DF + SF
2. **核心空白**: 无 EDL, 无 Ordinal Regression, 无多标注者建模
3. **最强论据**: SFXRay 5 位放射科医生四分类平均一致率仅 **54.1%** — 直接验证 EDL 必要性
4. **数据集就绪**: `data/` 含 DF 200 + SF 80 + masks 59 + 5 医生完整标注
5. **可迁移方法**: ORCU (MedIA 2025) — 最直接的 CE loss 替代方案

## 下一步

→ `03_Innovation/` — 基于 gaps 设计创新点组合
