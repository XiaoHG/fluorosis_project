---
date: 2026-05-15
author: LitAgent
input_from: "02_Literature/01_initial_search/References/Xu 等 - 2025 - Convolutional state space model with multi-window cross-scan to advance the automated diagnosis of s.pdf"
output_to: "02_Literature/02_deep_read/16_Mwinc-Mamba_SF_Diagnosis_BSPC2025.md"
status: draft
---

# Mwinc-Mamba: Convolutional State Space Model for Skeletal Fluorosis Automated Diagnosis

## Metadata

| Field | Value |
|-------|-------|
| Authors | Hao Xu, Yun Wu, Rui Xie, Jun Xu, Junpeng Wu, Rongpin Wang, Youliang Tian |
| Venue | Biomedical Signal Processing and Control 103 (2025) 107439 (Elsevier) |
| Year | 2025 (online Dec 2024) |
| Affiliation | Guizhou University + Zhijin County People's Hospital + Guizhou Provincial People's Hospital |
| Code/Dataset | https://github.com/uxhao-o/Mwinc-Mamba |
| Category | A2 — 氟骨症 DL 诊断 (NEW, Tier 1) |

## Problem & Motivation

- SF diagnosis relies on manual X-ray interpretation of forearm, pelvis, and calf. Even among 5 radiologists with 5yr+ experience, **Cohen's kappa is poor** — substantial inter-rater disagreement documented with heatmap evidence
- Only 1 prior DL paper (Liu et al. 2021, our #01) on private dataset, forearm-only, no radiologist comparison
- Three key gaps: (i) no public SF X-ray dataset, (ii) lesion features subtle and hard to differentiate, (iii) clinical applicability unvalidated
- CNN lacks long-range dependency for high-res X-rays; Transformer has quadratic complexity; SSM (Mamba) offers linear complexity with global modeling

## Method

### Mwinc-Mamba Architecture — 4-Stage Hierarchical

**Core idea**: Dual-branch CNN+SSM with Multi-Window Cross-Scan (MWCS)

1. **Patch Embedding Layer**: Non-flattened — patches retain 2D spatial structure (unlike Vim which disrupts spatial dependencies)
2. **Mwinc-SSM Block** (core, repeated per stage):
   - **CNN Branch**: Conv layers extract local texture features (bone trabeculae, periosteal calcification)
   - **MWCS-SSM Branch**: SSM with multi-window cross-scan
     - Divides patches into multiple non-overlapping windows at different scales
     - Cross-scans in 4 directions per window: H-scan, H-flip-scan, V-scan, V-flip-scan
     - Each direction → separate S6 block → cross-aggregated into unified token sequence
     - Captures **multi-grained lesion features**: small windows for local calcifications, large windows for overall bone deformity
   - Channel splitting sends half to each branch, outputs concatenated via 1×1 conv
3. **Patch Merging Layer** (Stages 1-3): Halves spatial dims, doubles channels
4. **Classifier Head**: After Stage 4, produces lesion grade

### SSM Background (S6/Mamba)
- Continuous SSM: h'(t) = Ah(t) + Bx(t); y(t) = Ch(t)
- Discretized via zero-order hold with timescale parameter Δ
- Parallel computation via global convolution: y = x ∗ K where K = (CB, CAB, ..., CA^(L-1)B)
- Input-specific parameterization for selectivity

### Training Setup
- Data augmentation: fuzziness, flipping, shifting, scaling, 180° rotation (p=0.5 each)
- Loss: Cross-Entropy
- Input resize: 224×224
- AdamW optimizer

## SFXRay Dataset (World's First Open-Source)

| Property | Value |
|----------|-------|
| Total images | 80 (selected from 122 collected) |
| Collection period | 2020-2023, Zhijin County, Guizhou |
| Device | UIH uDR 780i Pro |
| Body parts | Forearm, Pelvis, Calf |
| Class distribution | Normal: 21, Mild: 34, Moderate: 13, Severe: 12 |
| Annotation protocol | 3 specialists (>10yr SF diagnosis), independent → majority vote |
| Diagnostic standard | WS/T 192-2021 (China endemic SF standard) |
| Train/Test split | 56/24 (7:3) |
| Resolution | Forearm 2792×1224, Calf 2754×2537, Pelvis 2502×2984 |
| Bits stored | 12-bit DICOM |

## Key Results

### Binary Classification (Normal vs Abnormal)
| Model | Acc | Prec | Recall | F1 | AUC |
|-------|-----|------|--------|----|-----|
| **Mwinc-Mamba** | **83.33%** | **85.71%** | **90.00%** | **87.80%** | **0.88** |
| Vim-Tiny | 79.17% | — | — | — | — |
| VMamba-Tiny | 75.00% | — | — | — | — |
| ResNet-50 | 79.17% | — | — | — | — |
| Swin-Tiny | 75.00% | — | — | — | — |
| ViT-Base | 70.83% | — | — | — | — |

### Multi-Classification (4-Grade: Normal/Mild/Moderate/Severe)
| Model | Acc | Prec | Recall | F1 |
|-------|-----|------|--------|-----|
| **Mwinc-Mamba** | **66.67%** | **67.08%** | **66.67%** | **66.08%** |
| Vim-Tiny | 62.50% | — | — | — |
| VMamba-Tiny | 58.33% | — | — | — |
| ResNet-50 | 62.50% | — | — | — |
| Swin-Tiny | 54.17% | — | — | — |

### Human Baseline Comparison (Critical Finding)
- 5 radiologists (>5yr experience) graded the same 24 test images
- Average radiologist multi-class accuracy: **70.00%**
- Mwinc-Mamba accuracy: **66.67%** — only **3.33% difference**
- **No statistically significant difference** from radiologists in all evaluation metrics
- Radiologist inter-rater agreement: **poor** (Cohen's kappa mapped in heatmap)
- Key implication: model is more consistent than individual radiologists; ground truth uncertainty is inherent

## Relevance

**DIRECT COMPETITOR — same exact task (SF 4-grade X-ray diagnosis), public code + dataset.**

1. SFXRay is the benchmark for any SF DL research — we must compare against Mwinc-Mamba
2. Our SF dataset (80 X-rays, 59 masks, 4-grade) is directly comparable — can reproduce and improve
3. SSM-based architecture is a third paradigm beyond CNN/Transformer — should be in our search space
4. Poor radiologist agreement (documented with heatmap) is the strongest motivation yet for EDL uncertainty quantification
5. Mwinc-Mamba is the new SOTA baseline replacing Liu et al. 2021

## Key Takeaways for Method Design

1. **SSM backbone is validated for SF**: Mwinc-Mamba outperforms all CNN and Transformer baselines at lower complexity. Include Mamba variants in architecture search
2. **Multi-window cross-scan = critical for subtle lesions**: The MWCS mechanism solves the core SF challenge (diffuse, ill-defined lesions). Similar multi-scale attention should be in our design
3. **Target >70% multi-class accuracy**: Radiologist average is the clinical bar. Mwinc-Mamba at 66.67% leaves room for improvement — EDL + ordinal regression could bridge the 3.33% gap
4. **EDL is the natural next step**: The documented inter-rater disagreement directly validates EDL uncertainty modeling as our innovation vector over Mwinc-Mamba
5. **12-bit DICOM requires careful preprocessing**: Standard 8-bit JPEG conversion may lose subtle bone density information critical for SF grading
