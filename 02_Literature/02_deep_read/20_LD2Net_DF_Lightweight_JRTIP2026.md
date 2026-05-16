---
date: 2026-05-17
author: LitAgent
input_from: "02_Literature/01_initial_search/References/s11554-026-01874-4.pdf"
output_to: "02_Literature/02_deep_read/20_LD2Net_DF_Lightweight_JRTIP2026.md"
status: draft
---

# LD2Net: Real-Time Lightweight Fluorosis Grading with Depthwise Separable Convolution and Dual-Axis Attentional Intelligence

## Metadata

| Field | Value |
|-------|-------|
| Authors | Minghan Li, Yun Wu, Zhihao Li, Chengdong Ye |
| Venue | Journal of Real-Time Image Processing 23:85 (2026), Springer |
| Year | 2026 (received Dec 2025, accepted Mar 2026) |
| Affiliation | State Key Lab of Public Big Data + College of CS, Guizhou University |
| DOI | 10.1007/s11554-026-01874-4 |
| Code/Dataset | Uses DFID dataset (same as MLTrMR) |
| Category | A1 — 氟斑牙 DL 诊断 (NEW, Tier 1) |

## Problem & Motivation

- Clinical DF diagnosis relies on manual assessment: high subjectivity, low efficiency
- Severe fluorosis regions are in economically underdeveloped remote areas with limited healthcare
- DL research on DF is "in its infancy" — only recent papers emerging (2024+)
- Existing methods (MLTrMR) use heavy models (556M params); need lightweight solution for real clinical deployment
- Transfer learning from large-scale general datasets (ImageNet-1000) used to address 200-sample data scarcity

## Method

### LD2Net Architecture

Three main components:

**1. Transfer Learning Backbone**
- Knowledge transferred from MobileNetV2 pretrained on ImageNet-1000
- MobileNetV2 chosen because it also uses depthwise separable convolutions → architectural compatibility
- General visual features (texture, color) from source domain enhance feature extraction on DFID

**2. SDW (Strengthened Depthwise Separable Convolution) Module**
- Replaces standard convolution with depthwise separable convolution (Params ↓ by factor of 1/C_out + 1/K²)
- Enhanced with pointwise convolution for channel expansion, SE (Squeeze-and-Excitation) submodule for Z-axis (channel) spatial modeling
- Pointwise + Depthwise + SE → richer features than vanilla depthwise separable

**3. DAIA (Dual-Axis Integrated Attention) Module**
- Mimics dentist diagnostic logic: observing tooth surface structure and lesions
- Processes spatial features along both horizontal and vertical axes
- Concatenates dual-axis information → convolution → non-linear transform → spatial weighting
- Significantly enhances recognition of fluorosis-affected areas

### Loss Function
Standard cross-entropy (CE) loss. No ordinal, no uncertainty components.

## DFID Dataset (Same as MLTrMR)

Same 200-image, 4-grade (Normal/Mild/Moderate/Severe), 50/class dataset. Train/Test split likely 140/60 (same as MLTrMR).

## Key Results

### SOTA Comparison (Table 2)
| Model | Year | Acc (%) | F1 (%) | Params (M) | FLOPs |
|-------|------|---------|--------|------------|-------|
| ResNet-50 | 2016 | 72.12 | 68.99 | 24 | 20.9G |
| DenseNet-169 | 2017 | 73.59 | 71.36 | 12 | 17.1G |
| ViT-B/16 | 2017 | 44.05 | 47.51 | 86 | 287.2M |
| Swin-S | 2021 | 51.72 | 47.17 | 49 | 75.0M |
| HiFuse | 2024 | 78.23 | 70.45 | 164 | 24.1G |
| FusionDentNet | 2024 | 80.00 | 79.25 | 201 | 40.6G |
| **RMLT-H (MLTrMR best)** | 2025 | **80.19** | 75.79 | 556 | 21.3G |
| **LD2Net** | 2026 | **80.00** | **79.88** | **3.31** | **917.11M** |

### Lightweight Model Comparison (Table 3)
| Model | Year | Acc (%) | F1 (%) | Params (M) |
|-------|------|---------|--------|------------|
| ShuffleNetV2_x1.5 | 2018 | 78.33 | 78.04 | 2.48 |
| EfficientNetB0 | 2019 | 70.00 | 68.69 | 4.01 |
| GhostNetV1 | 2020 | 75.00 | 72.99 | 3.91 |
| **LD2Net** | 2026 | **80.00** | **79.88** | **3.31** |

### Ablation Findings
- Channel expansion ratio: ×4 optimal (80.00%), ×2 drops to 66.67%, ×6 drops to 76.67%
- Transfer learning alone: 73.33% (from 66.67% baseline MobileNetV2)
- + SDW: 75.00%
- + DAIA (full LD2Net): 80.00%

### Fivefold Cross-Validation
Performed (Fig 14) — stability confirmed. Specific values in figure (image, not extractable from PDF text).

### Error Analysis
- Mild fluorosis → Normal: early-stage lesions extremely subtle
- Normal → Moderate: insufficient image brightness + tooth surface pigmentation interference
- All errors attributed to insufficient image contrast

## Relevance

**DIRECT COMPETITOR — same task, same DFID dataset, same research group as MLTrMR.**

1. LD2Net = lightweight complement to MLTrMR: same group's answer to "MLTrMR too heavy (556M)"
2. Acc 80.00% ties MLTrMR 80.19% with **168× fewer params** (3.31M vs 556M)
3. Our EDL (83.33%) surpasses both on accuracy while providing uncertainty quantification
4. LD2Net + EDL: complementary directions — lightweight deployment vs. reliable diagnosis

## Key Takeaways for Our Paper

1. **80.00% Acc / 79.88% F1 is the lightweight SOTA** — cite alongside MLTrMR
2. **ViT-B/16 = 44.05% in their hands** vs our ViT CE = 81.67% — pretraining strategy difference is a major discussion point (ImageNet-21k HuggingFace vs likely from-scratch)
3. **Channel expansion ablation** — systematic hyperparameter tuning we should emulate for evidence head
4. **Transfer learning: +6.7pp** — confirms pretraining critical for 200-sample DFID
5. **Lightweight is NOT our competition** — they target deployment efficiency; we target diagnostic reliability + uncertainty. Complementary framing in related work
6. **Same Wu group: MLTrMR + LD2Net + Mwinc-Mamba** — building fluorosis AI portfolio. Our EDL fills the "uncertainty-aware diagnosis" gap
7. **No QWK/ECE/Unim reported** — our systematic 7-metric evaluation is a methodological contribution over both MLTrMR and LD2Net
