---
date: 2026-05-15
author: LitAgent
input_from: "02_Literature/01_initial_search/References/Xu 等 - 2025 - Masked latent transformer with random masking ratio to advance the diagnosis of dental fluorosis.pdf"
output_to: "02_Literature/02_deep_read/17_MLTrMR_DF_Diagnosis_JCVI2025.md"
status: draft
---

# MLTrMR: Masked Latent Transformer for Dental Fluorosis Automated Diagnosis

## Metadata

| Field | Value |
|-------|-------|
| Authors | Hao Xu, Yun Wu, Junpeng Wu, Rui Xie, Maohua Gu, Rongpin Wang |
| Venue | J. Visual Communication and Image Representation 111 (2025) 104496 (Elsevier) |
| Year | 2025 (accepted May, online June 2025) |
| Affiliation | Guizhou University + Zhijin County People's Hospital + Guizhou Provincial People's Hospital |
| Code/Dataset | https://github.com/uxhao-o/MLTrMR |
| Category | A1 — 氟斑牙 DL 诊断 (NEW, Tier 1) |

## Problem & Motivation

- DF diagnosis relies on dentists' visual assessment. Study shows: non-physicians cannot distinguish DF from normal; non-dentists struggle with mild vs moderate; even dentists may misdiagnose (documented with 20/10/10 participant survey)
- Only 5 prior studies on automated DF diagnosis (as of April 2025): 4 use traditional ML (fuzzy C-means, K-means, KNN clustering on color features), only 1 prior DL work (Xu et al.'s own two-stage U-Net+classifier from their previous study [7])
- No open-source DF image dataset existed — all prior work used private data
- Prior two-stage method (U-Net dental segmentation + dual-branch CNN/Transformer classifier) is complex; need end-to-end solution

## Method

### MLTrMR: Masked Latent Transformer with Random Masking Ratio

**Three-component architecture:**

#### Latent Embedder
- CNN backbone (ResNet/DenseNet w/o FC layer) → Adaptive Avg Pool → Embedding → **latent tokens** z_le ∈ R^(1×D)
- Extracts high-level lesion features serving as semantic conditions to guide Transformer attention
- **Hot-pluggable**: supports independent pre-training or end-to-end joint training

#### Encoder (L_e LT Blocks)
- Input image → patch embedding → **random masking** (ρ ∈ [0.3, 0.8]) → unmasked tokens + latent tokens processed
- **Random Masking Ratio**: adaptively selected per batch — small lesions get lower masking (preserve local details); extensive lesions get higher masking (force global inference)
- **LT Block** (Latent Transformer): replaces LayerNorm with **adaLN** (adaptive LN) — regresses γ, β, α from latent tokens for conditional guidance
- **RelPos-MSA**: multi-head self-attention with learnable relative position biases B_r

#### Decoder (L_d LT Blocks)
- Encoder output + randomly initialized masked tokens → unshuffled → processed with latent token guidance
- **Masked Shortcut**: before/after decoder, ensures prediction from unmasked tokens only
- Predicts masked tokens from visible context — forces learning lesion characteristics from partial information

### Auxiliary Loss (Key Innovation)
```
L = CE Loss(y', y) + λ · MSE(x', x)
```
where x' is decoder output reshaped to match original image dimensions. Binds **structural consistency** between feature map and original image. Unlike MAE's local patch reconstruction, this is **global feature map reconstruction** — encourages cross-regional semantic associations (infer overall severity from few visible patches).

### Model Variants
MLTrMR-B (Base), MLTrMR-S, MLTrMR-L, MLTrMR-H — explore depth, hidden dim, MLP ratio, heads

## DFID Dataset (World's First Open-Source DF Dataset)

| Property | Value |
|----------|-------|
| Total images | 200 (50/class × 4 classes) |
| Classes | Normal, Mild, Moderate, Severe |
| Resolution | 560 × 448 pixels |
| Participants | 200 (80 children, 70 adults, 50 elderly) from Guizhou |
| Camera | Optical camera (intraoral photography) |
| Annotation | Professional dentists, 4-grade classification |
| Train/Test | 140/60 (7:3), stratified per class |
| Ethics | Declaration of Helsinki, Chinese Medical Ethics Committee, anonymized |

## Key Results

### SOTA Comparison (MLTrMR-B)
| Model | Acc (%) | F1 (%) | QWK (%) |
|-------|---------|--------|---------|
| **MLTrMR (ResNet backbone)** | **80.19** | **75.79** | **81.28** |
| MLTrMR (DenseNet backbone) | 79.34 | 75.28 | 80.01 |
| ResNet-50 | 72.28 | 68.79 | 78.84 |
| DenseNet-121 | 77.28 | 73.31 | 74.82 |
| ViT-Base | 74.30 | 71.71 | 79.38 |
| Swin-Tiny | 76.21 | 72.42 | 79.55 |
| MAE | 74.65 | 70.05 | 79.03 |

### Ablation Findings
- Removing auxiliary loss: ↓ ~2% Acc, ↓ ~3% F1 — structural consistency constraint is important
- Fixed masking (0.5) vs random (0.3-0.8): random ~1.5% Acc gain — adaptivity matters
- Without latent embedder (standard ViT): significant degradation — adaLN guidance from latent tokens is essential
- Pre-trained CNN backbone vs joint training: comparable results, joint training sufficient

### Model Efficiency
- MLTrMR-B: lower FLOPs than ViT-B and Swin-T with better accuracy — best accuracy-efficiency trade-off

## Relevance

**DIRECT COMPETITOR — identical task, IDENTICAL dataset specs to our DF project!**

1. DFID: 200 images, 50/class, 4-grade, intraoral photos — **exactly matches our DF dataset structure** (200 images, 50/class, Dean Index 0-3)
2. MLTrMR (80.19% Acc, 81.28% QWK) is the **DF grading baseline we must surpass**
3. Same Guizhou University group as Mwinc-Mamba (SF) — they cover both diseases we study
4. DFID can serve as **external validation set** for our model's generalization
5. The 19.81% error concentrates at Mild-Moderate boundary — exactly where EDL + ordinal regression can help

## Key Takeaways for Method Design

1. **80.19% Acc / 81.28% QWK is the target to beat** for 4-grade DF classification. Adding EDL uncertainty + ORCU ordinal loss should push past this baseline
2. **Single-stage > two-stage**: MLTrMR outperforms the authors' prior U-Net+classifier pipeline — validates end-to-end design
3. **Random masking ratio (0.3-0.8) > fixed masking**: Adaptive masking preserves local details for small lesions while forcing global reasoning for extensive ones — directly applicable to our training
4. **Latent tokens from CNN backbone are critical**: CNN features guiding Transformer attention is an elegant way to inject inductive bias — could enhance our ViT-based architectures
5. **Auxiliary reconstruction loss provides useful signal**: MSE between feature map and original image — consider similar structural consistency constraint in our loss design
6. **EDL opportunity at Mild-Moderate boundary**: The confusion matrix shows this is the hardest distinction — EDL uncertainty quantification can flag ambiguous cases and improve calibration
