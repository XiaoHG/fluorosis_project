---
date: 2026-05-15
author: LitAgent
input_from: "02_Literature/01_initial_search/References/2405.14444-DuEDL_Scribble_Segmentation.pdf"
output_to: "02_Literature/02_deep_read/12_DuEDL_Scribble_Segmentation.md"
status: draft
---

# DuEDL: Dual-Branch Evidential Deep Learning for Scribble-Supervised Medical Image Segmentation

## Metadata
| Field | Value |
|-------|-------|
| Authors | Yitong Yang, Xinli Xu, Haigen Hu, Haixia Long, Qianwei Zhou, Qiu Guan |
| Venue | arXiv:2405.14444 (Zhejiang University of Technology) |
| Year | 2024 |

## Problem & Motivation
- Medical image segmentation requires expensive pixel-level annotations from experts. Scribble-based weak supervision reduces annotation cost but produces less robust and generalizable models.
- Evidential deep learning (EDL) can quantify predictive uncertainty but, when applied naively to scribble supervision, incurs an accuracy-reliability tradeoff: unlabeled pixels learn incorrect evidence.
- No prior work has combined EDL with scribble-supervised medical image segmentation.

## Method
- **Backbone**: U-Net with encoder-decoder structure. The decoder is extended to two branches (Decoder1 main, Decoder2 auxiliary; Decoder2 adds a dropout layer before input with rate=0.5, otherwise shares parameters).
- **Evidence representation**: Softplus activation (replacing Softmax) produces non-negative class-wise evidence e. Dirichlet distribution parameters alpha = e + 1. Belief mass b and uncertainty u are derived per pixel via Dempster-Shafer theory.
- **Partial evidence loss (Lp-EDL)**: LpMSE + lambda_t * LpKL computed ONLY on scribble-annotated pixels, preventing the model from learning incorrect evidence on unlabeled regions. Annealing coefficient lambda_t = min(1, t/T).
- **Dual-branch evidence fusion**: Belief masses and uncertainties from both branches are fused using Dempster's combination rule. The fused prediction generates hard pseudo-labels (argmax) for unlabeled pixels.
- **Evidence consistency loss (LECL)**: Dice loss between fused hard pseudo-labels and each branch's Dirichlet prediction probabilities.
- **Total loss**: L_total = L_s + lambda_u * LECL, where L_s averages Lp-EDL over both branches plus the fused branch.

## Key Results
- **ACDC clean data**: Dice = 0.875, ASSD = 2.34 (2nd best among WSL methods after ScribbleVC at 0.881, but best ASSD). Fully supervised DMPLSF: Dice = 0.912.
- **Robustness (Gaussian blur sigma=0.1)**: DuEDL Dice = 0.797 vs DMPLSF Dice = 0.626 -- DuEDL OUTPERFORMS the fully supervised model under distribution shift.
- **OOD generalization (ACDC→MSCMRseg)**: DuEDL Dice = 0.582, ASSD = 5.76 (beats fully supervised DMPLSF at ASSD=5.83). Next best WSL: ScribbleVC Dice = 0.502.
- **Uncertainty quality**: UEO = 0.802 (highest among all WSL methods), ECE = 0.014 (2nd best, close to ScribbleVC at 0.003).
- **Ablation**: Removing evidence fusion drops Dice from 0.875 to 0.841 and severely degrades robustness (sigma=0.1: Dice 0.706 vs 0.797). Partial evidence loss alone improves OOD Dice (0.310→0.472) but requires fusion for full benefit.

## Relevance to Our Project (氟中毒 DL 诊断)
- **Weak supervision for fluorosis segmentation**: Our intraoral photos have no pixel-level masks. Scribble-based annotation (drawing rough boundaries of fluorosis-affected enamel regions) is far more feasible. DuEDL shows this can work with good accuracy.
- **Uncertainty quantification is critical for clinical deployment**: When classifying DF severity grades, knowing when the model is uncertain can trigger a "refer to specialist" pathway. DuEDL provides per-pixel uncertainty maps -- we can adapt this for per-image classification uncertainty.
- **Robustness under image quality variation**: Intraoral photos vary in lighting, angle, and focus. DuEDL's strong performance under Gaussian blur (sigma up to 0.15) suggests dual-branch EDL is robust to acquisition noise -- directly relevant to our non-standardized photo collection.
- **Small-dataset compatible**: DuEDL was trained on 70 patients (ACDC). Our 200 intraoral photos are a similar scale. The partial evidence loss is explicitly designed to maximize learning from sparse annotations.

## Key Takeaways for Method Design
1. **Dual-branch decoder with evidence fusion is a lightweight robustness booster**: Adding a dropout-augmented second decoder branch costs minimal parameters but significantly improves OOD generalization and noise robustness. This pattern can be added to any U-Net-style backbone.
2. **Partial loss (compute only on labeled pixels) is essential for weak supervision**: Standard EDL loss applied to all pixels degrades performance under scribble supervision. The Lp-EDL trick is simple but critical -- analogous to how we would only supervise on clinically annotated regions of intraoral photos.
3. **Hard pseudo-labels from fused evidence maintain branch independence**: Using argmax-hardened pseudo-labels for the consistency loss cuts the gradient between branches, preserving diversity. This is superior to soft consistency.
4. **EDL provides dual output (prediction + uncertainty)**: Every inference produces both segmentation and uncertainty. For clinical DF grading, this means we can output both a grade prediction and a confidence score per image -- useful for triage.
5. **Evidence fusion via Dempster's rule is a principled alternative to averaging**: For our multi-view or multi-crop inference, fusing evidence from different views using Dempster's rule may yield more reliable predictions than simple probability averaging.
