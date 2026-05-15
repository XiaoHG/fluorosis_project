---
date: 2026-05-15
author: LitAgent
input_from: "02_Literature/01_initial_search/References/2404.06177-Uncertainty_Evidential_Fusion_SemiSeg.pdf"
output_to: "02_Literature/02_deep_read/11_Uncertainty_Evidential_Fusion_SemiSeg.md"
status: draft
---

# Uncertainty-aware Evidential Fusion-based Learning for Semi-supervised Medical Image Segmentation

## Metadata
| Field | Value |
|-------|-------|
| Authors | Yuanpeng He, Lijian Li |
| Venue | arXiv:2404.06177 (2024); precursor to MEDL (Paper 5) |
| Year | 2024 |

## Problem & Motivation
- Existing uncertainty-based semi-supervised medical segmentation methods typically rely on a **single uncertainty evaluation source** (e.g., only teacher network uncertainty, or only Monte Carlo dropout variance). This single-source approach fails to comprehensively capture prediction credibility -- it can lead to over-trusting biased data or systematically over/underestimating true uncertainty.
- Standard pseudo-labeling in semi-supervised learning suffers from **confirmation bias**: the model becomes overconfident in its own pseudo-labels and stops learning from challenging cases. When uncertainty from a single source is miscalibrated, the model may either trust incorrect pseudo-labels (false confidence) or discard useful information from correct ones (false uncertainty).
- Current methods treat uncertainty as a static filter (threshold-based pseudo-label selection) rather than as a dynamic signal to guide the learning curriculum. Uncertainty should determine not just WHAT to trust but also WHEN and HOW to learn from examples of different difficulty levels.
- This is the **foundational work** that later evolved into MEDL (Paper 5). It establishes the core framework of evidential fusion-based uncertainty estimation and voxel-wise curriculum learning, which MEDL extends to multi-network mutual learning.

## Method
- **Overall Framework**: Teacher-student architecture with two stages. Pre-training stage: initial network N learns from labeled data with mixed augmentation (copy-paste of foreground regions between samples). Self-training stage: student network N_s learns from both labeled and unlabeled data; teacher network N_t updated via Exponential Moving Average (EMA). Mixed samples M_l (labeled-only mixing) and M_ul (labeled+unlabeled mixing) are used.
- **Improved Probability Assignments Fusion (IPAF)**: The core innovation. Fuses evidential predictions from two sources (original sample prediction + mixed/restored sample prediction) at the same voxel position. The fusion formula: g_fused(C_n) = g_Ml(C_n) * g_Tl(C_n) + (|C_n|/|C_N|) * (g_Ml(C_n)*g_Tl(C_N) + g_Tl(C_n)*g_Ml(C_N)), where |C_n| = 1 and |C_N| = K (number of classes). The coefficient |C_n|/|C_N| = 1/K controls the degree of interaction between class-specific confidence and overall uncertainty mass. This is a modification of standard Dempster's rule that prevents uncertainty from dominating the fused probability while still allowing confidence-uncertainty interaction. The fused uncertainty for the multi-objective set: g_fused(C_N) = g_Ml(C_N) * g_Tl(C_N).
- **Generalized Probabilistic Framework**: Extends standard EDL by mapping model uncertainty to multi-objective subsets in DS theory. The probability assignment for a voxel: P = {{g(C_n)}_{n=0}^{K-1}, g(C_N)}, where C_N = {C_0, ..., C_{K-1}} represents the full frame of discernment. The uncertainty u = 1 - Σ g(C_n) is mapped to g(C_N), creating a direct correspondence between EDL uncertainty and DS theory's universal set. This enables principled evidence fusion operations.
- **Voxel-wise Asymptotic Learning (VWAL)**: A curriculum learning strategy. Voxels are sorted by their refined uncertainty in descending order within each sample. Refined uncertainty uses Shannon entropy weighted by the fused uncertainty mass: U(P_fused) = -g_fused(C_N) * Σ(δ_n log_2 δ_n) where δ_n = g_fused(C_n). This combines distributional entropy (from class probabilities) with evidential uncertainty (from belief mass). Dynamic weight function: φ(h, s(z)) = ε * Sigmoid(ζ(h) * ς(s(z))) where ζ(h) = 2h/H - 1 (epoch progress, -1→1), ς(s(z)) = 2s(z)/Z - 1 (voxel uncertainty rank, 1 for most uncertain). ε controls amplitude. Loss: L_w = (1/G) Σ Σ φ(h, s(z)_k) * L_{k,z}. Early epochs: model focuses on low-uncertainty voxels; late epochs: shifts to high-uncertainty voxels.
- **Loss Function**: Pre-training: L_ini = L_l1 + λ_1 L_w^{l1}. Self-training: L_Ns = L_l2 + L_u + λ_2 L_w^{l2} + λ_3 L_w^u. λ_1=0.8, λ_2=0.8, λ_3=0.4.

## Key Results
- **TBAD dataset (5% labeled)**: Dice 79.46 vs. UPCoL 76.19 (20% labeled) -- the method with 5% labeled data outperforms the previous SOTA using 20% labeled. At 10% labeled: Dice 79.99; at 20% labeled: Dice 82.67 vs UPCoL 76.19 (+6.48%).
- **Pancreas-CT (5% labeled)**: Dice 82.93 vs. BCP 80.33 (+2.60%). At 10%: Dice 83.54 vs. BCP 82.49. At 20%: Dice 84.73 vs. BCP 84.01.
- **LA dataset (5% labeled)**: Dice 90.50 vs. A&D 89.93. At 10%: Dice 90.98 vs. A&D 90.31. At 20%: Dice 91.86 vs. A&D 90.42 (+1.44%).
- **ACDC (5% labeled)**: Dice 89.29 vs. BCP 87.59 (+1.70%); Jaccard 81.29 vs. Co-BioNet 78.67 (+2.62%). At 10%: Dice 90.16 vs. BCP 88.84. At 20%: Dice 90.80 vs. Co-BioNet 89.51.
- **Ablation (LA, 5% labeled)**: Removing core losses has the most impact: without L_l1 Dice drops from 90.50 to 86.77; without L_u drops to 75.34. Among weighted losses on LA: removing L_w^{l2} causes largest drop (90.50→88.94); on TBAD: removing L_w^u causes largest drop (79.46→74.74), suggesting dataset-dependent importance of labeled vs. unlabeled uncertainty weighting.
- **λ sensitivity analysis (LA, 5% labeled)**: λ_1 optimal at 0.8; λ_2 optimal at 0.8; λ_3 optimal at 0.4. The lower optimal λ_3 suggests that uncertainty-guided learning on unlabeled data requires more conservative weighting than on labeled data -- excessive weight on unlabeled uncertainty degrades performance.

## Relevance to Our Project (氟中毒 DL 诊断 + EDL)
- **IPAF as a principled multi-annotator fusion alternative to majority voting**: Instead of taking the mode of 5 radiologist grades, IPAF fuses evidential predictions while the 1/K coefficient prevents uncertain annotators from dominating. The fused uncertainty naturally captures cases where annotators split (e.g., 3 say Dean 1, 2 say Dean 2).
- **VWAL directly addresses the fluorosis grading difficulty spectrum**: Dean 0 (clearly normal) and Dean 3 (clearly severe) are low-uncertainty cases that the model should master first. Dean 1 (questionable) and Dean 2 (mild) are where inter-rater disagreement concentrates -- higher uncertainty cases that VWAL automatically emphasizes later in training.
- **Confirmation bias prevention for fluorosis**: When majority voting produces a single Dean label, minority annotator uncertainty is lost. IPAF preserves this information and the VWAL curriculum prevents the model from becoming overconfident on ambiguous cases early in training.
- **The mixed augmentation strategy** (copy-pasting foreground regions between samples) could be adapted to mix fluorosis images of adjacent Dean grades, creating synthetic borderline cases that improve the model's ability to discriminate at class boundaries.
- **Dataset-dependent weighting insight**: The ablation shows that labeled and unlabeled data require different uncertainty weights (λ_1=0.8, λ_3=0.4). For fluorosis, the optimal balance between fully-labeled (all 5 annotators agree) and partially-labeled (only 1-2 annotators) cases will likely follow a similar pattern -- partially-labeled data needs more conservative uncertainty weighting.

## Key Takeaways for Method Design
1. **Fuse multi-annotator labels using IPAF, not majority voting**: Each radiologist's annotation becomes an evidential prediction with belief masses g_i(C_n). The IPAF formula fuses them while the 1/K coefficient prevents uncertainty dominance. Cases where 3 annotators say Dean 1 and 2 say Dean 2 will produce a soft distribution (≈60%/40%) with higher uncertainty -- exactly the desired behavior.
2. **Implement VWAL for fluorosis curriculum learning**: Sort training samples by their fused annotation uncertainty (from IPAF). Use φ(h, s(z)) = ε * Sigmoid(ζ(h)*ς(s(z))) to dynamically weight samples. Early training: high weight on samples where all 5 annotators agree on Dean 0 or 3; late training: high weight on samples where annotators split between Dean 1/2.
3. **Use Shannon entropy-refined uncertainty as the sample difficulty score**: U = -g_fused(C_K) * Σ(δ_n log_2 δ_n) combines both evidential uncertainty AND distributional entropy. This dual-source signal is more informative for identifying ambiguous fluorosis cases than either metric alone.
4. **Teacher-student with EMA for handling partially-labeled data**: In fluorosis datasets where not all images have 5 annotator labels, use the EMA-teacher to generate pseudo-labels for images with only 1-2 annotations. The VWAL uncertainty weight prevents these pseudo-labels from dominating training.
5. **Use more conservative uncertainty weighting for unlabeled/pseudo-labeled data** (λ_3=0.4 vs λ_1=0.8). For fluorosis, set lower uncertainty weights for images with partial annotations or lower inter-rater agreement, and higher weights for fully-annotated consensus cases.
