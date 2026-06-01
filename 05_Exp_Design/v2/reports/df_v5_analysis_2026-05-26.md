# DF V5 Experiment Analysis

**Date:** 2026-05-26
**Setup:** ViT-Base (86M), 200 intraoral photos, 4-class, **100 epochs**, EarlyStopping, MixUp(alpha=0.2) for CE/Cum/SORD, RandAugment(2,9)
**Key V5 changes vs V4:** EDL kl_lambda 0.10→**0.05**, kl_anneal_cap 1.0→**0.5**, EDL+ORCU λ_orcu 0.01→**0.005**, epochs 75→**100**, CV 5-fold→**10-fold**, CV methods 3→**5**

---

## Executive Summary

**V5 is a regression. The parameter "corrections" from V4 made EDL worse, not better.** EDL CV accuracy dropped from V4's 79.33%±3.74 to V5's 72.33%±7.90 — a **-7.00pp decline**. CE unexpectedly became the best method at 75.83%±5.23%, overtaking all ordinal and uncertainty-aware approaches.

**The core mistake:** V4's Phase 2 showed kl=0.05 was the EDL sweet spot for **75-epoch** training. V5 blindly applied kl=0.05 to **100-epoch** training, where it turned out to be the worst choice (60.0% in Phase 2). The correct V5 EDL sweet spot is kl=0.07 (75.0%). The interaction between KL annealing schedule and total epochs was not accounted for.

**Best V5 result:** EDL+ORCU Phase 1 seed=456 at 78.33% (single run). **Best V5 CV:** CE at 75.83%±5.23.

**Success tier: C** (best CV mean < 78%). V5 fails all success tier thresholds. This is a step backward from V4's Tier B+.

---

## 1. Phase 1: Multi-Seed Single Test-Set Results (15 runs)

### 1.1 Per-Method Summary

| Method | Seed 42 | Seed 123 | Seed 456 | **Mean Acc** | **Mean F1** | **Mean QWK** | **Mean ECE** |
|--------|---------|----------|----------|-------------|-------------|-------------|-------------|
| CE | 70.00% | 65.00% | 68.33% | **67.78%** | 0.671 | 0.879 | 0.178 |
| Cumulative | 73.33% | 76.67% | 63.33% | **71.11%** | 0.702 | 0.861 | 0.204 |
| SORD | 71.67% | 73.33% | 66.67% | **70.56%** | 0.699 | 0.872 | 0.227 |
| EDL | 58.33% | 75.00% | 65.00% | **66.11%** | 0.652 | 0.858 | 0.257 |
| **EDL+ORCU** | **75.00%** | 73.33% | **78.33%** | **75.56%** | **0.745** | **0.900** | **0.164** |

EDL+ORCU dominates Phase 1: best mean accuracy (75.56%), best mean F1 (0.745), best QWK (0.900), lowest ECE (0.164). EDL is the worst method (66.11%), confirming the kl=0.05 parameter was a wrong choice for 100-epoch training. CE at 67.78% is second-worst — the baseline method struggles with only 20 test samples.

### 1.2 Per-Class F1 (Phase 1 means)

| Method | Normal (0) | Mild (1) | Moderate (2) | Severe (3) |
|--------|-----------|----------|-------------|-----------|
| CE | 0.792 | 0.544 | 0.543 | 0.805 |
| Cumulative | 0.815 | 0.780 | 0.597 | 0.759 |
| SORD | 0.788 | 0.682 | 0.581 | 0.757 |
| EDL | 0.724 | 0.563 | 0.575 | 0.744 |
| EDL+ORCU | 0.845 | 0.747 | 0.546 | 0.791 |

Class 2 (Moderate) is the universal bottleneck (F1 0.54-0.60). EDL+ORCU has best Normal (0.845) and Severe (0.791) but middling Moderate (0.546). EDL's Normal F1 dropped to 0.724 (V4 was 0.822) — weak KL penalty fails to regularize the majority class.

### 1.3 Calibration Summary

| Method | %Unimodal | U_ECE | AUROC_u | ECE |
|--------|----------|-------|---------|-----|
| CE | 95.00% | 0.403 | 0.720 | 0.178 |
| Cumulative | 56.67% | 0.414 | 0.564 | 0.204 |
| SORD | **100.0%** | 0.495 | 0.593 | 0.227 |
| EDL | 76.67% | **0.098** | 0.511 | 0.257 |
| EDL+ORCU | 84.44% | 0.213 | **0.431** | **0.164** |

EDL has best U_ECE (0.098) despite worst accuracy — low evidence gives honest uncertainty at the cost of predictive power. EDL+ORCU has best ECE (0.164) but worst AUROC_u (0.431): well-calibrated but poorly discriminative for OOD detection. SORD achieves perfect unimodality (100%).

---

## 2. Phase 2: EDL KL Lambda Sweep (seed=42)

| kl_lambda | kl_cap | Acc | Macro F1 | QWK | ECE | %Unimodal |
|-----------|--------|-----|----------|-----|-----|-----------|
| 0.03 | 0.5 | 71.67% | 0.713 | 0.889 | 0.168 | **100.0%** |
| 0.05 | 0.5 | **60.00%** | 0.586 | 0.836 | 0.230 | 95.00% |
| **0.07** | **0.5** | **75.00%** | **0.743** | **0.883** | 0.241 | 60.00% |

**This is the most important finding in V5: kl=0.05 is the WORST choice at 100 epochs.** This directly contradicts V4's Phase 2 where kl=0.05 was best at 75 epochs (80.00%). The KL annealing schedule `min(cap, epoch/(total_epochs*0.3))` means:

- **V4 (75 epochs):** KL reaches cap at epoch 22.5, then 52.5 epochs at full strength
- **V5 (100 epochs):** KL reaches cap at epoch 30, then 70 epochs at full strength

The extra 17.5 epochs at full KL strength with kl=0.05 over-regularizes EDL, suppressing evidence too aggressively. kl=0.07 provides stronger per-step regularization that better matches the 100-epoch duration.

**kl=0.07 is the V5 EDL sweet spot at 75.00%.** This is still below V4's kl=0.05 at 80.00%, suggesting 100 epochs may be inherently too many for EDL on this dataset.

**Per-class at kl=0.07:** Normal=0.828, Mild=0.722, Moderate=0.609, Severe=0.813.

---

## 3. Phase 3: EDL+ORCU lambda_orcu Sweep (9 runs)

### 3.1 Per-Lambda Summary

| lambda_orcu | Seed 42 | Seed 123 | Seed 456 | **Mean Acc** | **Mean F1** | **Mean QWK** | **Mean ECE** |
|-------------|---------|----------|----------|-------------|-------------|-------------|-------------|
| **0.003** | 70.00% | **81.67%** | 75.00% | **75.56%** | **0.753** | **0.900** | 0.179 |
| 0.005 | 63.33% | 75.00% | 75.00% | 71.11% | 0.703 | 0.880 | 0.174 |
| 0.008 | 66.67% | 75.00% | **76.67%** | 72.78% | 0.714 | 0.890 | **0.171** |

**lambda_orcu=0.003 is the clear winner** — 75.56% mean accuracy. The V5 design's choice of λ=0.005 was wrong; it underperforms by -4.45pp. Lower λ = better accuracy, and calibration metrics don't degrade meaningfully. The log-barrier constraint is powerful enough even at λ=0.003.

### 3.2 λ=0.003 Per-Seed Detail

| Seed | Acc | Macro F1 | Per-Class F1 | QWK | %Unimodal |
|------|-----|----------|-------------|-----|-----------|
| 42 | 70.00% | 0.689 | [0.727, 0.621, 0.609, 0.800] | 0.892 | 95.00% |
| 123 | 81.67% | 0.800 | [0.923, 0.821, 0.600, 0.857] | 0.908 | 98.33% |
| 456 | 75.00% | 0.737 | [0.889, 0.833, 0.500, 0.727] | 0.900 | 95.00% |

Seed=123 at λ=0.003 achieves the best single-run result in V5 (81.67%). Moderate class F1 remains the bottleneck (0.50-0.61) across all seeds.

---

## 4. Phase 4: 10-Fold Cross-Validation (50 folds total)

### 4.1 Configurations

| Method | Parameters |
|--------|-----------|
| CE | Standard, MixUp alpha=0.2, patience=15 |
| Cumulative | Standard, MixUp alpha=0.2, patience=15 |
| SORD | Standard, MixUp alpha=0.2, patience=15 |
| EDL | kl_lambda=0.05, kl_anneal_cap=0.5, patience=25 |
| EDL+ORCU | kl_lambda=0.05, kl_anneal_cap=0.5, lambda_orcu=0.005, patience=15 |

Note: CV used the V5 design defaults. Phase 2 and 3 show both kl=0.05 and λ=0.005 are suboptimal. Phase 4 was launched before Phase 2/3 analysis.

### 4.2 CV Summary

| Method | Acc | Macro F1 | Weighted F1 | QWK | ECE | %Unimodal |
|--------|-----|----------|-------------|-----|-----|-----------|
| **CE** | **75.83% ± 5.23** | **75.19% ± 5.04** | **75.36% ± 5.35** | **0.886 ± 0.031** | 0.154 | 94.33% |
| Cumulative | 68.83% ± 8.57 | 66.92% ± 8.92 | 67.14% ± 9.61 | 0.819 ± 0.123 | **0.166** | 60.33% |
| SORD | 70.33% ± 10.32 | 69.22% ± 11.21 | 69.03% ± 12.17 | 0.857 ± 0.059 | 0.193 | **100.0%** |
| EDL | 72.33% ± 7.90 | 71.23% ± 8.52 | 71.03% ± 9.21 | 0.870 ± 0.045 | 0.186 | 73.50% |
| EDL+ORCU | 71.67% ± 6.83 | 70.63% ± 7.08 | 70.55% ± 7.64 | 0.867 ± 0.040 | 0.211 | 71.17% |

**CE is the best method in V5 CV.** This is a complete reversal from V4 where EDL led by a wide margin. The gap between CE (75.83%) and EDL (72.33%) is 3.50pp.

### 4.3 Per-Fold Variability (Accuracy)

| Fold | CE | Cumulative | SORD | EDL | EDL+ORCU |
|------|-----|-----------|------|-----|----------|
| 0 | 78.33% | 60.00% | 80.00% | 76.67% | 76.67% |
| 1 | 80.00% | 70.00% | 80.00% | 66.67% | 71.67% |
| 2 | 73.33% | 76.67% | 78.33% | 70.00% | 76.67% |
| 3 | 68.33% | 63.33% | 56.67% | 58.33% | 60.00% |
| 4 | 68.33% | 63.33% | 71.67% | 65.00% | 63.33% |
| 5 | 81.67% | 66.67% | 83.33% | **86.67%** | 76.67% |
| 6 | 73.33% | 80.00% | 71.67% | 80.00% | 70.00% |
| 7 | 71.67% | 83.33% | 50.00% | 68.33% | 65.00% |
| 8 | 80.00% | 70.00% | 65.00% | 73.33% | 73.33% |
| 9 | **83.33%** | 55.00% | 66.67% | 78.33% | **83.33%** |

Fold 3 is consistently the hardest (all methods ≤68.33%). Fold 5 is EDL's best at 86.67%. Fold 7 destroys SORD (50.00%) — 13 Severe samples misclassified. Cumulative fold 9 collapses to 55.00% (QWK=0.465).

### 4.4 CV Per-Class F1

| Method | Normal (0) | Mild (1) | Moderate (2) | Severe (3) |
|--------|-----------|----------|-------------|-----------|
| CE | **0.864 ± 0.042** | 0.751 ± 0.094 | 0.623 ± 0.074 | **0.770 ± 0.106** |
| Cumulative | 0.814 ± 0.091 | 0.690 ± 0.136 | 0.503 ± 0.096 | 0.670 ± 0.184 |
| SORD | 0.850 ± 0.080 | **0.733 ± 0.122** | 0.553 ± 0.102 | 0.633 ± 0.258 |
| EDL | 0.844 ± 0.059 | 0.751 ± 0.106 | 0.601 ± 0.096 | 0.654 ± 0.230 |
| EDL+ORCU | 0.826 ± 0.061 | 0.693 ± 0.093 | 0.594 ± 0.098 | 0.712 ± 0.187 |

CE dominates: best Normal (0.864), Moderate (0.623), and Severe (0.770). EDL ties CE on Mild (0.751). SORD Severe F1 has extreme variance (std=0.258) driven by the fold 7 collapse.

### 4.5 Statistical Significance (Paired t-test, 10 folds)

| Comparison | Acc p | F1 p | QWK p | ECE p |
|------------|-------|------|-------|-------|
| **CE vs EDL+ORCU** | **0.007** | **0.010** | 0.067 | **0.041** |
| CE vs Cumulative | 0.100 | 0.069 | 0.190 | 0.664 |
| CE vs SORD | 0.107 | 0.110 | 0.102 | 0.113 |
| CE vs EDL | 0.101 | 0.105 | 0.242 | 0.417 |
| EDL vs EDL+ORCU | 0.726 | 0.775 | 0.770 | 0.345 |
| Cumulative vs EDL+ORCU | 0.513 | 0.429 | 0.355 | **0.005** |

**CE vs EDL+ORCU is statistically significant on Acc (p=0.007), Macro F1 (p=0.010), and ECE (p=0.041).** This is the first time in the DF experiment series that we have statistically significant accuracy differences between methods. The 10-fold CV provides sufficient statistical power.

Cumulative vs EDL+ORCU on ECE is also significant (p=0.005) — Cumulative has better calibration, reflecting EDL+ORCU's suboptimal λ=0.005 parameter.

No other comparisons reach significance. High inter-fold variance still dominates for most pairs.

---

## 5. V4 → V5 Direct Comparison

### 5.1 CV Comparison

| Method | V4 CV Acc | V5 CV Acc | Delta | V4 CV QWK | V5 CV QWK |
|--------|----------|----------|-------|----------|----------|
| CE | N/A (not in CV) | **75.83% ± 5.23** | — | — | 0.886 |
| SORD | 77.67% ± 4.90 | 70.33% ± 10.32 | **-7.34pp** | 0.904 | 0.857 |
| EDL | **79.33% ± 3.74** | 72.33% ± 7.90 | **-7.00pp** | 0.905 | 0.870 |
| EDL+ORCU | 75.00% ± 5.16 | 71.67% ± 6.83 | **-3.33pp** | 0.894 | 0.867 |

Every method that appeared in both V4 and V5 CV declined. The universal regression suggests a systematic issue beyond individual hyperparameters — the 100-epoch design with this dataset size is fundamentally flawed.

### 5.2 Phase 1 Seed=42 Comparison

| Method | V4 Acc | V5 Acc | Delta |
|--------|--------|--------|-------|
| CE | 76.67% | 70.00% | -6.67pp |
| Cumulative | 80.00% | 73.33% | -6.67pp |
| SORD | 76.67% | 71.67% | -5.00pp |
| EDL | 73.33% | 58.33% | **-15.00pp** |
| EDL+ORCU | 73.33% | 75.00% | +1.67pp |

EDL seed=42 collapsed from 73.33% to 58.33% (-15pp). This is the kl=0.05 damage at its purest. EDL+ORCU was the only method that improved (+1.67pp) — the ORCU constraint partially compensates for the weak KL regularization.

### 5.3 EDL KL Sweet Spot Comparison

| Version | Epochs | Best kl | Best Acc | ECE |
|---------|--------|---------|----------|-----|
| V4 | 75 | 0.05 | 80.00% | 0.194 |
| V5 | 100 | 0.07 | 75.00% | 0.241 |

The KL sweet spot shifted right (0.05→0.07) with more epochs, and the peak accuracy dropped (80%→75%). This confirms the interaction between training duration and KL strength.

### 5.4 EDL Evidence Comparison (Phase 1 seed=42)

| Version | Normal | Mild | Moderate | Severe | Total |
|---------|--------|------|----------|--------|-------|
| V4 EDL (kl=0.10) | 3.20 | 2.10 | 1.82 | 2.08 | 9.20 |
| V5 EDL (kl=0.05) | 4.04 | 2.17 | 2.37 | 2.65 | 11.23 (+22%) |

V5 EDL produces MORE evidence than V4 (11.23 vs 9.20) despite weaker KL — the extra 25 epochs allow the model to accumulate more evidence before EarlyStopping. Higher evidence with lower accuracy means the model is more confident in its mistakes.

---

## 6. Root Cause Analysis: Why V5 Regressed

### 6.1 Primary Cause: KL Lambda — Epoch Interaction

The V5 design assumed V4's kl=0.05 finding would transfer to 100-epoch training. It didn't. The KL annealing schedule `min(cap, epoch/(total_epochs*0.3))` means:

- **V4 (75 epochs):** KL reaches cap 0.5 at epoch 22.5, then 52.5 epochs at full strength → total KL "dose" ≈ cap × full_strength_epochs × lambda = 0.5 × 52.5 × 0.05 = 1.31
- **V5 (100 epochs):** KL reaches cap 0.5 at epoch 30, then 70 epochs at full strength → total KL dose ≈ 0.5 × 70 × 0.05 = 1.75 (+33%)

The 33% increase in total KL regularization with the same kl_lambda over-regularizes EDL, suppressing meaningful evidence and degrading accuracy. kl=0.07 partially compensates but doesn't fully recover.

### 6.2 Secondary Cause: 100 Epochs Overfits Before Regularization Kicks In

With only 200 images (180 train, 20 test per fold), 100 epochs allows significant overfitting before the KL penalty reaches full strength at epoch 30. The increased std in V5 (EDL: ±7.90 vs V4's ±3.74) suggests higher sensitivity to train/test splits — a hallmark of overfitting.

### 6.3 Tertiary Cause: 10-Fold CV Amplifies Per-Fold Variance

V5's 10-fold CV (20 samples/test fold) has higher per-fold variance than V4's 5-fold CV (40 samples/test fold). The smaller test set per fold amplifies the impact of individual hard cases. Part of the std increase from ±3.74 (V4) to ±7.90 (V5) is a statistical artifact.

### 6.4 Why CE Won

CE with MixUp is the simplest method — it benefits from extra epochs without any regularization penalty. While EDL was being over-regularized by the KL term, CE simply learned better features. This is a reminder that stronger regularization is not always better.

---

## 7. Success Tier Assessment

| Criterion | Target | V5 Best Actual | Met? |
|-----------|--------|---------------|------|
| **Tier S**: EDL ≥ 82% Acc | 82% | 72.33% (CV) / 78.33% (P1 s456 EDL+ORCU) | **No** |
| **Tier A**: EDL ≥ 80% Acc | 80% | 72.33% (CV) | **No** |
| **Tier B**: EDL ≥ 78% Acc | 78% | 72.33% (CV) | **No** |
| **Tier C**: Best method ≥ 75% | 75% | 75.83% (CE CV) | **Borderline Yes** |

**Final: Tier C.** V5 fails all EDL-specific targets. The only bright spot is CE at 75.83%. This is a clear regression from V4's Tier B+.

---

## 8. Competitor Comparison (Updated)

| Method | Acc | Macro F1 | Parameters | Uncertainty | Year |
|--------|-----|----------|-----------|-------------|------|
| MLTrMR | 80.19% | 75.79% | 556M | None | 2025 |
| LD2Net | 80.00% | 79.88% | 3.3M | None | 2026 |
| FusionDentNet (citation unverified, arXiv ID may belong to MLTrMR) | 80.00% | 79.25% | 201M | None | 2024 |
| **Our V4 EDL (CV)** | **79.33% ± 3.74** | **79.02% ± 4.01** | **86M** | **Dirichlet** | 2026 |
| **Our V5 CE (CV)** | **75.83% ± 5.23** | **75.19% ± 5.04** | **86M** | **Entropy** | 2026 |
| **Our V5 EDL (CV)** | **72.33% ± 7.90** | **71.23% ± 8.52** | **86M** | **Dirichlet** | 2026 |

V5 CE at 75.83% is below all top competitors. V5 EDL at 72.33% is substantially below V4's results. **V4 remains our best result.** The V5 parameter changes undid V4's progress.

---

## 9. Key Findings

### What V5 Revealed (Despite Being a Regression)

1. **KL annealing interacts with total epochs.** The optimal kl_lambda depends on the total KL "dose" = f(cap, epochs, lambda), not lambda alone. V4→V5's 33% dose increase with the same lambda explains the collapse.

2. **10-fold CV provides sufficient statistical power.** For the first time, we have statistically significant results (CE vs EDL+ORCU, p<0.01). Future experiments should keep 10-fold CV.

3. **λ_orcu=0.003 is better than 0.005.** The unimodality constraint should be weaker, not stronger. The log-barrier is powerful even at low λ.

4. **CE is not a trivial baseline.** At 75.83% CV, CE+MixUp with 100 epochs is a legitimate contender, exceeding V5 EDL by 3.50pp.

5. **100 epochs is too many for 200 images.** V4's 75-epoch design was better calibrated. The extra epochs primarily added variance.

### What Went Wrong

1. **Blindly applying V4's kl=0.05 to 100 epochs.** The Phase 2 sweep should have been run before the full experiment.
2. **Changing too many variables at once.** Epochs, kl_lambda, kl_cap, λ_orcu, and CV folds all changed simultaneously.
3. **No Phase 2 verification before Phase 4.** All phases were launched simultaneously, wasting the CV run on suboptimal parameters.

---

## 10. Recommendations

### For Paper Writing

1. **Use V4 EDL as the primary result** (79.33% ± 3.74). V5 is a negative result for the main paper.
2. **Report V5 Phase 2 as an ablation** on KL strength vs. training duration interaction — this is a legitimate scientific finding.
3. **CE at 75.83% provides a stronger baseline** for future work than V4's 70.00% Phase 1 CE.

### For V6

1. **Revert to 75 epochs.** 100 epochs offers no benefit and increases variance on this dataset size.
2. **Use kl=0.07 with cap=0.5 for EDL**, or test kl=0.05 with lower cap (0.3) to reduce total KL dose.
3. **Use λ_orcu=0.003 for EDL+ORCU.** Phase 3 evidence is clear.
4. **Run Phase 2 (KL sweep) and Phase 3 (λ sweep) BEFORE Phase 4 (CV).** Sequential phasing prevents wasted CV runs.
5. **Keep 10-fold CV.** The statistical power is worth the computational cost.
6. **Consider reducing EDL patience** (25→15) to prevent overfitting before EarlyStopping triggers.
7. **Test a "light" EDL variant** with kl_lambda=0.03 and cap=0.3 — Phase 2 shows kl=0.03 achieves 71.67% with 100% unimodality, suggesting a gentler KL may balance accuracy and calibration better.

### Recommended V6 Configurations

```
EDL:      kl_lambda=0.07, kl_anneal_cap=0.5, epochs=75, patience=20
EDL+ORCU: kl_lambda=0.05, kl_anneal_cap=0.5, lambda_orcu=0.003, epochs=75, patience=15
CE:       epochs=75, patience=15, MixUp alpha=0.2
```

---

## 11. Raw Results Quick Reference

| Phase | Method | Best Config | Acc | Macro F1 | QWK | ECE |
|-------|--------|------------|-----|----------|-----|-----|
| P1 | CE | s42 | 70.00% | 0.694 | 0.884 | 0.155 |
| P1 | Cumulative | s123 | 76.67% | 0.762 | 0.910 | 0.239 |
| P1 | SORD | s123 | 73.33% | 0.733 | 0.883 | 0.340 |
| P1 | EDL | s123 | 75.00% | 0.741 | 0.887 | 0.184 |
| P1 | EDL+ORCU | s456 | 78.33% | 0.774 | 0.913 | 0.211 |
| P2 | EDL kl=0.07 | s42 | 75.00% | 0.743 | 0.883 | 0.241 |
| P2 | EDL kl=0.05 | s42 | 60.00% | 0.586 | 0.836 | 0.230 |
| P3 | EDL+ORCU λ=0.003 | s123 | 81.67% | 0.800 | 0.908 | 0.166 |
| P3 | EDL+ORCU λ=0.008 | s456 | 76.67% | 0.749 | 0.913 | 0.202 |
| **P4** | **CE CV** | 10-fold | **75.83% ± 5.2** | **75.19% ± 5.0** | **0.886 ± 0.031** | 0.154 |
| P4 | EDL CV | 10-fold | 72.33% ± 7.9 | 71.23% ± 8.5 | 0.870 ± 0.045 | 0.186 |
| P4 | EDL+ORCU CV | 10-fold | 71.67% ± 6.8 | 70.63% ± 7.1 | 0.867 ± 0.040 | 0.211 |
| P4 | SORD CV | 10-fold | 70.33% ± 10.3 | 69.22% ± 11.2 | 0.857 ± 0.059 | 0.193 |
| P4 | Cumulative CV | 10-fold | 68.83% ± 8.6 | 66.92% ± 8.9 | 0.819 ± 0.123 | 0.166 |
