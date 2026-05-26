# SF (Skeletal Fluorosis) Experiment Analysis

**Date:** 2026-05-26
**Data:** 80 X-ray images, 4-class (Normal=21, Mild=34, Moderate=13, Severe=12), 3 body parts, 512x1024 PNG
**Setup:** ResNet-50 backbone, 5-fold CV, single-run test, 100 epochs, no EarlyStopping

---

## 1. Single-Run Test Results

| Method     | Acc ↑  | Macro F1 ↑ | QWK ↑   | ECE ↓  | %Unimodal ↑ | U_ECE ↓ |
|------------|--------|------------|---------|--------|-------------|---------|
| CE         | 50.00% | 0.375      | 0.245   | 0.097  | 54.17%      | 0.328   |
| Cumulative | 54.17% | 0.333      | 0.317   | 0.149  | 25.00%      | 0.378   |
| SORD       | 41.67% | 0.263      | 0.091   | 0.070  | 87.50%      | 0.310   |
| EDL        | 45.83% | 0.281      | **-0.019** | 0.287 | 70.83%      | 0.022   |
| EDL+ORCU   | **54.17%** | **0.408**  | 0.280   | 0.215  | **91.67%**  | 0.089   |

**Key observations:**
- Best accuracy (54.17%) barely above random (25%), far below DF's best (70%)
- EDL has **negative QWK** (-0.019) — predictions are anti-correlated with ground truth
- EDL+ORCU achieves 91.67% unimodality — ORCU constraint works even on small data
- SORD has highest unimodality (87.5%) but worst accuracy (41.67%) — over-regularized

## 2. EDL Evidence Distribution (Single-Run)

| Method   | Class 0 (Normal) | Class 1 (Mild) | Class 2 (Moderate) | Class 3 (Severe) |
|----------|------------------|----------------|--------------------|--------------------|
| EDL      | **2.52**         | 1.86           | 0.49               | **0.34**           |
| EDL+ORCU | 1.61             | 1.57           | 0.58               | 0.36               |

**Skew:** EDL evidence concentrates on Class 0 (Normal), nearly 7.4x Class 3. The model is "confidently wrong" on the majority class while ignoring minority classes. ORCU partially balances this but Class 3 still lags.

## 3. 5-Fold Cross-Validation

| Method     | CV Acc ↑       | CV QWK ↑      | QWK p-value (vs CE) |
|------------|----------------|---------------|---------------------|
| CE         | 55.83% ±8.98   | 0.132 ±0.177  | -                   |
| Cumulative | **59.17%** ±8.08 | **0.408** ±0.130 | **p=0.027** (sig) |
| EDL        | **59.17%** ±4.86 | 0.285 ±0.210  | p=0.073             |
| EDL+ORCU   | 55.00% ±6.12   | 0.224 ±0.086  | p=0.237             |

**Key observations:**
- Cumulative has best CV QWK (0.408) and only significant QWK improvement over CE (p=0.027)
- EDL has lowest QWK variability — high CV Acc but inconsistent ordinal ranking
- Only 3 significant pairwise tests total (all for QWK or ECE, none for accuracy)
- EDL vs EDL+ORCU ECE: p=0.007 (EDL+ORCU better calibrated)

## 4. Training Dynamics Issues

| Method   | Peak Epoch | Peak Val Acc | Final Val Acc | Test Acc | Overfitting Gap |
|----------|-----------|-------------|---------------|----------|-----------------|
| CE       | 4         | 65.00%      | 65.00%        | 50.00%   | **-15%**        |
| Cumulative | 15      | 80.00%      | 60.00%        | 54.17%   | **-26%**        |
| SORD     | 11        | 70.00%      | 60.00%        | 41.67%   | **-28%**        |
| EDL      | 20        | 80.00%      | 65.00%        | 45.83%   | **-34%**        |
| EDL+ORCU | 3         | 65.00%      | 65.00%        | 54.17%   | **-11%**        |

**All methods show severe train→test overfitting.** Validation peaks at 65-80% but test drops to 42-54%. **No EarlyStopping** means models train 100 epochs despite peaking as early as epoch 3.

## 5. DF vs SF Comparison

| Metric        | DF (200 images) | SF (80 images) | Δ       |
|---------------|-----------------|----------------|---------|
| Best Test Acc | 70.0% (CE/SORD) | 54.2% (Cum/EDL+ORCU) | **-15.8%** |
| Best CV Acc   | 75.7% ±3.3 (EDL)| 59.2% ±8.1 (Cum/EDL)| **-16.5%** |
| Best QWK      | 0.882 (CE CV)   | 0.408 (Cum CV)  | **-0.474** |
| EDL QWK       | 0.862 (V2)      | **-0.019** (V2)  | Broken    |
| ORCU Unimodal | 100%            | 91.7%           | -8.3%     |
| Best CV ECE   | 0.139 (EDL+ORCU)| 0.214 (EDL+ORCU)| +0.075    |

## 6. Root Cause Analysis

### Primary: Extreme Data Scarcity (80 images, 20/class avg)
- 200 DF images already borderline; 80 SF images is **below critical mass for deep learning**
- ViT-Base (86M params) / ResNet-50 (25M) need 1000+ samples typically
- Class imbalance compounds: Normal=21, Severe=12 (1.75:1 ratio)

### Secondary: No regularization or early stopping
- Training runs 100 epochs despite peaking at epoch 3-20
- No EarlyStopping → severe overfitting
- No MixUp, no RandAugment in current SF transforms (added in v3 for DF only)
- EDL KL annealing unbounded (pre-v3 fix, kl_anneal_cap=1.0)

### Tertiary: Task difficulty
- SF X-rays (512x1024) have subtle bone texture changes vs. obvious DF tooth discoloration
- 3 body parts (hands, pelvis, forearm) add domain shift within 80-image set
- 4-grade classification with ordinal structure is harder than binary (where Mwinc-Mamba got 83.3%)

## 7. Recommendations

### Priority 1 — Must Fix (for any publishable result)
1. **Add EarlyStopping(patience=15)**: Prevents 80-epoch overfitting after peaks
2. **Reduce epochs to 50**: Match DF v3 settings
3. **Apply v3 fixes**: KL cap=0.5, lambda_kl=0.02, lambda_orcu=0.05, lambda_reg=0.005
4. **Add RandAugment(2,9)**: Was added for DF v3, also needed for SF

### Priority 2 — High Impact
5. **Stronger augmentation for X-rays**: RandomAffine, ElasticTransform, GridDistortion (torchvision or albumentations)
6. **MixUp(α=0.2)** for CE/Cumulative/SORD modes
7. **Higher weight decay or dropout**: Regularize ResNet-50 for 80-sample regime
8. **Per-body-part stratified splitting**: Ensure each fold has proportionally equal parts (hand/pelvis/forearm)

### Priority 3 — Medium Impact
9. **Class-weighted loss**: Address Normal=21 vs Severe=12 imbalance
10. **5-fold repeated CV** (3-5 repeats): Get reliable error bars on 80-sample set
11. **SF-specific lambda sweep**: Current sweep tuned for DF (200 samples); re-sweep λ_orcu for SF

### Priority 4 — Exploratory
12. **Pretrain on larger X-ray dataset** (MIMIC-CXR, CheXpert) → finetune on SF
13. **Siamese/triplet learning**: Learn bone-texture embeddings from body-part labels
14. **Test-time augmentation (TTA)**: 8-16 augmented views averaged at inference
15. **Synthetic minority oversampling (SMOTE)** in feature space

## 8. V3 DF Results (from Kaggle — for reference)

V3 kernel completed successfully (error in log was post-hoc dataset upload, not training):

| Method     | Single Test Acc | CV Acc       | CV QWK      |
|------------|-----------------|--------------|-------------|
| CE         | 70.00%          | 75.67% ±7.9  | 0.887 ±0.05 |
| Cumulative | 53.33%          | 70.67% ±4.3  | 0.820 ±0.08 |
| SORD       | 70.00%          | -            | -           |
| EDL        | **75.00%**      | **78.33%** ±3.8 | **0.905 ±0.02** |
| EDL+ORCU   | 70.00%          | 74.00% ±4.3  | 0.890 ±0.02 |

**V3 improvements over V2:** EDL Acc +8.3% (66.7→75.0%), EDL+ORCU unchanged. EDL now clearly best method.
Lambda sweep: λ_orcu=0.01/0.08 both give Test=73.3% QWK=0.877.

## Summary

SF performance (54.17% test, 59.17% CV) is fundamentally limited by the 80-image dataset size. The immediate path forward is applying all V3 fixes (EarlyStopping, reduced epochs, capped KL, RandAugment) and adding X-ray-specific augmentations. Even with these improvements, expect SF accuracy ceiling around 60-65% — **pretraining on a larger X-ray dataset is likely necessary to reach publishable SF results (>75%).**
