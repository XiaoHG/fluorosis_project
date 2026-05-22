# V2 Experiment Plan

**Created:** 2026-05-22
**Based on:** v1.0 / v2.2 node (correct architecture: StandardClassifier + EDLClassifier)
**Status:** Planning → Execution

---

## v2.2 Baseline (v1 定稿数据)

| Item | Value |
|------|-------|
| DF EDL Acc | 83.33% (SOTA, DFID 200 images) |
| DF CE Acc | 81.67% (already beats all competitors) |
| DF EDL ECE | 0.0719 (best calibration) |
| Architecture | ViT-Base (86M) + StandardClassifier for CE/SORD, EDLClassifier for EDL/EDL+ORCU |
| Environment | PyTorch 2.5, Kaggle GPU |
| SF | Not run — dataset exists (80 images), SFDataset implemented, not integrated |

---

## v2 Goals

1. **Re-establish DF baseline** with per-sample prediction export (v2.2's missing data)
2. **SF standalone experiments** — apply v2.2 architecture to SF data
3. **DF+SF joint analysis** — cross-dataset generalization, transfer learning
4. **Address review feedback** — fix version inconsistency (MC1), soften generalizability claims (MC2)
5. **Fill code review gaps** — CV wrapper, ablation runner, MC Dropout baseline

---

## Phase 1: DF Baseline Re-establishment (P0)

Re-run v2.2 DF experiments with per-sample export. v2.2 has correct architecture but only aggregate numbers — no per-sample predictions for generating confusion matrices, reliability diagrams, uncertainty histograms.

### 1.1 Main Experiment (5 modes x 1 seed)
- CE, Cumulative, SORD, EDL, EDL+ORCU
- ViT-Base, seed=42, 100 epochs
- **Export**: per-sample y_true, y_pred, prob, u, alpha, evidence
- Expected: reproduce v2.2 Acc=83.33% (EDL)

### 1.2 Multi-Seed Validation (CE vs EDL x 3 seeds)
- Seeds: 42, 123, 456
- Expected: CE CV=31.4%, EDL CV=4.3%

### 1.3 5-Fold Cross-Validation
- Expected: EDL+ORCU QWK std=0.32%

### 1.4 Lambda Sweep (EDL+ORCU, 9 combos)
- lambda_orcu in {0.1, 0.3, 0.5}, lambda_reg in {0.005, 0.01, 0.02}
- Expected: best at lambda_orcu=0.5, lambda_reg=0.01

### 1.5 Temperature Calibration (EDL only)
- T in {0.5, 1.0, 1.5, 2.0, 3.0, 5.0}
- On v2.2 EDL checkpoint

---

## Phase 2: SF Standalone (P0)

Apply v2.2 architecture directly to SF data. SF poses unique challenges: 80 images (vs DF 200), 3 body parts (forearm/calf/pelvis), extreme class imbalance.

### 2.1 SF Data Audit
- Confirm GT.xlsx labels match images/
- Check radiologist annotations for label quality
- Stratified split across body parts

### 2.2 SF Baselines (ViT-Base, 5 modes)
- Train: ~48, Val: ~8, Test: 24 (7:3)
- Same 5-mode comparison as DF
- Expected: lower absolute accuracy than DF (80 images, 3 body parts)
- **Competitor**: Mwinc-Mamba SF binary 83.33%, 4-grade not reported

### 2.3 SF Multi-Seed
- 3 seeds on CE vs EDL
- Check if EDL stability advantage holds on even smaller dataset

### 2.4 SF Cross-Validation
- 5-fold, body-part-aware stratification

---

## Phase 3: Cross-Dataset Generalization (P1)

### 3.1 DF to SF Transfer
- Train on DF (120), test on SF (80) — zero-shot
- Fine-tune on SF with frozen backbone

### 3.2 SF to DF Transfer
- Train on SF (48), test on DF (60) — zero-shot
- Fine-tune on DF with frozen backbone

### 3.3 Joint Training
- Combine DF (train) + SF (train), test on each separately
- Shared backbone, task-specific heads

---

## Phase 4: Architectural Improvements (P1)

### 4.1 MC Dropout Baseline
- Compare vs EDL uncertainty quality
- 10 forward passes, variance as uncertainty

### 4.2 Deep Ensemble Baseline
- 5 models, different seeds
- Variance across ensemble as uncertainty

### 4.3 Body Part Conditional (SF)
- Model body part as auxiliary input
- Improve SF accuracy by conditioning on anatomy

---

## Phase 5: Per-Sample Export Format

```python
# Exported .npz per experiment
{
    "y_true":     (N,) int64,
    "y_pred":     (N,) int64,
    "prob":       (N, 4) float32,    # softmax probability
    "alpha":      (N, 4) float32,    # Dirichlet evidence (EDL modes only)
    "u":          (N,) float32,      # total uncertainty
    "evidence":   (N, 4) float32,    # e_k per class (EDL modes only)
    "entropy":    (N,) float32,      # predictive entropy
    "mode":       str,               # "ce"|"cumulative"|"sord"|"edl"|"edl_orcu"
    "task":       str,               # "df"|"sf"
    "seed":       int,
    "fold":       int,               # for CV, -1 for main
}
```

---

## Phase 6: Manuscript Revision (P2)

- Update all tables/figures with v2 data
- Fix version inconsistency (MC1 from review)
- Add SF results as supplementary or expanded main
- Address single-center limitation (MC2)
- Update SOTA comparison with SF results

---

## Deliverables Checklist

| Phase | Deliverable | Priority | Status |
|-------|-------------|:--------:|:------:|
| 1.1 | DF main experiment + per-sample .npz | P0 | [ ] |
| 1.2 | DF multi-seed + .npz | P0 | [ ] |
| 1.3 | DF 5-fold CV + .npz | P0 | [ ] |
| 1.4 | DF lambda sweep results | P0 | [ ] |
| 1.5 | DF temperature calibration | P1 | [ ] |
| 2.1 | SF data audit report | P0 | [ ] |
| 2.2 | SF 5-mode baselines | P0 | [ ] |
| 2.3 | SF multi-seed | P1 | [ ] |
| 2.4 | SF cross-validation | P1 | [ ] |
| 3.1 | DF to SF transfer | P1 | [ ] |
| 3.2 | SF to DF transfer | P1 | [ ] |
| 3.3 | Joint DF+SF training | P1 | [ ] |
| 4.1 | MC Dropout baseline | P1 | [ ] |
| 4.2 | Deep Ensemble baseline | P2 | [ ] |
| 4.3 | Body-part conditional SF | P2 | [ ] |
| 6.1 | Updated manuscript | P2 | [ ] |

---

## Execution Order (Kaggle)

**Batch 1** (today): DF Phase 1.1 + SF Phase 2.2 (parallel, ~3h)
**Batch 2**: DF Phase 1.2 + 1.3 (~2h)
**Batch 3**: DF Phase 1.4 + 1.5 + SF Phase 2.3 (~2h)
**Batch 4**: Phase 3 cross-dataset (~2h)
**Batch 5**: Phase 4 baselines (~1h)

---

*Plan will be updated as experiments progress.*
