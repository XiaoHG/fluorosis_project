# V6 Experiment Plan

**Version:** v3
**Date:** 2026-06-01
**Status:** Planning
**Objective:** DF V6 experiments + SF multi-rater soft labels + code infrastructure improvements

## Background

V4 achieved best results: EDL CV Acc 79.33% ± 3.74, QWK 0.905 ± 0.017. V5 regressed due to KL annealing issues. V6 aims to:
1. Fix KL annealing from V5 regression
2. Implement log-barrier ORCU (replace hinge approximation)
3. 10-fold CV for more stable evaluation
4. Complete SF experiments with multi-rater soft labels

## Experiment Matrix

### Phase 1: Code Infrastructure (Priority: CRITICAL)

| # | Task | Description |
|---|------|-------------|
| 1.1 | ORCU log-barrier | Replace ReLU hinge with log-barrier penalty per design doc |
| 1.2 | EarlyStopping | Add patience-based early stopping with best-model restoration |
| 1.3 | CUDA determinism | Add seed_all(), cudnn.deterministic, cudnn.benchmark |
| 1.4 | YAML config loading | Parse YAML configs in train.py (currently hardcoded defaults) |
| 1.5 | Class weighting | Add class_weight to EDL loss for imbalanced SF data |
| 1.6 | AMP support | Add torch.cuda.amp autocast + GradScaler |
| 1.7 | SF soft labels | Load 5-rater annotations, support soft-label EDL training |

### Phase 2: DF V6 Experiments

| # | Mode | KL | ORCU | Folds | Seeds | Notes |
|---|------|-----|------|-------|-------|-------|
| 2.1 | CE | - | - | 10 | 42, 123, 456 | Baseline |
| 2.2 | Cumulative | - | - | 10 | 42, 123, 456 | Coral baseline |
| 2.3 | SORD | - | - | 10 | 42, 123, 456 | SORD baseline |
| 2.4 | EDL | 0.07 | - | 10 | 42, 123, 456, 789, 1024 | KL annealing fix |
| 2.5 | EDL+ORCU | 0.07 | 0.10 (log-barrier) | 10 | 42, 123, 456, 789, 1024 | New log-barrier |

**Key changes from V5:**
- KL lambda: 0.1 → 0.07 (milder regularization)
- ORCU: hinge → log-barrier (matches design doc)
- Folds: 5 → 10 (more stable CV estimates)
- Seeds: add 789, 1024 for 5-seed stability analysis

### Phase 3: SF V2 Experiments

| # | Mode | Backbone | Labels | Notes |
|---|------|----------|--------|-------|
| 3.1 | CE | ResNet-50 | Hard (majority vote) | Baseline |
| 3.2 | Cumulative | ResNet-50 | Hard | Coral |
| 3.3 | EDL | ResNet-50 | Hard | EDL baseline |
| 3.4 | EDL | ResNet-50 | Soft (5-rater) | **NEW**: soft-label EDL |
| 3.5 | EDL+ORCU | ResNet-50 | Hard | With log-barrier |
| 3.6 | EDL+ORCU | ResNet-50 | Soft (5-rater) | **NEW**: full pipeline |

### Phase 4: Analysis & Ablation

| # | Task | Description |
|---|------|-------------|
| 4.1 | Lambda sweep | kl ∈ {0.02, 0.05, 0.07, 0.10, 0.15}, orcu ∈ {0.05, 0.10, 0.20} |
| 4.2 | A6 ablation | log-barrier vs hinge comparison (requires ORCU loss fix first) |
| 4.3 | Seed stability | 5-seed boxplots for Acc, QWK, ECE |
| 4.4 | SF soft vs hard | Compare soft-label EDL vs hard-label EDL on SF |

## Success Criteria

- DF V6 EDL 10-fold CV Acc > 80% (exceed MLTrMR 80.19%)
- QWK > 0.91 (maintain or exceed V4 0.905)
- ECE < 0.15 (improve calibration from V4 0.173)
- Log-barrier ORCU shows better unimodality than hinge
- SF soft-label EDL outperforms hard-label EDL

## Timeline

| Phase | Duration | Priority |
|-------|----------|----------|
| Phase 1: Infrastructure | 2-3 days | CRITICAL |
| Phase 2: DF V6 | 3-4 days | HIGH |
| Phase 3: SF V2 | 2-3 days | HIGH |
| Phase 4: Analysis | 1-2 days | MEDIUM |

## Dependencies

- Phase 2 depends on Phase 1.1 (log-barrier), 1.3 (determinism), 1.4 (YAML config)
- Phase 3 depends on Phase 1.2 (EarlyStopping), 1.5 (class weighting), 1.7 (soft labels)
- Phase 4 depends on Phase 2 and Phase 3 completion
