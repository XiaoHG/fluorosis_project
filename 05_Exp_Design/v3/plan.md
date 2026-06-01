# V6 Experiment Plan — DF Only

**Version:** v3
**Date:** 2026-06-01
**Status:** Ready
**Scope:** DF (Dental Fluorosis) only — ViT-Base (86M), 200 intraoral photos, 4-class
**Objective:** Beat MLTrMR (80.19%) and LD2Net (80.00%) with log-barrier ORCU + 5-seed stability

## Background

V4 best: EDL CV Acc 79.33% ± 3.74, QWK 0.905 ± 0.017. V5 regressed (KL annealing).
V6 fixes: log-barrier ORCU, kl=0.07, 10-fold CV, 5 seeds.

**Code infrastructure (Phase 1) already completed in audit:**
- ✅ ORCU log-barrier (replaced hinge)
- ✅ EarlyStopping (patience-based, best-model restoration)
- ✅ CUDA determinism (set_seed)
- ✅ YAML config loading (--config)
- ✅ AMP mixed precision (--use_amp)
- ✅ Gradient clipping (max_norm=1.0)
- ✅ lambda_kl unified to 0.07

## DF V6 Experiment Matrix

### Phase 2: Single-Seed Baseline (5 methods × seed=42, ~1.5h)

Quick baseline with V6 defaults to verify no regressions.

| # | Mode | KL | ORCU | Seed | Notes |
|---|------|-----|------|------|-------|
| 2.1 | CE | - | - | 42 | Standard baseline |
| 2.2 | Cumulative | - | - | 42 | Coral K-1 BCE |
| 2.3 | SORD | - | - | 42 | Soft ordinal encoding |
| 2.4 | EDL | 0.07 | - | 42 | KL annealing fix |
| 2.5 | EDL+ORCU | 0.07 | 0.10 (log-barrier) | 42 | New log-barrier |

### Phase 3: Hyperparameter Sweep (~3h)

#### 3A: EDL KL Lambda Sweep

| KL | Seeds |
|----|-------|
| 0.03, 0.05, **0.07**, 0.10, 0.15 | 42, 123, 456 |

9 runs. Find best KL for CV calibration.

#### 3B: EDL+ORCU Lambda Sweep

| ORCU λ | KL | Seeds |
|---------|-----|-------|
| 0.05, **0.10**, 0.20 | 0.07 | 42, 123, 456 |

9 runs. Find best ORCU strength with log-barrier.

### Phase 4: Multi-Seed Stability (5 seeds, ~5h)

| Mode | KL | ORCU | Seeds |
|------|-----|------|-------|
| CE | - | - | 42, 123, 456, 789, 1024 |
| Cumulative | - | - | 42, 123, 456, 789, 1024 |
| SORD | - | - | 42, 123, 456, 789, 1024 |
| **EDL** | **0.07** | - | **42, 123, 456, 789, 1024** |
| **EDL+ORCU** | **0.07** | **0.10 (log-barrier)** | **42, 123, 456, 789, 1024** |

25 runs. Compare 3-seed vs 5-seed stability.

### Phase 5: 10-Fold Cross-Validation (~15h)

| Mode | KL | ORCU | Folds |
|------|-----|------|-------|
| CE | - | - | 10 |
| Cumulative | - | - | 10 |
| SORD | - | - | 10 |
| **EDL** | **0.07** | - | **10** |
| **EDL+ORCU** | **0.07** | **0.10 (log-barrier)** | **10** |

50 folds total. Paired t-tests (McNemar for paper).

### Phase 6: A6 Ablation — Log-Barrier vs Hinge (~1h)

| Mode | KL | ORCU λ | Regularizer | Seeds |
|------|-----|--------|-------------|-------|
| EDL+ORCU (log-barrier) | 0.07 | 0.10 | **log-barrier** | 42, 123, 456 |
| EDL+ORCU (hinge) | 0.07 | 0.10 | hinge | 42, 123, 456 |

6 runs. **Key paper contribution:** prove log-barrier > hinge for unimodality.

## Run Summary

| Phase | Runs | Est. Time | Priority |
|-------|------|-----------|----------|
| Phase 2: Baseline | 5 | ~1.5h | HIGH |
| Phase 3: Sweeps | 18 | ~3h | HIGH |
| Phase 4: Multi-Seed | 25 | ~5h | HIGH |
| Phase 5: 10-Fold CV | 50 | ~15h | **CRITICAL** |
| Phase 6: A6 Ablation | 6 | ~1h | MEDIUM |
| **Total** | **104** | **~25h** | |

## Success Criteria

| Tier | Acc (10-fold CV) | QWK | ECE | Status |
|------|-----------------|-----|-----|--------|
| **S** | ≥ 82% | ≥ 0.92 | < 0.12 | Paper-ready SOTA, beats all by ≥1.8pp |
| **A** | ≥ 80% | ≥ 0.91 | < 0.15 | Beats all competitors |
| B | ≥ 78% | ≥ 0.90 | < 0.18 | Competitive |
| C | < 78% | < 0.90 | > 0.18 | Investigate |

**Key metric:** 10-fold CV Acc > 80% = beats MLTrMR (80.19%) and LD2Net (80.00%).

## Competitors (DF Only)

| Method | Acc | Params | Uncertainty | Ordinal |
|--------|-----|--------|-------------|---------|
| MLTrMR (2025) | 80.19% | 556M | None | None |
| LD2Net (2026) | 80.00% | 3.3M | None | None |
| **Ours EDL (V4)** | **79.33% ± 3.74** | **86M** | **Dirichlet** | **Implicit** |
| **Ours EDL+ORCU (V6 target)** | **>80%** | **86M** | **Dirichlet** | **Log-barrier** |
