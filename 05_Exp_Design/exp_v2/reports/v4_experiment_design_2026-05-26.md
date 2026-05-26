# V4 DF Experiment Design

**Date:** 2026-05-26
**Baseline:** V3 best (SORD 78.33% single, EDL+ORCU λ=0.01→83.33%, CE CV 75.67%)
**Target:** Match/exceed V1 paper (EDL 83.33%, CE 81.67%) with full F1 + multi-seed validation

---

## 1. V3 Root Cause Analysis

### 1.1 Same-Seed Comparison (seed=42, identical test split)

| Method | V1 (s42) | V3 (s42) | Δ | F1 V3 |
|--------|----------|----------|------|------|
| CE | 61.67% / QWK=0.859 | **70.00%** / QWK=0.882 | **+8.3%** | 0.696 |
| SORD | 73.33% / QWK=0.900 | **78.33%** / QWK=0.896 | **+5.0%** | 0.768 |
| EDL | **83.33%** / QWK=0.936 | 61.67% / QWK=0.844 | **-21.7%** | 0.601 |
| EDL+ORCU | 70.00% / QWK=0.872 | 66.67% / QWK=0.856 | -3.3% | 0.651 |

V3 improved CE (+8.3pp) and SORD (+5pp) vs V1. **EDL collapsed (-21.7pp) due to kl_lambda=0.02 being 5x too low.**

### 1.2 EDL Collapse Root Cause

| Parameter | V1 EDL | V3 EDL | Impact |
|-----------|--------|--------|--------|
| kl_lambda | **0.10** | 0.02 | KL weight ↓5x — primary cause |
| kl_anneal_cap | None (1.0) | 0.5 | Prevents full KL |
| Epochs | 100 | 25 (ES) | Less KL annealing |
| MixUp | None | None | Not the cause |

Effective KL regularization is ~10x weaker in V3 EDL.

### 1.3 V1 Seed Sensitivity (critical context)

| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|--------|---------|----------|----------|------------|
| CE | 61.67% | **80.00%** | 35.00% | 58.89% ± 18.48% |
| EDL | **83.33%** | 80.00% | 75.00% | **79.44%** ± 3.42% |

CE is extremely seed-sensitive (CV=31.4%). EDL is robust (CV=4.3%). V3's seed=42 is CE's 2nd-worst but EDL's best — multi-seed is mandatory.

### 1.4 Lambda Sweep Finding

λ_orcu=0.01, λ_reg=0.005 → Test=83.33%, QWK=0.932 at seed=42. Unimodal=61.67% (low). Needs multi-seed validation — could be a lucky fold.

---

## 2. V4 Design

### 2.1 Parameter Changes

| Parameter | V3 Default | V4 Default | Reason |
|-----------|-----------|-----------|--------|
| kl_lambda (EDL only) | 0.02 | **0.10** | Restore V1 value |
| kl_lambda (EDL+ORCU) | 0.02 | **0.05** | Moderate increase |
| kl_anneal_cap (EDL) | 0.5 | **1.0** | Remove cap |
| kl_anneal_cap (EDL+ORCU) | 0.5 | 0.5 | Keep |
| patience (EDL) | 15 | **25** | More room for KL |
| patience (others) | 15 | 15 | Keep |
| epochs | 50 | **75** | More KL annealing |
| seeds | [42] | **[42, 123, 456]** | Multi-seed |
| MixUp (CE/Cum/SORD) | α=0.2 | α=0.2 | Keep |
| MixUp (EDL/EDL+ORCU) | No | No | Keep off |
| **F1 tracking** | Partial | **Full** | Per-class, Macro, Weighted |

### 2.2 Experiment Matrix

#### Phase 1: Core Methods × Multi-Seed (15 runs total)

| Method | Seeds | MixUp | kl_lambda | λ_orcu | Patience |
|--------|-------|-------|-----------|--------|----------|
| CE | [42,123,456] | α=0.2 | — | — | 15 |
| Cumulative | [42,123,456] | α=0.2 | — | — | 15 |
| SORD | [42,123,456] | α=0.2 | — | — | 15 |
| EDL | [42,123,456] | No | **0.10** | — | **25** |
| EDL+ORCU | [42,123,456] | No | **0.05** | **0.01** | 15 |

#### Phase 2: EDL kl_lambda Sweep (4 runs, seed=42)

| kl_lambda | kl_cap | Purpose |
|-----------|--------|---------|
| 0.02 | 0.5 | V3 default (expected: poor) |
| 0.05 | 0.5 | Intermediate |
| **0.10** | **1.0** | Restored V1 (expected: best) |
| 0.20 | 1.0 | Over-regularized? |

#### Phase 3: EDL+ORCU λ_orcu Validation (9 runs)

| λ_orcu | Seeds |
|--------|-------|
| 0.005 | [42,123,456] |
| 0.01 | [42,123,456] |
| 0.03 | [42,123,456] |

#### Phase 4: 5-Fold CV — Best Configs (3 runs)
Best EDL + Best EDL+ORCU + SORD → CV

**Total: 15 + 4 + 9 + 3 = 31 runs, ~8h on T4 x2**

### 2.3 F1 Tracking (mandatory for all V4 runs)

Every test_metrics.json includes:
```json
{
  "acc": 0.XX,
  "macro_f1": 0.XX,
  "weighted_f1": 0.XX,
  "per_class_f1": [0.XX, 0.XX, 0.XX, 0.XX],
  "qwk": 0.XX,
  "ece": 0.XX,
  "sce": 0.XX,
  "pct_unimodal": 0.XX,
  "u_ece": 0.XX,
  "auroc_u": 0.XX
}
```

Per-sample .npz includes: y_true, y_pred, prob, logits (alpha/evidence/u for EDL modes), plus _mode, _task, _seed, _fold.

---

## 3. V4 vs Competitors Target Table

| Method | Acc | Macro F1 | Weighted F1 | QWK | ECE | Params | Uncertainty |
|--------|-----|----------|-------------|-----|-----|--------|-------------|
| MLTrMR (2025) | 80.19% | 75.79% | — | 0.813 | — | 556M | None |
| LD2Net (2026) | 80.00% | 79.88% | — | — | — | 3.3M | None |
| FusionDentNet (2024) | 80.00% | 79.25% | — | — | — | 201M | None |
| HiFuse (2024) | 78.23% | 70.45% | — | — | — | 164M | None |
| Ours CE (V4) | TBD | TBD | TBD | TBD | TBD | 86M | Entropy |
| Ours SORD (V4) | TBD | TBD | TBD | TBD | TBD | 86M | Entropy |
| **Ours EDL (V4)** | **TBD ★** | TBD | TBD | TBD | TBD | 86M | **Dirichlet** |
| **Ours EDL+ORCU (V4)** | **TBD ★** | TBD | TBD | TBD | TBD | 86M | **Dirichlet** |

★ Target: multi-seed mean ≥ 80%, F1 ≥ 79%

---

## 4. Success Criteria

| Tier | EDL Multi-Seed Mean | Action |
|------|--------------------|--------|
| **S** | ≥ 82% Acc, ≥ 81% F1 | Paper-ready SOTA |
| **A** | ≥ 80% Acc, ≥ 79% F1 | Beats all competitors |
| **B** | ≥ 78% Acc, ≥ 77% F1 | Competitive, needs analysis |
| **C** | < 78% Acc | Investigate V1/V3 protocol gap |

---

## 5. Notebook Structure (kaggle_train_v4_df.ipynb)

1. Environment Setup (clone repo, cache weights)
2. V4 Training Components (updated defaults, `train_model_v4()`, F1 tracking)
3. Phase 1: Core Methods × Multi-Seed (15 runs)
4. Phase 2: EDL kl_lambda Sweep (4 runs)
5. Phase 3: EDL+ORCU λ_orcu Validation (9 runs)
6. Phase 4: 5-Fold CV (3 runs)
7. Results Summary (unified table with per-class F1, competitor comparison)
