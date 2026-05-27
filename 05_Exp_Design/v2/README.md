# V2 Experiments — Systematic Validation (Current Best)

**Period:** 2026-05-22 to 2026-05-27
**Status:** Active (V4 = current best)
**Scope:** DF + SF, ViT-Base (86M), 4-phase experiment pipeline

## Key Results (V4 — 5-fold CV)

| Method | Acc | QWK | ECE | %Unimodal | AUROC(u) |
|--------|-----|-----|-----|-----------|----------|
| **EDL** | **79.33% +/- 3.74** | **0.905 +/- 0.017** | 0.173 +/- 0.038 | 60.67% | 0.664 |
| SORD | 77.67% +/- 4.90 | 0.904 +/- 0.027 | 0.213 +/- 0.038 | 100% | 0.578 |
| EDL+ORCU | 75.00% +/- 5.16 | 0.894 +/- 0.024 | **0.166 +/- 0.057** | 82.33% | 0.431 |

**Best single fold:** EDL Fold 3 = 85.00% Acc, 85.13% F1

## Sub-versions

| Sub-version | Epochs | KL Config | Best EDL CV | Key Finding |
|-------------|--------|-----------|-------------|-------------|
| V3 | 50 | kl=0.02 | 61.67% | KL too weak → EDL collapse |
| **V4** | **75** | **kl=0.10, cap=1.0** | **79.33% +/- 3.74** | **Current best** |
| V5 | 100 | kl=0.05, cap=0.5 | 72.33% +/- 7.90 | KL sweet spot shifts with epochs |

## 4-Phase Pipeline

| Phase | Purpose | Key Finding |
|-------|---------|-------------|
| Phase 1 | Multi-seed benchmark (5 modes x 3 seeds) | EDL+ORCU s123=81.67% best single |
| Phase 2 | KL lambda sweep | kl=0.05 at 75ep = 80.00% sweet spot |
| Phase 3 | ORCU lambda sweep (3x3) | lambda=0.005 better than 0.01 |
| Phase 4 | 5-fold CV (EDL/EDL+ORCU/SORD) | EDL best, no stat. sig. between methods |

## V5 Regression Analysis (Negative Result)

V5 attempted 100 epochs + kl=0.05 → EDL regressed to 72.33%. Lessons:
- KL dose increases with epochs at same lambda (annealing window expands)
- kl=0.07 is the V5 sweet spot (75.00%)
- CE won at 100 epochs (75.83%) — more epochs favor simpler methods
- 10-fold CV enables statistical significance (first time in project)

## Contents

- `plan.md` — V2 6-phase experiment plan
- `reports/` — 9 analysis reports (DF v3/v4/v5, SF, experiment designs)
- `results/df/v4/` — Structured JSON (master summary + per-phase)
- `results/df/v5/` — V5 archived results
- `notebooks/` — 14 Kaggle notebooks (training + analysis)

## V6 Recommendation (→ v3/)

```
EDL:       kl_lambda=0.07, kl_cap=0.5, epochs=75, patience=20
EDL+ORCU:  kl_lambda=0.05, lambda_orcu=0.003, kl_cap=0.5
10-fold CV, KL/lambda sweep first, then full CV
```
