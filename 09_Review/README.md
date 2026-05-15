# 09_Review — Pre-Submission Review

## Documents

| File | Purpose |
|------|---------|
| `internal_review_checklist.md` | 42-point checklist: method, experiments, writing, reproducibility |
| `code_review_checklist.md` | Code quality: correctness, architecture, missing features, test coverage |
| `submission_gap_analysis.md` | Prioritized gaps (P0-P3) blocking submission |

## Key Findings

- **Overall readiness**: ~70%
- **P0 blocking issues**: 4 (no experimental data, no CV/t-tests, SF proxy validity, missing uncertainty baselines)
- **P1 important gaps**: 4 (incomplete references, OOD data, extra baselines, multi-rater asymmetry)
- **P2 nice-to-have**: 5 (external validation, reader study, supplementary, visual abstract, cover letter)

## Recommended Execution Order

1. Run SF experiment first (fastest, verifies pipeline)
2. Run DF experiment (larger dataset, main results)
3. Implement 5-fold CV + t-tests
4. Add MC Dropout + Deep Ensemble comparison
5. Run all ablation experiments
6. Fill manuscript tables with real numbers
