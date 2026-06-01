# V3 — V6 Experiments

**Status:** Planned
**Start Date:** TBD
**Objective:** DF V6 (KL annealing fix + log-barrier ORCU + 10-fold CV) + SF multi-rater soft-label EDL

## Key Changes from V4 (v2)

- ORCU: hinge → log-barrier penalty (matches design doc)
- KL lambda: 0.1 → 0.07
- Cross-validation: 5-fold → 10-fold
- SF: hard labels → soft 5-rater labels
- Code: EarlyStopping, CUDA determinism, AMP, YAML config loading

## Expected Deliverables

- 5 loss modes × 10-fold CV results
- Lambda sweep (kl, orcu)
- A6 ablation (log-barrier vs hinge)
- 5-seed stability analysis
- SF soft vs hard label comparison
