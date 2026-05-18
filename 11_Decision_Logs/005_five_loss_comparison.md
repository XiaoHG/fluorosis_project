# Decision Log #005: Systematic 5-Loss Comparison Design

**Date:** 2026-05-15
**Status:** Final
**Decided by:** Xiaohong Gao

## Context

Need to evaluate EDL+ORCU against fair baselines. Prior fluorosis papers only used CE. No controlled comparison of ordinal loss functions existed for this domain.

## Decision

Run a systematic 5-loss comparison under identical ViT-Base backbone:
1. Cross-Entropy (CE baseline)
2. Cumulative Link Model
3. SORD (Soft-Encoded Ordinal Regression)
4. EDL only
5. EDL + ORCU (proposed)

## Rationale

- Controlled backbone isolates loss function effect from architecture effect
- Covers nominal baseline + pure ordinal (Cumulative, SORD) + uncertainty (EDL) + combined (EDL+ORCU)
- First systematic ordinal loss comparison for small-sample medical image classification

## Consequences

- Revealed EDL's seed robustness advantage (CV 4.3% vs CE 31.4%)
- Showed EDL+ORCU CV stability (QWK CV 0.36%) at cost of single-split accuracy
- Provided practical guidance for loss selection in similar small-sample scenarios
