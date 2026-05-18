# Decision Log #001: Core Methodology — EDL + ORCU

**Date:** 2026-05-15
**Status:** Final
**Decided by:** Xiaohong Gao

## Context

Automated fluorosis diagnosis needed to address three gaps:
1. No uncertainty quantification (all prior methods used CE point estimates)
2. Ordinal structure (Normal < Mild < Moderate < Severe) ignored
3. Small dataset (120 training samples, 200 total)

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Standard CE + ensemble | Simple, well-understood | 5x compute, no ordinal structure |
| CE + temperature scaling | Easy post-hoc fix | Only calibrates confidence, no uncertainty decomposition |
| EDL only | Uncertainty + small-sample regularization | No ordinal constraints |
| EDL + ORCU (chosen) | Uncertainty + ordinal + unimodality | Two regularizers may compete |

## Decision

Adopt EDL + ORCU via shared-logit architecture. Single evidence head outputs logits feeding both Dirichlet (EDL) and ordinal calibration (ORCU) paths.

## Rationale

- EDL's Dirichlet parameterization provides regularization against overfitting on small data
- ORCU's log-barrier ensures unimodal probability distributions (clinically interpretable)
- Shared-logit avoids separate prediction heads, ensuring consistency
- First application of both methods to fluorosis → novelty contribution

## Consequences

- EDL+ORCU underperformed EDL-only on single split (70% vs 83.33%) due to regularizer competition
- EDL+ORCU achieved best 5-fold CV stability (QWK CV 0.36%)
- Future: decoupled optimization or Dirichlet-space ordinal prior
