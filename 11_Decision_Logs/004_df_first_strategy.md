# Decision Log #004: DF-First Strategy

**Date:** 2026-05-15
**Status:** Final
**Decided by:** Xiaohong Gao

## Context

Two fluorosis datasets: DF (dental, 200 images, single-rater) and SF (skeletal X-ray, 5-radiologist annotations). Needed scope for v1 manuscript.

## Decision

Focus v1 on DF only. Defer SF to future work.

## Rationale

- DF has published benchmarks (MLTrMR, LD2Net, FusionDentNet, HiFuse) enabling direct SOTA comparison
- SF experiments incomplete at time of writing
- v1 establishes EDL+ORCU framework; SF with multi-rater labels is natural extension

## Consequences

- All 5-loss comparisons and SOTA table use DFID dataset
- SF mentioned only in Discussion as future work
- SF multi-rater annotations (5 radiologists, 54.1% pairwise agreement) remain unexploited
