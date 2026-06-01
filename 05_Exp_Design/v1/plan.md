# V1 Experiment Plan

**Version:** v1
**Period:** 2026-05-15 to 2026-05-22
**Status:** Archived (superseded by v2/V4 best results)
**Scope:** DF only, ViT-Base (86M), 5 loss modes

## Objective
Initial validation of EDL + ORCU on dental fluorosis diagnosis (DFID dataset).
Compare 5 loss modes: CE, Cumulative, SORD, EDL, EDL+ORCU.

## Experiment Matrix

| Mode | Epochs | Seeds | Notes |
|------|--------|-------|-------|
| CE | 100 | 42, 123, 456 | Standard baseline |
| Cumulative | 100 | 42, 123, 456 | Coral K-1 BCE |
| SORD | 100 | 42, 123, 456 | Soft ordinal encoding |
| EDL | 100 | 42, 123, 456 | KL lambda = 0.10 |
| EDL+ORCU | 100 | 42, 123, 456 | lambda_orcu = 0.5, lambda_kl = 0.10 |

## Sub-versions

- v2.0: Initial implementation (Acc 78.33%)
- v2.1: Bug fixes, EDL unimodality collapsed to 28.3%
- v2.2: Recovered kl=0.10, final config (Acc 83.33%)

## Key Findings

- EDL single best: Acc 83.33%, QWK 0.938, ECE 0.072 (seed 42)
- EDL multi-seed: 79.44% +/- 3.42 (CV=4.3%) — extremely stable
- CE: 58.89% +/- 18.48 (CV=31.4%) — highly unstable
- EDL+ORCU underperformed pure EDL (70.00% vs 83.33%)
