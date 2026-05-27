# V1 Experiments — Initial EDL+ORCU Validation

**Period:** 2026-05-15 to 2026-05-22
**Status:** Archived (superseded by v2/V4 best results)
**Scope:** DF only, ViT-Base (86M), 5 loss modes

## Key Results

| Mode | Acc (best single) | QWK | ECE | Notes |
|------|-------------------|-----|-----|-------|
| EDL | **83.33%** (seed 42) | 0.938 | 0.072 | Historical SOTA on DFID |
| CE | 81.67% | 0.933 | 0.121 | Strong baseline |
| EDL+ORCU | 70.00% | 0.872 | 0.252 | Underperformed pure EDL |
| SORD | 73.33% | 0.900 | 0.167 | Safe choice |
| Cumulative | 68.33% | 0.819 | 0.188 | Unstable |

**Multi-seed (3 seeds):**
- EDL: 79.44% +/- 3.42 (CV=4.3%) — extremely stable
- CE: 58.89% +/- 18.48 (CV=31.4%) — highly unstable, seed 456 collapsed to 35%

## Sub-versions

| Sub-version | Key Change | Best EDL |
|-------------|-----------|----------|
| v2.0 | Initial implementation | 78.33% |
| v2.1 | Bug fixes | 80.00% |
| v2.2 | Recovered kl=0.10, final config | 83.33% |

## Contents

- `reports/` — Analysis reports for v2.0, v2.1, v2.2, comprehensive
- `results/` — 14 .npz files (5 modes + 9 lambda sweep combos)
- `notebooks/` — 5 Kaggle training notebooks
- `figures/` — 20 figures (F1-F10 PDF/PNG) + generation script

## Critical Findings

1. EDL at 83.33% is the highest ever reported on DFID benchmark
2. KL regularization provides implicit initialization robustness (CV 4.3% vs CE 31.4%)
3. EDL+ORCU underperformance due to KL vs log-barrier competition
4. Single-seed results are unreliable — need multi-seed + CV validation (addressed in v2)
