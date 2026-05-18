# Release v1.0-submission — 投稿版本

**Date:** 2026-05-18
**Git tag:** v1.0-submission
**Status:** Ready for submission to Medical Image Analysis (MedIA)

---

## Manuscript

| Item | Description |
|------|-------------|
| Title | Evidential Deep Learning with Ordinal Calibration for Automated Diagnosis of Endemic Fluorosis |
| Target Journal | Medical Image Analysis (MedIA, Elsevier) |
| Pages | 34 |
| Sections | 7 (Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion) |
| Figures | 16 (F1–F16) |
| Tables | 5 |
| References | 31 BibTeX entries |

## Key Results

| Metric | CE (Baseline) | EDL (Ours) |
|--------|:------------:|:----------:|
| Accuracy | 81.67% | **83.33%** |
| QWK | 0.9329 | **0.9376** |
| ECE | 0.1213 | **0.0719** (−40.7%) |
| Seed CV (Acc) | 31.4% | **4.3%** |

## Archive Contents

```
manuscript_source_v1.0-submission.zip (588 KB)
├── main.tex                    Root LaTeX file (elsarticle)
├── references.bib              31 BibTeX entries
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_method.tex
│   ├── 04_experiments.tex
│   ├── 05_results.tex
│   ├── 06_discussion.tex
│   └── 07_conclusion.tex
└── figures/
    ├── F1_confusion_matrices.pdf
    ├── F2_reliability.pdf
    ├── F3_uncertainty.pdf
    ├── F4_evidence.pdf
    ├── F5_mode_comparison.pdf
    ├── F6_lambda_sweep.pdf
    ├── F7_temperature.pdf
    ├── F8_prediction_dist.pdf
    ├── F9_summary.pdf
    ├── F10_per_class.pdf
    ├── F11_architecture.pdf
    ├── F12_multiseed.pdf
    ├── F13_sota_comparison.pdf
    ├── F14_autodiagnosis.pdf
    ├── F15_clinical_triage.pdf
    └── F16_example_reports.pdf
```

Also included: `main.pdf` (34 pages, 1.2 MB, zero errors/warnings)

## Revision History

| Commit | Description |
|--------|-------------|
| `2d35eda` | Complete Priority 1 revisions for submission readiness |
| `9f74dcb` | Project retrospective report |
| `da1ec2c` | Pre-submission review fixes (remove versions, soften claims, fix notation) |
| `c3c64dd` | Expand references 10→31 citations |
| `223a5ed` | Formatting fixes (line numbers, keywords, package order) |
| `e75af7b` | Complete manuscript v1 first draft |

## Revision Changes Since v1 First Draft

1. Removed all version info (v2.2/v6/PyTorch refs)
2. Distinguished Dirichlet vs softmax probability notation
3. Softened SOTA/generalization claims
4. Added AUROC(u) and per-bin uncertainty analysis in Results
5. Added McNemar statistical significance test discussion
6. Added label smoothing vs KL regularization analysis
7. Removed unverifiable FusionDentNet citation
8. Filled acknowledgment placeholders

---

*Generated 2026-05-18*
