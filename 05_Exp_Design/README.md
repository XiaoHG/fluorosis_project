# Experiment Design — Version Index

**One version = one folder.** When starting a new experiment round, create `vN/` following the template below.

## Version Overview

| Version | Status | Scope | Best Result | Period |
|---------|--------|-------|-------------|--------|
| [v1/](v1/) | Archived | DF only, initial EDL+ORCU validation | EDL 83.33% (single seed 42) | May 15-22 |
| [v2/](v2/) | **Active** | DF+SF, 4-phase systematic validation | EDL CV 79.33% +/- 3.74 | May 22-27 |
| [v3/](v3/) | Planned | DF V6 + SF multi-rater | TBD | — |

## New Version Template

```
vN/
├── plan.md              # Experiment plan (hypotheses, matrix, configs)
├── README.md            # Version summary with key results
├── reports/             # Analysis reports (one per sub-version)
├── results/             # Structured outputs (JSON for final, NPZ for raw)
│   ├── df/              # DF experiment results
│   └── sf/              # SF experiment results
├── notebooks/           # Kaggle training + analysis notebooks
└── figures/             # Generated figures
```

## Version Management Rules

1. New experiment round → new `vN/` folder. Never overwrite previous versions.
2. Each version has its own isolated `plan.md`, `reports/`, `results/`.
3. Final results stored as structured JSON; intermediate may be archived as .zip.
4. `vN/README.md` documents what changed, key results, and lessons learned.
5. When archiving, update this index.
