# Fluorosis DL Diagnosis — Project Guide

## Overview

Endemic fluorosis automated diagnosis using Evidential Deep Learning (EDL) + Ordinal Calibration (ORCU). Two imaging modalities:
- **DF (Dental Fluorosis):** 200 intraoral photos, 4-class (Normal/Mild/Moderate/Severe), ViT-Base backbone
- **SF (Skeletal Fluorosis):** 80 X-rays, 5-radiologist multi-rater labels, ResNet-50 backbone

Target journal: Medical Image Analysis (MedIA, Elsevier). Manuscript in `08_Manuscript/v1_first_draft/`.

## Version Management (CRITICAL)

**One version = one folder under `05_Exp_Design/`.**

```
05_Exp_Design/
├── v1/                    # Initial experiments (DF only, ViT-Base, v1.0-v2.2)
│   ├── reports/           # Analysis reports
│   ├── results/           # .npz prediction files
│   ├── notebooks/         # Kaggle training notebooks
│   └── figures/           # Paper figures (PDF/PNG)
├── v2/                    # Current active (V4 best results, DF+SF)
│   ├── plan.md
│   ├── reports/           # Per-version analysis + SOTA comparison
│   ├── results/
│   │   ├── df/            # v4/ (best JSON), v5/ (archived)
│   │   └── sf/            # SF results
│   ├── notebooks/         # Kaggle training + analysis notebooks
│   └── figures/
├── v3/                    # Next version (V6 experiments)
│   └── ...
├── README.md              # Experiment overview index
└── exp_v1/                # (LEGACY, gitignored — use v1/ instead)
```

**Rules:**
- When starting a new round of experiments, create `vN/` with its own `plan.md`, `reports/`, `results/`, `notebooks/`
- Never scatter version results across root-level files
- Archive old intermediate results as `.zip`; keep final results as structured JSON
- Each version has its own `README.md` documenting what was done and key findings

## Agent Skills

### Experiment Agent (`exp`)
**Role:** Design and analyze experiments for fluorosis DL diagnosis.
**Responsibilities:**
- Plan experiment matrices (loss modes, hyperparameters, seeds, folds)
- Analyze results (Acc, QWK, ECE, F1, AUROC) from JSON/NPZ outputs
- Generate comparison tables and radar charts
- Identify best configurations and regressions
- Output structured analysis reports in `reports/`
**Key knowledge:** EDL (Type II MLE + KL annealing), ORCU (SORD + log-barrier), 5 loss modes (CE/Cumulative/SORD/EDL/EDL+ORCU), ViT-Base 86M ImageNet-21k, DFID 200-image 4-class, 3-stage training

### Implementation Agent (`impl`)
**Role:** Write, debug, and optimize PyTorch training code.
**Responsibilities:**
- Maintain `06_Implementation/` codebase (train.py, eval.py, losses/, models/, data/)
- Fix bugs in EDL loss (KL annealing), ORCU loss (log-barrier), combined loss
- Add new loss modes or backbone support
- Optimize training configs (YAML in configs/)
- Ensure per-sample export (preds.npz with alpha/z/evidence/uncertainty)
**Key knowledge:** PyTorch, ViT (transformers), AdamW, CosineAnnealing, 3-stage training strategy, shared-logit EvidenceHead

### Writing Agent (`write`)
**Role:** Draft and revise manuscript sections in LaTeX.
**Responsibilities:**
- Write/edit sections in `08_Manuscript/v1_first_draft/sections/`
- Maintain Chinese translation (`main_zh.tex`, `manuscript_cn.md`)
- Manage references in `references.bib`
- Ensure consistency between results data and manuscript claims
- Format tables (booktabs) and figures (pdf)
**Key knowledge:** MedIA formatting, LaTeX, bib management, all experiment results and competitor comparisons

### Visualization Agent (`viz`)
**Role:** Generate publication-quality figures from experiment data.
**Responsibilities:**
- Confusion matrices (5-method comparison)
- Reliability diagrams (ECE calibration curves)
- Uncertainty/evidence distributions (ridge density, heatmaps)
- Seed stability boxplots
- Clinical triage workflow diagrams
- Architecture overview figure
**Key knowledge:** matplotlib, seaborn, TikZ, experiment result formats (JSON/NPZ)

### Literature Agent (`lit`)
**Role:** Track and analyze related work in fluorosis DL diagnosis.
**Responsibilities:**
- Maintain literature notes in `02_Literature/`
- Track competitor methods: MLTrMR (80.19%, 556M), LD2Net (80.00%, 3.3M)
- Identify gaps our method fills (no ordinal, no uncertainty, no calibration)
- Update SOTA comparison tables
**Key knowledge:** DFID benchmark, fluorosis grading (Dean's scale), medical image classification, uncertainty quantification

### Review Agent (`review`)
**Role:** Review manuscript and code for quality and correctness.
**Responsibilities:**
- Check statistical rigor (McNemar test, CV reporting)
- Verify all claims have supporting data
- Review LaTeX for formatting issues
- Check code for reproducibility
- Flag overclaims or unsupported statements
**Key knowledge:** MedIA review standards, statistical testing, experiment methodology

## Directory Conventions

| Directory | Purpose | Tracked? |
|-----------|---------|----------|
| `00_Admin/` | Releases, agent prompts, project plans | Yes |
| `02_Literature/` | Literature notes and PDFs | Notes yes, PDFs no |
| `05_Exp_Design/vN/` | Experiment versions (N=1,2,3...) | Yes |
| `06_Implementation/` | Training code, models, losses | Yes |
| `08_Manuscript/v1_first_draft/` | Manuscript source (sections/, figures/) | Yes |
| `11_Decision_Logs/` | Architectural decisions | Yes |
| `data/` | Raw datasets | No (gitignored) |
| `Kaggle_Setup/` | Kaggle env setup | No (gitignored) |

## Current State (2026-05-27)

**Best results (v2 = V4 experiments):**
- EDL CV: Acc 79.33% +/- 3.74, QWK 0.905 +/- 0.017, ECE 0.173 (5-fold)
- EDL single best (V1 seed 42): Acc 83.33%, QWK 0.936, ECE 0.072
- EDL+ORCU CV: Acc 75.00% +/- 5.16, best CV stability (QWK CV 0.36%)

**Competitors:** MLTrMR (80.19%, 556M), LD2Net (80.00%, 3.3M)

**Next steps (V6 → v3/):**
1. DF V6: kl=0.07, 10-fold CV, fix KL annealing from V5 regression
2. SF: ResNet-50 + EDL+ORCU with multi-rater soft labels
3. Paper: integrate V6 results, finalize figures, submit to MedIA

## Key Commands

```bash
# Train DF model
python train.py --mode edl --config configs/df_vit_edl_orcu.yaml

# Cross-validate
python scripts/cross_validate.py --mode edl --folds 5

# Evaluate and export per-sample predictions
python eval.py --checkpoint checkpoints/best.pth --export results/preds.npz

# Generate ablation sweep
python scripts/ablation_runner.py --sweep kl_lambda --values 0.02,0.05,0.10,0.20
```
