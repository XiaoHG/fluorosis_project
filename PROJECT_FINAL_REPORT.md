# Fluorosis DL Project — Final Progress Report

**Date:** 2026-05-18
**Author:** Xiaohong Gao
**Repository:** github.com/XiaoHG/fluorosis_project

---

## Project Overview

Automated diagnosis of endemic fluorosis using Evidential Deep Learning with Ordinal Calibration. Target journal: Medical Image Analysis (MedIA).

## Module Status

### 01_Knowledge_Base — Domain Knowledge ✅
- Fluorosis clinical background, 4-grade ordinal scale (Normal/Mild/Moderate/Severe)
- DF (dental) and SF (skeletal) distinctions documented

### 02_Literature — Literature Review ✅
- 20 deep-read papers with structured notes
- 19 literature files covering DF diagnosis, SF diagnosis, EDL methods, ordinal regression, uncertainty quantification
- Research gap analysis: no prior work combines EDL + ordinal + fluorosis
- Literature comparison table across 12+ methods
- Key references: MLTrMR (80.19%), LD2Net (80.00%), Mwinc-Mamba (SF)

### 03_Innovation — Innovation Design ✅
- Innovation blueprint: EDL + ORCU shared-logit framework
- Auto-diagnosis report design with uncertainty-gated triage
- Gap analysis: uncertainty, ordinality, small-sample robustness

### 04_Model_Design — Model Architecture ✅
- Evidence head design (shared-logit: EDL path + ORCU path)
- Backbone-agnostic architecture
- Loss function specifications: EDL (Type II MLE + KL), ORCU (SORD + log-barrier)

### 05_Exp_Design — Experiment Design ✅
- Experiment plan: 5-loss comparison, multi-seed, cross-validation, ablation
- 34 figures generated (F1-F16 used in manuscript)
- All figures in PDF format, symlinked to manuscript

### 06_Implementation — Code ✅
- **17 Python files** across data/models/losses/scripts:
  - Data: dataset.py (DFID loading, augmentations, 120/40/40 split)
  - Models: backbones.py, evidence_head.py, ordinal_head.py
  - Losses: edl_loss.py, orcu_loss.py, cumulative_loss.py, combined_loss.py
  - Scripts: train.py, eval.py, ablation_runner.py, cross_validate.py, report_generator.py, uncertainty_baselines.py

### 07_Visualization — Figures ✅
- All 34 figures generated and stored

### 08_Manuscript — Manuscript ✅
- **v1_first_draft complete** (43 pages)
- 7 sections: Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion
- 17 references in BibTeX
- 16 figures inserted (F1-F16) with proper labels and cross-references
- 5 tables: SOTA comparison, main results, multi-seed, 5-fold CV, ablation
- elsarticle format with frontmatter wrapper
- Final PDF: **zero errors, zero warnings, zero overfull hboxes**

### 09_Review — Pre-submission Review ✅
- Internal review checklist
- Code review checklist
- Submission gap analysis

### 10_Submission — Submission Package ✅
- Cover letter draft
- Highlights (3-5 bullet points)
- Author statement
- Graphical abstract brief
- Response to reviewers template
- Supplementary material plan

### 11_Decision_Logs — Decision Records ✅
- 5 key decisions documented: methodology, backbone, journal, scope, experiment design

---

## Key Results

| Metric | Value | Context |
|--------|-------|---------|
| EDL Accuracy | **83.33%** | New SOTA on DFID (+3.14pp over MLTrMR) |
| CE Baseline | 81.67% | Already beats MLTrMR (80.19%) |
| EDL ECE | 0.0719 | 40.7% reduction vs CE (0.1213) |
| EDL Seed CV | **4.3%** | vs CE 31.4% (7.3x more stable) |
| EDL+ORCU QWK CV | **0.36%** | Best cross-validation stability |
| Dataset | 200 images | 120 train / 40 val / 40 test |

---

## Remaining Items (Post-v1)

- [ ] External multi-center validation (India, East Africa datasets)
- [ ] Multi-rater DF annotations for inter-dentist variability study
- [ ] SF (skeletal fluorosis) experiments with 5-radiologist annotations
- [ ] EDL+ORCU regularizer conflict resolution (decoupled optimization or Dirichlet-space prior)
- [ ] Prospective reader study (AI-assisted vs unassisted dentist diagnosis)
- [ ] Mamba backbone integration for SF (pending CUDA kernel availability)

---

## Git History

```
fa37b62 Fix frontmatter, insert all 16 figures, eliminate overfull hbox warnings
9e906c1 Write complete manuscript v1 draft with all v2.2 experimental data
7b2bdb1 Add v5 comparison + paper figure/tables data audit to v2.2 analysis
c0e079b Fix torch.load weights_only=False for PyTorch 2.6 compatibility
c275d4f Add comprehensive DF experiment suite v5 with per-sample prediction export
```
