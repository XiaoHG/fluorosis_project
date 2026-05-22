# V1.0 Release Manifest — 正式发布版本

**Date:** 2026-05-22
**Git Tag:** v1.0
**Git Commit:** f36cde4
**Status:** Complete — Archived

---

## Project Identity

| Field | Value |
|-------|-------|
| Title | Evidential Deep Learning with Ordinal Calibration for Automated Diagnosis of Endemic Fluorosis |
| Target Journal | Medical Image Analysis (MedIA, Elsevier) |
| Framework | EDL (Evidential Deep Learning) + ORCU (Ordinal Regression with Calibrated Uncertainty) |
| Task | 4-grade ordinal classification (Normal / Mild / Moderate / Severe) |
| Dataset | DFID: 200 dental fluorosis images |

---

## Deliverables Inventory

### A. Manuscript — English (v1.0-submission)

| File | Description |
|------|-------------|
| `08_Manuscript/v1_first_draft/main.tex` | Root LaTeX (elsarticle, MedIA format) |
| `08_Manuscript/v1_first_draft/main.pdf` | Compiled PDF, 34 pages (1.2 MB, zero errors) |
| `08_Manuscript/v1_first_draft/sections/01-07*.tex` | 7 sections: Introduction through Conclusion |
| `08_Manuscript/v1_first_draft/references.bib` | 31 BibTeX entries |
| `08_Manuscript/v1_first_draft/figures/` | 16 figures (F1–F16, PDF) |

### B. Manuscript — Chinese Translation

| File | Description |
|------|-------------|
| `08_Manuscript/v1_first_draft/manuscript_overleaf_zh/` | Full Chinese translation with all 16 figures |

### C. Implementation Code (17 Python files)

| File | Description |
|------|-------------|
| `06_Implementation/train.py` | Main training loop |
| `06_Implementation/eval.py` | Evaluation script |
| `06_Implementation/data/dataset.py` | DFID data loader (120/40/40 split) |
| `06_Implementation/data/__init__.py` | Data module init |
| `06_Implementation/models/backbones.py` | ViT backbone factory (vit_base, vit_large) |
| `06_Implementation/models/evidence_head.py` | Shared-logit evidence network |
| `06_Implementation/models/ordinal_head.py` | Ordinal regression head |
| `06_Implementation/models/__init__.py` | Models module init |
| `06_Implementation/losses/edl_loss.py` | EDL (Type II MLE + KL divergence) |
| `06_Implementation/losses/orcu_loss.py` | ORCU (SORD + log-barrier) |
| `06_Implementation/losses/cumulative_loss.py` | Cumulative link model loss |
| `06_Implementation/losses/combined_loss.py` | EDL+ORCU combined loss |
| `06_Implementation/losses/__init__.py` | Losses module init |
| `06_Implementation/scripts/ablation_runner.py` | Ablation study runner |
| `06_Implementation/scripts/cross_validate.py` | 5-fold cross-validation |
| `06_Implementation/scripts/report_generator.py` | Auto-diagnosis report |
| `06_Implementation/scripts/uncertainty_baselines.py` | Baseline uncertainty methods |

### D. Experiment Results

| Item | Count | Location |
|------|-------|----------|
| Figures generated | 34 | `07_Visualization/02_final_figures/` |
| Manuscript figures (F1–F16) | 16 | `08_Manuscript/v1_first_draft/figures/` |
| Experiment configs | 5+ | `06_Implementation/configs/` |
| Ablation results | 6 variants | `06_Implementation/results/` |
| Cross-validation | 5-fold | `06_Implementation/results/` |

### E. Literature Review

| Item | Count |
|------|-------|
| Deep-read papers | 20 |
| Comparison table | 12+ methods |
| Gap analysis | 3 gaps identified |
| Dataset analysis | DF + SF survey |

### F. Submission Package

| File | Description |
|------|-------------|
| `10_Submission/cover_letter.md` | Cover letter to MedIA editor |
| `10_Submission/highlights.md` | 5 research highlights |
| `10_Submission/author_statement.md` | Author contributions |
| `10_Submission/graphical_abstract_brief.md` | Graphical abstract |
| `10_Submission/supplementary_material_plan.md` | Supplementary material |
| `10_Submission/response_to_reviewers_template.md` | Response template |

### G. Project Governance

| File | Description |
|------|-------------|
| `11_Decision_Logs/001-005*.md` | 5 key decision records |
| `PROJECT_FINAL_REPORT.md` | Final progress report (all modules) |
| `PROJECT_RETROSPECTIVE.md` | Comprehensive retrospective |

---

## Key Results Summary

| Metric | CE Baseline | EDL (Ours) | Improvement |
|--------|:----------:|:----------:|:-----------:|
| Accuracy | 81.67% | **83.33%** | +1.66 pp |
| Quadratic Weighted Kappa | 0.9329 | **0.9376** | +0.0047 |
| ECE (Expected Calibration Error) | 0.1213 | **0.0719** | −40.7% |
| Seed Stability (Acc CV) | 31.4% | **4.3%** | 7.3× improvement |

---

## Archive ZIP Files

| File | Size | Description |
|------|------|-------------|
| `00_Admin/releases/v1.0/manuscript_source_v1.0.zip` | 1.0 MB | v1 manuscript source + figures |
| `00_Admin/releases/v1.0/implementation_code_v1.0.zip` | 33 KB | All 17 Python files |
| `00_Admin/releases/v1.0-submission/manuscript_source_v1.0-submission.zip` | 588 KB | Submission-ready source |

---

## Git Tags

| Tag | Commit | Date | Description |
|-----|--------|------|-------------|
| `v0.0_project_init` | `05b770c` | 2026-05-15 | Initial project setup |
| `v1.0_manuscript_v1_draft` | `e75af7b` | 2026-05-18 | First manuscript complete draft |
| `v1.0-submission` | `478693c` | 2026-05-18 | Submission-ready (on `b2fe90e`) |
| `v1.0` | `f36cde4` | 2026-05-22 | **Final v1 release (this version)** |

---

*Archived 2026-05-22 — Project v1 is complete and frozen.*
