# Release v1.0 — Manuscript First Draft

**Date:** 2026-05-18
**Git Tag:** v1.0_manuscript_v1_draft
**Git Commit:** e75af7b
**Status:** Complete

## Scope

This release archives the complete first draft of the manuscript "Evidential Deep Learning with Ordinal Calibration for Automated Diagnosis of Endemic Fluorosis" plus all supporting research artifacts.

## Deliverables

### A. Manuscript Package
| File | Description |
|------|-------------|
| `08_Manuscript/v1_first_draft/main.tex` | Root LaTeX file (elsarticle, MedIA format) |
| `08_Manuscript/v1_first_draft/main.pdf` | Compiled PDF (43 pages, zero errors) |
| `08_Manuscript/v1_first_draft/sections/01_introduction.tex` | Introduction |
| `08_Manuscript/v1_first_draft/sections/02_related_work.tex` | Related Work |
| `08_Manuscript/v1_first_draft/sections/03_method.tex` | Method |
| `08_Manuscript/v1_first_draft/sections/04_experiments.tex` | Experiments |
| `08_Manuscript/v1_first_draft/sections/05_results.tex` | Results (16 figures, 5 tables) |
| `08_Manuscript/v1_first_draft/sections/06_discussion.tex` | Discussion |
| `08_Manuscript/v1_first_draft/sections/07_conclusion.tex` | Conclusion |
| `08_Manuscript/v1_first_draft/references.bib` | 17 BibTeX references |
| `08_Manuscript/v1_first_draft/figures/` | Figures F1-F16 (symlink to 05_Exp_Design/figures) |

### B. Implementation Code (17 Python files)
| File | Description |
|------|-------------|
| `06_Implementation/train.py` | Main training script |
| `06_Implementation/eval.py` | Evaluation script |
| `06_Implementation/data/dataset.py` | DFID data loader |
| `06_Implementation/models/backbones.py` | ViT backbone factory |
| `06_Implementation/models/evidence_head.py` | Evidence head (shared-logit) |
| `06_Implementation/models/ordinal_head.py` | Ordinal head |
| `06_Implementation/losses/edl_loss.py` | EDL loss (Type II MLE + KL) |
| `06_Implementation/losses/orcu_loss.py` | ORCU loss (SORD + log-barrier) |
| `06_Implementation/losses/cumulative_loss.py` | Cumulative link model loss |
| `06_Implementation/losses/combined_loss.py` | EDL+ORCU combined loss |
| `06_Implementation/scripts/ablation_runner.py` | Ablation study runner |
| `06_Implementation/scripts/cross_validate.py` | 5-fold cross-validation |
| `06_Implementation/scripts/report_generator.py` | Auto diagnosis report generator |
| `06_Implementation/scripts/uncertainty_baselines.py` | Uncertainty baseline methods |

### C. Research Documentation
| File | Description |
|------|-------------|
| `02_Literature/` | 20 deep reads, comparison table, gap analysis |
| `03_Innovation/01_innovation_blueprint.md` | Innovation blueprint |
| `03_Innovation/02_idea_proposals/auto_diagnosis_report.md` | Auto-diagnosis report design |
| `04_Model_Design/01_model_architecture.md` | Model architecture spec |
| `05_Exp_Design/01_experiment_plan.md` | Experiment plan |
| `05_Exp_Design/figures/` | 34 generated figures |

### D. Submission Package
| File | Description |
|------|-------------|
| `10_Submission/cover_letter.md` | Cover letter draft |
| `10_Submission/highlights.md` | Research highlights |
| `10_Submission/author_statement.md` | Author statement |
| `10_Submission/graphical_abstract_brief.md` | Graphical abstract brief |
| `10_Submission/supplementary_material_plan.md` | Supplementary material plan |
| `10_Submission/response_to_reviewers_template.md` | Response template |

### E. Project Governance
| File | Description |
|------|-------------|
| `11_Decision_Logs/` | 5 key decision records |
| `PROJECT_FINAL_REPORT.md` | Final progress report |
| `09_Review/` | Internal review + submission gap analysis |

## Key Results

| Metric | Value |
|--------|-------|
| EDL Accuracy | 83.33% (SOTA, +3.14pp) |
| CE Baseline | 81.67% |
| EDL ECE | 0.0719 (40.7% vs CE) |
| EDL Seed CV | 4.3% (vs CE 31.4%) |
| Dataset | DFID: 200 images, 4 classes |

## Git History (v1.0 relevant commits)

```
e75af7b Add final project progress report
f974ac1 Add decision logs documenting key project choices
fa37b62 Fix frontmatter, insert all 16 figures, eliminate overfull hbox warnings
9e906c1 Write complete manuscript v1 draft with all v2.2 experimental data
```
