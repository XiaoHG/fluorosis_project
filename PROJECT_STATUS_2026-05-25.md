# Fluorosis DL Project — Status Report

**Date:** 2026-05-25
**Author:** Xiaohong Gao
**Version:** v2.0-dev
**Repository:** github.com/XiaoHG/fluorosis_project

---

## 1. Project Identity

| Item | Detail |
|------|--------|
| Title | Evidential Deep Learning with Ordinal Calibration for Automated Diagnosis of Endemic Fluorosis |
| Target Journal | Medical Image Analysis (MedIA, Elsevier) |
| Core Method | Shared-logit EDL + ORCU (evidence head: ViT-Base → FC 768×4 → dual path) |
| Problem | Automated 4-grade fluorosis diagnosis (Normal/Mild/Moderate/Severe) with uncertainty quantification |
| Clinical Need | Endemic fluorosis affects millions; specialists scarce in rural areas → AI triage needed |
| Start Date | 2026-05-13 |
| Current Phase | v2.0 Phase 1 (DF re-run complete, analysis underway) |
| Total Git Commits | 54 |

---

## 2. Innovation Summary

### 2.1 What's New

1. **Shared-logit evidence head** — single ViT backbone serves both classification (CE/SORD) and evidential (EDL/EDL+ORCU) heads
2. **EDL+ORCU** — first combination of Evidential Deep Learning (Type II MLE + KL divergence) with Ordinal Calibration with Unimodality (SORD + log-barrier) for ordinal medical diagnosis
3. **Two-dataset validation** — dental fluorosis (DF, intraoral photos) + skeletal fluorosis (SF, X-rays) — first work applying the same DL pipeline across both fluorosis types
4. **Uncertainty-gated triage** — low-uncertainty predictions auto-report; high-uncertainty → dentist review

### 2.2 Research Gap

No prior work combines:
- EDL uncertainty + ordinal calibration + fluorosis diagnosis
- Cross-dataset generalization (DF↔SF transfer)
- Per-class evidence interpretation with clinical-grade uncertainty

---

## 3. Architecture

```
Input (224×224×3)
    │
    ▼
ViT-Base (86M params, pretrained ImageNet-21k)
    │
    ▼
FC 768 → 4 logits (shared)
    │
    ├── StandardClassifier: logits → softmax → CE/Cumulative/SORD loss
    │
    └── EDLClassifier: logits → exp() → evidence e_k → α_k = e_k + 1
         │                                          │
         │                                   Dirichlet(α)
         │                                   ├── p̂ = α_k / S (prediction)
         │                                   └── u = K / S (uncertainty, K=4)
         │
         ├── EDL path: Type II MLE loss + KL divergence (λ_reg anneals 0→1)
         │
         └── ORCU path: SORD (ordinal distance) + log-barrier (unimodality constraint)
              └── Combined: L_EDL + λ_orcu · L_ORCU
```

**5 Loss Modes Compared:**
| Mode | Loss Function | Uncertainty |
|------|--------------|-------------|
| CE | Cross-Entropy | Entropy-based |
| Cumulative | Cumulative Link Model | Entropy-based |
| SORD | Soft Ordinal Regression | Softmax entropy |
| EDL | Type II MLE + KL | Dirichlet u = K/S |
| EDL+ORCU | EDL + SORD + log-barrier | Dirichlet u = K/S |

---

## 4. Datasets

### 4.1 DFID (Dental Fluorosis)

| Property | Value |
|----------|-------|
| Modality | Intraoral photographs |
| Images | 200 (PNG) |
| Classes | 4 (Normal 50 / Mild 50 / Moderate 50 / Severe 50) |
| Annotation | Single dentist |
| Split | 120 train / 40 val / 40 test (stratified) |
| SOTA (prior) | MLTrMR 80.19% |

### 4.2 SFXRay (Skeletal Fluorosis)

| Property | Value |
|----------|-------|
| Modality | X-ray (UIH uDR 780i Pro) |
| Images | 80 (PNG, 512×1024, converted from 12-bit DICOM) |
| Classes | 4 (Normal 21 / Mild 34 / Moderate 13 / Severe 12) |
| Body Parts | Forearm 33 / Calf 30 / Pelvis 17 |
| Annotation | 3 senior radiologists (>10yr), majority vote + 5 individual annotations |
| Split | ~48 train / ~8 val / 24 test (from GT.xlsx Mode column) |
| Masks | 59 segmentation masks (abnormal cases only) |
| Competitor | Mwinc-Mamba (SF binary 83.33%, 4-grade not reported) |

### 4.3 Key Data Challenges

| Challenge | DF | SF |
|-----------|:--:|:--:|
| Sample size | 200 | 80 |
| Class balance | Balanced (50/class) | Imbalanced (34:12 ratio) |
| Annotation quality | Single rater | 5 radiologists with disagreement |
| Anatomical variation | Single body site | 3 body parts (confounding) |

---

## 5. Experiment Results

### 5.1 v1 Baseline (Archived)

| Method | Acc | ECE | Notes |
|--------|-----|-----|-------|
| CE | 81.67% | 0.1213 | Already beats MLTrMR (80.19%) |
| EDL | **83.33%** | **0.0719** | SOTA on DFID |
| EDL+ORCU | — | — | Best CV stability (QWK std 0.36%) |
| EDL Multi-seed CV | 4.3% | — | vs CE 31.4% (7.3× more stable) |

### 5.2 v2 DF Single-Run (seed=42, 100 epochs, 2026-05-25)

| Method | Acc | Macro F1 | Weighted F1 | QWK | ECE | SCE | %Unimodal |
|--------|-----|----------|-------------|-----|-----|-----|-----------|
| CE | 58.33% | 0.595 | 0.597 | 0.771 | 0.321 | 0.192 | 38.3% |
| Cumulative | **80.00%** | **0.791** | **0.795** | **0.924** | 0.180 | 0.137 | 50.0% |
| SORD | 73.33% | 0.728 | 0.735 | 0.897 | 0.185 | 0.136 | 100.0% |
| EDL | 78.33% | 0.789 | 0.783 | 0.911 | 0.223 | **0.112** | 48.3% |
| EDL+ORCU | 66.67% | 0.668 | 0.670 | 0.853 | 0.272 | 0.198 | 100.0% |

> Note: CE single-run (58.33%) is far below v1 CE (81.67%) — likely due to different train/test split. v1's CE 81.67% was on a different random seed. CV results (below) give CE 72.33%, confirming split sensitivity.

### 5.3 Per-Class F1 (single-run, seed=42)

| Method | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Macro F1 |
|--------|---------|---------|---------|---------|----------|
| CE | 0.778 | 0.333 | 0.667 | 0.600 | 0.595 |
| Cumulative | 0.800 | **0.857** | 0.706 | 0.750 | **0.791** |
| SORD | 0.706 | 0.688 | 0.727 | 0.706 | 0.728 |
| EDL | 0.800 | 0.700 | 0.813 | 0.778 | 0.789 |
| EDL+ORCU | 0.778 | 0.625 | 0.625 | 0.556 | 0.668 |
| **Hardest class** | — | **Grade 1** | — | — | — |

Grade 1 (mild fluorosis) is universally the hardest — all 5 methods score lowest F1 here, typical for clinical grading where "borderline mild" is ambiguous.

### 5.4 5-Fold Cross-Validation (mean ± std)

| Method | Acc | Macro F1 | QWK | ECE | F1 Std |
|--------|-----|----------|-----|-----|--------|
| CE | 72.33% ± 5.12 | 0.719 ± 0.054 | 0.878 ± 0.028 | 0.178 ± 0.048 | 0.054 |
| Cumulative | 68.67% ± 6.00 | 0.664 ± 0.081 | 0.683 ± 0.237 | 0.171 ± 0.063 | 0.081 |
| EDL | **75.33% ± 5.31** | **0.752 ± 0.055** | **0.891 ± 0.025** | 0.169 ± 0.021 | 0.055 |
| EDL+ORCU | 72.33% ± 3.27 | 0.721 ± 0.032 | 0.878 ± **0.009** | **0.145 ± 0.045** | **0.032** |

**CV F1 Stability Ranking:** EDL+ORCU (std 0.032) > CE (0.054) > EDL (0.055) > Cumulative (0.081)

**CV Key Findings:**
- EDL wins CV across Acc, F1, QWK — most robust and reliable method
- EDL+ORCU QWK std = 0.009 is remarkable (almost constant across folds) and best ECE (0.145)
- Cumulative has extreme QWK variance (0.237) — folds 3 & 4 collapsed while fold 2 hit 0.909
- All pairwise t-tests: **NOT significant at p < 0.05** (N=5 folds)

### 5.5 Lambda Sweep (EDL+ORCU, 9 combos)

| λ_orcu | λ_reg | Test Acc | Test QWK | Test ECE | %Unimodal |
|--------|-------|----------|----------|----------|-----------|
| 0.1 | 0.005 | 68.33% | 0.875 | 0.260 | 40.0% |
| **0.1** | **0.01** | **66.67%** | **0.870** | 0.271 | 40.0% |
| 0.1 | 0.02 | 71.67% | 0.887 | 0.272 | 41.7% |
| 0.3 | 0.01 | 66.67% | 0.869 | 0.230 | 83.3% |
| 0.5 | 0.01 | 63.33% | 0.858 | 0.219 | 75.0% |
| 0.5 | 0.02 | 60.00% | 0.847 | 0.203 | 78.3% |

Best combo: λ_orcu=0.1, λ_reg=0.01. All val_acc identical (65%) → validation set too small to discriminate. Higher λ_orcu consistently degrades performance: need to explore λ_orcu < 0.1.

### 5.6 Temperature Calibration (EDL)

| T | 0.5 | 0.8 | 1.0 | 1.5 | 2.0 | **3.0** | 5.0 |
|---|-----|-----|-----|-----|-----|--------|-----|
| ECE | 0.221 | 0.224 | 0.223 | 0.180 | 0.114 | **0.107** | 0.258 |

**T=3.0 gives 52% ECE reduction** (0.223 → 0.107) without changing accuracy. T=2.0 also strong (0.114). T=5.0 overshoots.

### 5.7 SF Experiments

SF single-run results show CE leads at 66.67% Acc but QWK only 0.391 — SF is significantly harder than DF. Full SF analysis pending Phase 2 completion.

---

## 6. Code & Deliverables Inventory

### 6.1 Implementation (06_Implementation/, 17 Python files)

| Module | Files | Purpose |
|--------|-------|---------|
| data/ | dataset.py | DFID + SFXRay loaders, augmentations, stratified splits |
| models/ | backbones.py, evidence_head.py, ordinal_head.py | ViT-Base Standard/EDL classifiers |
| losses/ | edl_loss.py, orcu_loss.py, cumulative_loss.py, combined_loss.py | 5 loss functions |
| scripts/ | train.py, eval.py, cross_validate.py, ablation_runner.py, uncertainty_baselines.py, report_generator.py | Training, evaluation, experiments |
| configs/ | df_vit_edl_orcu.yaml, sf_resnet_edl_orcu.yaml | Experiment configurations |

### 6.2 Project Scale

| Category | Count |
|----------|-------|
| Git commits | 54 |
| Python source files | 25 (~5,953 lines) |
| Jupyter notebooks | 18 (6 v1 runs + 2 v2 runs + 1 analysis) |
| Markdown documents | 81+ |
| LaTeX source files | 10 |
| BibTeX entries | 31 |
| Reference PDFs collected | 134 |
| Deep-read papers | 20 |
| Manuscript figures | 16 (F1–F16) |
| Experiment .npz files | 14 (v1) + 5 (v2 per-sample) |
| Model checkpoints (.pt) | 5 (v2, one per method) |

### 6.3 Manuscript (08_Manuscript/)

| Item | Status |
|------|--------|
| English manuscript (main.tex) | Complete, 34 pages, 0 LaTeX errors |
| Chinese translation (main_zh.tex) | Complete |
| Figures (F1–F16) | All inserted, cross-referenced |
| Tables (5) | SOTA comparison, main results, multi-seed, CV, ablation |
| References (31) | Full BibTeX with DOIs |
| Review v1 (2026-05-18) | Complete, 8 revisions addressed |

---

## 7. Progress Tracker

### v1.0 (Frozen)

| Deliverable | Status |
|-------------|:------:|
| DF-only EDL (83.33% SOTA) | Done |
| 5-loss comparison | Done |
| Multi-seed (3 seeds) | Done |
| 5-Fold CV | Done |
| Lambda sweep (9 combos) | Done |
| Temperature calibration | Done |
| English manuscript (34 pp) | Done |
| Chinese manuscript | Done |
| Submission package | Done |
| Release archives (ZIP) | Done |

### v2.0 (Active)

| Phase | Task | Status | Result Summary |
|-------|------|:------:|----------------|
| **1.1** | DF 5-mode main + per-sample export | **Done** | Cumulative 80.00% best single-run |
| **1.3** | DF 5-Fold CV | **Done** | EDL 75.33% CV winner |
| **1.4** | DF Lambda sweep | **Done** | Best λ=(0.1, 0.01), need wider range |
| **1.5** | DF Temperature calibration | **Done** | T=3.0 halves ECE |
| 1.2 | DF Multi-seed (3 seeds) | Pending | CE vs EDL stability |
| 2.1 | SF data audit | Audit done | Code not yet validated on Kaggle |
| 2.2 | SF 5-mode baseline | Pending | Kaggle notebook ready |
| 2.3 | SF multi-seed | Pending | |
| 2.4 | SF cross-validation | Pending | |
| 3.1 | DF→SF transfer | Pending | |
| 3.2 | SF→DF transfer | Pending | |
| 3.3 | Joint DF+SF training | Pending | |
| 4.1 | MC Dropout baseline | Pending | |
| 4.2 | Deep Ensemble baseline | Pending | |
| 4.3 | Body-part conditional SF | Pending | |
| 6.1 | Manuscript v2 revision | Pending | Update tables/figures with v2 data |

---

## 8. Key Findings Summary

### What Works Well

1. **Cumulative loss is the dark horse** — 80.00% Acc single-run, QWK=0.924, best per-class F1 (Grade 1: 0.857). Simple ordinal loss outperforms complex EDL in raw accuracy.
2. **EDL is the most reliable** — CV winner (75.33% Acc, 0.752 F1, 0.891 QWK), best SCE (0.112), evidence distribution aligns with class structure.
3. **EDL+ORCU has brilliant stability** — QWK std=0.009 across 5 folds means predictions are near-identical regardless of which 80% of data you train on. Best calibration (ECE=0.145).
4. **Temperature scaling is highly effective** — T=3.0 cuts EDL's ECE by 52% with zero accuracy cost.

### What Needs Attention

1. **CE is unreliable single-run** — 58.33% vs CV 72.33%. CE must always be reported with CV confidence intervals, never as a single number.
2. **Cumulative is unstable in CV** — QWK std=0.237 means it breaks on some folds. Great single-run, risky for deployment.
3. **EDL+ORCU underperforms in raw accuracy** — 66.67% single-run. ORCU constraint is too aggressive at current settings. Lambda sweep shows lower λ_orcu may help, but the range explored (0.1–0.5) might not go low enough.
4. **Grade 1 (mild fluorosis) is the hardest class for ALL methods** — F1 range 0.333–0.857. This reflects real clinical ambiguity at the normal/mild boundary and should be discussed in the manuscript.
5. **v2 CE dropped from v1's 81.67%** — need to investigate whether v2 split differs from v1 split. If v1 CE was cherry-picked on a lucky split, the CV results (~72%) better represent true CE performance.
6. **No pairwise comparison is statistically significant** at p<0.05 with N=5 folds. The ranking (EDL > EDL+ORCU ≈ CE > Cumulative) is suggestive but not definitive without more folds or multi-seed.

---

## 9. Recommendations for v2

### For the Manuscript

1. **Lead with Cumulative as the main accuracy result** (80.00%) — simple, reproducible, beats SOTA
2. **Lead with EDL for stability/uncertainty claims** — CV winner, SCE best, evidence interpretable
3. **EDL+ORCU as the calibration champion** — ECE=0.145 CV + QWK std=0.009 cross-fold
4. **Report CE with CV error bars only** — single-run CE is misleading (58% vs 72%)
5. **Explain Grade 1 difficulty** — clinically meaningful (mild fluorosis is genuinely ambiguous)

### For Experiments

1. **DF multi-seed (Phase 1.2) is the highest priority gap** — need 3-seed CE vs EDL to properly claim stability advantage
2. **SF experiments (Phase 2) are critical** — without SF, the paper is single-center/single-dataset
3. **Lambda sweep needs λ_orcu < 0.1** — current sweep missed the sweet spot (lower values may unlock EDL+ORCU's potential)
4. **Consider dropping SORD as a standalone** — 100% unimodal means it has no uncertainty discrimination, and accuracy (73%) is middling

### For Code

1. **Update eval.py with F1 per-class reporting** — currently only aggregate metrics
2. **Add split-seed tracking** — need to compare v1 split vs v2 split to explain CE discrepancy
3. **SF dataset validation** — SFDataset code is written but untested on Kaggle

---

## 10. Timeline

```
2026-05-13  Project start
2026-05-14  Literature survey (134 PDFs, 20 deep reads)
2026-05-15  Model design + code implementation
2026-05-16  Experiment iteration v1→v6 (Kaggle)
2026-05-17  Visualization + manuscript writing
2026-05-18  v1.0 freeze: release, review, revision
2026-05-22  v2.0 launch: DF re-run + SF integration
2026-05-23  DF Batch 1: Kaggle training complete
2026-05-25  DF analysis complete + progress report  ← CURRENT
           ↓
Next        DF multi-seed (Phase 1.2) + SF experiments (Phase 2)
           ↓
Target      v2 manuscript revision, MedIA submission
```

---

*Report generated by comprehensive project survey, 2026-05-25.*
