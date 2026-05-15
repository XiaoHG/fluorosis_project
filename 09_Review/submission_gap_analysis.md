# Submission Gap Analysis — EDL+ORCU Fluorosis Paper

## Current Status: ~70% Ready

Full pipeline from literature to manuscript is in place, but paper is not yet submittable.

---

## P0 — Blocking (Must Fix Before Submission)

### G1: No Experimental Results (Gap: Execution)

All tables contain [X] placeholders. This is the single biggest gap.

**Actions:**
1. Run train.py --task df (ViT-B + EDL+ORCU, 100 epochs)
2. Run train.py --task sf (ResNet-50 + EDL+ORCU, 150 epochs)
3. Run all baselines: CE, EDL-only, ORCU-only on both tasks
4. Run all 6 ablation experiments (A1-A6)
5. Run 5-fold CV for main results + statistical tests
6. Generate all figures from 07_Visualization with real data
7. Fill all [X] placeholders in 05_results.tex

**Estimated effort**: 2-4 weeks depending on GPU availability

### G2: No Statistical Significance Tests (Gap: Rigor)

Reviewers will reject any ML paper without error bars and significance.

**Actions:**
1. Implement 5-fold CV wrapper for train.py
2. Run paired t-test (alpha=0.05) between our method and each baseline
3. Add significance markers (*, dagger, double-dagger) to all tables
4. Report mean +/- std for all metrics

### G3: SF Baseline Validity (Gap: Architecture Mismatch)

ResNet-50 is used as Mamba proxy. If ResNet+CE performs drastically differently from Mwinc-Mamba (66.67%), the comparison loses credibility.

**Actions:**
1. Run ResNet-50 + CE on SF test set
2. If within 5pp of 66.67%: acceptable, add justification paragraph
3. If >5pp gap: either (a) implement true Mwinc-SSM backbone, or (b) reframe as loss-function ablation under fixed backbone

### G4: Missing Uncertainty Baselines (Gap: Completeness)

Reviewer will ask: "How does EDL uncertainty compare to MC Dropout / Deep Ensembles?"

**Actions:**
1. Implement MC Dropout baseline (dropout at inference, 30 forward passes)
2. Implement Deep Ensemble baseline (5 models with different seeds)
3. Compare AUROC(u), U-ECE, ECE across EDL / MC Dropout / Ensemble
4. Add comparison rows to main results tables

---

## P1 — Important (Should Fix Before Submission)

### G5: Incomplete References
14 bib entries contain [Author], [Journal], [Year] placeholders. Complete all from literature review.

### G6: OOD Detection Experiment (U3)
U3 planned but needs actual OOD test data. Source non-fluorosis dental photos and non-SF X-rays.

### G7: Missing S6 — Additional Baselines
Beyond MC Dropout/Ensemble, reviewer may expect:
- Temperature Scaling (calibration baseline)
- Label Smoothing (soft target baseline for A2 comparison)

### G8: Multi-Rater on DF
We apply multi-rater soft labels only to SF (no multi-rater DF annotations). Add explicit justification paragraph in Discussion.

---

## P2 — Nice to Have

### G9: External Validation
Single-dataset, single-center. Acknowledge as limitation (already done).

### G10: Reader Study
Would strengthen clinical impact. Acknowledge as future work.

### G11: Supplementary Material
Auto-report examples, full ablation tables, per-fold 5-fold CV results.

### G12: Visual Abstract
Required by many journals (MedIA, etc.).

### G13: Cover Letter
Draft explaining significance to target journal.

---

## Recommended Execution Order

1. Run train.py --task sf first (faster: 150 epochs x 16 batch x 80 images)
2. Run train.py --task df (100 epochs x 32 batch x 200 images)
3. Implement 5-fold CV + t-test
4. Implement MC Dropout + Deep Ensemble baselines
5. Run all ablations (A1-A6)
6. Generate all figures with real data
7. Fill manuscript tables with numbers
8. Complete references
9. Write supplementary material
10. Final proofread
