# Internal Review Checklist — EDL+ORCU Fluorosis Diagnosis Paper

## 1. Method Soundness

| # | Check | Status | Notes |
|---|-------|:------:|-------|
| M1 | Loss functions mathematically complete? | PASS | L_EDL, L_ORCU, L_total fully defined |
| M2 | Shared-logit bridge justified? | PASS | Documented in 04_Model_Design |
| M3 | alpha > 1 guarantee correct? | PASS | Softplus(z) + 1.0 |
| M4 | 3-stage training motivation clear? | PASS | CE -> EDL -> Joint progression |
| M5 | lambda annealing optimal? | CHECK | Need A3 experiment to confirm linear 0->0.5 |
| M6 | EDL compatible with soft targets? | CHECK | Type II MLE with soft y_k needs verification |
| M7 | ORCU tau=3.0 optimal for fluorosis? | CHECK | Need A4 experiment |
| M8 | KL annealing 30% inflection correct? | CHECK | Need A5 experiment |

## 2. Experiment Completeness

| # | Check | Status | Notes |
|---|-------|:------:|-------|
| E1 | All [X] placeholders filled with real data? | GAP | **Blocking** — must run all experiments |
| E2 | 5-fold CV executed? | GAP | train.py currently single-split only |
| E3 | Paired t-tests computed? | GAP | Requires 5-fold data |
| E4 | DF baseline (ViT+CE) near MLTrMR 80.19%? | GAP | If >5pp gap, comparison fairness at risk |
| E5 | SF baseline (RN50+CE) near Mwinc-Mamba 66.67%? | GAP | ResNet proxy validity to verify |
| E6 | All ablations (A1-A6) completed? | GAP | Need ablation runner script |
| E7 | U5 (rater agreement vs uncertainty correlation)? | GAP | Need Spearman correlation computation |
| E8 | OOD test data for U3? | GAP | Need non-fluorosis dental/X-ray images |
| E9 | All figures regenerated from real data? | GAP | 07_Visualization scripts ready, no data yet |

## 3. SOTA Comparison Fairness

| # | Check | Status | Notes |
|---|-------|:------:|-------|
| S1 | DF test split aligned with MLTrMR? | PASS | 120/20/60, test 60 aligned |
| S2 | SF test split aligned with Mwinc-Mamba? | PASS | 48/8/24, test 24 unchanged |
| S3 | ViT+CE baseline accuracy verified? | GAP | Must match or approach 80.19% |
| S4 | ResNet-as-Mamba-proxy justified? | CHECK | Add explicit paragraph in Discussion |
| S5 | Missing metrics from competitor papers? | CHECK | MLTrMR/Mwinc-Mamba may lack F1, QWK, ECE |
| S6 | Missing uncertainty baselines? | GAP | MC Dropout, Deep Ensemble, Temperature Scaling |

## 4. Writing Quality

| # | Check | Status | Notes |
|---|-------|:------:|-------|
| W1 | Title accurately reflects contribution? | PASS | EDL + Ordinal Calibration |
| W2 | Abstract covers method + results + impact? | PASS | Results section needs numbers |
| W3 | 4 contributions map to experiments? | PASS | C1-C4 each validated in experiments |
| W4 | Symbol consistency (K vs C, tau vs t)? | CHECK | Full pass needed |
| W5 | Missing key citations? | CHECK | Complete [Author] placeholders first |
| W6 | Limitations honest and complete? | PASS | Sample size, single-rater DF, proxy backbone |
| W7 | Table/figure numbering correct? | CHECK | Verify after inserting actual figures |

## 5. Reproducibility

| # | Check | Status | Notes |
|---|-------|:------:|-------|
| R1 | Code runnable? | PASS | train.py + eval.py pass sanity checks |
| R2 | Dependencies documented? | PASS | PyTorch, transformers, sklearn |
| R3 | Data access documented? | CHECK | Dataset from collaborators; add access statement |
| R4 | Hyperparameters in config YAML? | PASS | df/sf config files complete |
| R5 | Random seed fixed? | PASS | seed=42 default |
| R6 | Code comments sufficient? | PASS | Docstrings on all functions |

## 6. Innovation Self-Assessment

| Dimension | Score (1-5) | Rationale |
|-----------|:-----------:|-----------|
| EDL first application to fluorosis | 4 | New domain, but EDL itself is established |
| Shared-logit bridge architecture | 4 | Clean design; enables EDL+ORCU consistency |
| ORCU first medical classification use | 4 | New task domain for ORCU |
| Multi-rater soft labels | 3 | Established technique, but 5-radiologist resource is unique |
| Auto-diagnosis reports | 2 | Engineering extension, not core innovation |
| Small-sample EDL evidence | 3 | 48-training-sample extreme is valuable evidence |
| **Overall** | **3.3** | Appropriate for MedIA-level submission |

## Summary: Blocking Issues (P0)

1. **No experimental data**: All [X] placeholders unfilled — paper incomplete
2. **No 5-fold CV + significance tests**: Required even with single-split results
3. **SF baseline Acc unknown**: If ResNet+CE far from Mwinc-Mamba 66.67%, comparison loses validity
4. **Missing uncertainty baselines**: MC Dropout, Deep Ensemble needed for fair comparison
