# Supplementary Material — Planned Contents

## S1: Extended Method Details
- Full Type II MLE derivation for Dirichlet distribution
- Detailed log-barrier penalty derivation for ORCU
- Pseudocode for three-stage training algorithm
- Hyperparameter sensitivity: full lambda and tau sweep tables

## S2: Extended Experimental Results
- Per-fold results for 5-fold CV (DF and SF)
- Full confusion matrices for all methods (CE / EDL / ORCU / EDL+ORCU)
- Per-class precision, recall, F1 for all methods
- Statistical test details (t-statistics, p-values for all pairwise comparisons)

## S3: Ablation Study Details
- Full results for A1-A6 with all 8 metrics
- Learning curves for each ablation variant
- Wall-clock training time comparison across methods

## S4: Uncertainty Analysis Details
- Uncertainty distribution histograms per class and per method
- OOD detection: full ROC/PR curves with AUROC/AUPR values
- SF uncertainty by body part: per-part breakdown tables
- Rater agreement vs model uncertainty: scatter plots with Spearman rho

## S5: Auto-Diagnosis Report Examples
- 3 representative DF cases (Normal, Mild, Severe) with full report
- 3 representative SF cases with multi-rater annotation comparison
- RecAcc and SafePred full theta sweep (0.05 to 0.95, step 0.05)

## S6: Implementation Details
- Full training hyperparameters (table format)
- Hardware/software environment specification
- Data preprocessing pipeline details
- Estimated carbon footprint (GPU hours, CO2 equivalent)

## S7: Additional Baseline Details
- MC Dropout: implementation details and hyperparameters
- Deep Ensemble: per-model training details
- Temperature Scaling: calibration temperature values
