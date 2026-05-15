---
date: 2026-05-15
author: LitAgent
input_from: "02_Literature/01_initial_search/References/s41598-026-43429-4_reference.pdf"
output_to: "02_Literature/02_deep_read/10_SF_ML_Prediction_s41598.md"
status: draft
---

# Predicting Skeletal Fluorosis Severity Using Machine Learning Across Diverse Fluoride-Exposed Populations in China

## Metadata
| Field | Value |
|-------|-------|
| Authors | Hongjiang Long, Jiayi Zeng, Shaofeng Wei, Ting Hu, Yun Lu, Qingzhen Jia, Jinshu Li, Hongbing Ye, Yanhui Gao, Zhenting Zhang, Peng Luo |
| Venue | Scientific Reports (2026), in press |
| Year | 2026 |
| DOI | 10.1038/s41598-026-43429-4 |

## Problem & Motivation
- Skeletal fluorosis (SF) affects ~100 million globally and is diagnosed via X-ray changes that appear only at advanced, often irreversible stages.
- No existing predictive model integrates multi-dimensional clinical, environmental, and biomarker data for early risk screening.
- Three distinct fluoride exposure pathways in China (coal-burning, drinking-water, brick-tea) require a model generalizable across all types.

## Method
- **Study design**: 1,309 individuals from the China Fluorosis Cohort (CFC), three provinces representing the three exposure pathways.
- **Feature selection**: LASSO regression reduced 80 candidate variables to 22 non-zero coefficients (lambda.min via 10-fold CV).
- **Models**: Random Forest (RF), XGBoost, SVM (linear kernel), KNN, Decision Tree -- all tuned via grid search + 10-fold CV within training set only.
- **Primary metric**: AUC; complementary: Accuracy, Sensitivity, Specificity, F1 score; calibration curves + decision curve analysis (DCA).
- **Interpretability**: SHAP (Shapley Additive exPlanations) for feature importance, beeswarm plots, and dependence plots.
- **Data split**: 7:3 training/test; minimum-maximum scaling for continuous variables; one-hot encoding for categorical; multiple imputation (mice) for <=10% missing data.

## Key Results
- **RF best model**: Training AUC = 0.875 (95% CI: 0.851-0.913), Test AUC = 0.832; Test accuracy = 0.793, sensitivity = 0.906, specificity = 0.639, F1 = 0.841.
- **XGBoost second**: Test AUC = 0.830, accuracy = 0.786.
- **Top 5 SHAP predictors**: pain level score, knee function, age, shoulder function, urinary fluoride (UF).
- **Nonlinear UF threshold effect**: Risk increases sharply up to ~4 mg/L then plateaus -- suggesting a critical exposure threshold.
- **Bone metabolism signature**: Elevated PINP and osteocalcin with simultaneously LOWER bone mass index and density in moderate-to-severe SF ("high turnover, low quality" bone state).
- **Regional gradient**: Coal-burning (Guizhou) > drinking-water (Shanxi) > brick-tea (Sichuan) in severity proportion.
- **Lower specificity** (0.588-0.639) attributed to overlapping clinical/biochemical profiles between severity grades.

## Relevance to Our Project (氟中毒 DL 诊断)
- **Directly on topic**: This is the largest and most recent ML study specifically on skeletal fluorosis severity prediction. Provides the strongest reference for our work.
- **Feature selection methodology**: LASSO pipeline (80->22 features) can inspire variable selection for our tabular clinical data accompanying intraoral photos.
- **SHAP interpretability**: Demonstrates how to make ML predictions clinically actionable -- we should adopt SHAP for our classification models to identify which image regions drive DF/SF grading decisions.
- **Biomarker integration**: Shows the value of combining imaging with lab markers (UF, bone metabolism). Our project could explore multi-modal fusion of intraoral photos + clinical biomarkers.
- **Data limitation insight**: The specificity issue (overlap between mild and moderate/severe) mirrors the challenge we will face with 4-grade DF classification where adjacent grades have ambiguous boundaries.
- **No image data used**: This study predicts SF from tabular data only, not from radiographs. Our project differs by classifying directly from images, but the predictive framework logic is transferable.

## Key Takeaways for Method Design
1. **RF + SHAP is a strong baseline for tabular clinical data**: If we collect structured data alongside images, RF with SHAP should be our go-to baseline before deep learning. The 0.875 AUC on training data sets a benchmark.
2. **Preserve real-world class distribution**: The authors deliberately did not use SMOTE or balancing, arguing that preserving natural prevalence better reflects screening reality. With 200 intraoral photos potentially skewed across 4 grades, we should evaluate both balanced and imbalanced training strategies.
3. **LASSO for feature sparsification**: When combining imaging features with clinical metadata (age, UF, etc.), use LASSO to identify the most predictive subset. From 80->22, most biological features were preserved while noise variables were eliminated.
4. **Joint scoring (pain + function + biomarker) outperforms single modality**: The model shows that combining subjective (pain), functional (joint mobility), and objective (UF) measures yields strongest predictions. For us: fuse image features + clinical metadata.
5. **Calibration + DCA beyond AUC**: Even with high AUC, calibration curves and decision curve analysis revealed practical clinical utility thresholds. We should report these for our 4-grade DF classifier to show net benefit at different risk thresholds.
