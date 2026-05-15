---
date: 2026-05-15
author: LitAgent
input_from: "02_Literature/01_initial_search/References/Dental_Fluorosis_Segmentation_Using_Enhanced_Quantum-Inspired_Fuzzy_Clustering_Algorithm.pdf"
output_to: "02_Literature/02_deep_read/09_DF_Quantum_Fuzzy_Segmentation.md"
status: draft
---

# Dental Fluorosis Segmentation Using Enhanced Quantum-Inspired Fuzzy Clustering Algorithm

## Metadata
| Field | Value |
|-------|-------|
| Authors | Natchapon Petaitiemthong, Sansanee Auephanwiriyakul, Nipon Theera-Umpon, Chatpat Kongpun |
| Venue | 2022 37th International Technical Conference on Circuits/Systems, Computers and Communications (ITC-CSCC), IEEE |
| Year | 2022 |

## Problem & Motivation
- Dental fluorosis screening from digital images requires expert examination, which is time-consuming, prone to fatigue-induced errors, and not scalable for large-scale epidemiological surveys in endemic areas such as Chiang Mai, Thailand.
- Before a classification system can grade fluorosis severity, a robust segmentation step is needed to isolate fluorotic regions (white, yellow, brown discolored areas) from normal enamel and background. However, fluorosis lesions have fuzzy boundaries, variable colors, and irregular shapes that challenge traditional hard-clustering segmentation methods.
- The authors aim to develop an unsupervised segmentation pipeline that can automatically identify fluorotic tooth regions without pixel-level ground truth annotations, serving as a preprocessing step for downstream severity classification.

## Method
- **Core algorithm**: Enhanced Quantum-Inspired Evolutionary Fuzzy C-Means (EQIE-FCM) -- an evolutionary optimization wrapper around Fuzzy C-Means clustering that uses quantum bit (qubit) representations and quantum rotational gates to jointly optimize the fuzzifier parameter (m) and the number of clusters (C) for each color class.
- **Color space**: Images are converted from RGB to HSV (Hue-Saturation-Value) color space. Segmentation operates on H, S, and V channels as features, motivated by the observation that white, yellow, and brown fluorotic regions have characteristic HSV signatures.
- **Multi-prototype strategy**: Separate EQIE-FCM runs are performed on three color-specific training datasets: 4,888 white pixels, 4,888 yellow pixels, and 4,888 brown pixels, sampled from 7 training images (manually labeled by experts). Each run independently discovers the optimal number of cluster prototypes and fuzzifier for that color class.
- **Cluster validity**: The VIDSO index (combining intra-cluster compactness, inter-cluster separation, and inter-cluster overlap into a normalized score) serves as the fitness function. Lower VIDSO = better clustering (compact, well-separated, minimally overlapping clusters).
- **Quantum-inspired evolution**: Qubits represent fuzzifier m and cluster centers. Quantum rotational gates update qubit angles each generation based on whether local fitness exceeds global best fitness. The transformation process converts quantum states (probability amplitudes) into real-valued parameters via binary sampling and decoding.
- **Final segmentation**: After multi-prototypes are discovered for each color class, k-nearest neighbors (k=2) classifies every pixel in a test image into white, yellow, brown, or background based on HSV distance to the learned prototypes.
- **Data**: 137 images total (7 training, 16 validation, 114 blind test) of children's teeth collected by the Intercountry Centre for Oral Health (ICOH), Chiang Mai, Thailand, using a RICOH Caplio RX camera without special preparation.

## Key Results
- **Optimal parameters found**:
  - White pixels: m = 1.828, 59 prototypes, fitness = 0.01222
  - Yellow pixels: m = 1.934, 57 prototypes, fitness = 0.01120
  - Brown pixels: m = 1.991, 2 prototypes, fitness = 0.00747
- **Segmentation accuracy**: 83.68% on validation set (16 images), 84.61% on blind test set (114 images), compared against manual expert segmentation as ground truth.
- Brown pixels required only 2 cluster prototypes (most homogeneous), while white and yellow needed ~57-59 prototypes each (high intra-class variability), reflecting the diverse clinical appearance of fluorotic opacities.
- Fuzzifier values near 2.0 indicate that the optimal clusters have moderate fuzziness (close to standard FCM's typical m=2), with brown being the crispest (m=1.991).
- Segmentation errors were attributed primarily to inconsistent lighting conditions during image acquisition.

## Relevance to Our Project (氟中毒 DL 诊断)
- **Segmentation as preprocessing**: This work validates that color-based segmentation can effectively isolate fluorotic regions from intraoral photographs. For our 200-image Dean's Index grading task, incorporating a similar segmentation preprocessing step could (a) reduce background noise, (b) focus the DL model's attention on tooth surfaces, and (c) provide interpretable intermediate outputs showing which regions the model considers fluorotic.
- **HSV color space is validated for fluorosis**: The choice of HSV over RGB is biologically justified -- fluorosis manifests as changes in hue (white vs. yellow vs. brown staining), saturation (opacity vs. translucency), and value (brightness). Our model should consider HSV or LAB color representations as input channels, potentially alongside RGB.
- **Multi-class color taxonomy**: The white/yellow/brown categorization of fluorotic pixels aligns with the clinical spectrum: white = mild subsurface porosity, yellow/brown = moderate to severe with post-eruptive staining. This color taxonomy could inform a hierarchical grading approach (first classify color type, then grade severity).
- **Unsupervised approach is dataset-efficient**: EQIE-FCM requires only 7 labeled training images (and only pixel-level color sampling, not full segmentation masks). This low annotation burden is attractive given our limited 200-image dataset. However, the 83-85% segmentation accuracy indicates room for improvement that a supervised deep segmentation model (e.g., U-Net) might fill if pixel-level labels were available.
- **Lighting standardization**: The paper identifies lighting variation as the primary source of segmentation error. For our dataset, we should document and ideally standardize the photography protocol (camera, flash, ambient light, distance, angle) to minimize this confounder.

## Key Takeaways for Method Design
1. **HSV/LAB color channels should supplement RGB input**: The demonstrated effectiveness of HSV for fluorosis segmentation suggests concatenating HSV or LAB color space representations as additional input channels to our CNN, providing the model with explicit color information that is clinically relevant to fluorosis grading.
2. **Two-stage pipeline may improve performance**: Consider a segmentation-classification cascade: (Stage 1) segment tooth/fluorotic regions using a lightweight model (U-Net, DeepLabV3+, or even the unsupervised EQIE-FCM), then (Stage 2) classify severity from the segmented regions only. This removes irrelevant background information and forces the classifier to attend to diagnostically relevant areas.
3. **Color prototype analysis for interpretability**: The multi-prototype approach reveals that "white" fluorosis is not a single color but spans ~59 distinct HSV clusters. Visualizing and quantifying the color distribution of our dataset by Dean's grade could yield interpretable features and help identify systematic differences between grades.
4. **Data-efficient segmentation is feasible**: The EQIE-FCM approach achieves reasonable segmentation with minimal labeled data (pixel sampling, not full masks). If full pixel-level annotations are unavailable for our dataset, unsupervised or weakly-supervised segmentation strategies are viable.
5. **Brown pixels are diagnostically significant**: The finding that brown pixels form a tight cluster (only 2 prototypes needed) while white/yellow are highly diverse suggests that the presence of brown/discolored regions is a strong, reliable signal for moderate-to-severe fluorosis (Dean's Grade 2-3). Our model should explicitly learn to detect and weight brown/discolored regions.
6. **Lighting normalization is critical**: The paper's primary failure mode -- inconsistent lighting -- must be addressed in our pipeline. Consider histogram equalization, white-balance correction, or color constancy algorithms as preprocessing steps to normalize illumination across our 200 images.
