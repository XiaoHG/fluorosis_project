---
date: 2026-05-15
author: LitAgent
input_from: "02_Literature/01_initial_search/References/Deep_Learning_Techniques_for_Dental_Image_Diagnostics_A_Survey.pdf"
output_to: "02_Literature/02_deep_read/14_DL_Dental_Survey.md"
status: draft
---

# Deep Learning Techniques for Dental Image Diagnostics: A Survey

## Metadata
| Field | Value |
|-------|-------|
| Authors | Dipmala Salunke, Ram Joshi, Prasdu Peddi, D.T. Mane |
| Venue | 2022 International Conference on Augmented Intelligence and Sustainable Systems (ICAISS), IEEE |
| Year | 2022 |
| DOI | 10.1109/ICAISS55157.2022.10010576 |

## Problem & Motivation
- Dental disease detection traditionally relies on manual radiographic interpretation, which is time-consuming, error-prone, and dependent on clinician experience.
- Deep learning has shown promise in medical imaging, but the landscape of DL applications specifically in dentistry is fragmented across different diseases, image types, and architectures.
- No comprehensive survey existed that catalogs DL methods across multiple dental diagnostic tasks with standardized performance comparisons.

## Method
- **Literature search**: Reviewed over 200 articles from PubMed, IEEE Xplore, arXiv, and dental journals (Journal of Oral Diseases, Dental Research, Periodontology). 44 manuscripts included based on author interest and article requirements, all published 2000-2021.
- **Tasks covered**: (1) Caries detection, (2) Vertical root fracture detection, (3) Periodontal disease classification, (4) Tooth detection and numbering, (5) Dental plaque detection, (6) Image segmentation.
- **Architectures surveyed**: CNN, U-Net, MASK R-CNN, Faster R-CNN, GoogleNet Inception V3, VGG16, AlexNet, ResNet, back-propagation neural network, probabilistic neural network, DetectNet, 3D FCN.
- **Image modalities covered**: RVG X-rays, periapical radiographs, panoramic radiographs (OPG), CBCT, intraoral camera photos, intraoral color images, bitewing X-rays, occlusal X-rays.

## Key Results (Benchmark Summary Table)
| Task | Best Method | Best Accuracy | Dataset Size |
|------|------------|---------------|--------------|
| Caries detection | Back-propagation NN (Geetha 2020) | 97.1% | 105 images |
| Caries + fluorosis + 5 other diseases | MASK R-CNN (Liu 2015) | 90% avg | 2556 caries, 1075 fluorosis |
| Caries detection (7-class) | Mask R-CNN (Moutselos 2019) | 77.2% | 88 images |
| Caries (premolar/molar) | GoogleNet Inception V3 (Lee 2018) | 89%/88% | 3000 images |
| Vertical root fracture | Probabilistic NN (Johari 2017) | 96.6% (CBCT) | -- |
| Periodontal bone loss | 7-layer CNN (Krois 2019) | 81% acc/sens/spec | 109 images |
| Tooth detection/numbering | Faster R-CNN (Tuzoff 2019) | Sens 0.9941, Prec 0.9945 | 1352 images |
| Tooth segmentation | 3D FCN (Chen 2020) | Dice 0.936 | 25 CBCT, 770 teeth |
| Panoramic segmentation | U-Net ensemble (Koch 2019) | Dice 0.936 | 1500 images |
| Dental plaque | CNN (You 2020) | MIoU 0.724 | 886 images |

## Key Challenges Identified
1. **No public standard dental image dataset**: All studies used in-house data; objective cross-study comparison is impossible.
2. **Small training datasets**: Most studies used <1200 images; deep learning typically requires far more.
3. **Expert annotation bottleneck**: Images must be labeled by dentists, limiting dataset scale.
4. **Caries depth and plaque quantification**: Estimating severity depth from 2D X-rays remains unsolved.
5. **Architecture selection**: No consensus on which CNN architecture is best for each dental task.
6. **Region of interest cropping**: Manual ROI selection introduces error and limits automation.
7. **Data augmentation dependence**: Most studies relied heavily on augmentation to compensate for small datasets.

## Relevance to Our Project (氟中毒 DL 诊断)
- **Direct evidence of DL for dental fluorosis**: Liu et al. (2015) classified dental fluorosis using MASK R-CNN with ~90% accuracy on 1075 fluorosis images -- this is the closest published work to our DF classification task. Validates that DL can distinguish fluorosis from other dental conditions.
- **Benchmark performance floor**: The survey establishes that dental diagnostic DL accuracy ranges from 77% (7-class caries) to 99% (tooth detection). Our 4-grade DF classifier should target >85% accuracy to be competitive.
- **Architecture recommendations for small datasets**: The survey shows that CNN + transfer learning (Prajapati 2017) and MASK R-CNN work well with <200 images. For our 200 intraoral photos, transfer learning from ImageNet or dental pretrained models is strongly indicated.
- **Dental imaging diversity insight**: Intraoral photos (used by Moutselos 2019 and You 2020) achieved 77-91% accuracy, while radiographic modalities generally performed better. This suggests intraoral photography is viable but inherently noisier -- we should expect and account for this.
- **Gap identification**: No paper in this survey addressed fluorosis SEVERITY GRADING (only presence/absence classification). Our 4-grade DF severity classification appears to be a novel contribution for DL methods.

## Key Takeaways for Method Design
1. **Start with transfer learning on a pretrained backbone**: Every top-performing method in the survey used transfer learning (ImageNet pretrained -> dental fine-tuning). For our 200 photos, this is mandatory. VGG16, ResNet50, and Inception V3 are the most validated choices for dental imaging.
2. **U-Net for segmentation, classification head for grading**: The survey shows U-Net variants dominate dental segmentation (Dice 0.93-0.95). For our task, we could use a U-Net to segment fluorosis-affected enamel regions, then feed the segmentation mask into a classification head for 4-grade severity.
3. **Ensemble methods improve robustness**: Koch et al. (2019) improved U-Net Dice from 0.934 to 0.936 by ensembling 4 trained U-Nets with test-time augmentation. For clinical deployment reliability, ensemble + TTA should be our default inference strategy.
4. **Data augmentation is non-negotiable for small datasets**: Random rotation, flipping, and intensity scaling were used in virtually every surveyed study. Our augmentation pipeline should include dental-specific transforms (color jitter for lighting variation, slight affine transforms for angle variation in intraoral photos).
5. **Public benchmark needed**: A key survey finding is the absence of public dental datasets. If we release our fluorosis dataset (200 intraoral photos, 80 X-rays) publicly, it would be one of the first fluorosis-specific DL benchmarks -- a significant contribution opportunity.
