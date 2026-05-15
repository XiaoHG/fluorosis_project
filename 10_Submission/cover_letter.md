# Cover Letter — Medical Image Analysis Submission

---

Dear Editor-in-Chief,

We submit the manuscript entitled **"Evidential Deep Learning with Ordinal Calibration for Automated Diagnosis of Endemic Fluorosis"** for consideration for publication in Medical Image Analysis.

**Significance**: Endemic fluorosis affects millions globally, yet automated diagnosis using deep learning remains underexplored. Existing approaches (MLTrMR, Mwinc-Mamba) treat fluorosis grading as nominal classification, ignoring both the ordinal nature of severity grades and the critical need for prediction uncertainty in clinical settings—particularly given that five radiologists on our dataset achieved only 54.1% pairwise agreement.

**Novelty**: Our work makes four contributions:
1. A shared-logit evidence head architecture unifying Evidential Deep Learning (EDL) with Ordinal Regression for Calibration and Unimodality (ORCU), ensuring consistency between evidence accumulation and ordinal constraints.
2. The first application of EDL and ORCU to fluorosis diagnosis, providing calibrated prediction uncertainty that existing methods cannot offer.
3. Multi-rater soft label modeling leveraging the full annotation distribution from five radiologists rather than discarding inter-rater variability via majority voting.
4. A backbone-agnostic framework validated on both dental fluorosis (200 images, ViT-Base) and skeletal fluorosis (80 images, ResNet-50), demonstrating uncertainty-aware ordinal diagnosis with fewer than 100 training samples.

**Fit with Medical Image Analysis**: The paper advances uncertainty-aware medical image classification, introducing ordinal calibration alongside evidential learning. This aligns with MedIA's scope of methodological advances with clear clinical motivation. The multi-rater modeling component is timely given growing interest in learning from imperfect annotations.

**Declaration**: This manuscript is original, has not been published elsewhere, and is not under consideration by another journal. All authors have approved the manuscript and agree with its submission. The authors declare no conflicts of interest.

We suggest potential reviewers:
- [Reviewer 1: Expert in evidential deep learning / medical imaging]
- [Reviewer 2: Expert in ordinal regression / calibration]
- [Reviewer 3: Expert in dental/skeletal radiology AI]

Thank you for considering our manuscript.

Sincerely,
Xiaohong Gao
[Affiliation] | [Email] | [Date]
