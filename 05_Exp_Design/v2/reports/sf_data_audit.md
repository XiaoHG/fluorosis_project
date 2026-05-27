# SF Data Audit — V2 SFXRay Dataset Check

**Date:** 2026-05-22
**Dataset:** SFXRay (80 X-ray images, 4 grades, 3 body parts)

---

## Dataset Profile (from literature)

| Property | Value |
|----------|-------|
| Total images | 80 |
| Source | Zhijin County, Guizhou, 2020-2023 |
| Device | UIH uDR 780i Pro |
| Body parts | Forearm (33), Calf (30), Pelvis (17) |
| Classes | Normal: 21, Mild: 34, Moderate: 13, Severe: 12 |
| Annotation | 3 senior radiologists (>10yr), majority vote |
| Standard | WS/T 192-2021 (China endemic SF standard) |
| Resolution | 512x1024 PNG (converted from 12-bit DICOM) |
| Masks | 59 segmentation masks (grayscale PNG), abnormal cases only |
| Radiologists | 5 individual annotations provided |

## Audit Checklist (verify on Kaggle)

- [ ] GT.xlsx: 80 rows, columns ID/Binary/Multiple/Split/Part
- [ ] images/: 80 PNG files, naming matches GT.xlsx IDs
- [ ] masks/: 59 PNG files (abnormal cases only)
- [ ] Class distribution: N21/M34/Mo13/S12 in GT.Multiple
- [ ] Split: Mode column has train/test labels
- [ ] Body parts: Part column covers forearm/calf/pelvis
- [ ] Radiologist annotations: 5 Excel files readable

## Known Issues to Watch

1. **Class imbalance**: Mild=34 vs Severe=12 (3:1 ratio)
2. **Body part confounding**: part may correlate with grade
3. **Small test set**: 24 samples (7:3 split) — high variance
4. **No masks for normal cases**: masks only for SF-positive images
5. **Multi-radiologist disagreement**: documented in Mwinc-Mamba paper

## SFDataset Code Check (dataset.py lines 74-131)

- [ ] GT.xlsx: sheet_name="Sheet1", "Multiple" column for labels
- [ ] Split: train from Mode="train", val = every 7th, test from Mode="test"
- [ ] Image loading: RGB conversion, 224x224 resize
- [ ] Body part NOT used as input (Phase 4.3 improvement)

## Expected Split Distribution

| Split | Approx | Notes |
|-------|:------:|-------|
| Train | ~48 | From Mode="train", excluding every 7th |
| Val | ~8 | Every 7th from Mode="train" |
| Test | ~24 | From Mode="test" |

---

*To be validated when Kaggle kernel runs.*
