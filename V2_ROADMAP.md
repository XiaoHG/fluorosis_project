# V2 Development Roadmap

**Created:** 2026-05-22
**Based on:** v1.0 (frozen)
**Status:** Planning

---

## v1.0 Baseline Recap

| What v1 Delivered | Status |
|-------------------|--------|
| DF-only diagnosis (dental fluorosis) | Complete |
| EDL + ORCU framework on ViT backbone | Complete |
| 5-loss comparison + ablation + multi-seed CV | Complete |
| Manuscript draft (English + Chinese) | Complete |
| Submission package for MedIA | Complete |

**Key Limitation:** DF only (200 images). SF (skeletal fluorosis) not integrated.

---

## v2 Candidate Workstreams

### P0 — SF Integration
- Integrate skeletal fluorosis data into multi-modal framework
- Joint DF+SF diagnosis or cross-modal transfer learning
- Requires: SF dataset preparation, model architecture update

### P1 — Model Architecture Improvements
- Upgrade backbone (ViT-Large, ConvNeXt, Swin Transformer)
- Explore Mamba/SSM architectures per Mwinc-Mamba literature
- Multi-modal fusion: combine clinical metadata with images

### P2 — Experiment Expansion
- External validation on independent datasets
- Cross-population generalization study
- Robustness: adversarial attacks, distribution shift, corrupted inputs

### P3 — Auto-Diagnosis Report System
- End-to-end clinical report generation
- Uncertainty-gated triage pipeline
- Clinician-in-the-loop interface

### P4 — Manuscript Revision
- Incorporate reviewer feedback (when received)
- Update results with v2 improvements
- Prepare rebuttal materials

---

## Next Actions

- [ ] Define precise v2 scope based on priorities
- [ ] Set up SF dataset pipeline
- [ ] Design v2 model architecture
- [ ] Create v2 experiment plan

---

*To be updated as v2 scope solidifies.*
