# V2 Development Roadmap

**Created:** 2026-05-22
**Based on:** v1.0 / v2.2 experiment node
**Status:** Active — Phase 1 in progress

---

## v2.2 Starting Point (v1 Baseline)

| What v1 Delivered | Status | Key Metric |
|-------------------|:------:|------------|
| DF-only diagnosis (EDL+ORCU, ViT-Base) | Complete | EDL 83.33% Acc, ECE 0.072 |
| 5-loss systematic comparison | Complete | CE 81.67% already SOTA |
| Multi-seed (3) + 5-Fold CV | Complete | EDL CV 4.3% vs CE CV 31.4% |
| Manuscript (English + Chinese) | Complete | 34 pages, 16 figures, 31 refs |
| Submission package (MedIA) | Complete | Cover letter, highlights, etc. |

**Key Gaps:** SF not integrated, per-sample data missing, single-center only.

---

## v2 Scope (6 Phases)

### Phase 1: DF Baseline Re-establishment (P0)
Re-run v2.2 DF experiments with per-sample prediction export to fix v1's biggest data gap.

| # | Experiment | Expected Result |
|---|-----------|----------------|
| 1.1 | DF 5-mode main | Acc=83.33% (EDL) |
| 1.2 | DF multi-seed (3) | EDL CV 4.3% |
| 1.3 | DF 5-fold CV | EDL+ORCU QWK std 0.32% |
| 1.4 | Lambda sweep (9 combos) | best at lambda=0.5 |
| 1.5 | Temperature calibration | ECE vs T curve |

### Phase 2: SF Standalone (P0)
Apply v2.2 architecture to SF for the first time.

| # | Experiment | Target |
|---|-----------|--------|
| 2.1 | SF data audit | 80 images, 3 body parts |
| 2.2 | SF 5-mode baseline | ViT-Base, same architecture |
| 2.3 | SF multi-seed | EDL stability on 48-train samples |

**Competitor:** Mwinc-Mamba (Xu 2025, BSPC) — SF binary 83.33%, 4-grade not reported.

### Phase 3: Cross-Dataset (P1)
DF↔SF transfer learning and joint training.

### Phase 4: Baselines (P1)
MC Dropout + Deep Ensemble for uncertainty comparison.

### Phase 5: Architecture (P2)
Body-part conditioning, ViT-Large, Mamba exploration.

### Phase 6: Manuscript (P2)
Update paper with v2 results, fix review issues (MC1, MC2).

---

## Execution Plan (Kaggle)

**Batch 1** — DF 1.1 + SF 2.2 (parallel, ~3h)
**Batch 2** — DF 1.2 + 1.3 (~2h)
**Batch 3** — DF 1.4 + 1.5 + SF 2.3 (~2h)
**Batch 4** — Cross-dataset Phase 3 (~2h)
**Batch 5** — Baselines Phase 4 (~1h)

---

## Detailed Experiment Plan

See: `05_Exp_Design/exp_v2/V2_EXPERIMENT_PLAN.md`

---

## Review Feedback to Address

| # | Issue | Fix |
|---|-------|-----|
| MC1 | Version inconsistency (v2.2 vs v6) | Phase 1 re-runs v2.2 with per-sample export |
| MC2 | Single-center generalizability | Phase 2 adds SF as second dataset |

---

*Last updated: 2026-05-22*
