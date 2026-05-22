# Version Tracking

## Current Version: v2.0-dev

| Field | Value |
|-------|-------|
| Version | v2.0 (development) |
| Started | 2026-05-22 |
| Based on | v1.0 (tag: `v1.0`, commit `f36cde4`, v2.2 experiment node) |
| Branch | master |
| Architecture | StandardClassifier (CE/SORD) + EDLClassifier (EDL/EDL+ORCU) — v2.2 corrected dispatch |

---

## v2.2 Starting Point

v2 inherits from v1's v2.2 experiment node — the bug-fix release that established the correct architecture:

| Asset | Status |
|-------|--------|
| `StandardClassifier` in backbones.py | Correct CE/SORD dispatch |
| `EDLClassifier` in backbones.py | Correct EDL/EDL+ORCU dispatch |
| `SFDataset` in dataset.py | Implemented, not tested |
| SF data (80 X-ray images) | Available in `data/skeletal_fluorosis/` |
| DFID data (200 images) | Available in `data/dental_fluorosis/` |
| DF baseline results | EDL 83.33% Acc, ECE 0.072 |
| Per-sample predictions | Missing (v2 Phase 1 priority) |

---

## Version History

| Version | Tag | Date | Status |
|---------|-----|------|--------|
| v0.0 | `v0.0_project_init` | 2026-05-15 | Archived |
| v1.0-draft | `v1.0_manuscript_v1_draft` | 2026-05-18 | Archived |
| v1.0-submission | `v1.0-submission` | 2026-05-18 | Archived |
| **v1.0** | **`v1.0`** | **2026-05-22** | **Frozen** |
| v2.0 | — | 2026-05-22 → | Active |

---

## v1.0 Archive

All v1 deliverables are archived at:
- `00_Admin/releases/v1.0-final/V1_RELEASE_MANIFEST.md` — Complete deliverable inventory
- `00_Admin/releases/v1.0/` — Manuscript + code ZIP archives
- `00_Admin/releases/v1.0-submission/` — Submission-ready manuscript archive
- Git tag: `v1.0`

---

*Last updated: 2026-05-22*
