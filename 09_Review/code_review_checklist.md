# Code Review Checklist — 06_Implementation

## 1. Correctness

| # | Check | File | Status |
|---|-------|------|:------:|
| C1 | ORCU log-barrier on GPU tensors? | losses/orcu_loss.py | PASS (fixed) |
| C2 | EDL digamma/lgamma numerical stability? | losses/edl_loss.py | PASS |
| C3 | KL annealing formula? | losses/edl_loss.py:38 | PASS |
| C4 | 3-stage epoch boundaries? | losses/combined_loss.py:32-37 | PASS |
| C5 | Lambda annealing uses correct epochs? | losses/combined_loss.py:55-57 | PASS |
| C6 | EvidenceHead predict() returns correct u? | models/evidence_head.py:33-47 | PASS |
| C7 | SFDataset split matches GT.xlsx Mode? | data/dataset.py:98-103 | PASS |
| C8 | DFDataset split reproducible (seed)? | data/dataset.py:42-53 | PASS |

## 2. Architecture & Design

| # | Check | Status | Notes |
|---|-------|:------:|-------|
| D1 | Backbone-agnostic: EDLClassifier for any backbone? | PASS | Accepts any nn.Module with .dim attribute |
| D2 | EvidenceHead drop-in replacement? | PASS | Same interface as standard classifier head |
| D3 | Losses independent and composable? | PASS | EDLLoss + ORCULoss work separately or jointly |
| D4 | Train/eval separation clean? | PASS | train.py orchestrates, eval.py computes metrics |
| D5 | Config YAMLs match train.py defaults? | PASS | Verified parameter alignment |

## 3. Reproducibility

| # | Check | Status | Notes |
|---|-------|:------:|-------|
| R1 | Random seeds controllable? | PASS | --seed flag in train.py |
| R2 | Checkpoint saves full state? | PASS | model + optimizer + history + args |
| R3 | Data transforms deterministic for val/test? | PASS | No augmentations in val/test transforms |
| R4 | Multi-GPU support? | GAP | Single GPU only; add DDP/FSDP |
| R5 | Mixed precision (AMP)? | GAP | torch.cuda.amp not implemented |

## 4. Missing Functionality

| # | Feature | Priority | Notes |
|---|---------|:--------:|-------|
| F1 | 5-fold CV wrapper | P0 | Required for paper: mean +/- std + t-test |
| F2 | Ablation experiment runner | P0 | Script to sweep lambda, tau, stages, label types |
| F3 | MC Dropout baseline | P1 | Compare uncertainty quality vs EDL |
| F4 | Deep Ensemble baseline | P1 | Train 5 models with different seeds; compare u |
| F5 | OOD detection evaluation | P1 | ROC/PR curves for OOD vs ID uncertainty |
| F6 | TensorBoard / WandB logging | P2 | Better real-time training visualization |
| F7 | Config YAML loader in train.py | P2 | Currently CLI args only; YAML override support |
| F8 | ONNX/TorchScript export | P3 | Clinical deployment path |
| F9 | Gradient checkpointing | P3 | Memory optimization for larger backbones |

## 5. Edge Cases & Robustness

| # | Check | Status | Notes |
|---|-------|:------:|-------|
| E1 | Empty dataloader handling? | GAP | No explicit check before metrics |
| E2 | NaN loss detection? | GAP | EDL can produce NaN with extreme alpha values |
| E3 | GPU OOM recovery? | GAP | No auto batch size reduction |
| E4 | Corrupted image handling? | GAP | Dataset assumes all images loadable |
| E5 | Single-class batch (tiny SF val)? | CHECK | May cause sklearn metric warnings |

## 6. Test Coverage

| # | Test | Status |
|---|------|:------:|
| T1 | EvidenceHead output shape/range | PASS (sanity) |
| T2 | EDLLoss gradient flow | NOT TESTED |
| T3 | ORCULoss gradient flow | NOT TESTED |
| T4 | CombinedLoss stage transitions | PASS (sanity) |
| T5 | All eval metrics return valid ranges | PASS (sanity) |
| T6 | Checkpoint save/load roundtrip | NOT TESTED |
| T7 | Report generator output format | PASS (sanity) |
| T8 | DataLoader with actual fluorosis images | NOT TESTED |

## Summary

- **Ready to run**: train.py + eval.py functional
- **P0 before submission**: 5-fold CV, ablation runner, MC Dropout baseline
- **P1-P2**: WandB logging, YAML loader, test suite, AMP support
