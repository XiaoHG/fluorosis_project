# 06_Implementation — Core Implementation

EDL + ORCU dual-task fluorosis diagnosis model.

## Directory Structure

```
06_Implementation/
├── data/
│   ├── dataset.py          # DFDataset, SFDataset, transforms, create_dataloaders
│   └── __init__.py
├── models/
│   ├── evidence_head.py    # EvidenceHead: FC(D->4) -> z -> Softplus+1 -> alpha
│   ├── backbones.py        # ViTBackbone, ResNetBackbone, EDLClassifier, build_model
│   └── __init__.py
├── losses/
│   ├── edl_loss.py         # EDLLoss: Type II MLE + KL annealing
│   ├── orcu_loss.py        # ORCULoss: SORD encoding + log-barrier regularization
│   ├── combined_loss.py    # CombinedLoss: 3-stage (CE -> EDL -> Joint)
│   └── __init__.py
├── configs/
│   ├── df_vit_edl_orcu.yaml   # DF experiment config
│   └── sf_resnet_edl_orcu.yaml # SF experiment config
├── scripts/
│   ├── report_generator.py      # Auto-diagnosis report from EDL outputs
│   ├── cross_validate.py        # 5-fold CV with t-tests
│   ├── ablation_runner.py       # Automated ablation experiment sweep
│   └── uncertainty_baselines.py  # MC Dropout + Deep Ensemble baselines
├── train.py                # Training loop with 3-stage strategy + --mode flag
├── eval.py                 # Evaluation: 8 metrics + AUROC
├── checkpoints/            # Saved model checkpoints
├── logs/                   # Training logs
├── results/                # Experiment results
└── README.md
```

## Quick Start

### Train DF model (full EDL+ORCU)
```bash
cd 06_Implementation
python train.py --task df --data_root ../..
```

### Train SF model (full EDL+ORCU)
```bash
cd 06_Implementation
python train.py --task sf --data_root ../..
```

### Train pure CE baseline
```bash
python train.py --task df --data_root ../.. --mode ce
```

### Train EDL only (no ORCU)
```bash
python train.py --task df --data_root ../.. --mode edl
```

### Train ORCU only (no EDL)
```bash
python train.py --task df --data_root ../.. --mode orcu
```

### 5-fold Cross-Validation
```bash
python scripts/cross_validate.py --task df --data_root ../.. --methods ce edl edl_orcu
```

### Ablation sweep
```bash
python scripts/ablation_runner.py --task df --data_root ../.. --subset A1,A3
```

### Uncertainty baselines
```bash
python scripts/uncertainty_baselines.py --task df --data_root ../.. --method both
```

## Key Design Decisions

1. **Shared logit bridge**: `EvidenceHead.forward()` returns `(alpha, z)` — one FC output serves both paths
2. **Three-stage training**: CE warmup (stage 1) -> EDL only (stage 2) -> EDL + lambda*ORCU joint (stage 3)
3. **alpha > 1 guarantee**: `Softplus(z) + 1.0` ensures unimodal Dirichlet prior
4. **Backbone-agnostic**: DF uses ViT-Base, SF uses ResNet-50; loss/model is backbone-agnostic

## Metrics (7 + AUROC)

| Metric | Description | Direction |
|--------|-------------|:---------:|
| Acc | Accuracy | up |
| Macro-F1 | Class-balanced F1 | up |
| QWK | Quadratic Weighted Kappa | up |
| ECE | Expected Calibration Error (15 bins) | down |
| SCE | Static Calibration Error | down |
| %Unimodal | Fraction of unimodal predictions | up |
| U-ECE | Uncertainty-calibrated ECE | down |
| AUROC(u) | Uncertainty vs error discrimination | up |

## Dependencies

- PyTorch >= 2.0
- torchvision
- transformers (for ViT)
- scikit-learn
- numpy, pandas, PIL
