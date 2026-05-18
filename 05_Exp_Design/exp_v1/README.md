# exp_v1 — DF Fluorosis 实验归档 (2026-05-17)

论文定稿实验数据。数据源版本: **v2.2** (PyTorch 2.5)。

## 目录结构

```
exp_v1/
├── README.md                          # 本文件
├── reports/                           # 实验分析报告 (5 篇)
│   ├── 01_experiment_plan.md          # 实验计划
│   ├── 02_experiment_results_v2.md    # v2.0 分析
│   ├── 03_experiment_results_v2.1.md  # v2.1 分析
│   ├── 04_experiment_results_v2.2.md  # v2.2 分析 (论文主数据源)
│   └── 05_comprehensive_analysis.md   # 跨版本综合分析 (定稿)
├── figures/                           # 论文图表 (20 张图 + 生成脚本)
│   ├── generate_all_figures.py        # 图表生成脚本
│   ├── F1-F10_*.pdf                   # 矢量图 (论文用)
│   └── F1-F10_*.png                   # 位图 (预览, 300 DPI)
├── data/                              # Per-sample 预测数据 (14 个 .npz)
│   ├── exp1_ce_preds.npz              # CE 逐样本预测
│   ├── exp1_cumulative_preds.npz      # Cumulative
│   ├── exp1_sord_preds.npz            # SORD
│   ├── exp1_edl_preds.npz             # EDL (含 alpha/evidence)
│   ├── exp1_edl_orcu_preds.npz        # EDL+ORCU
│   └── exp2_sweep_lam*_reg*_preds.npz # Lambda Sweep (9 files)
└── notebooks/                         # Kaggle 训练脚本 (5 个)
    ├── kaggle_train_v2.ipynb          # v2.0 训练脚本
    ├── kaggle_train_v4.ipynb          # v2.2 训练脚本 (论文用)
    ├── kaggle_train_v6.ipynb          # v6 精简版 (per-sample 导出)
    ├── kaggle_multiseed_v1.ipynb      # Multi-seed 验证
    └── kaggle_sweep_calibration.ipynb # Lambda + Temp sweep
```

## 论文核心数据 (v2.2)

| Mode | Acc | F1 | QWK | ECE | %Unim |
|------|-----|----|----|-----|-------|
| CE | 0.8167 | 0.8085 | 0.9329 | 0.1213 | 100% |
| Cumulative | 0.6833 | 0.6574 | 0.8185 | 0.1881 | 91.7% |
| SORD | 0.7333 | 0.7343 | 0.8999 | 0.1665 | 100% |
| **EDL** | **0.8333** | **0.8278** | **0.9376** | **0.0719** | **96.7%** |
| EDL+ORCU | 0.7000 | 0.6948 | 0.8724 | 0.2524 | 100% |

## 图表清单

| 编号 | 内容 | 数据源 |
|------|------|--------|
| F1 | Confusion Matrices (5 modes) | v6 .npz |
| F2 | Reliability Diagrams | v6 .npz |
| F3 | Uncertainty Analysis | v6 .npz |
| F4 | Evidence Analysis (EDL) | v6 EDL .npz |
| F5 | Mode Comparison | v2.2 聚合指标 |
| F6 | Lambda Sweep Heatmap | v6 sweep .npz |
| F7 | Temperature Calibration | v6 temp data |
| F8 | Prediction Distribution | v6 .npz |
| F9 | Summary Table + Radar | v2.2 聚合指标 |
| F10 | Per-Class Analysis | v6 .npz |

## 环境信息

- 训练环境: Kaggle Notebooks, T4 x2 GPU
- PyTorch: 2.5.x (v2.2), 2.6.0 (v5/v6)
- Backbone: ViT-Base (google/vit-base-patch16-224-in21k, ImageNet-21k)
- 数据: DFID (200 images, 4-class, 50/class)
- Code: github.com/XiaoHG/fluorosis_project
