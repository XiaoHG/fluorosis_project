# 07_Visualization — Paper Figures

## Figure-to-Script Mapping

| Figure | Title | Script | Input |
|--------|-------|--------|-------|
| Fig 1 | Model Architecture | draw.io / OmniGraffle | manual |
| Fig 2 | Confusion Matrices | plot_metrics.py --compare | test_metrics.json x3 |
| Fig 3 | Reliability Diagrams | plot_metrics.py | ECE bin data |
| Fig 4 | Uncertainty vs Accuracy | plot_metrics.py | model outputs |
| Fig 5 | Training Curves (3-stage) | plot_metrics.py | history.json |
| Fig 6 | Evidence Distribution per Class | plot_metrics.py | test_metrics.json |
| Fig 7 | Ablation Bar Chart | plot_metrics.py --compare | test_metrics.json xN |
| Fig 8 | RecAcc & SafePred vs Theta | plot_metrics.py | report metrics sweep |

## Usage

```bash
cd 07_Visualization/04_plotting_scripts

# Single run (training curves + evidence)
python plot_metrics.py --results_dir ../../06_Implementation/checkpoints/df_experiment

# Comparison (confusion matrices + ablation)
python plot_metrics.py --results_dir ../../06_Implementation/checkpoints/df_edl_orcu \
    --compare ../../06_Implementation/checkpoints/df_ce \
              ../../06_Implementation/checkpoints/df_edl \
              ../../06_Implementation/checkpoints/df_orcu
```

## Directory Structure

```
07_Visualization/
├── 01_drafts/              # Exploratory drafts
├── 02_final_figures/       # Publication-ready PNG/PDF
├── 03_captions/            # LaTeX figure captions
├── 04_plotting_scripts/    # Generation scripts
│   ├── plot_utils.py       # Shared style/color constants
│   └── plot_metrics.py     # Main figure generator
└── README.md
```

## Style Guide

- Font: sans-serif, 10pt ticks, 12pt titles
- Colors: consistent across all figures (plot_utils.COLORS)
- DPI: 300 raster, PDF for vector
- Format: PNG (draft), PDF (final submission)
