#!/bin/bash
# ============================================================================
# Kaggle Data Preparation Script
# Creates zip archive ready for Kaggle Dataset upload.
#
# Output:
#   kaggle_fluorosis_data.zip  — upload as private Kaggle Dataset "fluorosis-data"
#
# Code is stored on GitHub (XiaoHG/fluorosis_project) — the notebook clones it.
#
# Usage:  chmod +x prepare_kaggle.sh && ./prepare_kaggle.sh
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Kaggle Data Preparation ==="

# ---- Data Dataset ----
echo "Creating data dataset archive..."
DATA_ZIP="$SCRIPT_DIR/kaggle_fluorosis_data.zip"
cd "$PROJECT_DIR"
zip -r "$DATA_ZIP" data/ -x "*.DS_Store" "*/__pycache__/*" "*.pyc" "._*"
echo "  Created: $DATA_ZIP ($(du -h "$DATA_ZIP" | cut -f1))"

echo ""
echo "=== Done ==="
echo "Next: Upload the zip as a private Kaggle Dataset 'fluorosis-data', then run the notebook."
echo "  Data root for notebook: /kaggle/input/fluorosis-data"
echo "  Code is cloned from:   https://github.com/XiaoHG/fluorosis_project.git"
