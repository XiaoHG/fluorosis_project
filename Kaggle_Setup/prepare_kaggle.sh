#!/bin/bash
# ============================================================================
# Kaggle Dataset Preparation Script
# Creates zip archives ready for Kaggle Dataset upload.
#
# Output:
#   kaggle_fluorosis_data.zip  — upload as private Kaggle Dataset "fluorosis-data"
#   kaggle_fluorosis_code.zip  — upload as private Kaggle Dataset "fluorosis-code"
#
# Usage:  chmod +x prepare_kaggle.sh && ./prepare_kaggle.sh
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Kaggle Dataset Preparation ==="

# ---- Data Dataset ----
echo "[1/2] Creating data dataset archive..."
DATA_ZIP="$SCRIPT_DIR/kaggle_fluorosis_data.zip"
cd "$PROJECT_DIR"
zip -r "$DATA_ZIP" data/ -x "*.DS_Store" "*/__pycache__/*" "*.pyc"
echo "  Created: $DATA_ZIP ($(du -h "$DATA_ZIP" | cut -f1))"

# ---- Code Dataset ----
echo "[2/2] Creating code dataset archive..."
CODE_ZIP="$SCRIPT_DIR/kaggle_fluorosis_code.zip"
cd "$PROJECT_DIR/06_Implementation"
zip -r "$CODE_ZIP" . -x "*.DS_Store" "*/__pycache__/*" "*.pyc" "checkpoints/*" "logs/*" "results/*"
echo "  Created: $CODE_ZIP ($(du -h "$CODE_ZIP" | cut -f1))"

echo ""
echo "=== Done ==="
echo "Next: Upload both zips as private Kaggle Datasets, then run the notebook."
echo "  Data root for notebook: /kaggle/input/fluorosis-data"
echo "  Code root for notebook: /kaggle/input/fluorosis-code"
