#!/bin/bash

set -e

echo "======================================"
echo " Android Recommender System Pipeline "
echo "======================================"

VENV_DIR=".venv"
PYTHON_BIN="python3.12"

echo ""
echo "[Setup] Creating virtual environment if needed..."
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_BIN -m venv $VENV_DIR
fi

echo ""
echo "[Setup] Activating virtual environment..."
source $VENV_DIR/bin/activate

echo ""
echo "[Setup] Installing required Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "[Step 1] Running Data Preprocessing (Building artifacts & Splitting Data)..."
python src/build_npy.py

echo ""
echo "[Step 2] Running Popularity Baseline..."
python -m baselines.baseline_popularity

echo ""
echo "[Step 3] Running User-Based KNN-CF Baseline..."
python -m baselines.baseline_knn_cf

echo ""
echo "[Step 4] Running LightGCN Variant..."
python variant_graph/variant_lightgcn.py

echo ""
echo "======================================"
echo " Pipeline Completed Successfully! "
echo " Results are saved in pipeline_output/results/ "
echo "======================================"
