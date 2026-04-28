#!/bin/bash

set -e

echo 
echo " Android Recommender System Pipeline with graph variants "
echo 

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
python variant_graph/graph_build_npy.py

echo ""
echo "[Step 2] Running Pure LightGCN Variant..."
python variant_graph/variant_lightgcn_colab.py

echo ""
echo "[Step 3] Running LightGCN + Metadata Variant..."
python variant_graph/variant_lightgcn_metadata_colab.py

