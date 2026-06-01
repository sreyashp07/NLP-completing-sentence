#!/bin/bash
set -e

echo "Starting CustomerIntent AI on HF Spaces..."
echo "Checking model artifacts..."

MODEL_DIR="ml/saved_models/baseline"
PIPELINE="$MODEL_DIR/pipeline.pkl"
LE="$MODEL_DIR/label_encoder.pkl"

if [ ! -f "$PIPELINE" ] || [ ! -f "$LE" ]; then
    echo "Model not found. Training now..."
    python data/generate_dataset.py
    python ml/training/train_baseline.py
    echo "Training complete!"
fi

echo "Starting Streamlit on port 7860..."
exec streamlit run app_hf.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
