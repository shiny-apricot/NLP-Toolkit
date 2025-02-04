#!/bin/bash

set -euo pipefail

# Load environment variables
if [ -f ../.env ]; then
    source ../.env
else
    echo "Error: .env file not found"
    exit 1
fi

# Check Python environment
if [ -d "../venv" ]; then
    source ../venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv ../venv
    source ../venv/bin/activate
    pip install --upgrade pip
    pip install -r ../requirements.txt
fi

# Setup logging directory
LOG_DIR=${LOG_DIR:-"../logs"}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pretrained_run_$(date +%Y%m%d_%H%M%S).log"

# Configure GPU settings if available
if [ "${CUDA_VISIBLE_DEVICES:-}" != "" ]; then
    echo "Using GPU devices: $CUDA_VISIBLE_DEVICES"
else
    echo "Running on CPU mode"
fi

# Run the summarization pipeline
echo "Starting summarization pipeline..."
python ../src/summarization_pretrained_pipeline.py \
    --model-name "${DEFAULT_MODEL_NAME}" \
    --max-length "${MODEL_MAX_LENGTH}" \
    --batch-size "${BATCH_SIZE}" \
    --cache-dir "${MODEL_CACHE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "$LOG_FILE"

exit_code=$?

# Check for errors
if [ $exit_code -ne 0 ]; then
    echo "Error: Pipeline failed with exit code $exit_code"
    echo "Check logs at: $LOG_FILE"
    exit $exit_code
fi

echo "Pipeline completed successfully"
echo "Logs saved to: $LOG_FILE"
