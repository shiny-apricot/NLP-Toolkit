#!/bin/bash

# Exit on error
set -e

# Load environment variables
if [ -f .env ]; then
    source .env
fi

# Default values
MODEL_TYPE="bart"
BATCH_SIZE=8
NUM_EPOCHS=3
LEARNING_RATE=2e-5
OUTPUT_DIR="checkpoints"
DATA_PATH=""
S3_BUCKET=""
USE_FP16=true
MAX_LENGTH=1024
NUM_GPUS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --s3-bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        --no-fp16)
            USE_FP16=false
            shift
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$DATA_PATH" ]; then
    echo "Error: --data-path is required"
    exit 1
fi

# Setup logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to upload to S3
upload_to_s3() {
    if [ -n "$S3_BUCKET" ]; then
        log "Uploading results to S3..."
        aws s3 cp "$OUTPUT_DIR" "s3://$S3_BUCKET/$OUTPUT_DIR" --recursive
        aws s3 cp "$LOG_FILE" "s3://$S3_BUCKET/logs/"
    fi
}

# Function to handle cleanup
cleanup() {
    log "Cleaning up..."
    # Upload results to S3 before exit
    upload_to_s3
    
    # Clean up any temporary files
    if [ -d "/tmp/cache" ]; then
        rm -rf /tmp/cache
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log configuration
log "Starting training with configuration:"
log "- Model Type: $MODEL_TYPE"
log "- Batch Size: $BATCH_SIZE"
log "- Number of Epochs: $NUM_EPOCHS"
log "- Learning Rate: $LEARNING_RATE"
log "- Output Directory: $OUTPUT_DIR"
log "- Data Path: $DATA_PATH"
log "- Number of GPUs: $NUM_GPUS"
log "- FP16: $USE_FP16"

# Prepare GPU arguments
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES

# Run training based on model type
case $MODEL_TYPE in
    "bart")
        log "Training BART model..."
        python -m src.models.fine_tuning.fine_tune_bart \
            --data-path "$DATA_PATH" \
            --output-dir "$OUTPUT_DIR" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$NUM_EPOCHS" \
            --learning-rate "$LEARNING_RATE" \
            --max-length "$MAX_LENGTH" \
            --fp16 "$USE_FP16" \
            2>&1 | tee -a "$LOG_FILE"
        ;;
    "t5")
        log "Training T5 model..."
        python -m src.models.fine_tuning.fine_tune_t5 \
            --data-path "$DATA_PATH" \
            --output-dir "$OUTPUT_DIR" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$NUM_EPOCHS" \
            --learning-rate "$LEARNING_RATE" \
            --max-length "$MAX_LENGTH" \
            --fp16 "$USE_FP16" \
            2>&1 | tee -a "$LOG_FILE"
        ;;
    "gpt")
        log "Training GPT model..."
        python -m src.models.fine_tuning.fine_tune_gpt \
            --data-path "$DATA_PATH" \
            --output-dir "$OUTPUT_DIR" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$NUM_EPOCHS" \
            --learning-rate "$LEARNING_RATE" \
            --max-length "$MAX_LENGTH" \
            --fp16 "$USE_FP16" \
            2>&1 | tee -a "$LOG_FILE"
        ;;
    *)
        log "Error: Unsupported model type: $MODEL_TYPE"
        exit 1
        ;;
esac

# Check if training was successful
if [ $? -eq 0 ]; then
    log "Training completed successfully"
    
    # Upload results to S3 if bucket is specified
    upload_to_s3
    
    log "All done!"
else
    log "Training failed with error code $?"
    exit 1
fi
