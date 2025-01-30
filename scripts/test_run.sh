#!/bin/bash

set -e

# Default values
export TEST_BATCH_SIZE=2
export TEST_NUM_SAMPLES=100
export TEST_MODEL_TYPE="bart"
export TEST_DATASET="cnn_dailymail"
export TEST_MAX_LENGTH=512
export TEST_NUM_EPOCHS=1
export TEST_OUTPUT_DIR="outputs/test_run"
export TEST_CACHE_DIR="cache"
export TEST_LOG_DIR="logs/test_run"

# Create required directories
mkdir -p "${TEST_OUTPUT_DIR}" "${TEST_CACHE_DIR}" "${TEST_LOG_DIR}"

# Setup logging
LOG_FILE="${TEST_LOG_DIR}/test_run_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE") 2>&1

echo "Starting test run at $(date)"
echo "Configuration:"
echo "- Model Type: ${TEST_MODEL_TYPE}"
echo "- Dataset: ${TEST_DATASET}"
echo "- Batch Size: ${TEST_BATCH_SIZE}"
echo "- Number of Samples: ${TEST_NUM_SAMPLES}"

# Test dataset loading
echo "Testing dataset loading..."
python -c "
from src.data.loader import HuggingFaceLoader
from src.utils.config import ConfigLoader

config_loader = ConfigLoader()
instance_config = config_loader.load_instance_config('t2.xlarge')

loader = HuggingFaceLoader(
    dataset_name='${TEST_DATASET}',
    instance_config=instance_config
)

dataset = loader.load()
print(f'Successfully loaded {len(dataset)} examples')
dataset.save_to_disk('${TEST_OUTPUT_DIR}/test_dataset')
"

# Test model loading
echo "Testing model loading..."
python -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = 'facebook/bart-base' if '${TEST_MODEL_TYPE}' == 'bart' else 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='${TEST_CACHE_DIR}')
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='${TEST_CACHE_DIR}')
print(f'Successfully loaded {model_name} model and tokenizer')
"

# Run training test
echo "Running training test..."
./scripts/train.sh \
    --model-type "${TEST_MODEL_TYPE}" \
    --batch-size "${TEST_BATCH_SIZE}" \
    --epochs "${TEST_NUM_EPOCHS}" \
    --data-path "${TEST_OUTPUT_DIR}/test_dataset" \
    --output-dir "${TEST_OUTPUT_DIR}" \
    --no-fp16 \
    --num-gpus 0 \
    --max-length "${TEST_MAX_LENGTH}"

# Cleanup
cleanup() {
    echo "Cleaning up..."
    if [ -d "${TEST_CACHE_DIR}/test_run" ]; then
        rm -rf "${TEST_CACHE_DIR}/test_run"
    fi
}

trap cleanup EXIT

echo "Test run completed at $(date)"