#!/bin/bash

# Exit on error
set -e

echo "Setting up r5.2xlarge memory-optimized environment..."

# System optimizations for memory performance
sudo sysctl -w vm.swappiness=10
sudo sysctl -w vm.dirty_ratio=40
sudo sysctl -w vm.dirty_background_ratio=10

# Create virtual environment with specific Python version
python3 -m venv summarization-env --prompt="(r5-mem)"
source summarization-env/bin/activate

# Upgrade pip and install memory-optimized dependencies
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# Install memory monitoring tools
pip install --no-cache-dir \
    psutil \
    memory_profiler \
    py-spy \
    numpy \
    torch>=2.0.0  # Ensure we have latest PyTorch with memory optimizations

# Create directories with appropriate permissions
mkdir -p logs/memory_profiles
mkdir -p cache
mkdir -p outputs
mkdir -p temp/model_cache
mkdir -p temp/dataset_cache

# Set environment variables for memory optimization
export MALLOC_TRIM_THRESHOLD_=100000
export PYTHONMALLOC=malloc
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export TRANSFORMERS_CACHE="./cache"
export TORCH_HOME="./cache"
export TORCH_CPU_NUM_THREADS=8
export AWS_DEFAULT_REGION=us-west-2

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Configure memory monitoring
echo "Setting up memory monitoring..."
cat > ./scripts/monitor_memory.py << EOL
import psutil
import time
import json
from pathlib import Path

def log_memory_usage(log_file):
    while True:
        memory = psutil.virtual_memory()
        with open(log_file, 'a') as f:
            json.dump({
                'timestamp': time.time(),
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_used_gb': memory.used / (1024**3)
            }, f)
            f.write('\n')
        time.sleep(60)

if __name__ == '__main__':
    log_memory_usage('logs/memory_profiles/memory_usage.jsonl')
EOL

# Start memory monitoring in background
python ./scripts/monitor_memory.py &
MONITOR_PID=$!

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Verify setup and available memory
echo "Verifying setup..."
python3 - << EOL
import torch
import psutil
import json
from pathlib import Path

memory = psutil.virtual_memory()
setup_info = {
    "pytorch_version": torch.__version__,
    "cpu_count": psutil.cpu_count(),
    "memory_total_gb": memory.total / (1024**3),
    "memory_available_gb": memory.available / (1024**3)
}

print("\nSystem Configuration:")
print(json.dumps(setup_info, indent=2))

# Save configuration
with open("logs/setup_info.json", "w") as f:
    json.dump(setup_info, f, indent=2)
EOL

echo "Setting up process priority..."
# Set nice level for Python processes to prioritize memory operations
sudo renice -n -10 $$

# Make runner script executable
chmod +x scripts/run_pretrained.py

# Update the final message
echo """
Setup complete! 

To run the pretrained pipeline:
./scripts/run_pretrained.py \\
    --model-name facebook/bart-large-cnn \\
    --dataset-split validation \\
    --sample-size 100 \\
    --device auto

Memory monitoring log: logs/memory_profiles/memory_usage.jsonl
Setup info: logs/setup_info.json
"""

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    kill $MONITOR_PID 2>/dev/null || true
}

# Register cleanup function
trap cleanup EXIT
