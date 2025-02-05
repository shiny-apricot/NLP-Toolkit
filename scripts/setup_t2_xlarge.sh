#!/bin/bash

# Exit on error
set -e

echo "Setting up t2.xlarge environment..."

# Create virtual environment
python3 -m venv summarization-env
source summarization-env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p logs
mkdir -p cache
mkdir -p outputs

# Set environment variables
export AWS_DEFAULT_REGION=us-west-2
export TRANSFORMERS_CACHE="./cache"
export TORCH_HOME="./cache"

# Verify AWS permissions
echo "Verifying AWS permissions..."
aws sts get-caller-identity || {
    echo "Error: AWS credentials not properly configured"
    exit 1
}

# Verify EC2 permissions
aws ec2 describe-instances --max-items 1 || {
    echo "Error: Missing required EC2 permissions"
    exit 1
}

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Verify setup
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "from transformers import AutoTokenizer; print('Transformers installation successful')"

echo "Setup complete! Run the following to start:"
echo "python scripts/run_aws_pipeline.py --config configs/aws_configs/t2.xlarge.yaml"
