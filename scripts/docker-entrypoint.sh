#!/bin/bash

set -e

# Function to check GPU availability
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found. Running without GPU."
        return 1
    fi
    return 0
}

# Function to setup AWS credentials
setup_aws() {
    if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
        aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
        aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
        aws configure set default.region ${AWS_DEFAULT_REGION:-us-west-2}
    fi
}

# Main entrypoint logic
case "$1" in
    train)
        setup_aws
        if check_gpu; then
            echo "Starting training with GPU..."
            exec python3 -m torch.distributed.launch \
                --nproc_per_node=$NUM_GPUS \
                scripts/train.sh \
                "$@"
        else
            echo "Starting training on CPU..."
            exec python3 scripts/train.sh "$@"
        fi
        ;;
    serve)
        setup_aws
        echo "Starting model server..."
        exec python3 -m src.serving.api "$@"
        ;;
    evaluate)
        setup_aws
        echo "Starting evaluation..."
        exec python3 scripts/evaluate.sh "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
