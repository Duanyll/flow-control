#!/bin/bash

#SBATCH --qos=batch
#SBATCH --job-name=hf_download
#SBATCH --output=logs/hf_download.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

cd "$SLURM_SUBMIT_DIR"

retry_until_success() {
    local cmd="$1"
    local max_retries=5
    local retry_count=0
    local delay=5

    until $cmd; do
        if [ $retry_count -ge $max_retries ]; then
            echo "Command failed after $max_retries attempts: $cmd"
            return 1
        fi
        echo "Command failed, retrying in $delay seconds... (Attempt $((retry_count + 1))/$max_retries)"
        sleep $delay
        retry_count=$((retry_count + 1))
    done
}

MODELSCOPE_CACHE="/gdata/hot3/huggingface/modelscope"

models=("$@")
for model in "${models[@]}"; do
    echo "Downloading model: $model"
    retry_until_success "uvx modelscope download $model"
done