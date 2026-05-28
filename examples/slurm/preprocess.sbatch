#!/bin/bash

#SBATCH --job-name=flow-control-preprocess
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --gpus=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%j/sbatch.out
#SBATCH --error=logs/slurm-%j/sbatch.err

set -euo pipefail
cd $SLURM_SUBMIT_DIR

CONFIG=${1:?usage: sbatch examples/slurm/preprocess.sbatch <config>}

uv run flow-control preprocess "$CONFIG"