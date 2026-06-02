#!/bin/bash
# =============================================================================
# Slurm batch script for flow_control training.
#
# Usage:
#   sbatch examples/slurm/train.sbatch path/to/config.jsonc
#
# Why these specific flags:
#   --requeue                     allow Slurm to automatically requeue us when
#                                 the cluster policy preempts this job.
#   --open-mode=append            keep old logs when the job is requeued.
#                                 The trainer resumes only when the config sets
#                                 auto_resume=true.
# =============================================================================

#SBATCH --job-name=flow-control-train
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# --output/--error capture bootstrap chatter. Python rank logs are written into
# logs/slurm-<job_id>/ by the launcher unless LOG_DIR is set explicitly.

set -euo pipefail
cd $SLURM_SUBMIT_DIR

CONFIG=${1:?usage: sbatch examples/slurm/train.sbatch <config>}

uv run flow-control launch "$CONFIG"

# Output layout (one place per run, stable across requeue):
#   runs/<exp>/<run_id>/
#     ├── metrics.jsonl / meta.json
#     └── checkpoints/step_*/
#   logs/slurm-<job_id>/
#     ├── rank0000.log
#     └── rank0000.traceback.log
