#!/bin/bash
# =============================================================================
# Slurm batch script for preemption-aware flow_control training.
#
# Usage:
#   sbatch examples/slurm/train.sbatch path/to/config.jsonc
#
# Why these specific flags:
#   --requeue                     allow Slurm to automatically requeue us when
#                                 we are preempted; see `scontrol requeue`
#                                 safety net at the end of the script.
#   --signal=USR1@300             Slurm broadcasts SIGUSR1 to every process in
#                                 the step 5 min before hitting the wall/ being
#                                 preempted. torchrun does NOT reap workers on
#                                 SIGUSR1 (unlike SIGTERM), so PreemptionMixin
#                                 gets a full 300 s to save DCP and exit.
#                                 Tune the lead time to (checkpoint_save_time
#                                 + safety margin).
#   --open-mode=append            keep old logs when the job is requeued.
#   FLOW_CONTROL_AUTO_RESUME=1    tell the trainer to auto-resume from the
#                                 newest step_XXXXXXX under checkpoint_root
#                                 whenever resume_from_dir is unset in config.
# =============================================================================

#SBATCH --job-name=flow-control-train
#SBATCH --partition=compute
#SBATCH --qos=batch
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# --output/--error capture bootstrap chatter only (Slurm doesn't know the
# run_id at parse time). The Python launcher writes structured rank logs into
# runs/<exp>/<run_id>/attempts/slurm-<job_id>/ — see CLAUDE.md for the layout.

set -euo pipefail
cd $SLURM_SUBMIT_DIR

CONFIG=${1:?usage: sbatch examples/slurm/train.sbatch <config>}

export FLOW_CONTROL_AUTO_RESUME=1

# pkill pattern matches Python worker cmdlines, e.g.
#   /.../python -m flow_control.scripts.launch <cfg> --type sft
# but deliberately NOT torchrun (argv[0] = "torchrun", no python token) --
# torchrun under Python's default USR1 handler would terminate and take the
# workers with it mid-save.
forward_preempt() {
    echo "[sbatch] USR1 received; forwarding to Python workers..." >&2
    pkill --signal USR1 -f 'python.* -m flow_control\.scripts\.launch' || true
}
trap forward_preempt USR1

uv run flow-control launch "$CONFIG" &
TRAIN_PID=$!

# wait is interruptible by signals; loop so the trap can fire during long
# training runs without us leaving the wait prematurely.
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    wait "$TRAIN_PID" 2>/dev/null || true
done
wait "$TRAIN_PID"

# Output layout (one place per run, stable across requeue):
#   runs/<exp>/<run_id>/
#     ├── metrics.jsonl / events.jsonl / meta.json
#     ├── attempts/slurm-<job_id>/
#     │     ├── rank0000.log
#     │     └── rank0000.traceback.log
#     └── checkpoints/step_*/