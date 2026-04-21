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
#SBATCH --output=logs/slurm-%j/sbatch.out
#SBATCH --error=logs/slurm-%j/sbatch.err

# Single-node multi-GPU: torchrun already handles the per-rank fan-out, so no
# srun step is needed. Running the training tree directly under the batch
# shell keeps the signal path short:
#
#   sbatch (bash, trap USR1) --> headless-tui-run --> tmux --> uv --> torchrun --> python workers
#
# The B: prefix tells Slurm to deliver the pre-termination USR1 only to the
# batch shell. If USR1 reached the whole step it would kill the bash-based
# `headless-tui-run` wrapper (bash default action on USR1 is terminate),
# taking the cgroup down before Python could save. Instead we trap USR1
# here and forward it straight to the Python workers via pkill.
#
# To trigger a preempt manually (testing), use:
#   scancel --batch --signal=USR1 <jobid>
# which targets the batch shell only and matches the auto-preempt path.

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

headless-tui-run --nvidia-smi --screen-file ./logs/slurm-$SLURM_JOB_ID/screen.txt -- \
    uv run flow-control launch "$CONFIG" &
TRAIN_PID=$!

# wait is interruptible by signals; loop so the trap can fire during long
# training runs without us leaving the wait prematurely.
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    wait "$TRAIN_PID" 2>/dev/null || true
done
wait "$TRAIN_PID"

# Log files:
# ./logs/slurm-<job_id>/rank0000.log: Live structured logs from the main process (rank 0).
# ./logs/slurm-<job_id>/rank0000.traceback.log: Tracebacks from the main process, if any.
# ./logs/slurm-<job_id>/screen.txt: Tmux rendered terminal output with nvidia-smi stats and progress bars, updated every 60 seconds.
# ./logs/slurm-<job_id>/sbatch.out: Stdout from the sbatch script itself. `headless-tui-run` also dumps the entire tmux buffer here after execution.