"""Slurm preemption support for training loops.

Slurm's default preemption (and ``--time`` timeout) sends ``SIGTERM`` to job
step processes and upgrades it to ``SIGKILL`` after a short, non-configurable
grace period enforced by ``torchrun``. That window is too short to write a
DCP checkpoint reliably. To give us enough time, we ask Slurm to send
``SIGUSR1`` ahead of the real termination via ``--signal=USR1@<seconds>``.
``torchrun`` does not attempt to reap workers on ``SIGUSR1``, so we can
handle it in Python and exit cleanly.

Integration:

1. Trainer's ``run()`` calls :meth:`PreemptionMixin.install_preempt_handler`
   once per worker process (after ``init_device_mesh`` and after any
   synchronous load-checkpoint step, so the handler is only active once we
   are in a state where saving makes sense).
2. At the end of every gradient-sync step the trainer calls
   :meth:`PreemptionMixin.check_preempt_and_maybe_exit` (a short cross-rank
   all-reduce). If any rank caught ``SIGUSR1`` every rank cooperatively saves
   a DCP checkpoint, requeues the Slurm job, and exits 0.

The mixin requires a ``save(step)`` method (provided by
:class:`CheckpointingMixin`) and the ``HsdpMixin`` attributes
(``rank``/``is_main_process``/``device``).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from typing import ClassVar

import torch
import torch.distributed as dist
from pydantic import BaseModel

from flow_control.utils.logging import get_logger

logger = get_logger(__name__)

_PREEMPT_EVENT = threading.Event()


def _handle_preempt_signal(signum: int, _frame) -> None:
    """Signal handler: only set a flag. No IO, no NCCL, no print."""
    _PREEMPT_EVENT.set()


class PreemptionMixin(BaseModel):
    """Coordinate a cooperative checkpoint-and-exit on Slurm preemption."""

    preempt_signal: int = signal.SIGUSR1
    """Signal to listen on. Set to 0 to disable preemption handling entirely."""

    preempt_exit_code: int = 0
    """Exit code used after a successful preempt-save. 0 lets the sbatch script
    continue (and run ``scontrol requeue`` itself) without triggering a
    "FAILED" job state."""

    _preempt_installed: ClassVar[bool] = False

    def install_preempt_handler(self) -> None:
        """Register the signal handler in this worker process.

        Safe to call multiple times: only the first call installs the handler.
        A no-op when ``preempt_signal`` is 0 or the thread is not the main one
        (signal handlers can only be installed from the main thread).
        """
        if self.preempt_signal == 0:
            return
        if PreemptionMixin._preempt_installed:
            return
        if threading.current_thread() is not threading.main_thread():
            logger.warning(
                "install_preempt_handler called off the main thread; skipping."
            )
            return
        signal.signal(self.preempt_signal, _handle_preempt_signal)
        PreemptionMixin._preempt_installed = True
        logger.info(
            "Preempt handler installed for signal %s. Send it (e.g. "
            "`scancel --signal=USR1 $SLURM_JOB_ID`) to trigger a graceful "
            "checkpoint + requeue.",
            signal.Signals(self.preempt_signal).name,
        )

    def _local_preempt_flag(self) -> bool:
        return _PREEMPT_EVENT.is_set()

    def _any_rank_preempted(self) -> bool:
        """All-reduce (MAX) the local flag across ranks. Cheap (1 int)."""
        local = 1 if self._local_preempt_flag() else 0
        if not dist.is_available() or not dist.is_initialized():
            return bool(local)

        device = getattr(self, "device", None)
        if not isinstance(device, torch.device):
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        flag = torch.tensor([local], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def check_preempt_and_maybe_exit(self, step: int) -> None:
        """Collective check; if any rank was preempted, save and exit.

        Must be called from *every* rank, at the same iteration boundary, to
        keep the ``all_reduce`` collective aligned.
        """
        if self.preempt_signal == 0:
            return
        if not self._any_rank_preempted():
            return

        logger.warning(
            "Preempt signal received; saving checkpoint at step %d and exiting.",
            step,
        )

        save = getattr(self, "save", None)
        if callable(save):
            try:
                save(step)
            except Exception:
                logger.exception(
                    "Failed to save checkpoint during preempt. Exiting anyway."
                )
        else:
            logger.warning(
                "PreemptionMixin user has no save(step) method; "
                "no checkpoint will be written."
            )

        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                logger.exception("dist.barrier() failed during preempt save.")

        is_main = bool(getattr(self, "is_main_process", True))
        if is_main:
            self._maybe_scontrol_requeue()

        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                logger.exception("destroy_process_group() failed during preempt save.")

        sys.exit(self.preempt_exit_code)

    @staticmethod
    def _maybe_scontrol_requeue() -> None:
        job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
        if not job_id:
            logger.info("SLURM_JOB_ID not set; skipping scontrol requeue.")
            return
        cmd = ["scontrol", "requeue", job_id]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning("scontrol requeue failed: %s", exc)
            return
        if result.returncode != 0:
            logger.warning(
                "scontrol requeue exited %d; stderr=%s",
                result.returncode,
                result.stderr.strip(),
            )
        else:
            logger.info("Requeued Slurm job %s via scontrol.", job_id)


if __name__ == "__main__":
    # Smoke test: simulate a preempt signal in a single-process setting.
    from rich import print as rprint

    class _FakeTrainer(PreemptionMixin):
        model_config = {"arbitrary_types_allowed": True}
        device: torch.device = torch.device("cpu")

        @property
        def is_main_process(self) -> bool:
            return True

        def save(self, step: int) -> None:
            rprint(f"[green]would save checkpoint for step={step}[/green]")

    trainer = _FakeTrainer()
    trainer.install_preempt_handler()
    rprint("[cyan]sending SIGUSR1 to self...[/cyan]")
    os.kill(os.getpid(), signal.SIGUSR1)
    # After the handler runs, event is set:
    assert _PREEMPT_EVENT.is_set()
    rprint(f"[cyan]local flag: {trainer._local_preempt_flag()}[/cyan]")
    rprint(f"[cyan]any rank preempted: {trainer._any_rank_preempted()}[/cyan]")
    # Reset so we do not sys.exit in the smoke test
    _PREEMPT_EVENT.clear()
    trainer.check_preempt_and_maybe_exit(123)
    rprint("[green]smoke test completed without exiting.[/green]")
