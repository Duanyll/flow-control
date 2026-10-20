"""``RunnerMixin``: the trainer's run-long :class:`~flow_control.training.rollouts.Runner`.

One device worker and reward loop serve every sampling phase of a run
(rollouts, validation): a batch sampled ahead keeps issuing reward requests
across the phases that follow, so the runner outlives them all and closes with
the trainer.
"""

import torch
from pydantic import PrivateAttr

from flow_control.adapters import ModelAdapter
from flow_control.training.rollouts import Runner

from .data import DataMixin

_CPU = torch.device("cpu")


class RunnerMixin(DataMixin):
    _runner: Runner | None = PrivateAttr(default=None)

    @property
    def runner(self) -> Runner:
        if self._runner is None:
            raise RuntimeError("make_runner() runs before the first sampling phase.")
        return self._runner

    def make_runner(self, model: ModelAdapter, storage: torch.device = _CPU) -> None:
        """Open the runner once the model and the processor are loaded;
        *storage* is where decoded rows go (the RL trainers' ``rollout_storage``)."""
        self._runner = Runner(model, self.processor, storage=storage, seed=self.seed)

    def close_runner(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None

    def cleanup(self) -> None:
        # Runs on every exit of run(), an exception included: batches sampled
        # ahead keep issuing reward requests until the runner is closed.
        self.close_runner()
        super().cleanup()
