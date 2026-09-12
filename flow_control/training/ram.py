"""RAM (Reinforce Adjoint Matching) preset, arXiv:2605.10759.

The math is :class:`~flow_control.training.objective.RamObjective`; the loop is
:class:`~flow_control.training.rollout_trainer.EndpointTrainer`. Rollouts are
sampled by the lagged EMA policy, which also supplies ``v_old``; training
timesteps are continuous power-law draws (``p(t) ∝ t``), ``K = 8`` per endpoint
(reference ``num_loss_targets_per_sample``). Reference RAM does not clip
gradients.
"""

from .ema import EMAConfig, LinearRampWarmup
from .mixins import trainer_registry
from .objective import Objective, RamObjective
from .rollout_trainer import EndpointTrainer
from .train_timesteps import ContinuousTimesteps, TrainTimesteps
from .weighting import PowerLawTimestepWeighting


@trainer_registry.register("ram")
class RamTrainer(EndpointTrainer):
    training_type: str = "ram"
    objective: Objective = RamObjective()
    train_timesteps: TrainTimesteps = ContinuousTimesteps(
        count=8, weighting=PowerLawTimestepWeighting(alpha=1.0)
    )
    ema_old: EMAConfig = EMAConfig(
        decay=0.9, warmup=LinearRampWarmup(flat_steps=0, ramp_rate=0.01)
    )
    """Old/lagged EMA config (stepped once per epoch): samples endpoints and
    supplies ``v_old`` in the loss target."""
    clip_grad_norm: float = 0.0
