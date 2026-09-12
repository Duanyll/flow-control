"""AWM (Advantage Weighted Matching) preset, arXiv:2509.25050.

The math is :class:`~flow_control.training.objective.AwmObjective`; the loop is
:class:`~flow_control.training.rollout_trainer.EndpointTrainer`. Rollouts are
sampled by the current policy (the lagged EMA when ``objective.off_policy``);
training timesteps are stratified rollout-grid indices over the noisiest 90%
of the plan, skipping the pure-noise step (reference ``discrete_wo_init``).
The lagged EMA is the reference's TRPO-EMA, ``decay = min(0.3, 0.001 * step)``.
"""

from .ema import EMAConfig, LinearRampWarmup
from .mixins import trainer_registry
from .objective import AwmObjective, Objective
from .rollout_trainer import EndpointTrainer
from .train_timesteps import GridTimesteps, TrainTimesteps


@trainer_registry.register("awm")
class AwmTrainer(EndpointTrainer):
    training_type: str = "awm"
    objective: Objective = AwmObjective()
    train_timesteps: TrainTimesteps = GridTimesteps(
        count=6, window=0.9, exclude_first=True, mode="stratified"
    )
    ema_old: EMAConfig = EMAConfig(
        decay=0.3, warmup=LinearRampWarmup(flat_steps=0, ramp_rate=0.001)
    )
    """TRPO-EMA config (stepped once per epoch)."""
