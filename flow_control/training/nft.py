"""Diffusion NFT (Negative-aware FineTuning) preset, arXiv:2509.16117.

The math is :class:`~flow_control.training.objective.NftObjective`; the loop is
:class:`~flow_control.training.rollout_trainer.EndpointTrainer`. Rollouts are
sampled by the old-teacher EMA and trained on the executed rollout grid.
"""

from .ema import EMAConfig, LinearRampWarmup
from .mixins import trainer_registry
from .objective import NftObjective, Objective
from .rollout_trainer import EndpointTrainer
from .train_timesteps import GridTimesteps, TrainTimesteps


@trainer_registry.register("nft")
class NftTrainer(EndpointTrainer):
    training_type: str = "nft"
    objective: Objective = NftObjective()
    train_timesteps: TrainTimesteps = GridTimesteps()
    ema_old: EMAConfig = EMAConfig(
        decay=0.5, warmup=LinearRampWarmup(flat_steps=0, ramp_rate=0.001)
    )
    """Old-teacher EMA config (stepped once per epoch)."""
