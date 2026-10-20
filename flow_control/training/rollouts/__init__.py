"""Rollout programs: how one prompt becomes its K scored rollouts.

``docs/rollout-reward-design.md`` is the spec. The in-tree programs are
``pointwise`` and ``round_robin``; out-of-tree programs register in
``rollout_registry`` from a config's ``imports``.
"""

from .base import BaseRollout, PointwiseRollout, Rollout, rollout_registry, write
from .pairwise import preferences, schedule, win_rates
from .round_robin import RoundRobinRollout
from .runner import Pending, RewardContext, Runner, SampleContext

__all__ = [
    "BaseRollout",
    "Pending",
    "PointwiseRollout",
    "RewardContext",
    "Rollout",
    "RoundRobinRollout",
    "Runner",
    "SampleContext",
    "preferences",
    "rollout_registry",
    "schedule",
    "win_rates",
    "write",
]
