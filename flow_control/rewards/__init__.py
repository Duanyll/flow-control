from typing import Annotated, Any

from pydantic import Discriminator, Tag, TypeAdapter

from .base import BaseReward
from .clip_score import CLIPScoreReward
from .composite import CompositeReward
from .geneval import GenevalReward
from .pickscore import PickScoreReward
from .unified_reward import UnifiedReward

Reward = Annotated[
    Annotated[CLIPScoreReward, Tag("clip_score")]
    | Annotated[PickScoreReward, Tag("pickscore")]
    | Annotated[GenevalReward, Tag("geneval")]
    | Annotated[UnifiedReward, Tag("unified_reward")]
    | Annotated[CompositeReward, Tag("composite")],
    Discriminator("type"),
]

_reward_ta = TypeAdapter(Reward)


def parse_reward(conf: dict[str, Any]) -> BaseReward:
    """Parse a reward config dict into the appropriate reward instance."""
    return _reward_ta.validate_python(conf)


__all__ = ["Reward", "parse_reward"]
