from typing import Annotated, Any

from pydantic import PlainValidator

from .base import BaseReward
from .clip_score import CLIPScoreReward
from .composite import CompositeReward
from .pickscore import PickScoreReward

REWARD_REGISTRY: dict[str, type[BaseReward]] = {
    "clip_score": CLIPScoreReward,
    "pickscore": PickScoreReward,
    "composite": CompositeReward,
}


def parse_reward(conf: dict[str, Any]) -> BaseReward:
    reward_type = conf["type"]
    reward_class = REWARD_REGISTRY.get(reward_type)
    if reward_class is None:
        raise ValueError(f"Unknown reward type: {reward_type}")
    return reward_class(**conf)


Reward = Annotated[BaseReward, PlainValidator(parse_reward)]

__all__ = ["Reward"]
