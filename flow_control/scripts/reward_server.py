import asyncio

import torch
import uvicorn
from pydantic import BaseModel, ConfigDict
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from flow_control.rewards import Reward, parse_reward
from flow_control.rewards.base import BaseReward
from flow_control.utils.config import load_config_file
from flow_control.utils.logging import get_logger
from flow_control.utils.remote import deserialize_batch, serialize_tensor
from flow_control.utils.types import TorchDevice

logger = get_logger(__name__)


class RewardServerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    host: str = "0.0.0.0"
    port: int = 8001
    device: TorchDevice = torch.device("cuda")

    reward: Reward | None = None


class _ServerState:
    """Mutable state for the reward server."""

    def __init__(self) -> None:
        self.reward: BaseReward | None = None
        self.config_json: dict | None = None

    def get_reward(self) -> BaseReward:
        if self.reward is None:
            raise RuntimeError("No reward model loaded. POST to /load first.")
        return self.reward


def create_app(config: RewardServerConfig) -> Starlette:
    gpu_lock = asyncio.Lock()
    model_lock = asyncio.Lock()
    state = _ServerState()

    # Pre-load if provided in config
    if config.reward is not None:
        config.reward._load_model(config.device)
        state.reward = config.reward
        state.config_json = config.reward.model_dump(mode="json", exclude={"endpoint"})

    async def load(request: Request) -> Response:
        body = await request.json()
        async with model_lock:
            if state.config_json == body:
                logger.info("Reward config unchanged, skipping reload.")
                return JSONResponse({"status": "ok", "loaded": False})
            logger.info(f"Loading reward from client config: {body.get('type', '?')}")
            reward = parse_reward(body)
            reward._load_model(config.device)
            state.reward = reward
            state.config_json = body
            logger.info("Reward loaded successfully.")
            return JSONResponse({"status": "ok", "loaded": True})

    async def score(request: Request) -> Response:
        reward = state.get_reward()
        body = await request.body()
        batch = deserialize_batch(body, config.device, torch.bfloat16)
        logger.info(f"Scoring batch with keys: {list(batch.keys())}")
        async with gpu_lock:
            with torch.no_grad():
                result = reward._score(batch)
            serialized = serialize_tensor(result)
        return Response(content=serialized, media_type="application/octet-stream")

    async def health(request: Request) -> Response:
        if state.reward is None:
            return JSONResponse({"status": "ok", "loaded": False})
        return JSONResponse({"status": "ok", "loaded": True, "type": state.reward.type})

    async def unload(request: Request) -> Response:
        async with model_lock:
            if state.reward is not None:
                state.reward._unload_model()
                state.reward = None
                state.config_json = None
                logger.info("Reward model unloaded.")
            return JSONResponse({"status": "ok"})

    routes = [
        Route("/load", load, methods=["POST"]),
        Route("/score", score, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
        Route("/unload", unload, methods=["POST"]),
    ]

    return Starlette(routes=routes)


def run(config_path: str) -> None:
    """Start the reward server with the given config file."""
    config = RewardServerConfig(**load_config_file(config_path))

    if config.reward is not None:
        logger.info(f"Pre-loading reward model: {config.reward.type}")

    logger.info(f"Starting reward server on {config.host}:{config.port}")
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start reward computation server.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    run(args.config_path)


if __name__ == "__main__":
    main()
