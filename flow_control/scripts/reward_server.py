"""Reward server: hosts local scorers from its own config for ``remote`` scorers.

- ``GET /info``: ``{name: {"labels": [...], "weights": [...], "fields": [...]}}``
- ``POST /score/{name}``: a ``torch.save``'d list of rows (loaded with
  ``weights_only=True``) -> the ``torch.save``'d ``[N, C]`` raw scores.
  Rows that arrive while a batch runs go into the scorer's next ``run_batch()``.
"""

import asyncio
import io
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
import uvicorn
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route
from torch import Tensor

from flow_control.rewards import BaseScorer, LocalReward, RemoteReward, Scorer
from flow_control.rewards.base import run_local_batch
from flow_control.rewards.remote import ScorerInfo
from flow_control.samplers.node import Row
from flow_control.utils import device as devutil
from flow_control.utils.logging import get_logger
from flow_control.utils.types import TorchDevice

logger = get_logger(__name__)


class RewardServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    imports: list[str] = []
    """Out-of-tree plugin modules to import (for registry side effects) before
    constructing this config. Loaded explicitly by the CLI, never via an env var."""

    host: str = "0.0.0.0"
    port: int = 8001
    device: TorchDevice = Field(default_factory=devutil.default_device)

    scorers: dict[str, Scorer]
    """Local scorers by the name clients ask for (``RemoteScorer.scorer``)."""

    @field_validator("scorers")
    @classmethod
    def _local_only(cls, scorers: dict[str, BaseScorer]) -> dict[str, BaseScorer]:
        remote = sorted(
            name for name, s in scorers.items() if isinstance(s, RemoteReward)
        )
        if remote:
            raise ValueError(
                f"The reward server hosts local scorers; {remote} already run "
                "remotely: score them from the client directly."
            )
        return scorers


class _Batcher:
    """One scorer's queue of rows and the task that drains it in ``run_batch`` calls
    of up to ``batch_size`` rows, run on the server's device thread."""

    def __init__(
        self, name: str, scorer: BaseScorer, device_thread: ThreadPoolExecutor
    ):
        assert isinstance(scorer, LocalReward)  # checked by RewardServerConfig
        self.name = name
        self.scorer: LocalReward[Row, Tensor] = scorer
        self._device_thread = device_thread
        self._queue: asyncio.Queue[tuple[Row, asyncio.Future[Tensor]]] = asyncio.Queue()

    async def score(self, rows: list[Row]) -> Tensor:
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in rows]
        for row, future in zip(rows, futures, strict=True):
            self._queue.put_nowait((row, future))
        return torch.stack(await asyncio.gather(*futures))

    async def drain(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self._queue.get()]
            while len(batch) < self.scorer.batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            rows = [row for row, _ in batch]
            try:
                outputs = await loop.run_in_executor(
                    self._device_thread, run_local_batch, self.scorer, rows
                )
            except Exception as exc:  # re-raised in each request of the batch
                logger.exception(
                    f"Scoring a batch of {len(rows)} with {self.name} failed"
                )
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
                continue
            for (_, future), output in zip(batch, outputs, strict=True):
                if not future.done():  # its request may have been cancelled
                    future.set_result(output)


def create_app(config: RewardServerConfig) -> Starlette:
    """Load every scorer onto ``config.device`` and build the app."""
    # One thread runs every scorer's batches, so one batch at a time is on the device.
    device_thread = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="reward-device",
        initializer=devutil.set_device,
        initargs=(config.device,),
    )
    batchers = {
        name: _Batcher(name, scorer, device_thread)
        for name, scorer in config.scorers.items()
    }
    for name, batcher in batchers.items():
        logger.info(f"Loading scorer {name!r} on {config.device}")
        device_thread.submit(batcher.scorer.load, config.device).result()
    info = {
        name: ScorerInfo(
            labels=scorer.labels, weights=scorer.weights, fields=sorted(scorer.fields)
        ).model_dump()
        for name, scorer in config.scorers.items()
    }

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        drainers = [asyncio.create_task(b.drain()) for b in batchers.values()]
        try:
            yield
        finally:
            for task in drainers:
                task.cancel()
            device_thread.shutdown(wait=False, cancel_futures=True)

    async def get_info(request: Request) -> Response:
        return JSONResponse(info)

    async def score(request: Request) -> Response:
        name = request.path_params["name"]
        batcher = batchers[name]
        rows = torch.load(io.BytesIO(await request.body()), weights_only=True)
        try:
            scores = await batcher.score(rows)
        except Exception as exc:  # a run_batch failure is logged by the drainer
            return PlainTextResponse(f"Scoring with {name} failed: {exc!r}", 500)
        buffer = io.BytesIO()
        torch.save(scores, buffer)
        return Response(buffer.getvalue(), media_type="application/octet-stream")

    return Starlette(
        routes=[
            Route("/info", get_info, methods=["GET"]),
            Route("/score/{name}", score, methods=["POST"]),
        ],
        lifespan=lifespan,
    )


def run(config_data: dict) -> None:
    """Start the reward server with the given config."""
    config = RewardServerConfig.model_validate(config_data)
    app = create_app(config)
    logger.info(f"Serving {sorted(config.scorers)} on {config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port)
