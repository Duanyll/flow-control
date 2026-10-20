"""A scorer hosted by a ``flow-control reward-server``.

The server owns its scorers' configs; a :class:`RemoteScorer` names where
(``url``) and which (``scorer``). The labels, component weights and row fields
come from the server's ``GET /info`` at :meth:`RemoteScorer.load`; the client
keeps ``weight`` and ``normalize``, since the server returns raw scores: the
hosted class's default ``normalize`` (pickscore's affine 1/26, say) does not
apply, so set it on the client.
"""

import asyncio
import io
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from pydantic import BaseModel, PrivateAttr
from torch import Tensor

from flow_control.samplers.node import Row
from flow_control.utils.aiohttp_session import AiohttpSession
from flow_control.utils.tensor import deep_apply_tensor_fn

from .base import BaseScorer, RemoteReward, scorer_registry

if TYPE_CHECKING:
    from flow_control.training.rollouts.runner import RewardContext


class ScorerInfo(BaseModel):
    """One scorer's entry in a reward server's ``GET /info``."""

    labels: list[str]
    weights: list[float]
    fields: list[str]


@scorer_registry.register("remote")
class RemoteScorer(BaseScorer, RemoteReward[Row, Tensor]):
    """A local scorer of a reward server, one ``POST /score/{scorer}`` per row."""

    type: Literal["remote"] = "remote"
    url: str
    """The reward server, e.g. ``http://node-12:8001``."""
    scorer: str
    """The server's name for the scorer (a key of its ``scorers``); also the
    default ``name``."""

    _info: ScorerInfo | None = PrivateAttr(default=None)
    _http: AiohttpSession | None = PrivateAttr(default=None)

    def _http_session(self) -> AiohttpSession:
        if self._http is None:
            self._http = AiohttpSession(timeout=300.0, max_concurrency=64)
        return self._http

    @property
    def label(self) -> str:
        return self.name or self.scorer

    def _loaded(self) -> ScorerInfo:
        if self._info is None:
            raise RuntimeError(
                f"RemoteScorer {self.label!r} reads its labels, weights and fields "
                f"from GET {self.url}/info; call load() first."
            )
        return self._info

    @property
    def labels(self) -> list[str]:
        return self._loaded().labels

    @property
    def weights(self) -> list[float]:
        return self._loaded().weights

    @property
    def fields(self) -> set[str]:
        return set(self._loaded().fields)

    def load(self, device: torch.device) -> None:
        """Fetch ``/info``; runs its own event loop, so call it outside of one."""

        http = self._http_session()

        async def fetch() -> dict[str, Any]:
            try:
                return await http.request(
                    "GET", f"{self.url}/info", read=lambda resp: resp.json()
                )
            finally:
                await http.close()

        info = asyncio.run(fetch())
        if self.scorer not in info:
            raise ValueError(
                f"The reward server at {self.url} hosts no scorer {self.scorer!r}; "
                f"it hosts {sorted(info)}."
            )
        self._info = ScorerInfo.model_validate(info[self.scorer])

    async def run_one(self, row: Row) -> Tensor:
        buffer = io.BytesIO()
        # A copy, not a view: torch.save writes a view's whole storage.
        torch.save([deep_apply_tensor_fn(row, _cpu_copy)], buffer)
        http = self._http_session()
        async with http.slot():
            body = await http.request(
                "POST",
                f"{self.url}/score/{self.scorer}",
                data=buffer.getvalue(),
                read=lambda resp: resp.read(),
            )
        return torch.load(io.BytesIO(body), weights_only=True)[0]


def _cpu_copy(tensor: Tensor) -> Tensor:
    return tensor.detach().to("cpu", copy=True)


if __name__ == "__main__":
    import socket
    import threading
    import time

    import uvicorn
    from rich import print

    from flow_control.scripts.reward_server import RewardServerConfig, create_app

    from .base import LocalReward, Reward

    batches: list[int] = []

    @scorer_registry.register("_remote_test")
    class _Doubler(BaseScorer, LocalReward[Row, Tensor]):
        """``[v, 2v]``, slowly, so that requests pile up during a batch."""

        type: Literal["_remote_test"] = "_remote_test"

        @property
        def labels(self) -> list[str]:
            return ["value", "double"]

        @property
        def weights(self) -> list[float]:
            return [1.0, 0.0]

        @property
        def fields(self) -> set[str]:
            return {"value"}

        def run_batch(self, rows: list[Row]) -> list[Tensor]:
            assert all(set(row) == {"value"} for row in rows), rows
            batches.append(len(rows))
            time.sleep(0.6)
            return [torch.tensor([row["value"], 2.0 * row["value"]]) for row in rows]

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    config = RewardServerConfig.model_validate(
        {"device": "cpu", "scorers": {"doubler": {"type": "_remote_test"}}}
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(config), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)

    url = f"http://127.0.0.1:{port}"
    remote = RemoteScorer(url=url, scorer="doubler", weight=0.5)
    remote.load(torch.device("cpu"))
    assert remote.label == "doubler"
    assert remote.labels == ["value", "double"] and remote.weights == [1.0, 0.0]
    assert remote.fields == {"value"}

    class _Ctx:
        async def run[I, O](self, reward: Reward[I, O], items: list[I]) -> list[O]:
            assert isinstance(reward, RemoteReward)
            return list(await asyncio.gather(*(reward.run_one(i) for i in items)))

    async def scenario() -> list[Tensor]:
        # The first request has the device to itself; the two sent while its
        # batch runs land in one run_batch().
        first = asyncio.create_task(remote.run_one({"value": 1.0}))
        await asyncio.sleep(0.2)
        ctx = cast("RewardContext", _Ctx())
        rest = await remote.score(ctx, [{"value": 2.0}, {"value": 3.0}])
        return [await first, *rest]

    scores = asyncio.run(scenario())
    assert torch.equal(torch.stack(scores), torch.tensor([[1, 2], [2, 4], [3, 6.0]]))
    assert batches == [1, 2], batches

    try:
        RemoteScorer(url=url, scorer="nope").load(torch.device("cpu"))
    except ValueError as exc:
        assert "['doubler']" in str(exc), exc
    else:
        raise AssertionError("an unknown server scorer must be rejected")

    server.should_exit = True
    thread.join()
    print("[green]rewards.remote self-test passed[/green]")
