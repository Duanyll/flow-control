"""The aiohttp session layer shared by HTTP clients (``LLMClient``, ``RemoteScorer``).

A client config outlives event loops: a trainer's run-long reward loop, an
``asyncio.run`` at load time, a short-lived loop in a script. aiohttp sessions
and asyncio primitives bind to the loop they were created on, so
:class:`AiohttpSession` keeps one session (and semaphore, and lock) per running
loop and recreates them when the loop changes. On top of that: a
``max_concurrency`` semaphore, retries with backoff on connection errors and
retryable statuses (408, 409, 425, 429, 5xx), and a bounded wait while a gateway
answers that it has no capacity.
"""

import asyncio
import contextlib
import json
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import aiohttp

from flow_control.utils.logging import get_logger

logger = get_logger(__name__)

RETRY_STATUS = frozenset({408, 409, 425, 429})


def _gateway_retry_after(
    status: int, retry_after: str | None, body: str
) -> float | None:
    if status not in (429, 503) or retry_after is None:
        return None
    try:
        error = json.loads(body).get("error") or {}
    except (AttributeError, json.JSONDecodeError):
        return None
    if error.get("type") != "capacity" or error.get("code") != "capacity_unavailable":
        return None
    try:
        return max(0.5, float(retry_after))
    except ValueError:
        return 5.0


def _retryable_status(status: int) -> bool:
    return status in RETRY_STATUS or 500 <= status < 600


class AiohttpSession:
    """Loop-bound aiohttp session with retries, capacity wait and a concurrency cap.

    Args:
        timeout: Total seconds per attempt (``aiohttp.ClientTimeout(total=...)``).
        retries: Retries on connection errors and retryable statuses.
        max_concurrency: Size of the :meth:`slot` semaphore and of the connection
            pool; 0 means no semaphore and a pool of 64 connections.
        capacity_wait_timeout: Wall-clock bound on waiting through a gateway's
            capacity 429/503 answers.
        headers: Default headers of every request.
    """

    def __init__(
        self,
        *,
        timeout: float,
        retries: int = 2,
        max_concurrency: int = 0,
        capacity_wait_timeout: float = 30 * 60,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.max_concurrency = max_concurrency
        self.capacity_wait_timeout = capacity_wait_timeout
        self.headers = headers or {}
        self._session: aiohttp.ClientSession | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._lock: asyncio.Lock | None = None
        self._bound_loop: asyncio.AbstractEventLoop | None = None

    def _rebind(self) -> None:
        """Drop session/semaphore/lock when the running event loop changed.

        Reusing them from a later loop raises "Event loop is closed". This must
        never happen while another loop still has requests of this session in
        flight.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._bound_loop is loop:
            return
        if self._session is not None:
            # The old loop is already closed, so a clean async close is not
            # possible; force_close=True on the connector means no live sockets
            # linger. Best-effort sync cleanup to quiet "Unclosed client
            # session" warnings, then abandon the reference. ``close()`` is a
            # coroutine (aiohttp 3.14) that needs the dead loop; its sync part
            # only marks the connector closed.
            connector = self._session.connector
            if connector is not None:
                with contextlib.suppress(Exception):
                    connector._close()
            with contextlib.suppress(Exception):
                self._session._connector = None
        self._session = None
        self._semaphore = None
        self._lock = None
        self._bound_loop = loop

    def session(self) -> aiohttp.ClientSession:
        """The running loop's session."""
        self._rebind()
        if self._session is None:
            connector = aiohttp.TCPConnector(
                # There has to be some limit to tcp connections or we might run
                # out of file descriptors.
                limit=self.max_concurrency if self.max_concurrency > 0 else 64,
                force_close=True,
            )
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self.headers,
                connector=connector,
            )
        return self._session

    def lock(self) -> asyncio.Lock:
        """A lock of the running loop, e.g. for a one-time async initialization."""
        self._rebind()
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one of ``max_concurrency`` slots around a logical request.

        :meth:`request` does not take a slot itself, so a caller can hold one
        across several requests and the work between them.
        """
        if self.max_concurrency <= 0:
            yield
            return
        self._rebind()
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        async with self._semaphore:
            yield

    async def request[T](
        self,
        method: str,
        url: str,
        read: Callable[[aiohttp.ClientResponse], Awaitable[T]],
        **kwargs: Any,
    ) -> T:
        """Send a request until it succeeds and return ``read(response)``.

        Connection errors and retryable statuses are retried ``retries`` times
        with jittered exponential backoff; a gateway's capacity answer is waited
        out for up to ``capacity_wait_timeout``. Other error statuses raise
        ``aiohttp.ClientResponseError`` at once, its message carrying the start
        of the response body.
        """
        errors = 0
        failure_delay = 1.0
        capacity_deadline = time.monotonic() + self.capacity_wait_timeout

        while True:
            capacity_delay: float | None = None
            try:
                async with self.session().request(method, url, **kwargs) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        capacity_delay = _gateway_retry_after(
                            resp.status,
                            resp.headers.get("Retry-After"),
                            body,
                        )
                        if capacity_delay is None:
                            raise aiohttp.ClientResponseError(
                                resp.request_info,
                                resp.history,
                                status=resp.status,
                                message=f"{resp.reason}: {body[:500]}",
                                headers=resp.headers,
                            )
                    else:
                        return await read(resp)
            except aiohttp.ClientResponseError as exc:
                if not _retryable_status(exc.status):
                    raise
                errors += 1
                if errors > self.retries:
                    raise
                await asyncio.sleep(failure_delay * (0.5 + random.random()))
                failure_delay = min(30.0, failure_delay * 2)
                continue
            except (
                aiohttp.ClientConnectionError,
                aiohttp.ClientPayloadError,
                TimeoutError,
            ):
                errors += 1
                if errors > self.retries:
                    raise
                await asyncio.sleep(failure_delay * (0.5 + random.random()))
                failure_delay = min(30.0, failure_delay * 2)
                continue

            assert capacity_delay is not None
            remaining = capacity_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{url} still had no capacity after "
                    f"{self.capacity_wait_timeout:.0f}s"
                )
            wait = min(capacity_delay * (0.5 + random.random()), remaining)
            logger.info(f"{url} has no capacity; retrying in {wait:.1f}s")
            await asyncio.sleep(wait)

    async def close(self) -> None:
        """Close the session, on the loop it belongs to."""
        if self._session is not None:
            session, self._session = self._session, None
            await session.close()

    def __del__(self) -> None:
        if self._session is None:
            return
        try:
            asyncio.get_running_loop().create_task(self._session.close())
        except RuntimeError:
            asyncio.run(self._session.close())


if __name__ == "__main__":
    import socket
    import threading

    from aiohttp import web
    from rich import print

    hits: dict[str, int] = {}

    async def handle(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        hits[name] = hits.get(name, 0) + 1
        if name == "flaky" and hits[name] == 1:
            return web.Response(status=503)
        if name == "busy" and hits[name] == 1:
            body = {"error": {"type": "capacity", "code": "capacity_unavailable"}}
            return web.json_response(body, status=429, headers={"Retry-After": "0"})
        if name in ("missing", "down"):
            status = 404 if name == "missing" else 500
            return web.Response(status=status, text=f"{name} body")
        return web.json_response({"name": name, "hits": hits[name]})

    # The server runs on its own loop in a thread, so it outlives the client's loops.
    server_loop = asyncio.new_event_loop()
    app = web.Application()
    app.router.add_get("/{name}", handle)
    runner = web.AppRunner(app)
    server_loop.run_until_complete(runner.setup())
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    base = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server_loop.run_until_complete(web.SockSite(runner, sock).start())
    threading.Thread(target=server_loop.run_forever, daemon=True).start()

    http = AiohttpSession(timeout=10, retries=2)

    async def get(name: str) -> Any:
        return await http.request("GET", f"{base}/{name}", read=lambda r: r.json())

    # A 503 and a capacity 429 are retried; each request ends on its second try.
    assert asyncio.run(get("flaky")) == {"name": "flaky", "hits": 2}
    first_session = http._session
    assert asyncio.run(get("busy")) == {"name": "busy", "hits": 2}
    # The second asyncio.run is another loop: a new session, the old one dropped.
    assert http._session is not first_session

    # A 404 is not retried; a 500 is, ``retries`` times.
    for name, attempts in (("missing", 1), ("down", 3)):
        try:
            asyncio.run(get(name))
        except aiohttp.ClientResponseError as exc:
            assert hits[name] == attempts, (name, hits[name])
            assert f"{name} body" in exc.message, exc.message
            print(f"{name}: {exc.status} after {attempts} attempt(s)")
        else:
            raise AssertionError(f"{name} must raise")
    asyncio.run(http.close())

    # slot() caps concurrent holders at max_concurrency.
    capped = AiohttpSession(timeout=10, max_concurrency=2)
    holding = peak = 0

    async def hold() -> None:
        global holding, peak
        async with capped.slot():
            holding += 1
            peak = max(peak, holding)
            await asyncio.sleep(0.02)
            holding -= 1

    async def crowd() -> None:
        await asyncio.gather(*(hold() for _ in range(6)))
        await capped.close()

    asyncio.run(crowd())
    assert peak == 2, peak

    server_loop.call_soon_threadsafe(server_loop.stop)
    print("[green]aiohttp_session self-test passed[/green]")
