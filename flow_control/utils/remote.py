import io
import pickle
from typing import Any

import httpx
import torch
from pydantic import BaseModel, PrivateAttr

from .logging import get_logger
from .tensor import deep_cast_float_dtype, deep_move_to_device

logger = get_logger(__name__)

REMOTE_TIMEOUT = httpx.Timeout(timeout=300.0)


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize tensor to bytes, converting to bf16 first."""
    tensor = tensor.to(torch.bfloat16).cpu()
    buffer = io.BytesIO()
    pickle.dump(tensor, buffer)
    return buffer.getvalue()


def deserialize_tensor(
    data: bytes, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Deserialize tensor from bytes and move to specified device/dtype."""
    buffer = io.BytesIO(data)
    tensor: torch.Tensor = pickle.load(buffer)  # noqa: S301
    return tensor.to(device=device, dtype=dtype)


def serialize_batch(batch: dict[str, Any]) -> bytes:
    """Serialize a batch dict (tensors + strings + metadata) to bytes.

    Tensors are converted to bf16 and moved to CPU before pickling.
    """
    cpu_batch = deep_move_to_device(
        deep_cast_float_dtype(batch, torch.bfloat16), torch.device("cpu")
    )
    buffer = io.BytesIO()
    pickle.dump(cpu_batch, buffer)
    return buffer.getvalue()


def deserialize_batch(
    data: bytes, device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    """Deserialize a batch dict from bytes, moving all tensors to device/dtype."""
    buffer = io.BytesIO(data)
    batch: dict[str, Any] = pickle.load(buffer)  # noqa: S301
    return deep_move_to_device(deep_cast_float_dtype(batch, dtype), device)


class RemoteClient:
    """HTTP client for remote model offloading with sync and async support."""

    def __init__(self, endpoint: str, timeout: httpx.Timeout = REMOTE_TIMEOUT):
        self.endpoint = endpoint
        self.timeout = timeout
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self.timeout)
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    def post_bytes(self, path: str, data: bytes) -> bytes:
        """Sync POST binary data, returns response bytes."""
        resp = self._get_sync_client().post(f"{self.endpoint}{path}", content=data)
        resp.raise_for_status()
        return resp.content

    async def async_post_bytes(self, path: str, data: bytes) -> bytes:
        """Async POST binary data, returns response bytes."""
        resp = await self._get_async_client().post(
            f"{self.endpoint}{path}", content=data
        )
        resp.raise_for_status()
        return resp.content

    def post_json(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Sync POST JSON, returns JSON response."""
        resp = self._get_sync_client().post(f"{self.endpoint}{path}", json=data)
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    def get_json(self, path: str) -> dict[str, Any]:
        """Sync GET, returns JSON response."""
        resp = self._get_sync_client().get(f"{self.endpoint}{path}")
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    def close(self) -> None:
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None


class RemoteOffloadable(BaseModel):
    """Mixin for models that can be offloaded to a remote server.

    Provides ``endpoint`` configuration, a ``RemoteClient`` instance, and helper
    methods for common remote call patterns (tensor-in/tensor-out and
    batch-in/tensor-out).  Subclasses call ``_init_remote()`` during their
    ``load_model`` path to establish the connection and send their pydantic config
    to the server's ``/load`` endpoint.
    """

    endpoint: str | None = None

    _remote_client: RemoteClient | None = PrivateAttr(default=None)
    _remote_device: torch.device | None = PrivateAttr(default=None)
    _remote_dtype: torch.dtype = PrivateAttr(default=torch.bfloat16)

    @property
    def is_remote(self) -> bool:
        return self._remote_client is not None

    def _init_remote(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        exclude_fields: set[str] | None = None,
    ) -> None:
        """Initialize the remote client and send config to the server's /load endpoint.

        Args:
            device: Device to place returned tensors on.
            dtype: Dtype for returned tensors.
            exclude_fields: Fields to exclude from the serialized config
                (``endpoint`` is always excluded).
        """
        if self.endpoint is None:
            raise ValueError("Cannot init remote: endpoint is None")
        excluded = {"endpoint"}
        if exclude_fields:
            excluded |= exclude_fields
        self._remote_client = RemoteClient(self.endpoint)
        self._remote_device = device
        self._remote_dtype = dtype

        config_json = self.model_dump(mode="json", exclude=excluded)
        result = self._remote_client.post_json("/load", config_json)
        status = result.get("status")
        if status != "ok":
            raise ConnectionError(
                f"Remote server at {self.endpoint} returned status={status}: {result}"
            )
        loaded = result.get("loaded", True)
        if loaded:
            logger.info(f"Remote server at {self.endpoint} loaded model from config.")
        else:
            logger.info(
                f"Remote server at {self.endpoint} already has the model loaded."
            )

    def _close_remote(self) -> None:
        """Close the remote client connection."""
        if self._remote_client is not None:
            self._remote_client.close()
            self._remote_client = None
            self._remote_device = None

    # ── Helper methods for remote calls ──────────────────────────────────

    def _remote_tensor_call(self, path: str, tensor: torch.Tensor) -> torch.Tensor:
        """Serialize tensor, POST to server, deserialize result tensor."""
        assert self._remote_client is not None
        assert self._remote_device is not None
        data = serialize_tensor(tensor)
        logger.debug(
            f"Sending {len(data) / 1024 / 1024:.2f} MB to {self.endpoint}{path}"
        )
        result = self._remote_client.post_bytes(path, data)
        return deserialize_tensor(result, self._remote_device, self._remote_dtype)

    async def _async_remote_tensor_call(
        self, path: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        """Async version of ``_remote_tensor_call``."""
        assert self._remote_client is not None
        assert self._remote_device is not None
        data = serialize_tensor(tensor)
        logger.debug(
            f"Sending {len(data) / 1024 / 1024:.2f} MB to {self.endpoint}{path}"
        )
        result = await self._remote_client.async_post_bytes(path, data)
        return deserialize_tensor(result, self._remote_device, self._remote_dtype)

    def _remote_batch_call(
        self,
        path: str,
        batch: dict[str, Any],
        fields: set[str] | None = None,
    ) -> torch.Tensor:
        """Serialize batch dict (optionally filtered by ``fields``), POST to server,
        deserialize result tensor."""
        assert self._remote_client is not None
        assert self._remote_device is not None
        if fields is not None:
            batch = {k: v for k, v in batch.items() if k in fields}
        data = serialize_batch(batch)
        logger.debug(
            f"Sending {len(data) / 1024 / 1024:.2f} MB batch to {self.endpoint}{path}"
        )
        result = self._remote_client.post_bytes(path, data)
        return deserialize_tensor(result, self._remote_device, self._remote_dtype)

    async def _async_remote_batch_call(
        self,
        path: str,
        batch: dict[str, Any],
        fields: set[str] | None = None,
    ) -> torch.Tensor:
        """Async version of ``_remote_batch_call``."""
        assert self._remote_client is not None
        assert self._remote_device is not None
        if fields is not None:
            batch = {k: v for k, v in batch.items() if k in fields}
        data = serialize_batch(batch)
        logger.debug(
            f"Sending {len(data) / 1024 / 1024:.2f} MB batch to {self.endpoint}{path}"
        )
        result = await self._remote_client.async_post_bytes(path, data)
        return deserialize_tensor(result, self._remote_device, self._remote_dtype)
