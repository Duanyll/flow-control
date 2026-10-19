"""Default batched inference: one velocity per logical sample, collective on every rank."""

import math
from abc import abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar, cast

import torch
import torch.distributed as dist
from pydantic import BaseModel, PositiveInt, PrivateAttr

from flow_control.utils.model_cache import cache_fields, clear_cache
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_detach,
    deep_move_to_device,
)


class MicrobatchMixin(BaseModel):
    """Chunks logical samples into dense or sequential forwards and keeps the ranks' forward counts equal."""

    micro_batch_size: PositiveInt = 1
    """Maximum logical samples per chunk. A throughput/memory knob: chunking is
    mathematically equivalent, but dense forwards are not bitwise reproducible
    on GPU."""
    use_cache: bool = True
    """Reuse adapter intermediates within a no-grad Executor run."""
    shared_cache_fields: ClassVar[tuple[str, ...]] = ()
    """Layout-only caches shared by compatible dense samples, independent of B.
    Sample-dependent caches require adapter-specific collation and splitting."""
    supports_dense_batching: ClassVar[bool] = False
    """Whether equal-shaped logical samples may use the default dense collator."""
    dense_batch_fields: ClassVar[tuple[str, ...]] = ()
    """Adapter inputs used by the default collator; unrelated metadata is ignored."""

    _dummy_sample: tuple[Any, torch.Tensor] | None = PrivateAttr(default=None)
    """Last real prepared sample, forwarded in place of samples another rank
    has and this rank lacks. Pins that sample's tensors on the device until
    the next call; only set in multi-rank runs, where a dummy can be needed."""

    # ------------------------- what the adapter provides ------------------------ #

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def active_variant(self) -> str: ...

    @abstractmethod
    def fp32_copy_active(self) -> bool: ...

    @abstractmethod
    def _predict_velocity(self, batch: Any, timestep: torch.Tensor) -> torch.Tensor:
        """One forward of one or more collated logical samples."""

    def _prepare_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        return timestep.to(device=self.device, dtype=self.dtype)

    # --------------------------------- entry ----------------------------------- #

    def predict_velocity_batched(
        self,
        batches: list[Any],
        timesteps: list[torch.Tensor],
        *,
        dummy_outputs: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Predict one velocity per logical sample; a collective on every rank.

        Logical samples keep their singleton leading batch dimension. Every
        rank runs the same chunk count and, within a chunk, the same forward
        count (see ``_forward_chunk``), padding with dummy forwards. An empty
        list is therefore a legal call returning ``[]``. Under autograd, dummy
        outputs are folded into the first real output with zero weight so FSDP
        backward stays aligned across ranks. A caller grouping multiple variants
        can supply ``dummy_outputs`` and fold them into its combined result;
        this also supports variants with no local real output.
        """
        if len(batches) != len(timesteps):
            raise ValueError(
                "batches and timesteps must have equal lengths, got "
                f"{len(batches)} and {len(timesteps)}."
            )
        for index, batch in enumerate(batches):
            if batch["noisy_latents"].shape[0] != 1:
                raise ValueError(
                    "Each logical sample must have a singleton leading batch "
                    f"dimension; sample {index} has shape "
                    f"{tuple(batch['noisy_latents'].shape)}."
                )

        with self._prepared_batches(batches) as prepared_batches:
            prepared_timesteps = [
                self._prepare_timestep(timestep) for timestep in timesteps
            ]
            if prepared_batches and dist.is_initialized() and dist.get_world_size() > 1:
                self._dummy_sample = (
                    deep_detach(
                        {
                            key: value
                            for key, value in prepared_batches[0].items()
                            if not key.startswith("_")
                        }
                    ),
                    prepared_timesteps[0].detach(),
                )
            size = self.micro_batch_size
            chunks, needs_dummy, donor = self._sync_max(
                [
                    math.ceil(len(prepared_batches) / size),
                    int(self._dummy_sample is None),
                    dist.get_rank() + 1
                    if dist.is_initialized() and self._dummy_sample is not None
                    else 0,
                ]
            )
            if chunks and needs_dummy and donor:
                self._share_dummy(donor - 1)
            velocities: list[torch.Tensor] = []
            dummies: list[torch.Tensor] = []
            autocast = torch.autocast(
                self.device.type, self.dtype, enabled=self.fp32_copy_active()
            )
            # Run the longest rank's chunk count; slicing past our own end is empty.
            with autocast:
                for start in range(0, chunks * size, size):
                    chunk_velocities, chunk_dummies = self._forward_chunk(
                        prepared_batches[start : start + size],
                        prepared_timesteps[start : start + size],
                    )
                    velocities.extend(chunk_velocities)
                    dummies.extend(chunk_dummies)
            dummies = [output for output in dummies if output.requires_grad]
            if dummy_outputs is not None:
                dummy_outputs.extend(dummies)
            elif dummies:
                if not velocities:
                    raise RuntimeError(
                        f"Rank {dist.get_rank()} ran dummy forwards under autograd "
                        "with no real sample to carry their zero-weight graph "
                        "dependency; a training call must give every rank at least "
                        "one item."
                    )
                velocities[0] = velocities[0] + sum(
                    output.sum() * 0 for output in dummies
                )
            return velocities

    # ------------------------------ preparation -------------------------------- #

    @contextmanager
    def _prepared_batches(self, batches: list[Any]) -> Iterator[list[Any]]:
        """Model inputs on the device; no-grad caches persist in `row["_cache"][variant]`."""
        reuse_cache = self.use_cache and not torch.is_grad_enabled()
        variant = self.active_variant
        prepared_batches: list[Any] = []
        for batch in batches:
            data = {
                key: value for key, value in batch.items() if not key.startswith("_")
            }
            prepared = deep_move_to_device(
                deep_cast_float_dtype(data, self.dtype), self.device
            )
            if reuse_cache:
                caches = cast(dict[str, Any], batch).setdefault("_cache", {})
                prepared.update(caches.get(variant, {}))
            prepared_batches.append(prepared)
        yield prepared_batches
        if reuse_cache:
            for batch, prepared in zip(batches, prepared_batches, strict=True):
                caches = cast(dict[str, Any], batch)["_cache"]
                caches.setdefault(variant, {}).update(cache_fields(prepared))

    # -------------------------------- chunks ----------------------------------- #

    def _forward_chunk(
        self,
        batches: list[Any],
        timesteps: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run one chunk as ``(velocities, dummy_outputs)``.

        A chunk is one dense forward when every rank holding data can collate
        it, else as many sequential forwards as the longest rank's chunk. This
        rank pads its shortfall (an empty chunk, or fewer samples than the
        longest rank) with forwards on the dummy sample.
        """
        collated = (
            self._collate_velocity_inputs(batches, timesteps) if batches else None
        )
        needs_fallback = bool(batches) and collated is None
        sequential, longest = self._sync_max([int(needs_fallback), len(batches)])
        if not sequential:
            if collated is None:  # Empty chunk: pad the peers' single dense forward.
                return [], [self._forward_dummy()]
            try:
                velocity = self._checked_velocity(
                    self._predict_velocity(*collated), len(batches)
                )
                shared = {
                    key: collated[0][key]
                    for key in self.shared_cache_fields
                    if key in collated[0]
                }
                for batch in batches:
                    cast(dict[str, Any], batch).update(shared)
                return list(velocity.split(1, dim=0)), []
            finally:
                clear_cache(collated[0])
        velocities = [
            self._forward_one(batch, timestep)
            for batch, timestep in zip(batches, timesteps, strict=True)
        ]
        dummies = [self._forward_dummy() for _ in range(longest - len(batches))]
        return velocities, dummies

    def _forward_one(self, batch: Any, timestep: torch.Tensor) -> torch.Tensor:
        return self._checked_velocity(self._predict_velocity(batch, timestep), 1)

    def _forward_dummy(self) -> torch.Tensor:
        assert self._dummy_sample is not None  # Seeded collectively before chunking.
        try:
            return self._forward_one(*self._dummy_sample)
        finally:
            clear_cache(self._dummy_sample[0])

    def _share_dummy(self, source: int) -> None:
        """Seed empty ranks once, transferring detached CPU data rather than GPU objects."""
        payload: list[Any] = [
            deep_move_to_device(self._dummy_sample, torch.device("cpu"))
            if dist.get_rank() == source
            else None
        ]
        dist.broadcast_object_list(payload, src=source, device=self.device)
        if self._dummy_sample is None:
            self._dummy_sample = deep_move_to_device(payload[0], self.device)

    def _sync_max(self, values: list[int]) -> list[int]:
        """MAX-reduce small integers across ranks."""
        if not dist.is_initialized():
            return values
        status = torch.tensor(values, device=self.device, dtype=torch.int64)
        dist.all_reduce(status, op=dist.ReduceOp.MAX)
        return status.tolist()

    def _checked_velocity(self, velocity: torch.Tensor, expected: int) -> torch.Tensor:
        if velocity.ndim == 0 or velocity.shape[0] != expected:
            raise ValueError(
                f"{type(self).__name__}._predict_velocity returned shape "
                f"{tuple(velocity.shape)}, expected leading dimension {expected}."
            )
        return velocity.float()

    # ------------------------------ dense collation ----------------------------- #

    def _collate_velocity_inputs(
        self,
        batches: list[Any],
        timesteps: list[torch.Tensor],
    ) -> tuple[Any, torch.Tensor] | None:
        """Densely collate fixed-shape batches, or return ``None`` to fall back."""
        if not self.supports_dense_batching:
            return None

        collated: dict[str, object] = {}
        for key in self.dense_batch_fields:
            present = [key in batch for batch in batches]
            if not any(present):
                continue
            if not all(present):
                return None
            values = [batch[key] for batch in batches]
            success, value = self._collate_velocity_values(values)
            if not success:
                return None
            collated[key] = value

        for key in self.shared_cache_fields:
            for batch in batches:
                if key in batch:
                    collated[key] = batch[key]
                    break
        return collated, torch.cat(timesteps, dim=0)

    def _collate_velocity_values(
        self,
        values: list[object],
    ) -> tuple[bool, object]:
        first = values[0]
        if isinstance(first, torch.Tensor):
            if not all(isinstance(value, torch.Tensor) for value in values):
                return False, first
            tensors = cast(list[torch.Tensor], values)
            if any(
                tensor.ndim == 0 or tensor.shape[0] != 1 or tensor.shape != first.shape
                for tensor in tensors
            ):
                return False, first
            return True, torch.cat(tensors, dim=0)

        if isinstance(first, list):
            if not all(
                isinstance(value, list) and len(value) == len(first) for value in values
            ):
                return False, first
            collated_items: list[object] = []
            lists = cast(list[list[object]], values)
            for index in range(len(first)):
                success, item = self._collate_velocity_values(
                    [value[index] for value in lists]
                )
                if not success:
                    return False, first
                collated_items.append(item)
            return True, collated_items

        if all(self._same_static_value(first, value) for value in values[1:]):
            return True, first
        return False, first

    @staticmethod
    def _same_static_value(left: object, right: object) -> bool:
        if type(left) is not type(right):
            return False
        try:
            equal = left == right
        except (RuntimeError, TypeError, ValueError):
            return False
        return bool(equal) if isinstance(equal, bool) else False
