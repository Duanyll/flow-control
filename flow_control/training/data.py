import math
import random
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import Dataset
from torch.utils.data import Sampler as TorchSampler

from flow_control.datasets import MapDataset
from flow_control.datasets.bucket_directory import BucketDataset
from flow_control.utils.logging import get_logger
from flow_control.utils.resize import (
    resize_short_side_and_random_crop,
    resize_to_multiple_of,
)
from flow_control.utils.tensor import ensure_alpha_channel

logger = get_logger(__name__)


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducible randomness across workers."""
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info is not None
    seed = worker_info.seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class SizedMapDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, item: Any, /) -> Any: ...


class PaddingAwareDatasetWrapper(Dataset):
    def __init__(self, dataset: MapDataset):
        self.dataset = dataset
        if isinstance(dataset, BucketDataset):
            self.bucket_lengths = dataset.bucket_lengths

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        real_index = item
        is_padding = False
        if isinstance(item, tuple):
            real_index, is_padding = item

        data = self.dataset[real_index]
        if is_padding and isinstance(data, dict):
            data = data.copy()
            data["__key__"] = "__padding__"
            data["_is_padding_sample"] = True
        return data


def _maybe_blend_rgba_to_bg(target: torch.Tensor, blend_prob: float) -> torch.Tensor:
    if blend_prob <= 0 or torch.rand(1).item() >= blend_prob:
        return target

    rgb = target[:, :3]
    alpha = target[:, 3:4]
    bg = torch.randint(0, 3, (1, 3, 1, 1), device=target.device).to(target.dtype) / 2.0
    blended_rgb = rgb * alpha + bg * (1 - alpha)
    return torch.cat([blended_rgb, torch.ones_like(alpha)], dim=1)


def prepare_vae_target_image(
    clean_image: torch.Tensor,
    *,
    key: str = "unknown",
    random_crop_size: int | None,
    resize_multiple: int,
    resize_pixels: int,
    blend_prob: float,
) -> torch.Tensor:
    target = clean_image.float()
    if target.shape[1] not in {3, 4}:
        raise ValueError(
            f"Expected clean_image to have 3 or 4 channels, got shape {tuple(target.shape)} "
            f"for sample {key!r}."
        )

    target = ensure_alpha_channel(target)

    if random_crop_size is not None:
        target = resize_short_side_and_random_crop(
            target,
            crop_size=random_crop_size,
            multiple=resize_multiple,
        )
    else:
        target = resize_to_multiple_of(target, resize_multiple, pixels=resize_pixels)

    target = _maybe_blend_rgba_to_bg(target, blend_prob)
    return target * 2 - 1  # [0, 1] -> [-1, 1]


class VaeTargetDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset: SizedMapDataset,
        *,
        random_crop_size: int | None,
        resize_multiple: int,
        resize_pixels: int,
        blend_prob: float,
    ):
        self.dataset = dataset
        self.random_crop_size = random_crop_size
        self.resize_multiple = resize_multiple
        self.resize_pixels = resize_pixels
        self.blend_prob = blend_prob
        if hasattr(dataset, "bucket_lengths"):
            self.bucket_lengths = cast(Any, dataset).bucket_lengths

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        if not isinstance(data, dict):
            return data

        target = prepare_vae_target_image(
            data["clean_image"],
            key=data.get("__key__", "unknown"),
            random_crop_size=self.random_crop_size,
            resize_multiple=self.resize_multiple,
            resize_pixels=self.resize_pixels,
            blend_prob=self.blend_prob,
        )
        data = data.copy()
        data["target_image"] = target
        return data


def _repeat_as_padding(
    indices: list[tuple[int, bool]], count: int
) -> list[tuple[int, bool]]:
    """Repeat indices as padding (marked True) until we have `count` items."""
    padding: list[tuple[int, bool]] = []
    while len(padding) < count:
        source = indices[: min(count - len(padding), len(indices))]
        padding.extend((x[0], True) for x in source)
    return padding


class DistributedBucketSampler(TorchSampler, Stateful):
    def __init__(
        self,
        dataset: SizedMapDataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        grad_acc_steps=1,
    ):
        super().__init__(None)  # type: ignore[arg-type]
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.mark_padding = isinstance(dataset, PaddingAwareDatasetWrapper)
        if not self.mark_padding:
            logger.warning(
                f"DistributedBucketSampler is used with a {dataset.__class__.__name__} that does not mark padding samples. "
                "To distinguish padding samples from real samples, please wrap your dataset with PaddingAwareDatasetWrapper."
            )

        if hasattr(dataset, "bucket_lengths"):
            self.lengths = cast(Any, dataset).bucket_lengths
        else:
            self.lengths = [len(dataset)]

        logger.info(
            f"Initialized DistributedBucketSampler with {len(self.lengths)} buckets and {sum(self.lengths)} samples."
        )

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.grad_acc_steps = grad_acc_steps

        self.epoch = 0
        self.counter = 0

        self.num_samples_per_rank = (
            math.ceil(
                sum(math.ceil(i / self.num_replicas) for i in self.lengths)
                / self.grad_acc_steps
            )
            * self.grad_acc_steps
        )

    def _build_bucket_blocks(
        self, g_cpu: torch.Generator
    ) -> list[list[tuple[int, bool]]]:
        """Build blocks by padding each bucket to be divisible by num_replicas."""
        blocks: list[list[tuple[int, bool]]] = []
        cumulative_size = 0
        for length in self.lengths:
            total_size = math.ceil(length / self.num_replicas) * self.num_replicas

            # 生成原始索引，并标记为 False (真实数据)
            if self.shuffle:
                raw_indices = torch.randperm(length, generator=g_cpu).tolist()
            else:
                raw_indices = list(range(length))

            # 加上 offset
            current_indices = [(idx + cumulative_size, False) for idx in raw_indices]

            # 补齐逻辑：从头复制，但标记为 True (重复数据/Padding)
            num_padding = total_size - length
            if num_padding > 0:
                current_indices.extend(_repeat_as_padding(current_indices, num_padding))

            cumulative_size += length

            # 分割成 Block，每个 Block 大小为 num_replicas
            for i in range(0, len(current_indices), self.num_replicas):
                blocks.append(current_indices[i : i + self.num_replicas])
        return blocks

    def _pad_blocks_for_grad_acc(
        self,
        blocks: list[list[tuple[int, bool]]],
        g_cpu: torch.Generator,
    ) -> list[list[tuple[int, bool]]]:
        """Shuffle and pad blocks so total count is divisible by grad_acc_steps."""
        total_blocks = (
            math.ceil(len(blocks) / self.grad_acc_steps) * self.grad_acc_steps
        )

        if self.shuffle:
            shuffled_idx = torch.randperm(len(blocks), generator=g_cpu).tolist()
            blocks = [blocks[i] for i in shuffled_idx]

        while len(blocks) < total_blocks:
            needed = total_blocks - len(blocks)
            source_blocks = blocks[: min(needed, len(blocks))]
            blocks.extend([(x[0], True) for x in blk] for blk in source_blocks)

        return blocks

    def __iter__(self):
        g_cpu = torch.Generator(device="cpu")
        g_cpu.manual_seed(self.seed + self.epoch)

        blocks = self._build_bucket_blocks(g_cpu)
        blocks = self._pad_blocks_for_grad_acc(blocks, g_cpu)

        # 分发给当前进程
        subsample = [b[self.rank] for b in blocks]
        if self.counter > 0:
            subsample = subsample[self.counter :]

        for item in subsample:
            yield item if self.mark_padding else item[0]
            self.counter += 1

        self.counter = 0

    def __len__(self):
        # 返回当前还剩下多少数据没跑
        return self.num_samples_per_rank

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.counter = 0

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "counter": self.counter,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.counter = state_dict["counter"]


class DistributedKRepeatSampler(TorchSampler, Stateful):
    """GRPO sampler: select M unique prompts, repeat each K times, distribute across GPUs.

    The total number of rollouts per batch is ``M * K`` (global, GPU-invariant).
    These are evenly distributed across ``num_replicas`` GPUs, so each rank
    gets ``M * K / num_replicas`` samples per batch.

    Each epoch yields ``num_batches * per_rank`` indices for the current rank.

    Unlike :class:`DistributedBucketSampler`, this sampler:

    - Ensures the same prompt appears *K* times for advantage estimation.
    - Yields one index at a time (batch_size=1 convention).
    - Runs for *num_batches* iterations per epoch.
    """

    def __init__(
        self,
        dataset: PaddingAwareDatasetWrapper,
        num_batches_per_epoch: int,
        num_prompts_per_batch: int,
        num_rollouts_per_prompt: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        keep_prompt_local: bool = False,
    ):
        super().__init__(None)  # type: ignore[arg-type]
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_prompts_per_batch = num_prompts_per_batch
        self.num_rollouts_per_prompt = num_rollouts_per_prompt
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.keep_prompt_local = keep_prompt_local

        self.total_samples = self.num_prompts_per_batch * self.num_rollouts_per_prompt
        if self.total_samples % self.num_replicas != 0:
            raise ValueError(
                f"num_prompts_per_batch * num_rollouts_per_prompt ({self.total_samples}) must be divisible "
                f"by num_replicas ({num_replicas}). "
                f"Got num_prompts_per_batch={num_prompts_per_batch}, num_rollouts_per_prompt={num_rollouts_per_prompt}."
            )
        self.per_rank = self.total_samples // self.num_replicas

        if keep_prompt_local and self.num_prompts_per_batch % self.num_replicas != 0:
            raise ValueError(
                f"keep_prompt_local requires num_prompts_per_batch ({num_prompts_per_batch}) "
                f"to be divisible by num_replicas ({num_replicas})."
            )

        self.epoch = 0
        self.batch_counter = 0
        self.within_batch_counter = 0

    def __iter__(self):
        for batch_idx in range(self.batch_counter, self.num_batches_per_epoch):
            g = torch.Generator()
            g.manual_seed(
                self.seed + self.epoch * self.num_batches_per_epoch + batch_idx
            )

            # Select m unique prompts
            dataset_size = len(self.dataset)
            indices = torch.randperm(dataset_size, generator=g)[
                : self.num_prompts_per_batch
            ].tolist()

            if self.keep_prompt_local:
                # Assign prompts to ranks round-robin, yield contiguously
                # per prompt (K rollouts grouped together)
                prompts_per_rank = self.num_prompts_per_batch // self.num_replicas
                rank_prompts = indices[
                    self.rank * prompts_per_rank : (self.rank + 1) * prompts_per_rank
                ]
                rank_indices = [
                    idx
                    for idx in rank_prompts
                    for _ in range(self.num_rollouts_per_prompt)
                ]
            else:
                # Repeat each K times
                repeated = [
                    idx for idx in indices for _ in range(self.num_rollouts_per_prompt)
                ]

                # Shuffle
                perm = torch.randperm(len(repeated), generator=g).tolist()
                shuffled = [repeated[i] for i in perm]

                # Split to ranks
                rank_indices = shuffled[
                    self.rank * self.per_rank : (self.rank + 1) * self.per_rank
                ]

            # Yield one at a time, resuming from within_batch_counter
            start = self.within_batch_counter if batch_idx == self.batch_counter else 0
            for i in range(start, len(rank_indices)):
                yield rank_indices[i]
                self.within_batch_counter = i + 1

            self.within_batch_counter = 0
            self.batch_counter = batch_idx + 1

        # Reset for next epoch
        self.batch_counter = 0
        self.within_batch_counter = 0

    def __len__(self):
        return self.num_batches_per_epoch * self.per_rank

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.batch_counter = 0
        self.within_batch_counter = 0

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "batch_counter": self.batch_counter,
            "within_batch_counter": self.within_batch_counter,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.batch_counter = state_dict["batch_counter"]
        self.within_batch_counter = state_dict["within_batch_counter"]


# This library is designed to work with batch size 1 datasets.
# For larger batch sizes, use gradient accumulation.
# Dataset should return tensors with batch dimension 1.
def collate_fn(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError(
            "Batch size greater than 1 is not supported. Use gradient accumulation instead."
        )
    item = batch[0]
    return item
