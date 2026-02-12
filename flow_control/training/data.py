import math

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import Dataset, Sampler

from flow_control.datasets import MapDataset
from flow_control.datasets.bucket_directory import BucketDataset
from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


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


def _repeat_as_padding(
    indices: list[tuple[int, bool]], count: int
) -> list[tuple[int, bool]]:
    """Repeat indices as padding (marked True) until we have `count` items."""
    padding: list[tuple[int, bool]] = []
    while len(padding) < count:
        source = indices[: min(count - len(padding), len(indices))]
        padding.extend((x[0], True) for x in source)
    return padding


class DistributedBucketSampler(Sampler, Stateful):
    def __init__(
        self,
        dataset: PaddingAwareDatasetWrapper,
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
            self.lengths = dataset.bucket_lengths
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
