import math

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import Dataset, Sampler

from .logging import get_logger

logger = get_logger(__name__)


def _patch_dataset_getitem(dataset: Dataset):
    """
    Dynamically change the dataset's class to handle (index, is_padding) tuples.
    """
    # 1. 防止重复 Patch
    if getattr(dataset, "_is_patched_for_padding", False):
        return

    # 2. 获取原始类
    OriginalClass = dataset.__class__
    original_class_name = OriginalClass.__name__
    if original_class_name == "PatchedDataset":
        # 已经被 Patch 过了
        return

    # 3. 定义一个新的子类
    class PatchedDataset(OriginalClass):
        def __getitem__(self, item):
            # 默认情况：假设是正常调用
            real_index = item
            is_padding = False

            # 检查是否是我们 Sampler 传来的特殊 Tuple
            if isinstance(item, tuple):
                real_index, is_padding = item

            # 4. 调用父类（原始 Dataset）的逻辑获取数据
            # 使用 super() 确保原本的文件读取/处理逻辑正常执行
            data = super().__getitem__(real_index)

            # 5. 如果是 Padding 数据（重复数据）
            if is_padding:
                data = data.copy()
                data["__key__"] = "__padding__"
                # 还可以加个 flag 方便后面过滤
                data["_is_padding_sample"] = True

            return data

    # 6. "偷天换日"：修改实例的类
    dataset.__class__ = PatchedDataset
    # 标记已处理
    dataset._is_patched_for_padding = True  # type: ignore[attr-defined]

    logger.info(f"Patched dataset {original_class_name} to handle padding indices.")


class DistributedBucketSampler(Sampler, Stateful):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        grad_acc_steps=1,
    ):
        super().__init__(None)
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        if hasattr(dataset, "lengths"):
            self.lengths = dataset.lengths  # type: ignore[attr-defined]
        else:
            self.lengths = [len(dataset)]  # type: ignore
        _patch_dataset_getitem(dataset)
        logger.info(
            f"Initialized DistributedBucketSampler with {len(self.lengths)} buckets and {sum(self.lengths)} samples."
        )

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.grad_acc_steps = grad_acc_steps  # 梯度累积步数

        self.epoch = 0
        self.counter = 0

        self.num_samples_per_rank = (
            math.ceil(
                sum(math.ceil(i / self.num_replicas) for i in self.lengths)
                / self.grad_acc_steps
            )
            * self.grad_acc_steps
        )

    def __iter__(self):
        # 保持随机种子逻辑不变
        g_cpu = torch.Generator(device="cpu")
        g_cpu.manual_seed(self.seed + self.epoch)

        blocks = []  # 存放的是 [(idx, is_padding), (idx, is_padding), ...] 的列表

        # -----------------------------------------------------------
        # 1. 桶内补齐 (Padding for World Size)
        # -----------------------------------------------------------
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

            # 计算需要补齐的数量
            num_padding = total_size - length

            # 补齐逻辑：从头复制，但标记为 True (重复数据/Padding)
            if num_padding > 0:
                padding_indices = []
                while len(padding_indices) < num_padding:
                    source = current_indices[
                        : min(num_padding - len(padding_indices), len(current_indices))
                    ]
                    padding_indices.extend([(x[0], True) for x in source])

                current_indices.extend(padding_indices)

            cumulative_size += length

            # 分割成 Block，每个 Block 大小为 num_replicas
            # current_indices 结构: [(idx, bool), (idx, bool)...]
            for i in range(0, len(current_indices), self.num_replicas):
                blocks.append(current_indices[i : i + self.num_replicas])

        # -----------------------------------------------------------
        # 2. 梯度累积补齐 (Padding for Grad Accumulation)
        # -----------------------------------------------------------
        total_blocks = (
            math.ceil(len(blocks) / self.grad_acc_steps) * self.grad_acc_steps
        )

        if self.shuffle:
            shuffled_idx = torch.randperm(len(blocks), generator=g_cpu).tolist()
            blocks = [blocks[i] for i in shuffled_idx]

        while len(blocks) < total_blocks:
            # 取出需要复制的 block
            needed = total_blocks - len(blocks)
            source_blocks = blocks[: min(needed, len(blocks))]

            padding_blocks = []
            for blk in source_blocks:
                # blk 结构: [(idx, bool), (idx, bool)...]
                # 创建新副本，全部设为 False
                padding_blocks.append([(x[0], True) for x in blk])

            blocks.extend(padding_blocks)

        # -----------------------------------------------------------
        # 3. 分发给当前进程
        # -----------------------------------------------------------
        # 每个 block 长度为 num_replicas，取出属于当前 rank 的那个元组
        subsample = [b[self.rank] for b in blocks]

        if self.counter > 0:
            subsample = subsample[self.counter :]

        for item in subsample:
            # item 是 (index, is_padding)
            yield item
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
