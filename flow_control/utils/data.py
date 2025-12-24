import math

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import Sampler


class DistributedBucketSampler(Sampler, Stateful):
    def __init__(
        self,
        lengths,
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

        self.lengths = lengths  # 每个 bucket 的长度列表
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
        # 为了保证不同 epoch 的补齐和打乱逻辑不同，使用 epoch 相关的随机种子
        g_cpu = torch.Generator(device="cpu")
        g_cpu.manual_seed(self.seed + self.epoch)

        # 1. 为每个长度创建索引列表，并补齐到 num_replicas 的整数倍
        blocks = []
        cumulative_size = 0
        for length in self.lengths:
            # 计算需要补齐多少个样本
            total_size = math.ceil(length / self.num_replicas) * self.num_replicas
            # 创建索引列表
            if self.shuffle:
                indices = torch.randperm(length, generator=g_cpu)
            else:
                indices = torch.arange(length)
            # 补齐索引，取前 total_size - length 个样本，考虑 length < total_size - length 的情况
            while len(indices) < total_size:
                indices = torch.cat(
                    (indices, indices[: min(total_size - len(indices), len(indices))])
                )
            indices += cumulative_size
            cumulative_size += length
            # 2. 将索引列表分割成 num_replicas 个块
            indices = indices.view(-1, self.num_replicas).tolist()
            blocks.extend(indices)

        # 3. 打乱 Blocks 并补齐到梯度累积步数的整数倍
        total_blocks = (
            math.ceil(len(blocks) / self.grad_acc_steps) * self.grad_acc_steps
        )
        if self.shuffle:
            # 使用同步的随机顺序
            shuffled_idx = torch.randperm(len(blocks), generator=g_cpu).tolist()
            blocks = [blocks[i] for i in shuffled_idx]
        while len(blocks) < total_blocks:
            blocks.extend(blocks[: min(total_blocks - len(blocks), len(blocks))])

        # 4. 每个进程取属于自己的索引
        subsample = [b[self.rank] for b in blocks]

        # 5. 断点恢复逻辑
        if self.counter > 0:
            subsample = subsample[self.counter :]

        for index in subsample:
            yield index
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
