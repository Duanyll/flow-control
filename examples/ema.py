import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

from flow_control.utils.ema import apply_ema_maybe, make_ema_optimizer


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.layers(x)


class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=4):
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x))


def install_lora_layers(model, r=4):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lora_module = LoRALinear(module, r=r)
            parent_module = model
            name_parts = name.split(".")
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name_parts[-1], lora_module)


def run_local_ema_test():
    model = ToyModel()
    model.requires_grad_(False)
    install_lora_layers(model, r=4)
    optimizer_cls = make_ema_optimizer(torch.optim.AdamW)
    optimizer = optimizer_cls(
        model.parameters(),
        lr=1e-3,
        ema_decay=0.999,
    )

    # Now the optimizer tracks EMA of LoRA parameters within its state
    # No need to separately manage EMA state
    for _ in range(10):
        inputs = torch.randn(16, 128)
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # For evaluation, apply EMA weights
    with apply_ema_maybe(optimizer):
        inputs = torch.randn(16, 128)
        outputs = model(inputs)


def setup_process(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def run_fsdp_ema_test(rank, world_size):
    setup_process(rank, world_size)
    model = ToyModel().cuda(rank)
    model.requires_grad_(False)
    install_lora_layers(model, r=4)
    model: nn.Module = fully_shard(model)  # type: ignore

    optimizer_cls = make_ema_optimizer(torch.optim.AdamW)
    optimizer = optimizer_cls(
        model.parameters(),
        lr=1e-3,
        ema_decay=0.999,
    )

    for _ in range(10):
        inputs = torch.randn(16, 128).cuda(rank)
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with apply_ema_maybe(optimizer):
        inputs = torch.randn(16, 128).cuda(rank)
        outputs = model(inputs)

    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["local", "fsdp"])
    args = parser.parse_args()
    if args.mode == "local":
        run_local_ema_test()
    elif args.mode == "fsdp":
        world_size = torch.cuda.device_count()
        mp.spawn(run_fsdp_ema_test, args=(world_size,), nprocs=world_size)
