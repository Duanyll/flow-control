import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import fully_shard


class PretrainedState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, [], options=StateDictOptions(strict=False)
        )
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            [],
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=StateDictOptions(strict=False),
        )


class LoraTrainingState(Stateful):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, [self.optimizer], options=StateDictOptions(strict=False)
        )
        # Filter out model state dict to only include LoRA parameters
        lora_model_state_dict = {
            k: v for k, v in model_state_dict.items() if "lora_" in k
        }
        return {"model": lora_model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            [self.optimizer],
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=StateDictOptions(strict=False),
        )


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


def create_seed_checkpoint(seed_path):
    model = ToyModel()
    model.requires_grad_(False)
    state = PretrainedState(model)
    dcp.save(state.state_dict(), checkpoint_id=seed_path)
    print(f"Created seed checkpoint at {seed_path}")


def setup_process(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train_with_lora(rank, world_size, seed_path, load_lora_from, save_lora_to):
    setup_process(rank, world_size)
    print(f"[Rank {rank}] initialized process group.")

    with torch.device("meta"):
        model = ToyModel()
        model.requires_grad_(False)
        install_lora_layers(model, r=4)
    fully_shard(model)
    model.to_empty(device="cuda")
    print(f"[Rank {rank}] model with LoRA layers installed and sharded.")

    pretrained_state = PretrainedState(model)
    dcp.load(
        pretrained_state.state_dict(),
        checkpoint_id=seed_path,
        planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
    )
    print(f"[Rank {rank}] loaded pretrained weights from {seed_path}.")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    if load_lora_from:
        lora_state = LoraTrainingState(model, optimizer)
        dcp.load(
            lora_state.state_dict(),
            checkpoint_id=load_lora_from,
            planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
        )
        print(f"[Rank {rank}] resumed LoRA training state from {load_lora_from}.")

    optimizer.zero_grad()
    inputs = torch.randn(32, 128).to(rank)
    outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()
    optimizer.step()
    print(f"[Rank {rank}] completed one training step.")

    if save_lora_to:
        lora_state = LoraTrainingState(model, optimizer)
        dcp.save(lora_state.state_dict(), checkpoint_id=save_lora_to)
        print(f"[Rank {rank}] saved LoRA training state to {save_lora_to}.")

    dist.destroy_process_group()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test_stage",
        choices=["create_seed", "train_lora", "resume_lora", "export_lora"],
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if args.test_stage == "create_seed":
        seed_path = "checkpoint_seed"
        create_seed_checkpoint(seed_path)
    elif args.test_stage == "train_lora":
        seed_path = "checkpoint_seed"
        save_lora_to = "checkpoint_lora"
        mp.spawn(
            train_with_lora,
            args=(world_size, seed_path, None, save_lora_to),
            nprocs=world_size,
            join=True,
        )
    elif args.test_stage == "resume_lora":
        seed_path = "checkpoint_seed"
        load_lora_from = "checkpoint_lora"
        save_lora_to = "checkpoint_lora_resumed"
        mp.spawn(
            train_with_lora,
            args=(world_size, seed_path, load_lora_from, save_lora_to),
            nprocs=world_size,
            join=True,
        )
    elif args.test_stage == "export_lora":
        dcp_to_torch_save("checkpoint_seed", "pretrained_model.pt")
        dcp_to_torch_save("checkpoint_lora_resumed", "lora_model.pt")


if __name__ == "__main__":
    main()
