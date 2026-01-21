import os
from typing import Any

import torch
from pydantic import model_validator
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.datasets import (
    DatasetConfig,
    DatasinkConfig,
    collate_fn,
    parse_dataset,
    parse_datasink,
)
from flow_control.processors import Processor
from flow_control.samplers import Sampler
from flow_control.samplers.simple_euler import SimpleEulerSampler
from flow_control.utils.common import (
    deep_cast_float_dtype,
    deep_move_to_device,
    tensor_to_pil,
)
from flow_control.utils.data import DistributedBucketSampler
from flow_control.utils.logging import console, get_logger

from .hsdp_engine import HsdpEngine, HsdpEngineConfig

logger = get_logger(__name__)


class HsdpInferenceConfig(HsdpEngineConfig):
    model: ModelAdapter
    sampler: Sampler = SimpleEulerSampler()
    processor: Processor
    dataset: DatasetConfig
    datasink: DatasinkConfig | None = None

    seed_checkpoint_dir: str | None = None
    checkpoint_dir: str | None = None
    save_preview_dir: str | None = None

    @model_validator(mode="after")
    def check_save_preview_dir(self):
        if self.datasink is None and self.save_preview_dir is None:
            raise ValueError("Either datasink or save_preview_dir must be specified.")
        return self


class HsdpInference(HsdpEngine):
    conf: HsdpInferenceConfig

    @property
    def model(self):
        return self.conf.model

    @property
    def transformer(self):
        return self.conf.model.transformer

    @property
    def sampler(self):
        return self.conf.sampler

    @property
    def processor(self):
        return self.conf.processor

    def __init__(self, **kwargs):
        self.conf = HsdpInferenceConfig(**kwargs)  # type: ignore
        super().__init__(**kwargs)

    dataloader: StatefulDataLoader

    def make_dataloader(self):
        dataset: Any = parse_dataset(self.conf.dataset)
        sampler = DistributedBucketSampler(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            seed=self.conf.seed,
            grad_acc_steps=1,
        )
        self.dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=collate_fn,
        )

    def state_dict(self):
        transformer_state_dict, optimizer_state_dict = get_state_dict(
            self.transformer,
            optimizers=[],
            options=StateDictOptions(strict=False),
        )
        return {
            "transformer": transformer_state_dict,
            "optimizer": optimizer_state_dict,
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.transformer,
            [],
            model_state_dict=state_dict["transformer"],
            optim_state_dict=state_dict["optimizer"],
            options=StateDictOptions(strict=False),
        )

    def make_progress_bar(self):
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn(" Batch: {task.completed}/{task.total}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        task = progress.add_task("Inference", total=len(self.dataloader))
        return progress, task

    @torch.no_grad()
    def run(self):
        self.init_device_mesh()
        self.set_seed()
        self.load_transformer_from_seed(self.model, self.conf.seed_checkpoint_dir)
        self.processor.load_models("decode", device=self.device)
        self.make_dataloader()

        if self.conf.checkpoint_dir is not None:
            self.load_dcp_checkpoint(self.conf.checkpoint_dir)

        if self.conf.datasink is not None:
            datasink = parse_datasink(self.conf.datasink)
        else:
            datasink = None

        if self.conf.save_preview_dir is not None:
            os.makedirs(self.conf.save_preview_dir, exist_ok=True)

        self.transformer.eval()
        console.rule("[bold green]Starting Inference[/bold green]")
        progress, task = self.make_progress_bar()
        progress.start()
        for batch in self.dataloader:
            batch = deep_cast_float_dtype(batch, self.model.dtype)
            batch = deep_move_to_device(batch, self.device)
            generator = torch.Generator(device=self.device).manual_seed(self.conf.seed)
            self.processor.initialize_latents(
                batch,
                generator=generator,
                device=self.device,
                dtype=self.model.dtype,
            )
            clean_latents = self.sampler.sample(self.model, batch)
            image = tensor_to_pil(self.processor.decode_output(clean_latents, batch))
            key = batch.get("__key__", None)
            if key == "__padding__":
                continue
            if datasink is not None:
                datasink.write(batch)
            if self.conf.save_preview_dir is not None:
                image.save(os.path.join(self.conf.save_preview_dir, f"{key}.png"))
            progress.advance(task)
        progress.stop()
        console.rule("[bold green]Inference Completed[/bold green]")

        self.cleanup()
