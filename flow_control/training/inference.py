import os
from typing import Any

import torch
from pydantic import ConfigDict, model_validator
from rich.progress import Progress
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
    parse_dataset,
    parse_datasink,
)
from flow_control.processors import Processor
from flow_control.samplers import Sampler
from flow_control.utils.logging import console, dump_if_failed, get_logger
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_move_to_device,
    tensor_to_pil,
)

from .data import DistributedBucketSampler, PaddingAwareDatasetWrapper, collate_fn
from .mixins import DcpMixin, HsdpMixin, LoggingMixin, distributed_main

logger = get_logger(__name__)


class Inference(HsdpMixin, DcpMixin):
    model_config = ConfigDict(extra="forbid")

    model: ModelAdapter
    sampler: Sampler
    processor: Processor
    dataset: DatasetConfig
    datasink: DatasinkConfig | None = None

    seed_checkpoint_dir: str | None = None
    checkpoint_dir: str | None = None
    save_preview_dir: str | None = None
    save_intermediate: bool = False

    @model_validator(mode="after")
    def check_save_preview_dir(self):
        if self.datasink is None and self.save_preview_dir is None:
            raise ValueError("Either datasink or save_preview_dir must be specified.")
        return self

    # ------------------------------- Lazy state --------------------------------- #
    _dataloader: StatefulDataLoader | None = None

    @property
    def transformer(self):
        return self.model.transformer

    @property
    def dataloader(self) -> StatefulDataLoader:
        if self._dataloader is None:
            raise RuntimeError("Dataloader not created yet.")
        return self._dataloader

    def make_dataloader(self):
        dataset = PaddingAwareDatasetWrapper(parse_dataset(self.dataset))
        sampler = DistributedBucketSampler(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            seed=self.seed,
            grad_acc_steps=1,
        )
        self._dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=collate_fn,
        )

    # ------------------------------- Checkpointing ------------------------------ #

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

    def load_state_dict(self, state_dict: dict[str, Any]):
        set_state_dict(
            self.transformer,
            [],
            model_state_dict=state_dict["transformer"],
            optim_state_dict=state_dict["optimizer"],
            options=StateDictOptions(strict=False),
        )

    # ------------------------------- Main loop ---------------------------------- #

    @torch.no_grad()
    @distributed_main
    def run(self):
        self.set_seed()
        self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)
        self.processor.load_models("decode", device=self.device)
        self.make_dataloader()

        if self.checkpoint_dir is not None:
            self.load_dcp_checkpoint(self.checkpoint_dir)

        datasink = parse_datasink(self.datasink) if self.datasink is not None else None

        if self.save_preview_dir is not None:
            os.makedirs(self.save_preview_dir, exist_ok=True)
            logger.info(f"Saving preview images to {self.save_preview_dir}")

        self.transformer.eval()
        console.rule("[bold green]Starting Inference[/bold green]")

        progress = Progress(
            *LoggingMixin.get_progress_columns(),
            console=console,
        )
        task = progress.add_task("Inference", total=len(self.dataloader))

        with progress:
            for batch in self.dataloader:
                with dump_if_failed(logger, batch):
                    batch = deep_cast_float_dtype(batch, self.model.dtype)
                    batch = deep_move_to_device(batch, self.device)
                    negative_batch: Any = (
                        self.processor.get_negative_batch(batch)
                        if self.sampler.cfg_scale > 1.0
                        else None
                    )
                    generator = torch.Generator(device=self.device).manual_seed(
                        self.seed
                    )
                    self.processor.initialize_latents(
                        batch,
                        generator=generator,
                        device=self.device,
                        dtype=self.model.dtype,
                    )
                    sample_output = self.sampler.sample(
                        self.model, batch, negative_batch=negative_batch
                    )
                    result = self.processor.decode_output(
                        sample_output.final_latents,
                        batch,
                    )
                    result = deep_move_to_device(result, torch.device("cpu"))
                    image = tensor_to_pil(result["clean_image"])
                    key = batch.get("__key__", None)
                    if key == "__padding__":
                        continue
                    if datasink is not None:
                        if self.save_intermediate:
                            batch.update(result)
                            result = batch
                        datasink.write(result)
                    if self.save_preview_dir is not None:
                        image.save(os.path.join(self.save_preview_dir, f"{key}.png"))
                progress.advance(task)

        console.rule("[bold green]Inference Completed[/bold green]")
