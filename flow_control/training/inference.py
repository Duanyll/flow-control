import os
from typing import Any

import torch
import torch.distributed as dist
from pydantic import ConfigDict, model_validator
from rich.progress import Progress
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torchdata.stateful_dataloader import StatefulDataLoader

from flow_control.adapters import ModelAdapter
from flow_control.datasets import DatasetConfig, DatasinkConfig, parse_datasink
from flow_control.processors import Processor
from flow_control.rewards import Reward, execute_reward
from flow_control.rewards.base import RewardResult
from flow_control.samplers import Sampler
from flow_control.samplers.sampler import derive_seed
from flow_control.utils.logging import console, dump_if_failed, get_logger
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_move_to_device,
    tensor_to_pil,
)

from .data import (
    DistributedBucketSampler,
    PaddingAwareDatasetWrapper,
    collate_fn,
    seed_worker,
)
from .mixins import DcpMixin, HsdpMixin, LoggingMixin, PreprocessMixin, distributed_main

logger = get_logger(__name__)


class Inference(PreprocessMixin, HsdpMixin, DcpMixin):
    model_config = ConfigDict(extra="forbid")

    model: ModelAdapter
    sampler: Sampler
    processor: Processor
    dataset: DatasetConfig
    datasink: DatasinkConfig | None = None
    reward: Reward | None = None

    seed_checkpoint_dir: str | None = None
    checkpoint_dir: str | None = None
    save_preview_dir: str | None = None
    save_extra: bool = False

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
        dataset = PaddingAwareDatasetWrapper(self.parse_inference_dataset(self.dataset))
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
            worker_init_fn=seed_worker,
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

    # ---------------------------------- Rewards ---------------------------------- #

    def _reward_fields(self, result: RewardResult) -> dict[str, Any]:
        aggregate = result.aggregate().detach().cpu()
        raw = result.raw.detach().cpu()
        normalized = result.normalized.detach().cpu()
        return {
            "reward": aggregate,
            "reward_raw": {
                label: raw[:, i].clone() for i, label in enumerate(result.labels)
            },
            "reward_normalized": {
                label: normalized[:, i].clone() for i, label in enumerate(result.labels)
            },
        }

    def _print_reward_result(self, key: str, result: RewardResult) -> None:
        aggregate = result.aggregate().detach().cpu()
        raw = result.raw.detach().cpu()
        parts = [
            f"{label}={raw[:, i].mean().item():.4f}"
            for i, label in enumerate(result.labels)
        ]
        console.print(
            f"[bold]{key}[/] reward={aggregate.mean().item():.4f} " + " ".join(parts)
        )

    def _print_reward_summary(self, results: list[RewardResult]) -> None:
        world_size: int = getattr(self, "world_size", 1)
        is_main: bool = getattr(self, "is_main_process", True)

        all_results = results
        if world_size > 1:
            if is_main:
                gathered: list[Any] = [None] * world_size
                dist.gather_object(results, gathered, dst=0)
                all_results = [
                    result for rank_results in gathered for result in rank_results
                ]
            else:
                dist.gather_object(results, None, dst=0)
                return

        if not all_results:
            return

        normalized = torch.cat([r.normalized.cpu() for r in all_results], dim=0)
        raw = torch.cat([r.raw.cpu() for r in all_results], dim=0)
        weights = all_results[0].weights.cpu().to(dtype=normalized.dtype)
        labels = all_results[0].labels
        aggregate = (normalized * weights).sum(dim=-1)
        console.rule("[bold green]Reward Summary[/bold green]")
        console.print(
            f"reward mean={aggregate.mean().item():.4f} "
            f"std={aggregate.std(correction=0).item():.4f}"
        )
        for i, label in enumerate(labels):
            console.print(
                f"{label}: raw_mean={raw[:, i].mean().item():.4f} "
                f"raw_std={raw[:, i].std(correction=0).item():.4f} "
                f"normalized_mean={normalized[:, i].mean().item():.4f}"
            )

    def _score_one(self, batch: dict[str, Any]) -> RewardResult:
        """Score a single generated batch, supporting async-only rewards.

        ``BaseReward.score()`` only implements the synchronous ``_score()`` path,
        which async-only rewards (e.g. the remote vLLM RationalRewards judge)
        raise ``NotImplementedError`` on. Route through ``execute_reward`` instead,
        which dispatches to ``async_score()`` when the reward needs the async path
        and falls back to the synchronous path otherwise.
        """
        assert self.reward is not None
        out: list[RewardResult] = []
        execute_reward(
            self.reward,
            ((batch, None) for _ in range(1)),
            lambda _tag, result: out.append(result),
        )
        return out[0]

    def _run_one_batch(
        self,
        batch: Any,
        datasink: Any,
        reward_results: list[RewardResult],
    ) -> None:
        batch = deep_move_to_device(batch, self.device)
        batch = self.preprocess_for_inference(batch, save_extra=True)
        batch = deep_cast_float_dtype(batch, self.model.dtype)
        negative_batch: Any = (
            self.processor.get_negative_batch(batch)
            if self.sampler.cfg_scale > 1.0
            else None
        )
        key = batch.get("__key__", "unknown")
        seed = derive_seed(self.seed, key)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.processor.initialize_latents(
            batch,
            generator=generator,
            device=self.device,
            dtype=self.model.dtype,
        )
        sample_output = self.sampler.sample(
            self.model,
            batch,
            negative_batch=negative_batch,
            generator=generator,
        )
        decoded = self.processor.decode_output(
            sample_output.final_latents,
            batch,
        )
        batch.update(decoded)

        reward_result: RewardResult | None = None
        if self.reward is not None and key != "__padding__":
            reward_result = self._score_one(batch)
            self._print_reward_result(key, reward_result)
            reward_results.append(
                deep_move_to_device(reward_result, torch.device("cpu"))
            )

        result = deep_move_to_device(decoded, torch.device("cpu"))
        if reward_result is not None:
            result.update(self._reward_fields(reward_result))
        image = tensor_to_pil(result["clean_image"])
        if key == "__padding__":
            return
        if datasink is not None:
            if self.save_extra:
                result = deep_move_to_device(batch, torch.device("cpu"))
                if reward_result is not None:
                    result.update(self._reward_fields(reward_result))
            datasink.write(result)
        if self.save_preview_dir is not None:
            image.save(os.path.join(self.save_preview_dir, f"{key}.png"))

    # ------------------------------- Main loop ---------------------------------- #

    @torch.no_grad()
    @distributed_main
    def run(self):
        self.set_seed()
        self.load_transformer_from_seed(self.model, self.seed_checkpoint_dir)
        self.load_processor()
        self.make_dataloader()
        if self.reward is not None:
            self.reward.load_model(self.device)

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
        reward_results: list[RewardResult] = []

        with progress:
            for batch in self.dataloader:
                with dump_if_failed(logger, batch):
                    self._run_one_batch(batch, datasink, reward_results)
                progress.advance(task)

        if self.reward is not None:
            self._print_reward_summary(reward_results)
        console.rule("[bold green]Inference Completed[/bold green]")
