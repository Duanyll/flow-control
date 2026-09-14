"""Transitional overlay for the rollout family (GRPO / endpoint trainers).

Still drives the old ``flow_control.datasets`` readers and the
``enable_preprocess`` switch; SFT, inference and validation already run on
``DataMixin``. Migrated and deleted in the rollout stage of the data rework.
"""

import asyncio
from typing import Any, cast

import torch

from flow_control.datasets import DatasetConfig, parse_dataset
from flow_control.processors import get_processor_input_typeddict
from flow_control.processors.base import ProcessedBatch

from .data import DataMixin


class PreprocessMixin(DataMixin):
    enable_preprocess: bool = False
    """
    Whether to enable online preprocessing using the processor. Will load processor models
    on each rank which takes more GPU memory. Currently online preprocessing does not support
    async calls, so it will block the training loop when calling external LLM endpoints.

    If disabled, you should load a preprocessed dataset with `flow-control preprocess` command.
    """
    enable_coercion: bool = True

    @staticmethod
    def _sample_if_distribution(t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] == 2:
            mean, std = t[0:1], t[1:2]
            return mean + std * torch.randn_like(mean)
        return t

    @staticmethod
    def _sample_latent_distributions(batch: Any) -> None:
        """For all *latents fields with shape[0]==2, sample mean + std * noise."""
        for key in list(batch.keys()):
            if not key.endswith("latents"):
                continue
            val = batch[key]
            if isinstance(val, torch.Tensor):
                batch[key] = PreprocessMixin._sample_if_distribution(val)
            elif isinstance(val, list):
                for i, v in enumerate(val):
                    if isinstance(v, torch.Tensor):
                        val[i] = PreprocessMixin._sample_if_distribution(v)

    def _finalize_processed_batch(
        self,
        original_batch: Any,
        processed_batch: Any,
        save_extra: bool = False,
    ) -> ProcessedBatch:
        if save_extra:
            original_batch.update(processed_batch)
            processed_batch = original_batch
        elif "__key__" in original_batch:
            processed_batch["__key__"] = original_batch["__key__"]
        self._sample_latent_distributions(processed_batch)
        return cast(ProcessedBatch, processed_batch)

    def load_processor(self) -> None:
        super().load_processor()
        if self.enable_preprocess:
            self.processor.load_models("encode", self.device)
            self._loop = asyncio.new_event_loop()

    def _run_processor(self, coro: Any) -> Any:
        if self._loop is None:
            raise RuntimeError("load_processor() must run before online preprocessing.")
        return self._loop.run_until_complete(coro)

    def parse_training_dataset(self, config: DatasetConfig):
        return parse_dataset(
            config,
            coerce_to=get_processor_input_typeddict(
                self.processor.__class__, "training"
            )
            if self.enable_preprocess and self.enable_coercion
            else None,
        )

    def parse_inference_dataset(self, config: DatasetConfig):
        return parse_dataset(
            config,
            coerce_to=get_processor_input_typeddict(
                self.processor.__class__, "inference"
            )
            if self.enable_preprocess and self.enable_coercion
            else None,
        )

    def preprocess_for_training(
        self, batch: dict, save_extra: bool = False
    ) -> ProcessedBatch:
        if self.enable_preprocess:
            res = self._run_processor(self.processor.prepare_training_batch(batch))
        else:
            res = batch
        return self._finalize_processed_batch(batch, res, save_extra=save_extra)

    def preprocess_for_inference(
        self, batch: dict, save_extra: bool = False
    ) -> ProcessedBatch:
        if self.enable_preprocess:
            res = self._run_processor(self.processor.prepare_inference_batch(batch))
        else:
            res = batch
        return self._finalize_processed_batch(batch, res, save_extra=save_extra)
