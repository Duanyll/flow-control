import asyncio
from typing import Any, cast

import torch

from flow_control.datasets import DatasetConfig, parse_dataset
from flow_control.processors import Processor, get_processor_input_typeddict
from flow_control.processors.base import ProcessedBatch

from .hsdp import HsdpMixin


class PreprocessMixin(HsdpMixin):
    processor: Processor
    enable_preprocess: bool = False
    enable_coercion: bool = True
    _processor_loop: asyncio.AbstractEventLoop

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
        processed_batch["__key__"] = original_batch.get("__key__")
        self._sample_latent_distributions(processed_batch)
        return cast(ProcessedBatch, processed_batch)

    def load_processor(self):
        if self.enable_preprocess:
            self.processor.load_models("encode", self.device)
            self._processor_loop = asyncio.new_event_loop()
        self.processor.load_models("decode", self.device)

    def parse_training_dataset(self, config: DatasetConfig):
        return parse_dataset(
            config,
            coerce_to=get_processor_input_typeddict(
                self.processor.__class__, "training"
            )
            if self.enable_coercion
            else None,
        )

    def parse_inference_dataset(self, config: DatasetConfig):
        return parse_dataset(
            config,
            coerce_to=get_processor_input_typeddict(
                self.processor.__class__, "inference"
            )
            if self.enable_coercion
            else None,
        )

    def preprocess_for_training(
        self, batch: dict, save_extra: bool = False
    ) -> ProcessedBatch:
        if self.enable_preprocess:
            res = self._processor_loop.run_until_complete(
                self.processor.prepare_training_batch(batch)
            )
        else:
            res = batch
        return self._finalize_processed_batch(batch, res, save_extra=save_extra)

    def preprocess_for_inference(
        self, batch: dict, save_extra: bool = False
    ) -> ProcessedBatch:
        if self.enable_preprocess:
            res = self._processor_loop.run_until_complete(
                self.processor.prepare_inference_batch(batch)
            )
        else:
            res = batch
        return self._finalize_processed_batch(batch, res, save_extra=save_extra)
