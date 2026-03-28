import asyncio
from typing import cast

from flow_control.datasets import DatasetConfig, parse_dataset
from flow_control.processors import Processor, get_processor_input_typeddict
from flow_control.processors.base import ProcessedBatch

from .hsdp import HsdpMixin


class PreprocessMixin(HsdpMixin):
    processor: Processor
    enable_preprocess: bool = False
    enable_coercion: bool = True
    _processor_loop: asyncio.AbstractEventLoop

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
            if save_extra:
                batch.update(res)
                res = batch
            res["__key__"] = batch.get("__key__")  # type: ignore
            return cast(ProcessedBatch, res)
        else:
            return cast(ProcessedBatch, batch)

    def preprocess_for_inference(
        self, batch: dict, save_extra: bool = False
    ) -> ProcessedBatch:
        if self.enable_preprocess:
            res = self._processor_loop.run_until_complete(
                self.processor.prepare_inference_batch(batch)
            )
            if save_extra:
                batch.update(res)
                res = batch
            res["__key__"] = batch.get("__key__")  # type: ignore
            return cast(ProcessedBatch, res)
        else:
            return cast(ProcessedBatch, batch)
