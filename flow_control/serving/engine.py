from __future__ import annotations

import asyncio
import time
from typing import Any

import json5
import torch
from PIL import Image
from pydantic import BaseModel

from flow_control.adapters import parse_model_adapter
from flow_control.adapters.base import BaseModelAdapter
from flow_control.processors import parse_processor
from flow_control.processors.base import BaseProcessor
from flow_control.samplers import Sampler
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import (
    deep_cast_float_dtype,
    deep_move_to_device,
    tensor_to_pil,
)

from .config import ServeConfig

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  HfModelLoader helpers                                                       #
# --------------------------------------------------------------------------- #


def _iter_hf_model_loaders(
    *roots: BaseModel,
) -> list[HfModelLoader]:
    """Walk Pydantic model trees and collect all HfModelLoader instances."""
    result: list[HfModelLoader] = []
    visited: set[int] = set()

    def _walk(obj: Any) -> None:
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)
        if isinstance(obj, HfModelLoader):
            result.append(obj)
        if isinstance(obj, BaseModel):
            for name in type(obj).model_fields:
                value = getattr(obj, name, None)
                if value is not None:
                    _walk(value)

    for root in roots:
        _walk(root)
    return result


def _transplant_models(
    old_roots: tuple[BaseModel, ...],
    new_roots: tuple[BaseModel, ...],
) -> list[str]:
    """Transfer loaded ``_model`` refs from old loaders to new ones with matching keys.

    Unloads old models that have no match in the new config. Returns human-readable
    messages describing what was unloaded/transplanted.
    """
    messages: list[str] = []

    old_models: dict[tuple, HfModelLoader] = {}
    for loader in _iter_hf_model_loaders(*old_roots):
        if loader.is_loaded:
            old_models[loader.config_key] = loader

    transplanted: set[tuple] = set()
    for new_loader in _iter_hf_model_loaders(*new_roots):
        key = new_loader.config_key
        if key in old_models:
            new_loader._model = old_models[key]._model  # noqa: SLF001
            transplanted.add(key)

    for key, old_loader in old_models.items():
        if key not in transplanted:
            name = f"{old_loader.class_name} ({old_loader.pretrained_model_id})"
            old_loader.unload_model()
            messages.append(f"Unloaded: {name}")

    return messages


# --------------------------------------------------------------------------- #
#  DCP checkpoint loading (single-process)                                     #
# --------------------------------------------------------------------------- #


class _TransformerStateful:
    """Minimal ``Stateful`` adapter so ``dcp.load`` can match the nested key
    structure used by training checkpoints (``app.transformer.<param>``).

    See ``flow_control/training/sft.py`` and ``flow_control/training/mixins/dcp.py``
    for the save-side counterpart.
    """

    def __init__(self, transformer: torch.nn.Module) -> None:
        self._transformer = transformer

    def state_dict(self) -> dict[str, Any]:
        return {"transformer": self._transformer.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "transformer" in state_dict:
            self._transformer.load_state_dict(state_dict["transformer"], strict=False)


def _load_dcp_seed(transformer: torch.nn.Module, path: str) -> None:
    """Load a *seed* checkpoint (flat keys, saved with ``no_dist=True``)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    state_dict = transformer.state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=path,
        no_dist=True,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    transformer.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded seed checkpoint from {path}")


def _load_dcp_training(transformer: torch.nn.Module, path: str) -> None:
    """Load a *training* checkpoint (nested ``app.transformer.<key>`` structure).

    Only the transformer weights are loaded; optimizer / scheduler / dataloader
    state is ignored via ``allow_partial_load``.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    adapter = _TransformerStateful(transformer)
    dcp.load(
        {"app": adapter},
        checkpoint_id=path,
        no_dist=True,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    logger.info(f"Loaded training checkpoint from {path}")


# --------------------------------------------------------------------------- #
#  ServingEngine                                                               #
# --------------------------------------------------------------------------- #


class ServingEngine:
    """Orchestrates model loading, config changes, and single-sample inference."""

    def __init__(self, config: ServeConfig) -> None:
        self.model_device = config.device
        self.processor_device = (
            config.processor_device
            if config.processor_device is not None
            else config.device
        )
        self.model: BaseModelAdapter = config.model
        self.processor: BaseProcessor = config.processor
        self.sampler: Sampler = config.sampler
        self._gpu_lock = asyncio.Lock()

        # Checkpoint state — tracked separately so we can detect changes.
        self._seed_checkpoint_dir: str | None = config.seed_checkpoint_dir
        self._checkpoint_dir: str | None = config.checkpoint_dir

    # ----------------------------- Model loading ----------------------------- #

    def load_all(self) -> None:
        """Initial full model load."""
        logger.info(
            f"Loading transformer on {self.model_device}, "
            f"processor on {self.processor_device}..."
        )
        self.model.load_transformer(device=self.model_device)
        self._apply_checkpoints()
        self.model.transformer.eval()

        self.processor.load_models("encode", self.processor_device)
        self.processor.load_models("decode", self.processor_device)
        logger.info("All models loaded.")

    def _apply_checkpoints(self) -> None:
        """Apply seed then training checkpoint to the current transformer."""
        if self._seed_checkpoint_dir:
            _load_dcp_seed(self.model.transformer, self._seed_checkpoint_dir)
        if self._checkpoint_dir:
            _load_dcp_training(self.model.transformer, self._checkpoint_dir)

    # ----------------------------- Checkpoint reload ------------------------- #

    def update_checkpoints(
        self,
        seed_checkpoint_dir: str | None,
        checkpoint_dir: str | None,
    ) -> list[str]:
        """Reload the transformer if checkpoint paths changed.

        The base model is re-loaded from HuggingFace (via ``HfModelLoader``) and
        then the new checkpoints are applied on top.  Processor models (VAE,
        encoder) are *not* reloaded — they are unaffected by training checkpoints.

        Returns human-readable messages describing what happened.
        """
        # Normalise empty strings to None
        seed_checkpoint_dir = seed_checkpoint_dir or None
        checkpoint_dir = checkpoint_dir or None

        if (
            seed_checkpoint_dir == self._seed_checkpoint_dir
            and checkpoint_dir == self._checkpoint_dir
        ):
            return ["Checkpoints unchanged."]

        messages: list[str] = []

        # Force-reload the transformer from HuggingFace base weights.
        self.model.hf_model.unload_model()
        self.model.load_transformer(device=self.model_device)

        self._seed_checkpoint_dir = seed_checkpoint_dir
        self._checkpoint_dir = checkpoint_dir
        self._apply_checkpoints()
        self.model.transformer.eval()

        if seed_checkpoint_dir:
            messages.append(f"Loaded seed checkpoint: {seed_checkpoint_dir}")
        if checkpoint_dir:
            messages.append(f"Loaded training checkpoint: {checkpoint_dir}")
        if not messages:
            messages.append("Checkpoints cleared, using base model weights.")
        return messages

    # ----------------------------- Config dump/apply ------------------------- #

    def config_dump(self) -> dict[str, Any]:
        """Dump current model/processor/sampler config as a JSON-serializable dict."""
        return {
            "model": self.model.model_dump(
                mode="json", warnings="none", exclude_defaults=True
            ),
            "processor": self.processor.model_dump(
                mode="json", warnings="none", exclude_defaults=True
            ),
            "sampler": self.sampler.model_dump(
                mode="json", warnings="none", exclude_defaults=True
            ),
        }

    def apply_config(self, full_config_jsonc: str) -> list[str]:
        """Parse a full config JSONC, rebuild Pydantic objects, transplant unchanged models.

        The input should be a complete config dict (not a partial override), typically
        obtained by editing the output of :meth:`config_dump`.

        Returns a list of human-readable messages about what happened.
        """
        conf = json5.loads(full_config_jsonc)
        if not isinstance(conf, dict):
            raise ValueError("Config must be a JSON object.")

        # Rebuild everything from the full config (cheap)
        new_model = parse_model_adapter(conf["model"])
        new_processor = parse_processor(conf["processor"])
        new_sampler = Sampler(**conf.get("sampler", {}))

        messages = _transplant_models(
            old_roots=(self.model, self.processor),
            new_roots=(new_model, new_processor),
        )

        # Load any new models that weren't transplanted (no-ops if already loaded)
        new_model.load_transformer(device=self.model_device)
        new_model.transformer.eval()
        new_processor.load_models("encode", self.processor_device)
        new_processor.load_models("decode", self.processor_device)

        # Replace engine state
        self.model = new_model
        self.processor = new_processor
        self.sampler = new_sampler

        if not messages:
            messages.append("Config updated (no model reload needed).")
        return messages

    # ----------------------------- Inference --------------------------------- #

    @torch.no_grad()
    async def generate(
        self,
        input_batch: dict[str, Any],
        seed: int | None = None,
    ) -> tuple[Image.Image, str]:
        """Run single-sample inference. Returns (PIL image, status message)."""
        async with self._gpu_lock:
            return await asyncio.to_thread(self._generate_sync, input_batch, seed)

    def _generate_sync(
        self,
        input_batch: dict[str, Any],
        seed: int | None,
    ) -> tuple[Image.Image, str]:
        t0 = time.perf_counter()
        seed = seed if seed is not None else self.sampler.seed

        # --- encode on processor_device ---
        batch = deep_move_to_device(input_batch, self.processor_device)

        loop = asyncio.new_event_loop()
        try:
            batch: Any = loop.run_until_complete(
                self.processor.prepare_inference_batch(batch)
            )
        finally:
            loop.close()

        batch = deep_cast_float_dtype(batch, self.model.dtype)
        negative_batch: Any = (
            self.processor.get_negative_batch(batch)
            if self.sampler.cfg_scale > 1.0
            else None
        )

        # --- sample on model_device ---
        # sampler.sample internally moves batch to model.device via
        # deep_move_to_device, so cross-device is handled automatically.
        generator = torch.Generator(device=self.model_device).manual_seed(seed)
        self.processor.initialize_latents(
            batch, generator=generator, device=self.model_device, dtype=self.model.dtype
        )
        sample_output = self.sampler.sample(
            self.model, batch, negative_batch=negative_batch, generator=generator
        )

        # --- decode on processor_device ---
        output_latent = sample_output.final_latents.to(self.processor_device)
        result = self.processor.decode_output(output_latent, batch)
        result = deep_move_to_device(result, torch.device("cpu"))
        image = tensor_to_pil(result["clean_image"])

        elapsed = time.perf_counter() - t0
        status = (
            f"Seed: {seed} | Steps: {self.sampler.steps} | "
            f"CFG: {self.sampler.cfg_scale} | Time: {elapsed:.1f}s"
        )
        return image, status
