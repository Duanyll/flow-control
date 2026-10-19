from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import json5
import torch
from pydantic import BaseModel, TypeAdapter

from flow_control.adapters import parse_model_adapter
from flow_control.adapters.base import BaseModelAdapter
from flow_control.processors import parse_processor
from flow_control.processors.base import BaseProcessor
from flow_control.samplers import (
    BaseGrid,
    BaseTrajectory,
    Cfg,
    Executor,
    RootFrame,
    Trajectory,
    data,
    variants,
    walk,
)
from flow_control.utils import device as devutil
from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger, warn_once
from flow_control.utils.progress import report_progress
from flow_control.utils.tensor import deep_cast_float_dtype, deep_move_to_device

from .config import ServeConfig

logger = get_logger(__name__)


def first_node[T: BaseModel](tree: BaseModel, kind: type[T]) -> T | None:
    """The first `kind` in a config tree: the UI's steps and CFG knobs."""
    return next((node for node in walk(tree) if isinstance(node, kind)), None)


def cfg_scale_for_display(sampler: BaseTrajectory) -> float:
    """The serving UI's ``cfg_scale`` value; 1.0 without a CFG node."""
    cfg = first_node(sampler, Cfg)
    if cfg is None:
        return 1.0
    return cfg.scale[0] if isinstance(cfg.scale, list) else cfg.scale


def steps_for_display(sampler: BaseTrajectory) -> int | None:
    """The serving UI's ``steps`` value; None without a grid."""
    grid = first_node(sampler, BaseGrid)
    return grid.steps if grid is not None else None


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


def _validate_checkpoint_path(path: str) -> None:
    """Raise if the checkpoint directory does not exist."""
    if not Path(path).is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")


def _load_dcp_seed(transformer: torch.nn.Module, path: str) -> None:
    """Load a *seed* checkpoint (flat keys, saved with ``no_dist=True``)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    _validate_checkpoint_path(path)
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

    Only the transformer weights, variant slots included, are loaded; optimizer /
    scheduler / dataloader state is ignored via ``allow_partial_load``.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    _validate_checkpoint_path(path)
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
    """Orchestrates model loading, config changes, and single-sample inference.

    The engine is **stateless between requests** (with caching).  Each
    :meth:`generate` call carries the full desired state (config JSONC,
    checkpoint paths, sampler overrides).  :class:`HfModelLoader`'s class-level
    cache makes unchanged models free to "reload".
    """

    def __init__(self, config: ServeConfig) -> None:
        self.model_device = config.device
        self.processor_device = (
            config.processor_device
            if config.processor_device is not None
            else config.device
        )
        self.offload_processor = config.offload_processor
        self.model: BaseModelAdapter = config.model
        self.processor: BaseProcessor = config.processor
        self.sampler: BaseTrajectory = config.sampler
        self.seed = config.seed
        self.variant = config.variant
        self._gpu_lock = asyncio.Lock()

        self._seed_checkpoint_dir: str | None = config.seed_checkpoint_dir
        self._checkpoint_root: str | None = config.checkpoint_root
        # Normalize checkpoint_dir relative to checkpoint_root so the Gradio
        # dropdown shows a bare subdirectory name that resolve_checkpoint_dir
        # can round-trip correctly.
        ckpt = config.checkpoint_dir
        if ckpt and self._checkpoint_root:
            with contextlib.suppress(ValueError):
                ckpt = str(Path(ckpt).relative_to(self._checkpoint_root))
        self._checkpoint_dir: str | None = ckpt
        self._last_state_key: tuple[str, str | None, str | None] | None = None

    def _iter_model_loaders(
        self, obj: Any, seen: set[int] | None = None
    ) -> Iterator[HfModelLoader[Any]]:
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, HfModelLoader):
            yield obj

        if isinstance(obj, BaseModel):
            for field_name in obj.__class__.model_fields:
                yield from self._iter_model_loaders(getattr(obj, field_name), seen)
            return

        if isinstance(obj, dict):
            for value in obj.values():
                yield from self._iter_model_loaders(value, seen)
            return

        if isinstance(obj, (list, tuple, set)):
            for value in obj:
                yield from self._iter_model_loaders(value, seen)

    def list_checkpoints(self) -> list[str]:
        """Return sorted subdirectory names under ``checkpoint_root``, or ``[]``."""
        if not self._checkpoint_root:
            return []
        root = Path(self._checkpoint_root)
        if not root.is_dir():
            return []
        return sorted(d.name for d in root.iterdir() if d.is_dir())

    def resolve_checkpoint_dir(self, value: str | None) -> str | None:
        """Resolve a checkpoint value to a full path.

        When ``checkpoint_root`` is set, a bare name (not an absolute path)
        is treated as a subdirectory of the root.  Empty or ``None`` → ``None``.
        """
        if not value:
            return None
        if self._checkpoint_root and not Path(value).is_absolute():
            return str(Path(self._checkpoint_root) / value)
        return value

    def _unload_loaders(self, *objects: Any) -> None:
        unloaded_keys: set[tuple[Any, ...]] = set()
        for obj in objects:
            for loader in self._iter_model_loaders(obj):
                if loader.config_key in unloaded_keys:
                    continue
                loader.unload_model()
                unloaded_keys.add(loader.config_key)

    def _cleanup_failed_reload(
        self,
        *,
        new_model: BaseModelAdapter,
        new_processor: BaseProcessor,
        cache_refs_before: dict[tuple[Any, ...], Any],
    ) -> None:
        for loader in self._iter_model_loaders((new_model, new_processor)):
            loaded_model = getattr(loader, "_model", None)
            if loaded_model is None:
                continue
            if cache_refs_before.get(loader.config_key) is not loaded_model:
                loader.unload_model()
            else:
                loader._model = None

        devutil.empty_cache()

    # ----------------------------- Model loading ----------------------------- #

    def load_all(self) -> None:
        """Initial full model load.  Sets the state key so the first generate
        call won't redundantly reload."""
        logger.info(
            f"Loading transformer on {self.model_device}, "
            f"processor on {self.processor_device}..."
        )
        self.model.load_transformer(device=self.model_device)
        self._apply_checkpoints()
        self.model.transformer.eval()

        self.processor.load_models("encode", self.processor_device)
        self.processor.load_models("decode", self.processor_device)

        # Compute config_jsonc *after* loading so it matches what the Gradio
        # app will put in the jsonc_box (also computed post-load).
        config_jsonc = json5.dumps(self.config_dump(), indent=2, ensure_ascii=False)
        self._last_state_key = (
            config_jsonc,
            self._seed_checkpoint_dir,
            self._checkpoint_dir,
        )
        logger.info("All models loaded.")

    def _apply_checkpoints(self) -> None:
        """Apply seed then training checkpoint to the current transformer."""
        if self._seed_checkpoint_dir:
            _load_dcp_seed(self.model.transformer, self._seed_checkpoint_dir)
        resolved = self.resolve_checkpoint_dir(self._checkpoint_dir)
        if resolved:
            _load_dcp_training(self.model.transformer, resolved)

    def _parse_requested_state(
        self, config_jsonc: str
    ) -> tuple[BaseModelAdapter, BaseProcessor, BaseTrajectory]:
        conf = json5.loads(config_jsonc)
        if not isinstance(conf, dict):
            raise ValueError("Config must be a JSON object.")

        processor_conf = conf.get("processor")
        if not isinstance(processor_conf, dict):
            raise ValueError("Config must contain a 'processor' object.")

        return (
            parse_model_adapter(conf["model"]),
            parse_processor(processor_conf),
            TypeAdapter(Trajectory).validate_python(conf["sampler"]),
        )

    def _needs_fresh_transformer(
        self,
        *,
        new_model: BaseModelAdapter,
        seed_checkpoint_dir: str | None,
        checkpoint_dir: str | None,
        force_reload: bool,
    ) -> bool:
        ckpt_changed = (
            seed_checkpoint_dir != self._seed_checkpoint_dir
            or checkpoint_dir != self._checkpoint_dir
        )
        transformer_key_changed = (
            new_model.hf_model.config_key != self.model.hf_model.config_key
        )
        return force_reload or ckpt_changed or transformer_key_changed

    def _commit_state(
        self,
        *,
        new_model: BaseModelAdapter,
        new_processor: BaseProcessor,
        new_sampler: BaseTrajectory,
        state_key: tuple[str, str | None, str | None],
    ) -> None:
        self.model = new_model
        self.processor = new_processor
        self.sampler = new_sampler
        self._last_state_key = state_key

    # ----------------------------- State reconciliation --------------------- #

    def _ensure_models(
        self,
        config_jsonc: str,
        seed_checkpoint_dir: str | None,
        checkpoint_dir: str | None,
        force_reload: bool = False,
    ) -> None:
        """Ensure engine state matches *config_jsonc* + checkpoint paths.

        Stateless: always rebuilds Pydantic config objects from JSONC.
        Cached: :class:`HfModelLoader`'s class-level cache makes unchanged
        models free to "reload".  Only truly new models hit the network/disk.
        """
        seed_checkpoint_dir = seed_checkpoint_dir or None
        checkpoint_dir = checkpoint_dir or None

        state_key = (config_jsonc, seed_checkpoint_dir, checkpoint_dir)
        if not force_reload and state_key == self._last_state_key:
            return

        new_model, new_processor, new_sampler = self._parse_requested_state(
            config_jsonc
        )
        need_fresh_transformer = self._needs_fresh_transformer(
            new_model=new_model,
            seed_checkpoint_dir=seed_checkpoint_dir,
            checkpoint_dir=checkpoint_dir,
            force_reload=force_reload,
        )
        cache_refs_before = dict(HfModelLoader._model_cache)

        try:
            if force_reload:
                self._unload_loaders(self.model, self.processor)
            elif need_fresh_transformer:
                self.model.hf_model.unload_model()  # frees GPU + clears cache

            # Load all models — cache handles dedup for unchanged components
            report_progress(0.0, "Loading transformer...")
            with HfModelLoader.loading_scope() as scope:
                new_model.load_transformer(device=self.model_device)
                report_progress(0.4, "Loading processor (encode)...")
                new_processor.load_models("encode", self.processor_device)
                report_progress(0.6, "Loading processor (decode)...")
                new_processor.load_models("decode", self.processor_device)

            # Apply DCP checkpoints on a freshly loaded transformer
            if need_fresh_transformer:
                self._seed_checkpoint_dir = seed_checkpoint_dir
                self._checkpoint_dir = checkpoint_dir
                if seed_checkpoint_dir:
                    report_progress(0.7, "Loading seed checkpoint...")
                    _load_dcp_seed(new_model.transformer, seed_checkpoint_dir)
                resolved_ckpt = self.resolve_checkpoint_dir(checkpoint_dir)
                if resolved_ckpt:
                    report_progress(0.85, "Loading training checkpoint...")
                    _load_dcp_training(new_model.transformer, resolved_ckpt)

            new_model.transformer.eval()

            self._commit_state(
                new_model=new_model,
                new_processor=new_processor,
                new_sampler=new_sampler,
                state_key=state_key,
            )

            # Free stale cached models
            scope.purge_stale()
        except Exception:
            self._cleanup_failed_reload(
                new_model=new_model,
                new_processor=new_processor,
                cache_refs_before=cache_refs_before,
            )
            raise

    # ----------------------------- Config dump ------------------------------ #

    def config_dump(self) -> dict[str, Any]:
        """Dump current model/processor/sampler config as a JSON-serializable dict."""
        return {
            "model": self.model.model_dump(
                mode="json", warnings="none", exclude_unset=True
            ),
            "processor": self.processor.model_dump(
                mode="json", warnings="none", exclude_unset=True
            ),
            "sampler": self.sampler.model_dump(
                mode="json", warnings="none", exclude_unset=True
            ),
        }

    # ----------------------------- Inference --------------------------------- #

    async def generate(
        self,
        input_row: dict[str, Any],
        *,
        seed: int | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        variant: str | None = None,
        config_jsonc: str | None = None,
        seed_checkpoint_dir: str | None = None,
        checkpoint_dir: str | None = None,
        force_reload: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run single-sample inference with optional config/sampler overrides.

        Returns ``(result_dict, info_dict)`` where *result_dict* contains
        decoded image tensors (CPU) and *info_dict* contains non-tensor
        metadata for display.

        All state mutations and GPU work are serialized under ``_gpu_lock``.
        """
        async with self._gpu_lock:
            # Reconcile models / checkpoints (fast no-op when nothing changed)
            if config_jsonc is not None:
                await asyncio.to_thread(
                    self._ensure_models,
                    config_jsonc,
                    seed_checkpoint_dir,
                    checkpoint_dir,
                    force_reload,
                )

            # Sampler overrides (config mutation at the request boundary)
            if steps is not None:
                grid = first_node(self.sampler, BaseGrid)
                if grid is not None:
                    grid.steps = steps
                else:
                    warn_once(
                        logger,
                        "The request's steps is ignored: the sampler has no grid.",
                    )
            # An untouched slider keeps a per-step scale schedule.
            if cfg_scale is not None and cfg_scale != cfg_scale_for_display(
                self.sampler
            ):
                cfg = first_node(self.sampler, Cfg)
                if cfg is not None:
                    cfg.scale = cfg_scale
                else:
                    warn_once(
                        logger,
                        "The request's cfg_scale is ignored: the sampler has no cfg node.",
                    )
            seed = seed if seed is not None else self.seed

            # --- encode on processor_device (async) ---
            row = deep_move_to_device(input_row, self.processor_device)
            row = await self.processor.prepare_inference_row(row)

            # --- sample & decode (sync GPU work) ---
            return await asyncio.to_thread(
                self._sample_and_decode, row, seed, variant or self.variant
            )

    async def reload(
        self,
        *,
        config_jsonc: str,
        seed_checkpoint_dir: str | None = None,
        checkpoint_dir: str | None = None,
    ) -> str:
        async with self._gpu_lock:
            await asyncio.to_thread(
                self._ensure_models,
                config_jsonc,
                seed_checkpoint_dir,
                checkpoint_dir,
                True,
            )
            return "Models reloaded."

    def _move_processor_models(self, device: torch.device) -> None:
        """Move all processor nn.Module models to *device*."""
        for field_name in dict.fromkeys(
            self.processor._encoding_components + self.processor._decoding_components
        ):
            loader: HfModelLoader = getattr(self.processor, field_name)
            if loader is not None and loader.is_loaded and hasattr(loader.model, "to"):
                loader.model.to(device)

    @torch.no_grad()
    def _sample_and_decode(
        self, row: Any, seed: int, variant: str | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sample and decode, returning ``(result_dict, info_dict)``.

        ``result_dict`` is the :class:`DecodedRow` from the processor
        (tensors on CPU).  ``info_dict`` contains non-tensor metadata
        extracted from the processed row for display.
        """
        t0 = time.perf_counter()
        row = deep_cast_float_dtype(row, self.model.dtype)

        # Offload processor to CPU before DiT sampling to free GPU memory
        if self.offload_processor:
            self._move_processor_models(torch.device("cpu"))

        try:
            # --- sample on model_device ---
            generator = torch.Generator(device=self.model_device).manual_seed(seed)
            self.processor.initialize_latents(
                row, generator=generator, device=self.model_device
            )
            total, done = self.sampler.estimate_calls(row), 0

            def on_call(_: Any) -> None:
                nonlocal done
                done += 1
                report_progress(min(done / total, 1.0), f"{done}/{total} model calls")

            row.update(
                sigma=1.0,
                _generator=generator,
                _frame=RootFrame(self, on_call=on_call),
            )
            report_progress(0.0, "Sampling...")
            with self.model.use_variant(variant):
                executor = Executor(self.model, variants(self.sampler))
                list(executor.stream([(row, self.sampler(row))]))

            # Reload processor back to GPU for decode
            if self.offload_processor:
                self._move_processor_models(self.processor_device)

            # --- decode on processor_device ---
            report_progress(1.0, "Decoding...")
            output_latent = row["noisy_latents"].to(self.processor_device)
            result = self.processor.decode_output(output_latent, row)
            result = deep_move_to_device(result, torch.device("cpu"))
        finally:
            if self.offload_processor:
                self._move_processor_models(self.processor_device)

        elapsed = time.perf_counter() - t0

        # Build info dict: sampler stats + non-tensor fields from the row
        info: dict[str, Any] = {
            "seed": seed,
            "steps": steps_for_display(self.sampler),
            "cfg_scale": cfg_scale_for_display(self.sampler),
            "time": f"{elapsed:.1f}s",
        }
        for k, v in data(row).items():
            if isinstance(v, torch.Tensor):
                continue
            if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                continue
            info[k] = v

        return dict(result), info
