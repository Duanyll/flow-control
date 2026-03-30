from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Literal

import aim
import numpy as np
import torch
import torch.distributed as dist
from pydantic import BaseModel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.status import Status

from flow_control.utils.logging import console, get_logger, get_version
from flow_control.utils.tensor import (
    BlendBackground,
    remove_alpha_channel,
    tensor_to_pil,
)

from .hsdp import main_process_only

logger = get_logger(__name__)


class LoggingMixin(BaseModel):
    aim_repo: str
    experiment_name: str
    training_type: str = ""
    aim_image_grouping: Literal["per_image", "per_step"] = "per_image"
    aim_image_background: None | BlendBackground = None

    _tracker: aim.Run | None = None
    _aggregated_metrics: dict[str, list[float]] = {}

    _status_fields: dict[str, str] = {}  # Should be overridden by subclasses
    _status_values: dict[str, float] = {}
    _rich_status: Status | None = None

    @property
    def tracker(self) -> aim.Run:
        if not self._tracker:
            raise RuntimeError("Aim tracker not initialized, or not on main process.")
        return self._tracker

    def _build_context(self, subset: str | None = None) -> dict[str, Any] | None:
        ctx: dict[str, Any] = {}
        if self.training_type:
            ctx["training_type"] = self.training_type
        if subset:
            ctx["subset"] = subset
        return ctx if ctx else None

    @main_process_only
    def init_tracker(self):
        self._tracker = aim.Run(repo=self.aim_repo, experiment=self.experiment_name)
        conf = self.model_dump(mode="json", warnings="none")
        conf["__version__"] = get_version()
        self.tracker["hparams"] = conf
        logger.info(
            f"Initialized Aim tracker at {self.aim_repo}, "
            f"experiment={self.experiment_name}."
        )

    def _track_image(
        self,
        aim_image: aim.Image,
        image_key: str,
        name: str,
        step: int,
        context: Any,
    ):
        if self.aim_image_grouping == "per_step":
            self.tracker.track(aim_image, name=name, step=step, context=context)
        else:
            self.tracker.track(
                aim_image, name=f"{name}/{image_key}", step=step, context=context
            )

    def log_image(
        self, image: torch.Tensor, image_key: str, step: int, name: str = "validation"
    ):
        ctx = self._build_context(subset=name)
        is_main: bool = getattr(self, "is_main_process", True)
        world_size: int = getattr(self, "world_size", 1)
        if self.aim_image_background is not None:
            image = remove_alpha_channel(image, self.aim_image_background)
        image_np = np.array(tensor_to_pil(image))

        if world_size == 1:
            self._track_image(aim.Image(image_np), image_key, name, step, ctx)
            return

        if is_main:
            images: list = [None] * world_size
            keys: list = [None] * world_size
            dist.gather_object(image_np, images, dst=0)
            dist.gather_object(image_key, keys, dst=0)
            for k, img in zip(keys, images, strict=True):
                if k == "__padding__":
                    continue
                self._track_image(aim.Image(img), k, name, step, ctx)
        else:
            dist.gather_object(image_np, None, dst=0)
            dist.gather_object(image_key, None, dst=0)

    def _track_metrics_with_context(self, metrics: dict[str, float], step: int):
        """Parse prefix/name keys, group by prefix, and track with aim context."""
        grouped: dict[str | None, dict[str, float]] = {}
        for k, v in metrics.items():
            if "/" in k:
                prefix, name = k.split("/", 1)
                grouped.setdefault(prefix, {})[name] = v
            else:
                grouped.setdefault(None, {})[k] = v

        for prefix, group in grouped.items():
            ctx: Any = self._build_context(subset=prefix)
            self.tracker.track(group, step=step, context=ctx)

    def log_metrics(self, metrics: Mapping[str, float | torch.Tensor], step: int):
        is_main: bool = getattr(self, "is_main_process", True)
        world_size: int = getattr(self, "world_size", 1)

        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }

        if world_size == 1:
            self._update_status_bar(metrics)
            self._track_metrics_with_context(metrics, step)
            return

        if is_main:
            all_metrics: list = [None] * world_size
            dist.gather_object(metrics, all_metrics, dst=0)

            avg_metrics = {
                k: sum(m[k] for m in all_metrics) / world_size for k in all_metrics[0]
            }

            self._update_status_bar(avg_metrics)
            self._track_metrics_with_context(avg_metrics, step)
        else:
            dist.gather_object(metrics, None, dst=0)

    def log_aggregated_metrics(self, metrics: Mapping[str, float | torch.Tensor]):
        for k, v in metrics.items():
            if k not in self._aggregated_metrics:
                self._aggregated_metrics[k] = []
            self._aggregated_metrics[k].append(
                v.item() if isinstance(v, torch.Tensor) else v
            )

    def flush_aggregated_metrics(self, step: int):
        avg_metrics = {k: sum(v) / len(v) for k, v in self._aggregated_metrics.items()}
        self.log_metrics(avg_metrics, step)
        self._aggregated_metrics = {}

    @contextmanager
    def status_bar(self, title: str):
        try:
            console.rule(f"[bold green]{title} Started[/bold green]")
            if self._status_fields:
                self._rich_status = Status(
                    "Initializing ...",
                    console=console,
                )
                self._rich_status.start()
            yield
            console.rule(f"[bold green]{title} Completed[/bold green]")
        finally:
            if self._rich_status:
                self._rich_status.stop()
                self._rich_status = None

    def _update_status_bar(self, metrics: dict):
        if self._rich_status:
            self._status_values.update(
                {k: v for k, v in metrics.items() if k in self._status_fields}
            )
            status_str = " · ".join(
                self._status_fields[k].format(v=self._status_values[k])
                for k in self._status_values
            )
            self._rich_status.update(f"{status_str}")

    @classmethod
    def get_progress_columns(cls):
        return [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description:<20}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
