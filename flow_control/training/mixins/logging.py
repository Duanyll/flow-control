from collections.abc import Mapping
from contextlib import contextmanager

import aim
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
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

from .hsdp import main_process_only

logger = get_logger(__name__)


class LoggingMixin(BaseModel):
    aim_repo: str
    experiment_name: str

    _tracker: aim.Run | None = None
    _aggregated_metrics: dict[str, list[float]] = {}

    _status_fields: dict[str, str] = {}  # Should be overridden by subclasses
    _status_title: str = ""
    _status_values: dict[str, float] = {}
    _rich_status: Status | None = None

    @property
    def tracker(self) -> aim.Run:
        if not self._tracker:
            raise RuntimeError("Aim tracker not initialized, or not on main process.")
        return self._tracker

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

    def log_image(
        self, image: Image.Image, image_key: str, step: int, name: str = "validation"
    ):
        is_main: bool = getattr(self, "is_main_process", True)
        world_size: int = getattr(self, "world_size", 1)

        image_np = np.array(image)
        if world_size == 1:
            self.tracker.track(
                aim.Image(image_np), name=f"{name}/{image_key}", step=step
            )
            return

        if is_main:
            images: list = [None] * world_size
            keys: list = [None] * world_size
            dist.gather_object(image_np, images, dst=0)
            dist.gather_object(image_key, keys, dst=0)
            for k, img in zip(keys, images, strict=True):
                if k == "__padding__":
                    continue
                self.tracker.track(aim.Image(img), name=f"{name}/{k}", step=step)
        else:
            dist.gather_object(image, None, dst=0)
            dist.gather_object(image_key, None, dst=0)

    def log_metrics(self, metrics: Mapping[str, float | torch.Tensor], step: int):
        is_main: bool = getattr(self, "is_main_process", True)
        world_size: int = getattr(self, "world_size", 1)

        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }

        if world_size == 1:
            self._update_status_bar(metrics)
            self.tracker.track(metrics, step=step)
            return

        if is_main:
            all_metrics: list = [None] * world_size
            dist.gather_object(metrics, all_metrics, dst=0)

            avg_metrics = {
                k: sum(m[k] for m in all_metrics) / world_size for k in all_metrics[0]
            }

            self._update_status_bar(avg_metrics)
            self.tracker.track(avg_metrics, step=step)
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
                self._status_title = title
                self._rich_status = Status(
                    f"{title} ...",
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
            self._rich_status.update(f"{self._status_title}: {status_str}")

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
