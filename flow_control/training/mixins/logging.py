import atexit
import json
import os
import secrets
import socket
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import torch
import torch.distributed as dist
import trackio
from pydantic import BaseModel, PrivateAttr
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

if TYPE_CHECKING:
    from trackio.run import Run as TrackioRun

logger = get_logger(__name__)


class LoggingMixin(BaseModel):
    experiment_name: str
    training_type: str = ""

    trackio_project: str | None = None
    """Trackio project name. Defaults to ``experiment_name`` if None."""
    trackio_space_id: str | None = None
    """Optional HF Space id for remote trackio sync."""

    runs_root: str = "./runs"
    """Root directory for append-only jsonl run logs."""
    run_id: str | None = None
    """Stable run identifier. Auto-generated at ``init_tracker()`` if None.

    Setting this explicitly lets a preempted job reuse the same on-disk run dir
    (and the same trackio run via ``resume='allow'``).
    """

    image_background: BlendBackground | None = None
    """Background to blend with alpha channel when logging images. If None, the
    alpha channel is preserved.
    """

    _trackio_run: "TrackioRun | None" = PrivateAttr(default=None)
    _run_dir: Path | None = PrivateAttr(default=None)
    _metrics_file: TextIO | None = PrivateAttr(default=None)
    _events_file: TextIO | None = PrivateAttr(default=None)
    _aggregated_metrics: dict[str, list[float]] = PrivateAttr(default_factory=dict)

    _status_fields: dict[str, str] = PrivateAttr(default_factory=dict)
    _status_values: dict[str, float] = PrivateAttr(default_factory=dict)
    _rich_status: Status | None = PrivateAttr(default=None)

    def _resolve_run_id(self) -> str:
        # env wins (launcher sets this); then explicit config; then generate.
        env_run_id = os.getenv("FLOW_CONTROL_RUN_ID")
        if env_run_id:
            return env_run_id
        if self.run_id:
            return self.run_id
        return f"{time.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"

    @main_process_only
    def init_tracker(self):
        self.run_id = self._resolve_run_id()

        conf = self.model_dump(mode="json", warnings="none")
        conf["flow_control_version"] = get_version()

        project = self.trackio_project or self.experiment_name
        self._trackio_run = trackio.init(
            project=project,
            name=self.run_id,
            config=conf,
            space_id=self.trackio_space_id,
            resume="allow",
            auto_log_gpu=True,
        )

        self._run_dir = Path(self.runs_root) / self.experiment_name / self.run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "meta.json").write_text(
            json.dumps(conf, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._metrics_file = (self._run_dir / "metrics.jsonl").open(
            "a", buffering=1, encoding="utf-8"
        )
        self._events_file = (self._run_dir / "events.jsonl").open(
            "a", buffering=1, encoding="utf-8"
        )
        attempt_fields: dict[str, Any] = {
            "training_type": self.training_type,
            "attempt_id": os.getenv("FLOW_CONTROL_ATTEMPT_ID", ""),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        }
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        if slurm_job_id:
            attempt_fields["slurm_job_id"] = slurm_job_id
            array_task = os.getenv("SLURM_ARRAY_TASK_ID")
            if array_task:
                attempt_fields["slurm_array_task_id"] = array_task
        self._write_event("run_start", **attempt_fields)

        atexit.register(self.finish_tracker)

        logger.info(
            f"Initialized trackio project={project} run={self.run_id} "
            f"attempt={attempt_fields['attempt_id'] or 'n/a'} "
            f"jsonl at {self._run_dir}"
        )

    @main_process_only
    def finish_tracker(self) -> None:
        if self._events_file is not None:
            self._write_event("run_finish", status="ok")
        for handle in (self._metrics_file, self._events_file):
            if handle is not None:
                handle.close()
        self._metrics_file = None
        self._events_file = None
        if self._trackio_run is not None:
            trackio.finish()
            self._trackio_run = None

    def _write_event(self, kind: str, **fields: Any) -> None:
        if self._events_file is None:
            return
        line = json.dumps(
            {"t": time.time(), "kind": kind, **fields}, ensure_ascii=False
        )
        self._events_file.write(line + "\n")
        self._events_file.flush()

    def _emit_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._trackio_run is not None:
            trackio.log(dict(metrics), step=step)
        if self._metrics_file is not None:
            t = time.time()
            for k, v in metrics.items():
                line = json.dumps(
                    {"t": t, "step": step, "key": k, "value": v},
                    ensure_ascii=False,
                )
                self._metrics_file.write(line + "\n")
            self._metrics_file.flush()

    def _emit_image(
        self, image: torch.Tensor, image_key: str, step: int, name: str
    ) -> None:
        if self.image_background is not None:
            image = remove_alpha_channel(image, self.image_background)
        pil = tensor_to_pil(image)
        full_key = f"{name}/{image_key}"
        if self._trackio_run is not None:
            trackio.log({full_key: trackio.Image(pil)}, step=step)
        self._write_event("image_logged", key=full_key, step=step)

    def log_image(
        self,
        image: torch.Tensor,
        image_key: str,
        step: int,
        name: str = "validation",
    ):
        is_main: bool = getattr(self, "is_main_process", True)
        world_size: int = getattr(self, "world_size", 1)

        if world_size == 1:
            self._emit_image(image, image_key, step, name)
            return

        if is_main:
            images: list = [None] * world_size
            keys: list = [None] * world_size
            dist.gather_object(image, images, dst=0)
            dist.gather_object(image_key, keys, dst=0)
            for k, img in zip(keys, images, strict=True):
                if k == "__padding__":
                    continue
                assert isinstance(img, torch.Tensor) and isinstance(k, str)
                self._emit_image(img, k, step, name)
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
            self._emit_metrics(metrics, step)
            return

        if is_main:
            all_metrics: list = [None] * world_size
            dist.gather_object(metrics, all_metrics, dst=0)

            avg_metrics = {
                k: sum(m[k] for m in all_metrics) / world_size for k in all_metrics[0]
            }

            self._update_status_bar(avg_metrics)
            self._emit_metrics(avg_metrics, step)
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


if __name__ == "__main__":
    import shutil
    import tempfile

    from rich import print

    tmp_root = Path(tempfile.mkdtemp(prefix="flow_control_logging_smoke_"))
    print(f"[dim]Smoke test workspace: {tmp_root}[/dim]")

    class SmokeTrainer(LoggingMixin):
        pass

    trainer = SmokeTrainer(
        experiment_name="smoke",
        training_type="smoke_test",
        runs_root=str(tmp_root / "runs"),
        trackio_project="flow-control-smoke",
        image_background="checkerboard",
    )
    trainer.init_tracker()
    assert trainer.run_id is not None
    run_dir = tmp_root / "runs" / "smoke" / trainer.run_id
    assert run_dir.exists(), run_dir
    print(f"[green]✓[/green] Created run dir {run_dir}")

    trainer.log_metrics({"loss": 0.1, "lr": 1e-4}, step=0)
    trainer.log_metrics({"loss": 0.05, "lr": 1e-4}, step=1)
    trainer.log_image(torch.rand(1, 3, 32, 32), "sample_0", step=1, name="val")

    trainer.log_aggregated_metrics({"agg": 1.0})
    trainer.log_aggregated_metrics({"agg": 3.0})
    trainer.flush_aggregated_metrics(step=2)

    trainer.finish_tracker()

    metrics_lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    events_lines = (run_dir / "events.jsonl").read_text().strip().splitlines()
    meta = json.loads((run_dir / "meta.json").read_text())

    print(f"metrics.jsonl ({len(metrics_lines)} lines):")
    for line in metrics_lines:
        print(f"  {line}")
    print(f"events.jsonl ({len(events_lines)} lines):")
    for line in events_lines:
        print(f"  {line}")
    print(f"meta.json keys: {sorted(meta)}")

    assert len(metrics_lines) == 5, metrics_lines  # 2+2+1
    parsed = [json.loads(line) for line in metrics_lines]
    assert {p["key"] for p in parsed} == {"loss", "lr", "agg"}
    agg_value = next(p for p in parsed if p["key"] == "agg")
    assert agg_value["value"] == 2.0, agg_value

    parsed_events = [json.loads(line) for line in events_lines]
    kinds = [e["kind"] for e in parsed_events]
    assert kinds == ["run_start", "image_logged", "run_finish"], kinds

    print("[bold green]All assertions passed.[/bold green]")
    shutil.rmtree(tmp_root)
