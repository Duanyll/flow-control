import atexit
import json
import os
import re
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

from flow_control.utils.logging import LOG_DIR, console, get_logger, get_version
from flow_control.utils.tensor import (
    BlendBackground,
    remove_alpha_channel,
    tensor_to_pil,
)

from .hsdp import main_process_only

if TYPE_CHECKING:
    from trackio.run import Run as TrackioRun

logger = get_logger(__name__)

AUTO_CHECKPOINT_ROOT_SENTINEL = "$auto"
_STEP_DIR_RE = re.compile(r"^(?:rolling_)?step_(\d+)$")
_RUN_ID_RE = re.compile(r"^\d{14}-[0-9a-f]{8}$")


def _generate_run_id() -> str:
    return f"{time.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"


def _attempt_id_from_env() -> str:
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id:
        array = os.getenv("SLURM_ARRAY_TASK_ID")
        return f"slurm-{job_id}_{array}" if array else f"slurm-{job_id}"
    return f"local-{time.strftime('%Y%m%d%H%M%S')}-{os.getpid()}"


def _has_checkpoint(run_dir: Path) -> bool:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.is_dir():
        return False
    for child in ckpt_root.iterdir():
        if _STEP_DIR_RE.match(child.name) and (child / ".metadata").is_file():
            return True
    return False


def _find_resumable_run_id(experiment_dir: Path) -> str | None:
    if not experiment_dir.is_dir():
        return None
    candidates: list[str] = []
    for child in experiment_dir.iterdir():
        if not child.is_dir():
            continue
        if not _RUN_ID_RE.match(child.name):
            continue
        if _has_checkpoint(child):
            candidates.append(child.name)
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return False


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
    attempt_id: str | None = None
    """One OS/Slurm invocation within a run. Auto-generated if unset."""

    image_background: BlendBackground | None = None
    """Background to blend with alpha channel when logging images. If None, the
    alpha channel is preserved.
    """

    _trackio_run: "TrackioRun | None" = PrivateAttr(default=None)
    _run_dir: Path | None = PrivateAttr(default=None)
    _metrics_file: TextIO | None = PrivateAttr(default=None)
    _aggregated_metrics: dict[str, list[float]] = PrivateAttr(default_factory=dict)

    _status_fields: dict[str, str] = PrivateAttr(default_factory=dict)
    _status_values: dict[str, float] = PrivateAttr(default_factory=dict)
    _rich_status: Status | None = PrivateAttr(default=None)

    def _sync_context_value(self, value: str) -> str:
        if not dist.is_available() or not dist.is_initialized():
            return value
        payload = [value if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        synced = payload[0]
        if not isinstance(synced, str):
            raise TypeError(
                f"Expected broadcast run context value to be str, got {synced!r}"
            )
        return synced

    def _choose_run_id(self) -> str:
        if self.run_id:
            return self.run_id

        checkpoint_root = getattr(self, "checkpoint_root", None)
        auto_resume = bool(getattr(self, "auto_resume", False))
        if checkpoint_root == AUTO_CHECKPOINT_ROOT_SENTINEL and auto_resume:
            experiment_dir = Path(self.runs_root) / self.experiment_name
            resumable = _find_resumable_run_id(experiment_dir)
            if resumable is not None:
                return resumable

        return _generate_run_id()

    def resolve_run_context(self) -> None:
        """Resolve run identity and any ``checkpoint_root='$auto'`` in-place.

        This runs after torchrun has initialized distributed state, so rank 0 can
        choose a fresh run id once and broadcast it to the other ranks.
        """
        self.run_id = self._sync_context_value(self._choose_run_id())
        self.attempt_id = self._sync_context_value(
            self.attempt_id or _attempt_id_from_env()
        )

        checkpoint_root = getattr(self, "checkpoint_root", None)
        if checkpoint_root == AUTO_CHECKPOINT_ROOT_SENTINEL:
            run_id = self.run_id
            if run_id is None:
                raise RuntimeError("run_id was not resolved.")
            resolved = (
                Path(self.runs_root) / self.experiment_name / run_id / "checkpoints"
            )
            self.checkpoint_root = str(resolved)

    @main_process_only
    def init_tracker(self):
        self.resolve_run_context()
        if self.run_id is None:
            raise RuntimeError("run_id was not resolved.")

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
        self._link_log_dir()
        (self._run_dir / "meta.json").write_text(
            json.dumps(conf, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._metrics_file = (self._run_dir / "metrics.jsonl").open(
            "a", buffering=1, encoding="utf-8"
        )
        attempt_fields: dict[str, Any] = {
            "training_type": self.training_type,
            "attempt_id": self.attempt_id or "",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        }
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        if slurm_job_id:
            attempt_fields["slurm_job_id"] = slurm_job_id
            array_task = os.getenv("SLURM_ARRAY_TASK_ID")
            if array_task:
                attempt_fields["slurm_array_task_id"] = array_task

        atexit.register(self.finish_tracker)

        logger.info(
            f"Initialized trackio project={project} run={self.run_id} "
            f"attempt={attempt_fields['attempt_id'] or 'n/a'} "
            f"jsonl at {self._run_dir}"
        )
        logger.info(f"Run started: {attempt_fields}")

    @main_process_only
    def finish_tracker(self) -> None:
        was_active = self._metrics_file is not None or self._trackio_run is not None
        if self._metrics_file is not None:
            self._metrics_file.close()
        self._metrics_file = None
        if self._trackio_run is not None:
            trackio.finish()
            self._trackio_run = None
        if was_active:
            logger.info("Run finished.")

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
        self,
        image: torch.Tensor,
        image_key: str,
        step: int,
        name: str,
        caption: str | None = None,
    ) -> None:
        if self.image_background is not None:
            image = remove_alpha_channel(image, self.image_background)
        pil = tensor_to_pil(image)
        full_key = f"{name}/{image_key}"
        if self._trackio_run is not None:
            trackio.log({full_key: trackio.Image(pil, caption=caption)}, step=step)
        logger.info(f"Image logged: key={full_key} step={step}")

    def _link_log_dir(self) -> None:
        if self._run_dir is None:
            return
        link_path = self._run_dir / "logs"
        target = LOG_DIR.resolve(strict=False)
        if link_path.is_symlink():
            if _same_path(link_path, target):
                return
            link_path.unlink()
        elif link_path.exists():
            if _same_path(link_path, target):
                return
            logger.warning(
                "Run log link target %s already exists and is not a symlink; "
                "leaving it unchanged.",
                link_path,
            )
            return
        try:
            link_path.symlink_to(target, target_is_directory=True)
        except OSError:
            logger.exception(
                "Failed to create run log symlink %s -> %s", link_path, target
            )

    def log_image(
        self,
        image: torch.Tensor,
        image_key: str,
        step: int,
        name: str = "validation",
        caption: str | None = None,
    ):
        is_main: bool = getattr(self, "is_main_process", True)
        world_size: int = getattr(self, "world_size", 1)

        if world_size == 1:
            self._emit_image(image, image_key, step, name, caption)
            return

        if is_main:
            images: list = [None] * world_size
            keys: list = [None] * world_size
            captions: list = [None] * world_size
            dist.gather_object(image, images, dst=0)
            dist.gather_object(image_key, keys, dst=0)
            dist.gather_object(caption, captions, dst=0)
            for k, img, cap in zip(keys, images, captions, strict=True):
                if k == "__padding__":
                    continue
                assert isinstance(img, torch.Tensor) and isinstance(k, str)
                self._emit_image(img, k, step, name, cap)
        else:
            dist.gather_object(image, None, dst=0)
            dist.gather_object(image_key, None, dst=0)
            dist.gather_object(caption, None, dst=0)

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
    log_link = run_dir / "logs"
    assert log_link.is_symlink(), log_link
    assert log_link.resolve(strict=False) == LOG_DIR.resolve(strict=False)
    print(f"[green]✓[/green] Created run dir {run_dir}")

    trainer.log_metrics({"loss": 0.1, "lr": 1e-4}, step=0)
    trainer.log_metrics({"loss": 0.05, "lr": 1e-4}, step=1)
    trainer.log_image(
        torch.rand(1, 3, 32, 32),
        "sample_0",
        step=1,
        name="val",
        caption="a quick brown fox jumps over the lazy dog",
    )

    trainer.log_aggregated_metrics({"agg": 1.0})
    trainer.log_aggregated_metrics({"agg": 3.0})
    trainer.flush_aggregated_metrics(step=2)

    trainer.finish_tracker()

    metrics_lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    meta = json.loads((run_dir / "meta.json").read_text())

    print(f"metrics.jsonl ({len(metrics_lines)} lines):")
    for line in metrics_lines:
        print(f"  {line}")
    print(f"meta.json keys: {sorted(meta)}")

    assert len(metrics_lines) == 5, metrics_lines  # 2+2+1
    parsed = [json.loads(line) for line in metrics_lines]
    assert {p["key"] for p in parsed} == {"loss", "lr", "agg"}
    agg_value = next(p for p in parsed if p["key"] == "agg")
    assert agg_value["value"] == 2.0, agg_value

    print("[bold green]All assertions passed.[/bold green]")
    shutil.rmtree(tmp_root)
