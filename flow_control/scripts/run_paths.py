"""Resolve canonical directory layout for a training run.

A "run" is identified by a stable ``run_id`` that survives preempt/requeue. A
single run can span multiple "attempts" — each attempt is one invocation of the
trainer (one Slurm job allocation, or one local process).

All artifacts for a run live under::

    <runs_root>/<experiment_name>/<run_id>/
    ├── meta.json
    ├── metrics.jsonl
    ├── events.jsonl
    ├── attempts/<attempt_id>/        # rank logs, sbatch.out, ...
    └── checkpoints/step_*/

Trackio's SQLite is shared at ``<runs_root>/.trackio/`` so the dashboard groups
runs by project (= experiment_name) cleanly.

Resolution happens in the launcher parent process *before* `os.execvp(torchrun)`
so the resolved values can be passed to workers via env vars and a tempfile
config.
"""

from __future__ import annotations

import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

AUTO_CHECKPOINT_ROOT_SENTINEL = "$auto"
AUTO_RESUME_ENV = "FLOW_CONTROL_AUTO_RESUME"
RUN_ID_ENV = "FLOW_CONTROL_RUN_ID"
ATTEMPT_ID_ENV = "FLOW_CONTROL_ATTEMPT_ID"
DEFAULT_RUNS_ROOT = "./runs"
TRACKIO_SUBDIR = ".trackio"

_STEP_DIR_RE = re.compile(r"^step_(\d+)$")
_RUN_ID_RE = re.compile(r"^\d{14}-[0-9a-f]{8}$")


@dataclass(frozen=True, slots=True)
class ResolvedPaths:
    runs_root: Path
    experiment_name: str
    run_id: str
    attempt_id: str
    run_dir: Path
    attempt_dir: Path
    checkpoint_root: Path
    trackio_dir: Path

    def to_env(self) -> dict[str, str]:
        """Env vars to export before `os.execvp(torchrun)`."""
        return {
            "LOG_DIR": str(self.attempt_dir),
            "TRACKIO_DIR": str(self.trackio_dir),
            RUN_ID_ENV: self.run_id,
            ATTEMPT_ID_ENV: self.attempt_id,
        }


def _generate_run_id() -> str:
    return f"{time.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"


def _attempt_id_from_env() -> str:
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id:
        array = os.getenv("SLURM_ARRAY_TASK_ID")
        return f"slurm-{job_id}_{array}" if array else f"slurm-{job_id}"
    return f"local-{time.strftime('%Y%m%d%H%M%S')}-{os.getpid()}"


def _has_checkpoint(run_dir: Path) -> bool:
    """A run dir counts as 'resumable' if it has at least one step_* with .metadata."""
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.is_dir():
        return False
    for child in ckpt_root.iterdir():
        if _STEP_DIR_RE.match(child.name) and (child / ".metadata").is_file():
            return True
    return False


def _find_resumable_run_id(experiment_dir: Path) -> str | None:
    """Return the newest run_id under experiment_dir whose checkpoints/ has a
    completed step_* dir. Newest is determined by the lexicographic run_id
    (timestamp-prefixed → sorts chronologically)."""
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


def resolve_run_paths(config_data: dict) -> ResolvedPaths:
    """Resolve canonical paths for one training launch.

    Reads from ``config_data`` (raw dict from the user's config file), the
    environment (`FLOW_CONTROL_RUN_ID`, `FLOW_CONTROL_AUTO_RESUME`,
    `SLURM_JOB_ID`, ...), and the existing filesystem state under
    ``runs_root/<experiment_name>/``.

    Mutating filesystem state (mkdir) is the caller's job — this function is
    pure-ish (only reads the filesystem to detect resumable runs).
    """
    experiment_name = config_data.get("experiment_name")
    if not experiment_name:
        raise ValueError("config is missing required field 'experiment_name'")

    runs_root = Path(config_data.get("runs_root", DEFAULT_RUNS_ROOT))
    experiment_dir = runs_root / experiment_name

    checkpoint_root_cfg = config_data.get("checkpoint_root")
    if checkpoint_root_cfg is None:
        raise ValueError("config is missing required field 'checkpoint_root'")
    is_auto_checkpoint = checkpoint_root_cfg == AUTO_CHECKPOINT_ROOT_SENTINEL

    # ---- run_id resolution ----
    env_run_id = os.getenv(RUN_ID_ENV)
    cfg_run_id = config_data.get("run_id")
    auto_resume = os.getenv(AUTO_RESUME_ENV) == "1"

    if env_run_id:
        run_id = env_run_id
    elif cfg_run_id:
        run_id = cfg_run_id
    elif is_auto_checkpoint and auto_resume:
        resumable = _find_resumable_run_id(experiment_dir)
        run_id = resumable if resumable is not None else _generate_run_id()
    else:
        run_id = _generate_run_id()

    attempt_id = _attempt_id_from_env()

    run_dir = experiment_dir / run_id
    attempt_dir = run_dir / "attempts" / attempt_id

    if is_auto_checkpoint:
        checkpoint_root = run_dir / "checkpoints"
    else:
        checkpoint_root = Path(checkpoint_root_cfg)

    trackio_dir = runs_root / TRACKIO_SUBDIR

    return ResolvedPaths(
        runs_root=runs_root,
        experiment_name=experiment_name,
        run_id=run_id,
        attempt_id=attempt_id,
        run_dir=run_dir,
        attempt_dir=attempt_dir,
        checkpoint_root=checkpoint_root,
        trackio_dir=trackio_dir,
    )


if __name__ == "__main__":
    import json
    import shutil
    import tempfile

    from rich import print

    tmp = Path(tempfile.mkdtemp(prefix="flow_control_run_paths_"))
    print(f"[dim]Tempdir: {tmp}[/dim]")

    # Make sure we don't inherit any caller env that would skew the test.
    for k in (RUN_ID_ENV, ATTEMPT_ID_ENV, AUTO_RESUME_ENV, "SLURM_JOB_ID"):
        os.environ.pop(k, None)

    base_config = {
        "experiment_name": "smoke",
        "runs_root": str(tmp),
        "checkpoint_root": "$auto",
    }

    # Case 1: no env, no existing dirs → fresh run_id
    p1 = resolve_run_paths(base_config)
    assert _RUN_ID_RE.match(p1.run_id), p1.run_id
    assert p1.run_dir == tmp / "smoke" / p1.run_id
    assert p1.attempt_dir.parent.parent == p1.run_dir
    assert p1.checkpoint_root == p1.run_dir / "checkpoints"
    assert p1.trackio_dir == tmp / ".trackio"
    assert p1.attempt_id.startswith("local-")
    print(f"[green]✓[/green] fresh: {p1.run_id}")

    # Case 2: FLOW_CONTROL_RUN_ID env wins
    os.environ[RUN_ID_ENV] = "20990101000000-deadbeef"
    p2 = resolve_run_paths(base_config)
    assert p2.run_id == "20990101000000-deadbeef"
    del os.environ[RUN_ID_ENV]
    print(f"[green]✓[/green] env override: {p2.run_id}")

    # Case 3: AUTO_RESUME with a resumable run on disk
    fake_run = tmp / "smoke" / "20250101000000-aaaaaaaa"
    (fake_run / "checkpoints" / "step_0000100").mkdir(parents=True)
    (fake_run / "checkpoints" / "step_0000100" / ".metadata").write_text("")
    os.environ[AUTO_RESUME_ENV] = "1"
    p3 = resolve_run_paths(base_config)
    assert p3.run_id == "20250101000000-aaaaaaaa", p3.run_id
    print(f"[green]✓[/green] auto-resume picked: {p3.run_id}")

    # Case 4: AUTO_RESUME with no resumable run → fresh
    shutil.rmtree(tmp / "smoke" / "20250101000000-aaaaaaaa")
    p4 = resolve_run_paths(base_config)
    assert _RUN_ID_RE.match(p4.run_id), p4.run_id
    assert p4.run_id != "20250101000000-aaaaaaaa"
    del os.environ[AUTO_RESUME_ENV]
    print(f"[green]✓[/green] auto-resume fallback: {p4.run_id}")

    # Case 5: AUTO_RESUME with multiple resumables → newest by lexicographic order
    older = tmp / "smoke" / "20250101000000-11111111"
    newer = tmp / "smoke" / "20250601000000-22222222"
    for d in (older, newer):
        (d / "checkpoints" / "step_0000100").mkdir(parents=True)
        (d / "checkpoints" / "step_0000100" / ".metadata").write_text("")
    os.environ[AUTO_RESUME_ENV] = "1"
    p5 = resolve_run_paths(base_config)
    assert p5.run_id == "20250601000000-22222222", p5.run_id
    del os.environ[AUTO_RESUME_ENV]
    print(f"[green]✓[/green] auto-resume newest: {p5.run_id}")

    # Case 6: explicit checkpoint_root passes through unchanged
    explicit_config = {**base_config, "checkpoint_root": "/some/explicit/path"}
    p6 = resolve_run_paths(explicit_config)
    assert p6.checkpoint_root == Path("/some/explicit/path"), p6.checkpoint_root
    print(f"[green]✓[/green] explicit checkpoint_root: {p6.checkpoint_root}")

    # Case 7: SLURM_JOB_ID gives slurm- attempt_id
    os.environ["SLURM_JOB_ID"] = "12345"
    p7 = resolve_run_paths(base_config)
    assert p7.attempt_id == "slurm-12345", p7.attempt_id
    del os.environ["SLURM_JOB_ID"]
    print(f"[green]✓[/green] slurm attempt_id: {p7.attempt_id}")

    # Case 8: env serialization round-trip
    env = p1.to_env()
    assert env["LOG_DIR"] == str(p1.attempt_dir)
    assert env["TRACKIO_DIR"] == str(p1.trackio_dir)
    assert env[RUN_ID_ENV] == p1.run_id
    print(f"[green]✓[/green] env export: {json.dumps(env, indent=2)}")

    shutil.rmtree(tmp)
    print("[bold green]All assertions passed.[/bold green]")
