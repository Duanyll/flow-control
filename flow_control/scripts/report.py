"""Append a Markdown experiment report to an existing trackio run.

``flow-control report <project> <run_id> --file report.md`` re-opens a run with
``trackio.init(resume="must")`` and logs the markdown under a single key so it
renders in the dashboard's report view, then finishes. Use it to record an
experiment's purpose, key results, and conclusions directly in trackio (once the
run is done, or at any point while it is paused/finished).

``project`` is the trackio project (== ``experiment_name``) and ``run_id`` is the
trackio run name. ``resume="must"`` raises if that run does not exist, so a typo
fails loudly instead of creating a stray run.
"""

from __future__ import annotations

import os

from flow_control.utils.logging import get_logger

logger = get_logger(__name__)


def run(
    project: str,
    run_id: str,
    text: str,
    *,
    key: str = "report",
    step: int | None = None,
    trackio_dir: str | None = None,
) -> None:
    """Log ``text`` as a ``trackio.Markdown`` value on an existing run.

    Args:
        project: Trackio project (== ``experiment_name``).
        run_id: Trackio run name (== ``run_id``).
        text: Markdown report body.
        key: Metric key the markdown is logged under (default ``"report"``).
        step: Optional step to log the report at.
        trackio_dir: Trackio DB directory; sets ``$TRACKIO_DIR`` for this process.
    """
    if not text.strip():
        raise ValueError("Report text is empty; refusing to log an empty report.")

    if trackio_dir:
        os.environ["TRACKIO_DIR"] = trackio_dir
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    import trackio

    trackio.init(project=project, name=run_id, resume="must")
    try:
        trackio.log({key: trackio.Markdown(text)}, step=step)
    finally:
        trackio.finish()

    suffix = f" step={step}" if step is not None else ""
    logger.info(
        f"Wrote {len(text)} chars of markdown to trackio "
        f"project={project} run={run_id} key={key!r}{suffix}"
    )


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from rich import print

    tmp = Path(tempfile.mkdtemp(prefix="flow_control_report_smoke_"))
    os.environ["TRACKIO_DIR"] = str(tmp)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    import trackio

    # Simulate a training run that finishes.
    trackio.init(project="report_smoke", name="r1", config={"lr": 1e-4})
    trackio.log({"loss": 0.3}, step=0)
    trackio.finish()

    # Append a report to the finished run.
    run(
        "report_smoke",
        "r1",
        "# Report\n\n**Goal:** smoke test\n\nFinal loss **0.3**.",
        trackio_dir=str(tmp),
    )

    from trackio.sqlite_storage import SQLiteStorage

    logs = SQLiteStorage.get_logs("report_smoke", "r1")
    assert any("report" in row for row in logs), logs
    print("[bold green]report smoke test passed[/bold green]")

    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
