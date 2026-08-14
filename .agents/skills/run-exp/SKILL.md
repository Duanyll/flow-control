---
name: run-exp
description: Launch a validated flow-control experiment on the Villa Slurm cluster, babysit it through its first checkpoint and validation cycle, diagnose and relaunch failed runs, verify GPU utilization, and hand a stable run to recurring monitoring. Use when the user invokes $run-exp or asks Codex to launch and watch a training config under examples/ or local/.
---

# Run Experiment

Launch the user's validated experiment promptly, prove that it is stable, and
keep it monitored. Treat the invocation text as the input: identify the config
path, any `--update` overrides, the purpose or hypothesis, a preferred machine,
and whether early stopping is authorized, regardless of their order. Expect
invocations shaped like:

```text
$run-exp <config.jsonc> [--update key=value ...] [purpose] [machine] [early-stop ok]
```

If the purpose is unclear and cannot be inferred from the config or diff, ask
one short question. Otherwise proceed. A target machine defaults to A800, but
always check current availability and choose a different eligible node when it
better satisfies the run's requirements. Never stop a healthy run early unless
the user explicitly authorized early stopping.

## Operating authority

Treat invocation of this skill as authority to submit the experiment, monitor
it, fix launch blockers or obvious divergence, cancel an unhealthy attempt, and
relaunch or resume it. Do not commit any fix unless the user explicitly asks.
Surface any non-obvious fix that could invalidate the experiment instead of
silently proceeding.

The agent session is a CPU-only `service` Sidecar. Never run CUDA workloads
there. Use `srun` only for short interactive GPU probes and `sbatch` with the
`batch` QoS for unattended or long-running GPU work. `/home` and `/gdata` are
shared across nodes; `runs/` points to `/gdata`, so inspect metrics, logs,
checkpoints, and validation media from the Sidecar.

## 1. Inspect current cluster state

Do not rely on a saved inventory. Before choosing resources, read
`/gdata/shared/docs/docs/user/README.md` and the relevant linked Slurm chapter,
then run:

```bash
sinfo -o "%P %a %l %G"
sacctmgr -n show qos format=Name,MaxWall,MaxTRESPU%30
sacctmgr -n show assoc user="$USER" format=QOS%40
sinfo -p compute -N -o "%N | %f | %G"
/usr/bin/squeue -u "$USER"
```

Select 24/48 GiB VRAM and patched P2P support through node features such as
`vram_24g`, `vram_48g`, and `p2p`; do not infer them from a GPU marketing name.
For multi-GPU jobs on eligible patched nodes, set `NCCL_P2P_LEVEL=SYS`.

## 2. Preflight the run

Read the config and check the failure-prone requirements before submitting:

- Confirm every referenced model is present in the Hugging Face cache.
- Confirm required datasets, latent caches, and output parents exist.
- Probe reward, VAE, or other external endpoints the config requires.
- Require `checkpoint_root: "$auto"` and `auto_resume: true` for a run that must
  survive requeue or preemption; correct the working config when appropriate.
- Preserve all launch overrides supplied by the user.

Use a stable `experiment_name` and a unique `run_id`:

- Set `experiment_name` to the Trackio project and local workspace directory.
  Use one snake-case family name such as
  `<model>_<task-or-reward>_<algo>[_<tag>]`.
- Put working configs and round reports under `local/<experiment_name>/`.
- Set `run_id` explicitly to `r<N>[-<variant>]`, for example `r2-cfg1.0`.
  Increment `N` for a fresh comparable attempt. Reuse a run ID only when
  intentionally resuming that run with `auto_resume`.
- Leave `attempt_id` automatic; it records the Slurm attempt.

## 3. Submit with Slurm

Submit a batch job that runs:

```bash
uv run flow-control launch <config> --update run_id=<run-id> <other-overrides>
```

Ensure the batch request uses the chosen partition, `--qos=batch`, an honest
time limit, a stable job name, requeue behavior, and sufficient GPU/CPU/memory.
Use `sbatch --test-only` when practical. Capture the initial ID from
`sbatch --parsable`; on later checks, resolve the active job from the stable job
name with `squeue` rather than embedding a job ID in scripts or assuming an old
attempt ID is still current.

## 4. Reuse the singleton Trackio panel

Only one `trackio show` process is needed for every project in
`runs/.trackio`. Check the listener and `pgrep -f 'trackio show'` before
starting anything. Reuse an existing panel whether the launcher or an earlier
run started it.

If none exists, start exactly one detached process on a node that can keep it
alive, bind it to all interfaces, and record the reachable node IP:

```bash
TRACKIO_DIR=runs/.trackio GRADIO_SERVER_PORT=7860 \
  GRADIO_ANALYTICS_ENABLED=False \
  uv run --no-sync trackio show --host 0.0.0.0
```

Use a detached `tmux` session or a suitable `service` batch job so the panel
outlives the current turn. Do not start a duplicate panel.

## 5. Watch until stable

Poll at 210-270 second intervals. Prefer the product's nonblocking recurring
monitor or wait mechanism over occupying a foreground shell. Do not emit dense
heartbeat updates.

At each poll:

1. Resolve job state, pending reason, node, and current job ID with `squeue`.
2. Tail `runs/<project>/<run-id>/metrics.jsonl` and inspect loss/reward trends.
3. Tail `runs/<project>/<run-id>/logs/rank0000.log` for steady progress.
4. Check `runs/<project>/<run-id>/logs/rank0000.traceback.log`; any non-empty
   traceback means the run crashed.
5. Inspect validation outputs, throughput, and GPU memory for plausibility.
6. Query `https://prometheus.villa.moe` for the allocated GPU's utilization,
   memory, and power. Investigate sustained low utilization or roughly 100 W
   power after warm-up; fix batch size, data loading, or host-sync stalls before
   allowing a long job to consume more card-hours.

Use `$hugging-face-trackio` and its metric-retrieval guidance for deeper trend
analysis rather than inventing Trackio storage queries.

Declare the run **stable** only after it is RUNNING past its first checkpoint
and validation cycle, has finite non-diverging metrics trending in the expected
direction, shows steady throughput and healthy GPU utilization, and has no
traceback.

## 6. Debug and relaunch failures

When the run fails to launch, crashes, OOMs, produces NaNs, explodes, collapses
its reward, or wastes the GPU:

1. Diagnose from Slurm state, stdout/stderr, rank logs, traceback, metrics, and
   GPU telemetry.
2. Make the smallest justified code or config fix.
3. Cancel the unhealthy job if it is still consuming resources.
4. Resume from the newest valid checkpoint or launch a fresh run ID, whichever
   preserves experimental validity.
5. Repeat stabilization checks after relaunch.

Keep fixes uncommitted and list them in the status report. Ask for direction
only when a required fix is ambiguous or would materially change the result.

## 7. Report stability and hand off monitoring

Once stable, report the node and current job ID, Trackio project and run ID,
panel URL, observed checkpoint/validation/metric/GPU evidence, and any
uncommitted changes. Tell the user to select the project and filter by `run_id`
when comparing runs in the panel. Tell the user that tight polling has ended.

When the current Codex surface exposes recurring monitoring or scheduled tasks,
create a same-chat check every two hours at an off-minute such as `:17`. Use a
self-contained prompt containing the project, run ID, stable job name,
last-known node, panel location, expected trend, and early-stop authorization.
Each check must rediscover the current job and node, inspect metrics/logs/
traceback/GPU utilization, return a one-line OK when healthy, and debug or
relaunch under this skill's authority when broken.

Codex CLI and the IDE do not provide the Scheduled management interface. If no
recurring-monitor capability is exposed, do not pretend to schedule one and do
not install an OS crontab by default. State the limitation and provide the
self-contained monitor prompt for the user to schedule from the desktop app or
web.

## 8. Convergence and reports

Only judge convergence and cancel a healthy job when the invocation explicitly
authorizes early stopping. Otherwise allow it to finish naturally.

When the run ends or the user requests a write-up:

1. Retrieve final metrics from `runs/.trackio` using
   `$hugging-face-trackio`.
2. Write `local/<experiment_name>/<run-id>-report.md` with the purpose or
   hypothesis, config highlights, final metric values, validation observations,
   conclusion, and next step.
3. Log the report into Trackio:

```bash
uv run --no-sync flow-control report <project> <run-id> \
  --file local/<experiment_name>/<run-id>-report.md
```

This logs the Markdown under the default `report` key in
`./runs/.trackio` and is safe while the singleton panel is running.
