---
description: Launch a validated experiment, babysit it to stability, then hand off to a bi-hourly cron.
argument-hint: <config.jsonc> [--update run_id=...] [purpose / machine / "early-stop ok"]
---

You are launching an experiment for the user. They are **confident in the code** —
it has been validated elsewhere (another repo or a past run) — so bias toward
**action**: get it running, prove it is stable, and watch it. You have authority
to debug and restart.

## Inputs

`$ARGUMENTS` is the config path (a `.jsonc` under `examples/` or `local/`) plus
optional extras, in any order:
- launch overrides, e.g. `--update run_id=...`
- the experiment's **purpose / hypothesis** (use it in the final report)
- a target machine (default A800)
- whether **early-stopping is authorized** (e.g. "early-stop ok", "stop when
  converged"). Without this, never stop a healthy run yourself.

If the purpose is unclear and you can't infer it from the config/diff, ask one
short question. Otherwise proceed.

## The cluster, as it is now

- The agent session runs on the **service node**: it has `/gdata` + `/home`, **no
  GPU**. Submit GPU work with `sbatch`.
- `runs/` is a symlink onto `/gdata`, so every run's `metrics.jsonl`, checkpoints,
  trackio db, and validation media are readable from the service node — you can
  inspect a run without touching the GPU node.
- Long jobs go through **`sbatch`** (they outlive the Bash-tool timeout); `srun`
  only for short interactive GPU probes.

## Naming standard (keep trackio panels comparable)

Trackio only overlays/compares runs **within one project**, so the project must be
stable and each attempt needs a meaningful run id.

- **`experiment_name` == trackio project == local workspace dir.** One stable,
  snake_case name per comparable family: `<model>_<task-or-reward>_<algo>[_<tag>]`
  (e.g. `qwen_rational_nft`, `flux2_rgba_t2i`). Working configs and per-round
  report files live in `local/<experiment_name>/` (gitignored), mirroring the
  project name so dir ↔ panel is obvious.
- **`run_id` == one comparable run == `r<N>[-<variant>]`** (`r1`, `r2-cfg1.0`,
  `r3-lr2e-4`). Bump `N` per fresh attempt; add a short `-<variant>` tag for the
  hyperparam you are varying so siblings read clearly on the panel. Set it
  explicitly: `--update run_id=<id>`. A **new** run_id = a new comparable run;
  **reusing** a run_id with `auto_resume` resumes that run/checkpoint after a
  Slurm requeue.
- `attempt_id` is auto (`slurm-<jobid>`) and logged as a field — leave it.

## Workflow

### 1. Pre-flight
- Read the config. Sanity-check the bits that bite unattended runs: every model
  it references is in the HF cache, data/latent caches exist, endpoints
  (reward/vae servers) are reachable, `checkpoint_root: "$auto"` +
  `auto_resume: true` so a requeue resumes cleanly.
- Pick `experiment_name` + `run_id` per the standard above. If you are tuning a
  config, copy it into `local/<experiment_name>/` and edit there.
- Check the queue (`/usr/bin/squeue -u duanyll`) and decide the machine.

### 2. Launch
Submit through `sbatch` (so it survives), running `uv run flow-control launch
<config> --update run_id=<id>` under it, on A800 (default) or a `compute` 4090
node. Requirements, not a specific script:
- Pin the stable `run_id` you chose; ensure the job requeues + resumes on preempt.
- Capture the **job id** robustly from `squeue` by job name — never hardcode it
  (it changes across requeues).
- Bring up the trackio panel (next section) if one is not already serving.

### 3. Trackio panel (one global singleton)
One `trackio show` against `runs/.trackio` serves **all** projects, so you only
ever need **one** panel — do not start a second.
- First check whether one is already up (a panel the launcher started on the GPU
  node, or one from a previous `/run-exp`): look for a listener on the port /
  `pgrep -f 'trackio show'`. If so, reuse it.
- Otherwise start exactly one, **detached** so it outlives this turn (a `tmux`
  pane, or a backgrounded process), bound to all interfaces:
  ```bash
  TRACKIO_DIR=runs/.trackio GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    uv run --no-sync trackio show --host 0.0.0.0
  ```
  It is reachable at `http://<node-ip>:7860`. The service node can host it (the
  panel needs no GPU and `/gdata` is mounted there).

### 4. Stabilization watch (tight loop)
Poll **every 3–5 minutes** until the run is provably stable — keep intervals in
that window (Anthropic's prompt cache TTL is 5 min: shorter wastes budget while
the cache is warm, longer forces a cold context reread). Within a turn,
`sleep 210–270` between checks.

Each poll, look at:
- job state — `/usr/bin/squeue -u duanyll` (PENDING → RUNNING, NODELIST);
- `runs/<exp>/<run_id>/metrics.jsonl` (tail it) and the panel — loss/reward trend;
- `runs/<exp>/<run_id>/logs/rank0000.log` — recent progress;
- `runs/<exp>/<run_id>/logs/rank0000.traceback.log` — **non-empty = a crash**;
- validation images / throughput / GPU memory look sane.

**STABLE** = job RUNNING, past the first checkpoint + a validation cycle, loss
decreasing / reward trending the expected way (not NaN, not diverging), no
traceback, steady throughput. Use the `trackio` skill for any deeper metric
pull/analysis.

### 5. Auto-debug authority
While stabilizing (and whenever a cron check finds trouble) you **may**:
- fix bugs that block launch or cause obvious divergence (NaN/exploding loss,
  reward collapse, OOM, config/cache/endpoint errors);
- `scancel` and relaunch from scratch, or resume from the newest checkpoint —
  whichever is correct.

Constraints:
- **Do not `git commit` any fix** unless the user explicitly tells you to. Leave
  fixes in the working tree and **list what you changed** in your status report.
- If a fix is non-obvious or could invalidate results, surface it instead of
  silently proceeding.

### 6. Report status to the user (once running / stable)
Tell them concisely:
- **node** the job landed on + **job id**, **project** + **run_id**;
- the **trackio panel URL** (`http://<node-ip>:7860`), and that on the panel they
  filter within the project by `run_id` to compare runs;
- any code you touched (uncommitted), and that the tight watch is now handed off
  to a bi-hourly cron.

### 7. Hand off to a low-frequency cron
Once STABLE, stop tight polling and schedule a recurring check. **Bi-hourly** is
plenty for multi-day training (pick a longer interval if the run is very slow).
Use an **off-minute** cron (e.g. minute 17, not :00/:30 — prompt-cache courtesy)
and keep it **session-only** unless the user wants it to persist. Note the 7-day
auto-expiry.

The cron prompt must be self-contained — embed `project`, `run_id`, job name, and
node so a fresh turn has context. Each check: job state, `metrics.jsonl`/panel
trend, traceback file. If healthy → one-line OK. If broken (job gone, traceback,
divergence) → wake up and debug/relaunch per §5.

### 8. Convergence & early-stopping (opt-in only)
**Only if the user authorized it** may you judge that a run has converged or is no
longer worth continuing, then `scancel` (early-stop) or move on to the next
experiment in the plan. Otherwise let healthy runs run.

### 9. End-of-run report → into trackio
When a run ends (naturally, early-stopped, or when the user asks for a writeup),
write a Markdown report **straight into trackio** so it lives with the run:
1. Compose it: **purpose/hypothesis**, config highlights, **key metrics with
   final values**, validation observations, conclusion / next step. Pull final
   numbers from `runs/.trackio` (reachable on `/gdata`) via the `trackio` skill.
2. Save it to `local/<experiment_name>/<run_id>-report.md`.
3. Log it:
   ```bash
   uv run --no-sync flow-control report <project> <run_id> \
     --file local/<experiment_name>/<run_id>-report.md
   ```
   (defaults: `--key report`, `--trackio-dir ./runs/.trackio`). It renders in the
   panel's report view, stored with the run. `flow-control report` re-opens the
   finished run (`resume="must"`) and logs a `trackio.Markdown` — safe to run
   while the singleton panel is up.
