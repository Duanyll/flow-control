# Output layout

A training run launched via `flow-control launch <config>` writes run artifacts under one stable directory, identified by a `run_id` that can survive preempt + Slurm requeue when `auto_resume=true`:

```
./runs/                                       # runs_root (config field, default "./runs")
├── .trackio/                                 # launch.trackio_dir default; one DB per experiment
│   └── <experiment_name>.sqlite
├── <experiment_name>/<run_id>/
│   ├── meta.json                             # config snapshot + flow_control_version
│   ├── metrics.jsonl                         # append-only; tail -F for live monitoring
│   ├── logs -> <LOG_DIR>/                    # symlink to rank logs for this launch
│   └── checkpoints/                          # DCP shards (see two tiers below)
│       ├── step_*/                           # archival: sparse cadence, kept up to max_checkpoints
│       └── rolling_step_*/                   # rolling: exactly one; time-gated preempt safety net
└── logs/slurm-<job_id>.{out,err}             # Slurm bootstrap chatter (sbatch only)

./logs/slurm-<job_id>/                        # rank logs under Slurm; appended across requeue
├── rank0000.log
├── rank0000.traceback.log
└── rankNNNN.log

./logs/local-<timestamp>-<pid>/               # rank logs for non-Slurm flow-control launch
├── rank0000.log
├── rank0000.traceback.log
└── rankNNNN.log
```

`run_id` format: `YYYYMMDDHHMMSS-<8hex>`. It can also be set explicitly in the config.

`flow_control.scripts.launch.run` no longer rewrites the training config. It only reads the `launch` section, sets process-level launch env such as `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, `LOG_DIR`, and `TRACKIO_DIR` (from `launch.trackio_dir`), then execs torchrun with the original config path. Run identity and `checkpoint_root="$auto"` are resolved inside the trainer after distributed initialization, with rank 0 broadcasting the chosen `run_id` to the other ranks.

Once the run directory exists, `LoggingMixin` creates a `logs` symlink inside it pointing to the active `LOG_DIR`, so browsing `runs/<exp>/<run_id>/logs/rank0000.log` reaches the lifecycle/rank log without duplicating log files. If code bypasses `flow-control launch` entirely, logging falls back to `./logs/` (or `./logs/slurm-<job_id>/` when Slurm env vars are present).

In configs, use `"checkpoint_root": "$auto"` to opt into the unified layout, and `"auto_resume": true` to resume from the newest complete checkpoint when `resume_from_dir` is unset. Explicit checkpoint paths still work.

Checkpoints come in two tiers (`CheckpointingMixin` in `flow_control/training/mixins/dcp.py`), both written only at clean boundaries: **archival** `step_*` on a sparse cadence (`checkpoint_interval`, in epochs for GRPO/NFT and steps for SFT/VAE), capped at `max_checkpoints`; and a single **rolling** `rolling_step_*` refreshed at most every `rolling_checkpoint_interval_seconds`. There is no preempt-time saving; on a hard kill / Slurm preempt the requeued job resumes from the newest complete checkpoint when `auto_resume=true`, so worst-case lost work is bounded by the rolling interval plus one boundary.

Cleanup: `rm -rf runs/<exp>/<run_id>` purges the run's jsonl/checkpoints but leaves trackio rows in `runs/.trackio/<exp>.sqlite` (cost of the shared project model). Use trackio's delete-run API to fully purge, or `rm runs/.trackio/<exp>.sqlite` to nuke the whole experiment's dashboard data.

Launch trackio dashboard against the project tree:
```bash
TRACKIO_DIR=./runs/.trackio uv run trackio show --project <experiment_name>
```
