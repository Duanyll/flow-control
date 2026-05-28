# Output layout

A training run launched via `flow-control launch <config>` writes everything for that run under one stable directory, identified by a `run_id` that survives preempt + Slurm requeue:

```
./runs/                                       # runs_root (config field, default "./runs")
├── .trackio/                                 # TRACKIO_DIR — one .sqlite per experiment (shared)
│   └── <experiment_name>.sqlite
├── <experiment_name>/<run_id>/
│   ├── meta.json                             # config snapshot + flow_control_version
│   ├── metrics.jsonl                         # append-only; tail -F for live monitoring
│   ├── events.jsonl                          # run_start (attempt metadata), image_logged, run_finish
│   ├── resolved_config.json                  # config with $auto already expanded
│   ├── attempts/<attempt_id>/                # one dir per OS invocation
│   │   ├── rank0000.log                      # structured rank logs (see flow_control/utils/logging.py)
│   │   ├── rank0000.traceback.log
│   │   └── rankNNNN.log
│   └── checkpoints/step_*/                   # DCP shards
└── logs/slurm-<job_id>.{out,err}             # Slurm bootstrap chatter (sbatch only)
```

`run_id` format: `YYYYMMDDHHMMSS-<8hex>`. `attempt_id` is `slurm-<JOB_ID>` under Slurm, else `local-<timestamp>-<pid>`.

Resolution happens once in `flow_control.scripts.launch.run` before exec-ing torchrun, then propagates to workers via env vars (`FLOW_CONTROL_RUN_ID`, `FLOW_CONTROL_ATTEMPT_ID`, `LOG_DIR`, `TRACKIO_DIR`) and a resolved-config tempfile.

In configs, use `"checkpoint_root": "$auto"` to opt into the unified layout. Explicit paths still work for back-compat.

Cleanup: `rm -rf runs/<exp>/<run_id>` purges the run's jsonl/logs/checkpoints but leaves trackio rows in `runs/.trackio/<exp>.sqlite` (cost of the shared project model). Use trackio's delete-run API to fully purge, or `rm runs/.trackio/<exp>.sqlite` to nuke the whole experiment's dashboard data.

Launch trackio dashboard against the project tree:
```bash
TRACKIO_DIR=./runs/.trackio uv run trackio show --project <experiment_name>
```
