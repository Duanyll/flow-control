# Repo layout & multi-machine workflow

## Top-level directories

Only `examples/` is tracked; the rest are gitignored (see `.gitignore`).

- `examples/` — **tracked**, machine-neutral canonical reference configs (jsonc),
  prompt lists, and small example scripts. Endpoints use placeholders like
  `127.0.0.1`; device counts use small defaults. Anything here must run on any
  machine after data is staged. This is the only config tree that lands on `main`.
- `local/` — **gitignored**, machine-specific launchers and in-progress
  experiment/tuning configs:
  - `local/slurm/` — `sbatch` launchers bound to a node/reservation/paths.
  - `local/<experiment-slug>/` — self-contained working configs for one run you
    are actively tuning. Paths are repo-root-relative, so these freely reference
    `examples/.../preprocess.jsonc` and `./data/...`. `local/` depends on
    `examples/`, never the reverse.

  To tinker: copy an `examples/` config into `local/<slug>/` and edit there. When
  it stabilizes, fold the machine-neutral parts back into `examples/` and delete
  the local copy.
- `data/` — datasets and preprocessed latent/text caches.
- `demo/` — generated outputs (`.pt`/`.png`) from running demo pipelines.
- `draft/` — one-off scratch scripts, benchmark dumps, notes, throwaway checkpoints.

For where a training *run* writes its artifacts (`runs/`, logs, checkpoints), see
`docs/output-layout.md`.

## Multi-machine workflow

This is a solo project worked on from several machines at once. Keep `main`
linear: pull/rebase your work onto `origin/main`, never merge-commit.

The one rule that keeps this working: **machine-specific or in-progress
experiment state never gets committed to `main`.** It lives in the gitignored
`local/` tree (and `data/`, `demo/`, `draft/`). Library changes under
`flow_control/` and machine-neutral configs under `examples/` are the only things
that land on `main`. Because machine config is gitignored, you do not need a
per-machine branch — just work on `main` everywhere.
