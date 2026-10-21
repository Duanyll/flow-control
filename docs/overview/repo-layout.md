# Repo layout

Where files go, and which of them reach `main`. The rule that keeps one `main` usable from several machines at once: machine-specific or in-progress experiment state is never committed.

## Top-level Directories

| Directory | Tracked | Holds |
|---|---|---|
| `flow_control/` | yes | The library. `flow_control/contrib/` holds tracked plugins. |
| `examples/` | yes | Machine-neutral reference configs (jsonc), prompt lists, example plugins (`examples/plugins/`), Slurm templates (`examples/slurm/`). Endpoints are placeholders such as `127.0.0.1`, device counts are small, and everything runs on any machine once its data is staged. |
| `tests/`, `docs/`, `frontend/` | yes | Tests, these docs, the Trackio media frontend. |
| `schema/` | no (generated) | JSON schemas for editor completion, written by `flow-control schema` ([config-plugins](config-plugins.md)). |
| `local/` | no | Machine-specific launchers (`local/slurm/`) and in-progress work (`local/<project>/`): configs, plugins, notes. |
| `data/` | no | Datasets and preprocessed caches ([data](../data/data.md)). |
| `demo/` | no | Outputs of demo pipelines. |
| `draft/` | no | Scratch scripts, benchmark dumps, throwaway checkpoints. |
| `runs/`, `logs/` | no | Run artifacts and rank logs ([outputs](../training/outputs.md)). |

Paths inside configs are relative to the repo root, so a `local/` config references `examples/.../preprocess.jsonc` and `./data/...` directly. `local/` depends on `examples/`, never the reverse.

To tinker, copy an `examples/` config into `local/<project>/` and edit it there. Once it stabilizes, fold its machine-neutral parts back into `examples/` and delete the local copy.

## Where Code Goes

| Work | Location | Activated by |
|---|---|---|
| Published algorithms and generic tools (adapters for released models, samplers, rewards anyone can use) | `flow_control/` core | always imported |
| Unpublished work that needs version control | `flow_control/contrib/<name>/` (e.g. `efficient_layered`, `iqa`, `momentum_guidance`, `rgba_vae_training`) | `"imports": ["flow_control.contrib.<name>"]` |
| Unpublished work without version control | `local/<project>/` | `"imports": ["local.<project>.<module>"]` |

Contrib and local code reaches the core only through registries; the mechanics are in [config-plugins](config-plugins.md). An experiment that fits an extension point adds files there and leaves the core untouched. Core modules never import `flow_control.contrib`.

## Branches

Keep `main` linear: rebase or squash onto `origin/main`, never a merge commit. Library changes under `flow_control/` and machine-neutral configs under `examples/` are what lands on `main`; everything machine-specific stays in the gitignored trees, so every machine works on `main` without a per-machine branch.
