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

## Extension points & plugins

Most algorithmic choices are **open registries**, not closed enums: model
adapters, rewards (and reward normalizers), samplers' `shift`/`solver`, processor
tasks and presets, encoders, VAEs, timestep/loss weightings, advantage estimators,
EMA warmups, datasets/datasinks, serving task templates, and trainers. The
mechanism is `Registry` / `RegistryUnion` / `load_plugins` in
`flow_control/utils/registry.py`.

### Adding a component (in-tree)

Subclass the family base and register it at its definition site:

```python
from flow_control.rewards.base import BaseReward, reward_registry

@reward_registry.register("my_reward")
class MyReward(BaseReward):
    type: Literal["my_reward"] = "my_reward"
    ...
```

The field alias (e.g. `Reward = Annotated[BaseReward, RegistryUnion(reward_registry,
"type")]`) picks it up automatically — validation, discriminated-union dispatch, and
JSON schema all follow with no other edits. The registry tag must equal the member's
discriminator value (its `type` field, or `f"{arch}_{type}"` for model adapters).

### Out-of-tree / experimental components (plugins)

A component registers the same way from a module that is **not** imported by any
core `__init__`. A config opts in via a top-level `imports` key; `load_plugins`
imports the module (for its `@register` side effects) before the config is built:

```jsonc
{ "imports": ["flow_control.contrib.rgba_vae_training"], /* ... */ }
```

`flow_control/contrib/` is the tracked home for such components — experimental or
unpublished algorithms that ship in the repo but stay out of the core import graph
(currently `efficient_layered` and `rgba_vae_training`). They never affect a run
that doesn't `import` them, so the published core stays stable and parallel
experiments live in their own files without editing core unions.

Plugins are passed **explicitly, never via an env var**: loaded in `_dispatch`
(`flow_control/scripts/cli.py`) and `_run_child` (`flow_control/scripts/launch.py`)
before any config is built, and threaded into the spawn-based preprocess pipeline
workers as an explicit `plugin_modules` argument (`flow_control/utils/pipeline/`), so
plugin datasets/processors work inside workers too.

### Trainers

Trainers are a registry too: `trainer_registry` (in
`flow_control/training/mixins/base.py`, next to `BaseTrainer`) maps a config's
`launch.type` to a trainer class. Built-ins (`sft`/`grpo`/`nft`/`inference`) are
registered by `import_builtin_trainers()` (`flow_control/training/__init__.py`),
which the entry points call before resolving; plugin trainers (e.g. `vae` from
`flow_control.contrib.rgba_vae_training`) register via `imports`. A trainer
subclasses `BaseTrainer` (`flow_control/training/mixins/base.py`) and
overrides `run()`; it may also override `seed_checkpoint()` / `export_checkpoint()`
(the base seeds a transformer adapter and rejects export). `flow-control
launch`/`seed`/`export` dispatch through the registry uniformly.

### Editor schemas with plugins

`flow-control schema` emits core-only schemas. For a config that uses plugins, run
`flow-control schema --config <file>`: it loads that config's `imports` first, so the
generated schemas include the plugin members. Point the config's `$schema` at that
per-experiment output for full autocompletion.

## Multi-machine workflow

This is a solo project worked on from several machines at once. Keep `main`
linear: pull/rebase your work onto `origin/main`, never merge-commit.

The one rule that keeps this working: **machine-specific or in-progress
experiment state never gets committed to `main`.** It lives in the gitignored
`local/` tree (and `data/`, `demo/`, `draft/`). Library changes under
`flow_control/` and machine-neutral configs under `examples/` are the only things
that land on `main`. Because machine config is gitignored, you do not need a
per-machine branch — just work on `main` everywhere.
