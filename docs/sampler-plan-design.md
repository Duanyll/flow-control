# Sampler execution and extensions

Updated 2026-09-10 for sampler-rethink R1–R5 and A1–A4.

`Sampler` owns the sigma grid, solver, start, transforms, guidance, projectors,
and optional tiling. `sample(model, requests, observer=...)` is the single
sampling path. Each request has its own batch, negative batch, and RNG.
`SampleOutput` contains final latents and the **executed** start-sigma grid.

## Execution

1. Build the shifted/custom grid and solver plan.
2. Slice for SDEdit, then apply transforms using the request generator.
3. Initialize latents at the resulting plan's first sigma.
4. For each transition, apply whole-image `pre_transition` projectors once.
5. Drive solver generators together. Each yielded model evaluation expands
   named branches, evaluates tiles if configured, and assembles whole images.
6. Combine whole-image branch velocities, then apply `post_combine` projectors.
7. Resume each solver with the velocity, observe the completed transition,
   and advance its private runtime state.

`Transition` contains `(solver, sigma, sigma_next, eta)`. Execution position is
`StepContext.item_index / num_items`; solver and guidance history also live in
that context. Shared configuration objects hold no per-sample numerical state.
SA retains its multi-evaluation generator and cross-request batching.

Distributed ranks use one per-round rendezvous path. Unequal plan lengths or
solver/guidance configurations raise collectively. In particular, SDEdit with
resolution-dependent shifting can produce unequal sliced lengths; use matching
resolution groups. Branch variants use a common forward schedule across ranks;
missing branches receive dummy forwards, with zero-valued graph dependencies
when backward is required. Tile-count padding belongs to the tiled wrapper.

## Configuration examples

SDEdit starts from a clean batch tensor, selecting the first grid point at or
below `strength`. Noise interpolation uses that selected sigma:

```jsonc
"sampler": {
  "start": {"source": "clean_latents", "strength": 0.6},
  "steps": 30,
  "guidance": 4.5
}
```

An SDE window preserves each selected transition's eta and sets the others to
zero. Window indices refer to the sliced execution plan; the terminal step is
excluded. GRPO records stochastic steps automatically:

```jsonc
"rollout_sampler": {
  "solver": {"type": "flow", "eta": 0.7},
  "transforms": [{"type": "sde_window", "size": 3, "range": [2, 12]}]
}
```

First-order CFG++ supports Flow and DDIM, using each solver's own coefficient
and the transition's actual eta. It requires a negative branch even at scale
one or below; unsupported solvers raise:

```jsonc
"guidance": {"type": "cfg_pp", "inner": {"type": "cfg", "scale": 0.5}}
```

Named branches separate condition selection from model weights. Existing LoRA
adapters can be selected per step; a short list holds its last entry. `null`
uses the current weights; `"base"` disables adapters. Comparing two variants on
the same positive condition gives the constant-lambda signed combination:

```jsonc
"guidance": {
  "type": "cfg",
  "scale": 2.0,
  "positive_variant": ["default", "default", "base"],
  "negative_variant": "base",
  "negative_condition": "positive"
}
```

The variant context restores active adapters, disabled state, and parameter
trainability, including checkpoint recomputation. An outer base/reference
context takes precedence over branch choices. Loading/training two independent
weight sets and learning a density-ratio classifier are separate work.

Differential Diffusion is a projector consuming whole-image inpaint tensors:

```jsonc
"projectors": [{"type": "differential"}]
```

It preserves the previous pre-transition mask schedule. `post_combine` is the
second extension hook, called after each evaluation; no post-step blend is
introduced. Momentum remains available through
`"imports": ["flow_control.contrib.momentum_guidance"]`.

## Tiled evaluation

```jsonc
"tiled": {"tile_size": [1024, 1024], "overlap": [128, 128], "position": "local"}
```

Sizes are pixels and must align with the adapter's packed pixel stride.
Overlaps are normalized uniformly. Each raw branch is stitched before guidance
and post-combine projection; the solver draws one whole-image noise tensor.

T2I processor inputs may include `tiles`, a row-major list of individual prompt
inputs with each tile's actual `image_size`. The resulting `batch["tiles"]`
contains complete conditioning batches, including each tile's negative prompt.
Without that list all tiles inherit the full-image condition. Spatial control
and inpaint tensors are cropped to each tile; independent reference images keep
their own geometry.

Local coordinates restart per tile. Global coordinates currently support
FLUX.1 base, d-concat, n-concat and Fill; unsupported adapters raise for global
mode. The wrapper requires one packed BND image, and rejects layered/multi-image
latent layouts. It does not skip tiles based on a mask.

## Training and migration

GRPO's `training/grpo_sampling.py` owns collection, records, likelihood replay,
and mean/std reconstruction. It records every supported step with eta > 0 and
rejects empty stochastic trajectories or stateful guidance. Flow/DDIM/CPS/
Dance/Flash retain their existing likelihood/KL conventions, including Flash's
Gaussian approximation when clipping noise. Inference solvers do not compute
log probabilities. Replay and NFT use the same branch/combine/projector function
as sampling, with actual executed step metadata supplied by the rollout.

The KRepeat data sampler groups bucketed prompts so corresponding rank positions
share a resolution while preserving exactly K rollouts per prompt. Incompatible
bucket capacities raise; repeats are never silently padded. Unbucketed datasets
retain their prior selection semantics.

Migration: move `rollout_recipe[0].transforms` into `rollout_sampler.transforms`
and remove `record`; inference uses `sampler.start/transforms`. `recipe`, phases,
inversion, `from_previous`, and core replay APIs have been removed. Old config
keys fail validation. Regenerate editor schemas with `uv run flow-control schema`.
DDNM, dual-weight training, time travel, and SamplingPipeline are deferred.

Validation retains the 21 pre-refactor solver fixtures unchanged, and includes
cross-module tests for CFG++/replay, variants, tiling, and KRepeat, plus real
distributed CPU/GPU worker harnesses.
