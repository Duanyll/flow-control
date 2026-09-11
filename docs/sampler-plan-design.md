# Sampler execution and extensions

Updated 2026-09-11 for sampler-rethink R1–R5/A1–A4 and processor-owned tiling.

`Sampler` owns the sigma grid, solver, start, transforms, guidance and projectors.
The processor stores the tile layout in each processed batch. `sample(model, requests, observer=...)` is the single
sampling path. Each request has its own batch, negative batch, and RNG.
`SampleOutput` contains final latents and the **executed** start-sigma grid.

## Execution

1. Build the shifted/custom grid and solver plan. Resolution-dependent shift
   reads `batch["model_image_size"]` (the size one forward sees, e.g. a tile)
   and falls back to `image_size`.
2. Slice for SDEdit, then apply transforms using the request generator.
3. Initialize latents at the resulting plan's first sigma.
4. For each transition, apply whole-image `pre_transition` projectors once.
5. Drive solver generators together. Each yielded model evaluation expands
   named branches; every branch reaches the model through `predict_velocity`,
   the single leaf call, which expands tiled batches, runs the model once and
   stitches whole images back.
6. Combine whole-image branch velocities, then apply `post_combine` projectors.
7. Resume each solver with the velocity, observe the completed transition,
   and advance its private runtime state.

`Transition` contains `(solver, sigma, sigma_next, eta)`. Execution position is
`StepContext.item_index / num_items`; solver and guidance history also live in
that context. Shared configuration objects hold no per-sample numerical state.
SA retains its multi-evaluation generator and cross-request batching.

Distributed ranks use one per-round rendezvous path: each round synchronizes
its evaluation count and raises when ranks disagree. No configuration
fingerprint is exchanged, so unequal plan lengths (SDEdit with
resolution-dependent shifting, mixed resolutions in one microbatch) surface as
that count mismatch; use matching resolution groups. The branch schedule is
derived from the guidance configuration alone, so ranks never exchange their
local branches; missing branches receive dummy forwards, with zero-valued graph
dependencies when backward is required. Tile counts are never padded: unequal
counts across ranks fail in the adapter's collation sync.

## Configuration examples

SDEdit starts from a clean batch tensor, selecting the first grid point at or
below `strength`. Noise interpolation uses that selected sigma. `start.source`
defaults to `noisy_latents`, or to `clean_latents` once `strength` is set:

```jsonc
"sampler": {
  "start": {"strength": 0.6},
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

Tiling is a processor task. Its settings are top-level processor fields:

```jsonc
"processor": {
  "task": "tiled_t2i",
  "preset": "flux1",
  "tile_size": 1024,
  "overlap": 128,
  "save_negative": true
}
```

`TileConfig` in `processors/tiles.py` holds the square pixel `tile_size` and
`overlap`; both must be multiples of the adapter's packed pixel stride
(`patch_size * vae_scale_factor`), as must the image. Layouts are planned on
the token grid by `utils/tiling.py` (`plan_tiles`): an axis no longer than the
tile is one tile of its own length; otherwise tiles of exactly `tile_size` are
spread evenly with the first and last flush to the edges, so adjacent overlaps
are at least `overlap` and every origin is token-aligned.

The processor preserves the full output `image_size` and writes:

- `batch["tiling"]`: a `TileLayout` (`tile_size`, `overlap`, `stride`).
- `batch["model_image_size"]`: the actual tile size. Resolution-dependent shift
  reads it, so a 4096-pixel output in 1024-pixel tiles gets the 1024-pixel grid.
- `batch["tiles"]`: complete row-major per-tile conditions. Inputs may supply a
  `tiles` list with individual prompts and negative prompts; otherwise the
  global encoded condition is shared by every tile without extra encoder calls
  (the list may also be omitted entirely). With `save_negative=true` the
  per-tile negative conditions go to `batch["negative"]["tiles"]`, so
  `get_negative_batch` needs no tile logic.

`samplers/evaluation.py:predict_velocity` is the single leaf call for sampling,
`get_guided_velocity()` and GRPO replay. It reads each batch's metadata, cuts
`noisy_latents` on the token grid into per-tile batches (the tile's condition
plus its latent slice and `image_size`), passes ordinary batches through, runs
one `predict_velocity_batched` over everything, and stitches each raw branch
back with `stitch_tiles` before guidance and `post_combine`. Stitching feathers
only edges shared with a neighbour (Hann ramps over the actual overlap) and
normalizes by the total weight, so constant inputs reconstruct exactly; the
solver draws one whole-image noise tensor. Adapters need no tile hooks and
positional coordinates restart per tile.

## Training and migration

GRPO's `training/grpo_sampling.py` owns collection, records, likelihood replay,
and mean/std reconstruction. It records every supported step with eta > 0 and
rejects empty stochastic trajectories or stateful guidance. Flow/DDIM/CPS/
Dance/Flash retain their existing likelihood/KL conventions, including Flash's
Gaussian approximation when clipping noise. Inference solvers do not compute
log probabilities. Replay and NFT use the same branch/combine/projector function
as sampling, with actual executed step metadata supplied by the rollout.

Tiling applies wherever the model is reached through `predict_velocity`:
sampling, guided evaluation, GRPO replay, NFT guided training predictions and
the direct SFT/AWM/RAM training forwards, so rollout and training see the same
tiles.

The KRepeat data sampler groups bucketed prompts so corresponding rank positions
share a resolution while preserving exactly K rollouts per prompt. Incompatible
bucket capacities raise; repeats are never silently padded. Unbucketed datasets
retain their prior selection semantics.

Migration: move `rollout_recipe[0].transforms` into `rollout_sampler.transforms`
and remove `record`; inference uses `sampler.start/transforms`. `recipe`, phases,
inversion, `from_previous`, and core replay APIs have been removed. Old config
keys fail validation. Regenerate editor schemas with `uv run flow-control schema`.
For tiled configurations, move the old `sampler.tiled` fields to the processor
and select `task="tiled_t2i"`; regenerate preprocessed batches to store their
layout. Plain `t2i` handles ordinary text-to-image preprocessing.

DDNM, dual-weight training, time travel and SamplingPipeline are deferred. The
next Pipeline round will define one microbatch upper limit that the container
can enforce internally and outer callers can read. This tiling correction adds
no separate microbatch control.

Validation retains the 21 pre-refactor solver fixtures unchanged, and includes
cross-module tests for CFG++/replay, variants, tiling, and KRepeat, plus real
distributed CPU/GPU worker harnesses.
