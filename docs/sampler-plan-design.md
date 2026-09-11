# Sampler execution and extensions

Updated 2026-09-11 for sampler-rethink R1–R5/A1–A4 and processor-owned tiling.

`Sampler` owns the sigma grid, solver, start, transforms, guidance and projectors.
The processor stores tile layout and blend settings in each processed batch. `sample(model, requests, observer=...)` is the single
sampling path. Each request has its own batch, negative batch, and RNG.
`SampleOutput` contains final latents and the **executed** start-sigma grid.

## Execution

1. Build the shifted/custom grid and solver plan. Resolution-dependent shift
   uses the actual tile size when the batch contains tiling metadata.
2. Slice for SDEdit, then apply transforms using the request generator.
3. Initialize latents at the resulting plan's first sigma.
4. For each transition, apply whole-image `pre_transition` projectors once.
5. Drive solver generators together. Each yielded model evaluation expands
   named branches, reads each batch's tiling metadata, and assembles whole images.
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

Tiling is a processor task. Its settings are top-level processor fields:

```jsonc
"processor": {
  "task": "tiled_t2i",
  "preset": "flux1",
  "tile_size": [1024, 1024],
  "overlap": [128, 128],
  "position": "local",
  "blend": "uniform",
  "save_negative": true
}
```

`TileConfig` in `processors/tiles.py` provides the shared layout rules. Sizes
are pixels and must align with the adapter's packed pixel stride. Each actual
tile dimension is the smaller of the configured size and the image dimension;
origins are row-major and the final tile reaches the image edge.

The processor preserves the full output `image_size` and writes a plain
`batch["tiling"]` dictionary containing `tile_size`, `overlap`, `position` and
`blend`. Inputs may supply a row-major `tiles` list with individual prompts and
negative prompts. Set processor `save_negative=true` to encode and retain those
negative prompts; its default is `false`. The processor derives each tile's
`image_size`; callers need not repeat it. Encoded tile dictionaries contain complete conditions. Without
individual prompts the processor shares the global encoded conditions across
tiles, avoiding repeated encoder calls. Negative batches retain the same layout
and use the corresponding tile's negative condition.

`TiledModel(model)` has no separate configuration. `Sampler.sample()`,
`get_guided_velocity()` and GRPO replay automatically apply this wrapper. It
reads each request's metadata, supports different layouts in one logical batch,
and passes ordinary batches through. Plain batches count as one model input
when synchronizing tile counts across ranks. The optional `batch["tiles"]` list
can be omitted when every tile shares the full-image condition.

Each raw branch is stitched before guidance and post-combine projection, and
the solver draws one whole-image noise tensor. The three blend modes normalize
by the total covering weight:

- `uniform` (default): equal weights.
- `gaussian`: a separable Gaussian with standard deviation one quarter of each
  tile dimension.
- `hann`: a separable Hann window evaluated at cell centers, so boundary weights
  stay positive even where only one tile covers a pixel.

Dynamic shift uses the actual model input tile rather than the full output
resolution. With 1024-pixel tiles, a 4096-pixel output receives the same shift as
a real 1024-pixel sample. `latent_length_from="actual"` scales packed token
count by tile area/full-image area; `"image_size"` uses tile area with the
existing 256-pixel divisor. An image smaller than a tile uses its own size.

Spatial control and inpaint tensors are cropped to each tile; independent
reference images retain their own geometry. Local coordinates restart per tile.
Global coordinates currently support FLUX.1 base, d-concat, n-concat and Fill;
unsupported adapters raise for global mode. The wrapper requires one packed BND
image, rejects layered/multi-image latent layouts, and does not skip masked tiles.

## Training and migration

GRPO's `training/grpo_sampling.py` owns collection, records, likelihood replay,
and mean/std reconstruction. It records every supported step with eta > 0 and
rejects empty stochastic trajectories or stateful guidance. Flow/DDIM/CPS/
Dance/Flash retain their existing likelihood/KL conventions, including Flash's
Gaussian approximation when clipping noise. Inference solvers do not compute
log probabilities. Replay and NFT use the same branch/combine/projector function
as sampling, with actual executed step metadata supplied by the rollout.

Automatic tiling covers sampling, guided evaluation, GRPO replay and NFT guided
training predictions. Direct SFT/AWM/RAM training forwards still call the raw
adapter; batch metadata does not switch those forwards to tiled training. The
next Pipeline round will unify this boundary.

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
