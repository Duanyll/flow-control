# Sampler execution and extensions

Updated 2026-09-12 for composable predictors with binding-local state.

`Sampler` owns the sigma grid, solver, start, transforms, guidance and projectors.
The processor stores the tile layout in each processed batch.
`sample(model, requests, collector=...)` accepts a lazy iterable and yields completed
`SampleRun` objects. Each request has its own batch, negative batch and RNG.
The result exposes final latents through `run.ctx.latents` and the **executed**
plan through `run.plan`; completion order may differ from submission order.

The [batch dictionary contract](batch-contract.md) lists data fields, coordinate
spaces and their producers/consumers. Execution state remains outside the batch.

## Execution

1. Build the shifted/custom grid and solver plan. Resolution-dependent shift
   reads `batch["model_image_size"]` (the size one forward sees, e.g. a tile)
   and falls back to `image_size`.
2. Slice for SDEdit, then apply transforms using the request generator.
3. Initialize latents at the resulting plan's first sigma.
4. Bind the configured prediction tree once for this run. For each transition,
   apply whole-image `pre_transition` projectors once.
5. The solver calls its predictor with `yield from predict(request, ctx)`.
   Each component evaluates its children and finishes its own calculation in
   the same generator. Independent children join with `yield from gather(...)`;
   the model leaf yields `ModelCall(batch, timestep, variant)` objects.
   `Executor` batches calls
   across runs, using the same variant order on every rank. The adapter chunks
   these calls by `model.micro_batch_size`, collates compatible inputs, and
   falls back to sequential forwards when shapes differ.
6. Child results return through the prediction tree: tiling stitches, CFG guides,
   Momentum updates its EMA, in the order configured. Once the root returns a
   whole-image velocity, apply `post_combine` projectors.
7. Resume each solver with the velocity, pass the completed `StepRecord` to
   the optional caller collector `(run, step)`, and advance private state.
   Sampling itself keeps no step history. GRPO retains only stochastic
   latents/log probabilities; serving uses the callback for step progress.

`Transition` contains `(solver, sigma, sigma_next, eta)`. Execution position is
`StepContext.item_index / num_items`; solver history lives in that context.
Prediction history belongs to closures created by `bind(batch, negative_batch)`.
Shared configuration objects hold no per-sample numerical state.
SA retains its multi-evaluation generator and cross-request batching.

The common execution protocol is `Calls[T]`: yield a list of model calls,
receive a list of fp32 velocities in the same order, return `T`. Local `gather`
joins any number of these generators, including nested joins and children that
finish without a model call. It has no model, microbatch, or collective logic.
Only `Executor` drives physical execution. `SampleRun` knows neither CFG branches
nor tile geometry, and solvers need no special handling for either.

The sampling stream admits new runs while its pending calls are below one
adapter microbatch (one tiled run can exceed that window). Training `evaluate()`
drives all supplied logical items together, with physical forwards still chunked.
Each round's all-reduce checks whether **any** rank has work; drained ranks keep
participating with empty lists. Unequal request, plan and tile counts are legal.
All ranks must use the same configured variant list and adapter microbatch limit.
The adapter aligns chunk/forward counts and dense/fallback decisions. A rank
with no cached dummy receives a detached sample through a cold-start collective;
subsequent rounds reuse its local cache. Under autograd, the executor combines
zero-valued dummy dependencies across variants into a local real output. A
training rank must have a local evaluation to carry that graph into backward.

Precision is an interface contract: adapter computation may use bf16 or lower,
but returns fp32 velocities. Start, guidance, projectors, tile accumulation,
solver state and loss arithmetic use fp32. Low-precision stored latents, teacher
predictions and GRPO records are promoted before arithmetic. Final sampler
latents are cast to the input latent storage dtype. Each `guided_velocity()`
evaluation gets fresh state even when several evaluations share one run.

## Configuration examples

SDEdit starts from a condition image's latents, selecting the first grid point
at or below `strength`. Noise interpolation uses that selected sigma.
`start.source` is a [condition-image selector](batch-contract.md#condition-image-roles)
defaulting to `noisy_latents`, or to `clean` once `strength` is set:

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
"guidance": {"type": "cfg_pp", "scale": 0.5}
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

Differential Diffusion is a projector consuming the whole-image inpaint mask
and a selectable source, defaulting to the `inpaint` condition image:

```jsonc
"projectors": [{"type": "differential", "source": "inpaint"}]
```

It preserves the previous pre-transition mask schedule. `post_combine` is the
second extension hook, called after each evaluation; no post-step blend is
introduced.

## Composition and state

The existing `guidance` field now accepts a recursive prediction tree. Core
nodes are `model`, `tiled`, `cfg`, and `cfg_pp`; the Momentum plugin adds
`momentum`. All use `bind(batch, negative_batch) -> Predictor`, where
`Predictor(request, ctx) -> Calls[Tensor]`. Configuration declares children and
execution requirements; a bound predictor owns its runtime history. Actual
evaluation latents always come from `request.latents`, including SA substeps
and tile slices, rather than the step-start latent in `ctx`.

For compatibility, CFG's default child is `Tiled(Model)`: numeric guidance still
handles tiled batches, and its renorm runs after whole-image reconstruction.
This is a default configuration choice; CFG's execution never reads tile data.
Set `inner="model"` to call the model directly, or place CFG inside tiling to
guide each tile before stitching:

```jsonc
"guidance": {
  "type": "tiled",
  "inner": {"type": "cfg", "scale": 4.5, "renorm": true, "inner": "model"}
}
```

Momentum composes with CFG instead of inheriting its configuration or methods:

```jsonc
"imports": ["flow_control.contrib.momentum_guidance"],
"sampler": {
  "guidance": {
    "type": "momentum", "alpha": 0.5, "beta": 0.25,
    "inner": {"type": "cfg", "scale": 4.5}
  }
}
```

`Momentum(CFG(...))` tracks the combined velocity. Conversely,
`CFG(Momentum(Model))` binds independent positive and negative histories, and
`Tiled(Momentum(...))` binds one history per tile. All bindings are separate
across samples, even when they share the same config instance. A branch's
history follows that logical branch across a variant schedule. Momentum updates
after every child evaluation, so SA's first transition updates twice. There is
no branch-key dictionary or mutable state in the config. A single Momentum
binding cannot be evaluated concurrently: ambiguous update ordering raises;
independent branches must bind separate predictors.

These orders are meaningful but need not be equivalent, particularly with
renorm. CFG++ also follows the specified order: `Momentum(CFG++)` applies EMA
extrapolation to CFG++'s effective velocity, while `CFG++(Momentum(Model))`
first updates the two condition histories and then applies CFG++'s
solver-dependent conversion. Neither denotes a separate momentum operation
on just the guided term while keeping a raw unconditional term.

CFG binds its child once per condition, so a child requesting another negative
condition, including nested CFG, is rejected instead of guessing its meaning.
CFG++ requires both conditions and a supported Flow/DDIM transition. Generic
tree traversal collects child weight variants and detects stateful nodes for
independent training/replay; the executor never dispatches on a concrete algorithm class.

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

- `batch["tiling"]`: a serialized `TileLayout` dictionary (`tile_size`, `overlap`, `stride`).
- `batch["model_image_size"]`: the actual tile size. Resolution-dependent shift
  reads it, so a 4096-pixel output in 1024-pixel tiles gets the 1024-pixel grid.
- `batch["tiles"]`: complete row-major per-tile conditions. Inputs may supply a
  `tiles` list with individual prompts and negative prompts; otherwise the
  global encoded condition is shared by every tile without extra encoder calls
  (the list may also be omitted entirely). With `save_negative=true` the
  per-tile negative conditions go to `batch["negative"]["tiles"]`, so
  `get_negative_batch` needs no tile logic.

`TiledPrediction` in `samplers/tiling.py` owns tile expansion and merging for
sampling and training trees that include it. It reads metadata, cuts
`noisy_latents` on the token grid into per-tile batches (the tile's condition
plus its latent slice and `image_size`), passes ordinary batches through, runs
child predictors through the executor and adapter, and stitches their results
with `stitch_tiles`. CFG may run inside or outside this node. Stitching feathers
only edges shared with a neighbour (Hann ramps over the actual overlap) and
normalizes by the total weight, so constant inputs reconstruct exactly; the
solver draws one whole-image noise tensor. Adapters need no tile hooks and
positional coordinates restart per tile.

## Training and migration

SFT/AWM/RAM/NFT/GRPO require a separate `train_predictor` tree. Nodes have no
training-mode flag: omitting a wrapper is its no-op, and different parameters
are ordinary configuration. Current, old/EMA, reference and precomputed targets
all use this tree; rollout and validation keep their own sampler guidance.

```jsonc
"rollout_sampler": {"steps": 20, "guidance": 4.5},
"train_predictor": "tiled" // conditional Tiled(Model), without CFG
```

Use `"train_predictor": "model"` to call the model directly even when batch
metadata describes tiles. To keep training CFG, specify for example
`{"type":"cfg","scale":4.5,"inner":"tiled"}`. Training resolves its own
negative conditions regardless of whether rollout needed them; missing required
negative data raises. SFT dropout retains the original negative condition.

`BasePrediction.velocity(batch, timestep, negative_batch)` evaluates any tree
at an independent timestep, replacing the removed `conditional_velocity`.
SFT and continuous training timesteps (`train_timesteps: continuous`, the RAM
preset) use this through `TrainingPredictionMixin.predict_training`; they have
no solver transition and use index 0 for variant schedules. Consequently CFG++
is undefined there and raises. Grid training timesteps (`train_timesteps:
grid`, the NFT and AWM presets) and GRPO replay instead use
`make_run(plan=executed_plan, predictor=train_predictor)` plus
`guided_velocity(x_t, grid_index)` to preserve actual sigma, successor sigma,
eta, execution index and post-projectors.

Training timesteps are independent, so stateful training nodes are rejected;
Momentum would otherwise reset on every call and silently become identity.
Stateful rollout trees remain valid, including with GRPO collection.

GRPO's `training/grpo_sampling.py` owns collection, records, likelihood replay,
and mean/std reconstruction. It records every supported step with eta > 0 and
rejects empty stochastic trajectories. Flow/DDIM/CPS/
Dance/Flash retain their existing likelihood/KL conventions, including Flash's
Gaussian approximation when clipping noise. Inference solvers do not compute
log probabilities. GRPO keeps the **actual rollout's recorded score** as the
ratio denominator and evaluates the current/reference model through the training
tree. Different trees can produce a non-unit initial ratio and nonzero clipfrac;
the denominator is not recomputed to hide that difference. This preserves the
existing GRPO surrogate (dimension-averaged log probabilities and the existing
CPS/Flash conventions), rather than introducing a new importance-weight formula.
All replay items/ranks share the configured training tree and grid length.

Migration: add `train_predictor` explicitly. `"tiled"` preserves the previous
SFT/AWM/RAM conditional path; copy the old rollout guidance into this field to
preserve stateless NFT/GRPO behavior. Examples have been migrated. Include tiling
in both trees when training and sampling should use the same tile evaluation.

Rollout prompts come from `RowCursor` (`chunked`: consecutive rows of the
cost-grouped plan, no repeats within a pass over the dataset; `independent`: a
fresh random subset per epoch) and `expand_rollouts` splits the `M × K` rollouts
across ranks with a stride, so the i-th rollout of every rank comes from
cost-adjacent prompts while every prompt keeps exactly K rollouts. Short plan
groups are padded at plan time (`__padding__`) and the cursor skips the padding;
there are no buckets. Pairwise rewards group completions by `__key__` in the
reward executor; `expand_rollouts(whole_prompts=True)` keeps all K rollouts of a
prompt on one rank and incomplete groups raise instead of comparing different
prompts. See the data section of [modules.md](modules.md).

Migration: move `rollout_recipe[0].transforms` into `rollout_sampler.transforms`
and remove `record`; inference uses `sampler.start/transforms`. `recipe`, phases,
inversion, `from_previous`, and core replay APIs have been removed. Old config
keys fail validation. Regenerate editor schemas with `uv run flow-control schema`.
CFG++ now owns its CFG parameters directly: replace
`{"type":"cfg_pp","inner":{"type":"cfg","scale":0.5}}` with
`{"type":"cfg_pp","scale":0.5}`. `inner` now means the predictor evaluated
separately for each condition. Move Momentum's old inherited CFG fields under
its `inner` CFG object. Predictor plugins register with `prediction_registry`
and implement `bind`, replacing `guidance_registry` and `branches/combine`.
For tiled configurations, move the old `sampler.tiled` fields to the processor
and select `task="tiled_t2i"`; regenerate preprocessed batches to store their
layout. Plain `t2i` handles ordinary text-to-image preprocessing.

DDNM, dual-weight training, time travel and SamplingPipeline are deferred. The adapter
exposes `model.micro_batch_size` as its physical forward limit. Remove obsolete
`micro_batch_size` from inference and `validation_micro_batch_size` /
`rollout_micro_batch_size` from trainers, moving their intended forward limit
under `model`. `train_micro_batch_size` still counts logical loss items per
backward; branch/tile expansion is independently chunked by the adapter.

Validation retains the 21 pre-refactor solver fixtures unchanged, and includes
cross-module tests for CFG++/replay, variants, tiling, and the rollout cursor, plus real
distributed CPU/GPU worker harnesses.

## Planned DDNM extensions

These components are not implemented yet. The intended split is:

- `DDNMProjector`: correct a whole-image clean estimate in `post_combine`, after
  the prediction tree has returned its velocity. Tiling and guidance remain
  ordinary predictor composition; the projector needs no tile knowledge.
- `DDNMPlusSampler`: own the supported trajectory, noise adjustment and time
  travel. It can restrict solver/parameterization choices to those for which its
  formulas are defined; it need not support every registered solver.

The projector's velocity conversion follows directly from this repository's
flow convention, `x_sigma = (1 - sigma) * x0 + sigma * noise`. For `sigma > 0`,
with linear degradation `A`, pseudoinverse `A_dagger`, and a measurement `y` in
the **same coordinate convention**:

```text
x0_hat = x_sigma - sigma * v
delta = A_dagger(y - A(x0_hat))
x0_projected = x0_hat + delta
v_projected = (x_sigma - x0_projected) / sigma = v - delta / sigma
```

Thus the existing velocity-returning projector interface can express the
clean-estimate projection without changing solver or executor interfaces.
`sigma == 0` requires an explicit endpoint policy instead of this division.
This algebra alone does not specify a DDNM+ transition or its noise covariance.

The [batch contract's DDNM boundary](batch-contract.md#ddnm-inputoutput-boundary-planned)
records the remaining observation, normalization and noise-unit requirements.
Sampler registration and richer/data-dependent transition streams are still
future work; existing replay and plan consumers need an explicit compatibility
contract when those streams are introduced.
