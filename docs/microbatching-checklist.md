# Microbatching Implementation Checklist

Status: core implementation complete; focused CPU, distributed, FSDP2, real
model, end-to-end GRPO, and a 512x512 SD3.5 AWM throughput sweep passed.

The real-model throughput measurements below predate the generator executor;
they are retained as historical results, not remeasured in this refactor.

This is the living implementation record for fixed-shape microbatching. It also
records the stable boundary intended for future padding and sequence packing.

## Scope and decisions

- [x] Treat each existing `[1, N, D]` item as one logical sample.
- [x] Let adapters own physical collation instead of the DataLoader.
- [x] Expose list-based adapter/replay APIs and a lazy sampler stream.
- [x] Remove singleton compatibility overloads and migrate all in-tree callers.
- [x] Use dense `[M, N, D]` forwards only for adapter-declared compatible fields.
- [x] Fall back to one forward per logical sample when inputs are incompatible.
- [x] Synchronize dense/fallback decisions and forward counts across distributed ranks.
- [x] Preserve independent per-request sigma grids, solver states, generators,
      transformed plans (SDE windows), and outputs.
- [x] Keep processor encode/decode, reward evaluation, and annotation per sample.
- [x] (superseded by plan-as-data) SA-Solver now batches across requests through
      the plan executor's rendezvous loop; no sequential special case remains.
- [x] Keep sequence packing out of the original microbatching change. Sampler-rethink
      expands tiled rows inside `TiledPrediction` into shared leaf generators; the `tiled_t2i` processor writes the layout into each
      row and the sampler has no independent tile configuration. Unequal
      tile counts are padded with dummy forwards by the adapter.
- [x] Do not require bitwise identity between dense and singleton GPU kernels.

## Stable APIs

### Adapter

```python
def predict_velocity_batched(
    self,
    batches: list[TBatch],
    timesteps: list[torch.Tensor],
    *,
    dummy_outputs: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Return one singleton-leading fp32 velocity per logical sample."""
```

- [x] Accept empty lists; reject unequal argument lengths and non-singleton logical inputs.
- [x] Cast and move inputs through the adapter's existing dtype/device path.
- [x] Chunk by `model.micro_batch_size`, align chunk/forward counts, and
      provide a dummy through a collective when a rank has no cached sample.
- [x] Fold autograd dummy outputs into real results; the executor collects
      dependencies across variants with no local real sample.
- [x] Preserve input order and singleton leading output dimensions.
- [x] Use `torch.cat(..., dim=0)` for singleton-leading tensor fields.
- [x] Ignore undeclared dataset metadata when deciding dense compatibility.
- [x] Recursively collate homogeneous tensor lists such as reference latents.
- [x] Require shared topology metadata such as image/reference sizes to match.
- [x] Validate dense and fallback output leading dimensions.
- [x] Keep timestep preparation overridable; HiDream-O1 retains float32 time.

The default implementation is conservative. An adapter opts into the dense
collator with `supports_dense_batching` and `dense_batch_fields`, or can
override the collation/public prediction method for a different physical
representation.

### Sampler

```python
@dataclass(slots=True)
class SampleRequest:
    row: Batch
    negative_row: Batch | None = None
    generator: torch.Generator | None = None


def sample(
    self,
    model: SamplerModel,
    requests: Iterable[SampleRequest],
    *,
    collector: StepCollector | None = None,
) -> Iterator[SampleRun]:
    ...
```

`sample()` executes all sampling configurations through one path. `start`
handles direct latent initialization and SDEdit; `transforms` applies SDE
windows to each request's plan. Named branch evaluation, row-described tile forwards,
whole-image guidance/projectors, and solver steps preserve cross-request batching
and FSDP alignment. Completed runs expose `ctx.latents` and `plan`; the
optional caller collector `(run, step)` consumes each completed transition
without the sampler retaining step tensors.
Dynamic shift reads `model_image_size` (the tile size for tiled rows), so a
4k output with 1k tiles follows the same sigma schedule as a real 1k model
input. Each request can carry its own layout; tiles are feathered with the
shared `utils.tiling.stitch_tiles` window; ordinary rows pass through.
Tile-specific negative prompts require processor `save_negative=true`
(default: false). Training explicitly configures `train_predictor`; include
`tiled` in both trees when rollout and training should use the same tiling.
There is no implicit training tiling or training-mode dispatch.

- [x] Yield one `SampleRun` per request in completion order without stacking outputs.
- [x] Build one shifted/custom sigma schedule per request.
- [x] Keep solver state in `StepContext` and prediction history in per-binding
      closures (independent across requests, CFG branches and tiles); SDE
      windows are per-request plan data and GRPO recording is a caller collector.
- [x] Batch the expensive conditional model forward once per rendezvous round.
- [x] Run solver math independently per request.
- [x] Support no-negative, all-negative, and mixed CFG request lists.
- [x] Execute the same configured variant order on every rank; validate required
      negative conditions while constructing runs.
- [x] Apply CFG renormalization per logical sample (`ClassifierFreeGuidance`).
- [x] Accept unequal request, plan and tile counts across ranks; exhausted
      no-grad ranks continue until the executor's activity collective is empty.
- [x] Give each standalone `guided_velocity()` evaluation fresh state, including
      concurrent evaluations of different steps from the same run.
- [x] Batch SA's multiple evaluations and replenish the stream as runs complete.
- [x] Compose CFG, tiling and Momentum with `yield from` and local `gather`;
      each algorithm issues child calls and computes its result in one generator.
      Executor and adapter retain all physical batching and collective logic.
- [x] Require a separate `train_predictor` for every diffusion trainer; all
      current/teacher/reference/cache predictions use it. Resolve its negative
      conditions independently of rollout, preserving SFT dropout inputs.

Replay uses `training.grpo_sampling.replay_steps(model, items)`. Each `ReplayItem`
carries its `SampleRun` and a training-owned `RecordedStep` containing the executed
transition, step index/count and optional Flash noise ramp snapshot. `GrpoCollector`
records stochastic latents and log probabilities immediately, dropping velocities;
`take(run)` consumes a completed trajectory. The denominator stays the actual
rollout score even when the stateless training tree differs from the behavior
tree; stateful rollouts are supported. NFT stores the executed plan on
`Rollout`, keeping branch schedule, successor sigma and transformed eta.

## Window RNG (plan transforms)

- [x] Window selection never touches Python's global `random` state.
- [x] `transforms.select_sde_window` draws the inclusive window start with the
      per-request `SampleRequest.generator`, before start noise or solver noise.
- [x] Use the generator's device for `torch.randint`.
- [x] Validate positive window size, valid range, and window fit.
- [x] Preserve the half-open runtime window `[train_start, train_end)`;
      `with_sde_window` gates eta; the GRPO collector records stochastic steps.
- [x] Test Python-RNG independence, generator isolation, and both endpoints
      (`tests/test_recipe_build.py::SelectSdeWindowTest`).

The same request generator drives initial latent creation, window selection at
plan construction, SDEdit start noise, and stochastic solver draws — one per-request RNG stream.

## Adapter support matrix

| Adapter | Initial behavior | Status |
| --- | --- | --- |
| FLUX.1 base | Dense for equal image/text topology | [x] |
| FLUX.1 d-concat / n-concat | Dense with homogeneous control latents | [x] |
| FLUX.1 Fill | Dense with inpaint latents and mask | [x] |
| FLUX.1 Kontext | Dense with equal reference topology | [x] |
| SD3 / SD3.5 | Dense for equal packed latent and prompt shapes | [x] |
| Krea2 | Dense including prompt mask | [x] |
| LongCat base/edit | Dense; edit requires equal references | [x] |
| FLUX.2 | Dense with equal references and batch-sized guidance/IDs | [x] |
| Qwen base/edit/layered | Dense only for homogeneous topology | [x] |
| Z-Image | Native transformer tensor-list input/output | [x] |
| HiDream-O1 | Safe synchronized sequential fallback | [x] |
| Efficient Layered Qwen | Safe synchronized sequential fallback | [x] |

- [ ] Run real-weight smoke tests for FLUX.2, Qwen, LongCat, Krea2, Z-Image,
      HiDream-O1, and every specialized FLUX.1 variant as those caches/configs
      become available.

## DataLoaders and callers

- [x] Make the shared collator return raw `list[dict]`.
- [x] Use `model.micro_batch_size` for physical forwards in inference, validation
      and rollout; remove their former caller-level forward-size fields.
- [x] Add `train_micro_batch_size` to GRPO, NFT, AWM, and RAM.
- [x] Use `PositiveInt` for every new configuration field.
- [x] Keep serving at one request and report per-step progress via its collector.
- [x] Group streamed pairwise rewards by original `key`, retaining K per prompt
      on the same rank and rejecting incomplete groups.
- [x] Migrate inference, validation, rollout, serving, SFT, GRPO, NFT, AWM,
      and RAM.
- [x] Regenerate JSON schemas with `uv run flow-control schema`.
- [x] Update the module API documentation.
- [x] Confirm no stale public `predict_velocity` or singleton sampler call remains.

Rollout seeds are derived from the base seed, epoch, rank, logical rollout
ordinal, and key. Repeated prompts therefore get distinct seeds independent of
physical microbatch boundaries. Cross-world-size reproducibility remains out of
scope because data assignment changes with world size.

## Training arithmetic

For global optimizer batch `G`, world size `W`, and per-rank logical
backward batch `M = train_micro_batch_size`:

```text
local_update_batch = G // W
gradient_accumulation_steps = local_update_batch // M
```

- [x] Validate `G % W == 0`.
- [x] Validate `local_update_batch % M == 0`.
- [x] Keep optimizer/scheduler/current-step semantics tied to optimizer updates.
- [x] Scale a short microbatch mean by `m / C` within an optimizer chunk of
      `C` logical items.
- [x] Enable FSDP gradient synchronization only on the final backward microbatch.
- [x] Share the chunk/microbatch slicing, loss scaling, tail-update warning, and
      non-finite loss check across GRPO/NFT/AWM/RAM via `MicrobatchTrainMixin`
      (`flow_control/training/mixins/microbatch.py`); trainer loss entry points
      take `(items, rollouts, advantages)` and GRPO caches reference means on
      `GrpoTrainItem` in place of a separate matrix.
- [x] Unify batch-option naming on the mixin fields: SFT and the VAE trainer
      now use `train_batch_size` (was `global_batch_size`) and SFT uses
      `train_micro_batch_size` (was `micro_batch_size`); the VAE trainer's
      `grad_acc_steps` raises on non-divisible world_size instead of silently
      truncating. Old field names fail fast via `extra="forbid"`.
- [x] Preserve independent SFT CFG drop, timestep, noise, target, and weight.
- [x] Batch GRPO current/reference replay and reference precomputation.
- [x] Batch NFT current/old/reference prediction, including CFG synchronization.
- [x] Batch AWM current/reference/EMA prediction.
- [x] Batch RAM current/base/old prediction.
- [x] Advance progress and aggregate metrics per logical item.

- [ ] Add focused fake-trainer loss/gradient equivalence tests for every RL
      objective; the shared scaling path is implemented and code-reviewed but
      complete logical-batch equivalence is not yet covered for every objective.
      Adapter gradients, real-model batched backward, and NFT/GRPO low-precision
      storage with fp32 loss/gradient parity are covered.
- [ ] Run full checkpoint/restore tests at SFT optimizer-step and RL outer-epoch
      boundaries. Raw list-batch StatefulDataLoader state round-trip is covered.
- [ ] Treat changing microbatch size while restoring an existing mid-epoch
      DataLoader state as unsupported until explicitly tested.

## Distributed invariants

- [x] All ranks use the same adapter limit and configured variant order.
- [x] Ranks may submit unequal logical counts; training still requires a local
      evaluation to carry dummy dependencies into backward.
- [x] All ranks agree on dense collation versus sequential fallback.
- [x] All ranks execute equal conditional/unconditional CFG forward counts.
- [x] All ranks execute equal backward counts per optimizer update.
- [x] Plan-time padding (`padding` rows) produces equal final logical microbatch lengths.
- [x] A rank with dense-compatible inputs safely follows another rank's fallback.
- [x] Two-rank CPU collective tests cover fallback, cold empty ranks, CFG, and tails.
- [x] Two-rank FSDP2/NCCL tests cover dense forward/backward, local fallback,
      and cross-rank synchronized fallback.

## Validation completed

The 2026-09-11 generator refactor passed 70 tests and 154 subtests without
changing the 21 numeric fixtures. Slurm job 30503 passed both two-rank Gloo
workers; job 30504 passed FSDP checkpoint/variant/tile gradients and complete
bf16 tiled sampling with fp32 steps. These short tiny-model correctness runs
do not establish sustained production throughput.

### Static and CPU

- [x] `uv run ruff format flow_control tests`
- [x] `uv run ruff check --fix flow_control tests`
- [x] `uv run ty check flow_control tests`
- [x] Preserve all 21 bitwise solver fixtures without regeneration.
- [x] Cover concurrent replay contexts, completion-order pairwise grouping,
      collector parity, and bf16 storage with fp32 sampler/loss arithmetic.
- [x] Two-rank Gloo distributed worker passes.
- [x] JSON schemas regenerate successfully.
- [x] `git diff --check` passes.

Covered unit behavior includes dense/fallback calls, recursive reference
topology, metadata independence, output order, dense/fallback gradients,
trajectory RNG, per-request schedules, stochastic generator isolation, CFG
none/all/mixed/renorm, named replay output, list collation, and StatefulDataLoader
resume.

### GPU/FSDP

- [x] Two-rank FSDP2 forward/backward smoke on 2x 48 GB RTX 4090.
- [x] Dense microbatch, both-rank fallback, and one-rank-only fallback all pass.
- [x] Real SD3.5-medium dense forward and batched backward pass.
- [x] Real FLUX.1-dev dense forward and batched backward pass.
- [x] End-to-end SD3.5 GRPO on two FSDP ranks with rollout, train, and
      validation microbatch size 2.

At 256x256 synthetic fixed-shape inputs, four logical forward samples:

| Model | Singleton time | Microbatch-2 time | Speedup | Peak allocated | Relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SD3.5-medium | 0.183 s | 0.099 s | 1.85x | 4.66 GiB | 0.00317 |
| FLUX.1-dev | 0.247 s | 0.161 s | 1.53x | 22.23 GiB | 0.00418 |

Separate batched backward checks produced finite nonzero gradient norms of
`2.16` (SD3.5) and `25.43` (FLUX.1).

The end-to-end 512x512 GRPO smoke completed four CPS rollouts, PickScore reward,
reference replay, two LoRA optimizer updates, and final DCP checkpoint creation.
It recorded `2.24` rollout items/s and `0.435` training items/s. Its active
training phase lasted only about ten seconds, shorter than a useful Prometheus
sampling window, so it is a correctness smoke rather than a utilization result.

These are adapter/kernel smoke benchmarks, not representative RL throughput
measurements.

### SD3.5 512x512 AWM throughput

The production-shape controlled sweep used one 48 GB RTX 4090 per run,
SD3.5-medium with
LoRA rank 32, fixed 512x512 PickScore caches, 14-step no-CFG AWM rollouts, 16
endpoints, 6 training timesteps per endpoint, precomputed reference/EMA
predictions, and gradient checkpointing disabled. Each run completed two
epochs; the table reports the second epoch after warm-up.

| Microbatch | Rollout items/s | Reference items/s | EMA items/s | Train items/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.568 | 4.804 | 8.738 | 3.583 |
| 2 | 1.000 | 8.664 | 15.840 | 6.645 |
| 4 | 1.674 | 17.069 | 32.314 | 12.825 |
| 8 | 1.681 | 18.100 | 34.029 | 13.398 |
| 16 | 1.682 | 17.361 | 32.280 | 12.804 |

Microbatch 4 is the default operating point: versus microbatch 1 it delivers
2.95x rollout and 3.58x training throughput. Microbatch 8 adds only 0.4% rollout
and 4.5% training throughput in eager mode, while microbatch 16 regresses the
auxiliary and training paths.

`torch.compile` is worthwhile once its one-time cost is amortized. The initial
SD3 block, LoRA enable/disable, and grad/no-grad states produce 12 finite
recompilations of shared forward frames. A limit of 8 makes Dynamo hit its limit
and locally fall back to eager; a limit of 16 compiles all observed variants.
The same-card microbatch 4 comparison on a stock 24 GB RTX 4090 was:

| Mode | Rollout items/s | Reference items/s | EMA items/s | Train items/s |
| --- | ---: | ---: | ---: | ---: |
| Eager | 1.534 | 15.030 | 27.703 | 12.178 |
| Compile16 | 1.743 | 19.183 | 36.239 | 15.195 |

That is a 13.6% rollout, 27.6% reference, 30.8% EMA, and 24.8% training
improvement. At microbatch 8 on the same 48 GB node, compile16 reached 2.110
rollout and 17.296 training items/s, 25.5% and 29.1% above eager. The first
compiled epoch is intentionally excluded: compilation made its train phase
51.2 seconds versus 5.6 seconds at steady state.

The recommended microbatch 4 plus compile16 configuration also completed the
same two-epoch workload on a stock 24 GB RTX 4090 without OOM or a recompile
limit hit. Prometheus observed up to 72% GPU utilization and 273 W, but its
coarse sampling missed some short phases, so these are health checks rather
than exact sustained averages. Slurm jobs 10908, 10914, 10920, and 10925 retain
the eager sweep, compile8 sweep, compile16 sweep, and 24 GB compile validation
logs respectively; job 10926 is the same-card 24 GB eager control.

## Remaining acceptance work

- [ ] Run complete FLUX.1 and SD3.5 rollout plus optimizer jobs at the actual RL
      resolutions and model/LoRA settings.
- [x] Run a paired rollout/train microbatch sweep for 512x512 SD3.5 AWM through
      the eager plateau and validate the selected setting on 24 GB and 48 GB
      RTX 4090 cards.
- [ ] Sweep rollout and train microbatch sizes independently until OOM or plateau.
- [ ] Measure CFG/no-CFG and inline/precomputed auxiliary prediction modes.
- [ ] Record samples/s, train items/s, VRAM, utilization, and power from cluster
      telemetry during sustained jobs.
- [ ] Compare loss/reward distributions and inspect rollout uniqueness against
      microbatch 1.
- [ ] Promote only machine-neutral settings to `examples/`; keep machine-specific
      launch choices under `local/`.

## Future variable-length evolution

The public list boundaries do not need to change:

```text
list of logical samples
        |
        +-- dense:  [M, N, D]
        +-- padded: [M, Nmax, D] + masks
        +-- packed: [1, sum(N_i), D] + offsets/cu_seqlens/block mask
        +-- native model-specific tensor lists
        |
list of [1, N_i, D] outputs
```

- [ ] Add adapter-owned padding and attention masks where a model supports them.
- [ ] Reset or construct position IDs per logical segment.
- [ ] Prevent cross-sample attention with block-diagonal masks or varlen metadata.
- [ ] Treat text, target-image, and reference-image streams as one model-specific
      topology rather than packing tensors independently.
- [ ] Split packed/padded outputs back to logical samples inside the adapter.

The sampler, solvers, replay items, trainers, and losses already operate on
logical lists, so future packing is localized to adapter collation, model
attention/position handling, and output splitting. Z-Image is the first native
ragged example. FLUX and Qwen still require architecture-specific masks and
position semantics; the new API removes the outer plumbing work but not that
model-level complexity.

## Forward and backward limits

`model.micro_batch_size` bounds physical adapter forwards, including branch/tile
expansion. The executor uses it to decide how many sampling runs to keep active.
`train_micro_batch_size` separately limits logical loss items per backward and
therefore controls retained autograd state. There is no separate tiling or rollout
forward-size setting.
