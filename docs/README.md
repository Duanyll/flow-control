# flow_control docs

`flow_control` trains and samples flow-matching DiTs: SFT, diffusion RL (GRPO, NFT, AWM, RAM) and batched inference, on FSDP2. These docs cover the designs the code does not show top-down, the extension points, and the contracts that need a discussion before they change. How to write them: [CLAUDE.md](CLAUDE.md).

## Chapters

| Chapter | Covers |
|---|---|
| **Overview** | |
| [repo-layout](overview/repo-layout.md) | Tracked vs gitignored trees; core vs `contrib/` vs `local/` |
| [config-plugins](overview/config-plugins.md) | Config loading, `imports`, `Registry` / `RegistryUnion`, every registry |
| [row](overview/row.md) | The row dict every stage shares: rules, reserved and runtime keys, shapes, precision |
| **Adapters** | |
| [adapter](adapters/adapter.md) | `_predict_velocity`, framework conventions, load order, meta init from the seed checkpoint |
| [precision](adapters/precision.md) | Storage / compute / trainable dtypes, layerwise upcasting, fp32 EMA |
| [microbatching](adapters/microbatching.md) | Logical samples outside the adapter, physical batches inside; the collective contract |
| [variants](adapters/variants.md) | LoRA / EMA / full / alias weight sets, O(1) selection, offline LoRA conversion |
| **Data** | |
| [data](data/data.md) | Raw source → preprocess → group → resample → consume → report; the three cache modes |
| [formats](data/formats.md) | On-disk random cache, packed cache, inference report |
| **Processors** | |
| [processors](processors/processors.md) | Task × preset × components; encode/decode lifecycle; negative rows; prompt enhancement |
| **Rewards** | |
| [rewards](rewards/rewards.md) | Placement (local / remote) × slot (scorer / judge); normalization; reward server |
| **Sampler** | |
| [runtime](sampler/runtime.md) | `Calls` generators, Executor rounds, rows and frames |
| [nodes](sampler/nodes.md) | Trajectory / transition / prediction nodes, shifts, custom nodes |
| **Training** | |
| [trainers](training/trainers.md) | Mixin composition, objectives and hooks, batch arithmetic, seeds |
| [distributed](training/distributed.md) | Where a collective may sit, collective-free zones, equal-count invariants |
| [rollouts](training/rollouts.md) | Rollout programs, the Runner's two threads, n-step-off |
| [outputs](training/outputs.md) | Run directory, metrics, checkpoints, resume |
| **Scripts** | |
| [cli](scripts/cli.md) | Every command; launch; seed checkpoint; preprocess pipeline |
| [serving](scripts/serving.md) | Outdated, needs an interface refactor: stateless engine, shared model cache, task templates |
| **Guides** | |
| [fsdp2-dcp](guides/fsdp2-dcp.md) | FSDP2 and DCP as this repo uses them ([runnable example](guides/fsdp/dcp.py)) |
| **Reading** | Library-independent notes |
| [flux2-blog](reading/flux2-blog.md), [models-and-reports](reading/models-and-reports.md) | FLUX.2 notes; DiT papers and reports |

## Where to build what

An experiment that fits one of these needs no design discussion. Unpublished work lives in `flow_control/contrib/` or `local/<project>/`, activated through `imports`; `flow_control/` holds published algorithms and generic tools only.

| To add | Extension point | Chapter |
|---|---|---|
| A model architecture, or a new type of one | `adapter_registry`: subclass `BaseModelAdapter`, implement `_predict_velocity` | [adapter](adapters/adapter.md) |
| Stacking, padding or packing for a family | Dense-batching declarations, or override `predict_velocity_batched` | [microbatching](adapters/microbatching.md) |
| Encoding for a new family | `preset_registry`, `encoder_registry`, `vae_registry` | [processors](processors/processors.md) |
| A new input/output task | `task_registry` | [processors](processors/processors.md) |
| A dataset reader | `source_registry`, in contrib even for a published dataset | [data](data/data.md) |
| Per-draw augmentation | `processor.resample` override | [data](data/data.md) |
| A sampling method | Any `BaseNode`; or `trajectory_registry` / `transition_registry` / `prediction_registry` / `shift_registry` | [nodes](sampler/nodes.md) |
| A reward model | `scorer_registry` / `judge_registry` + a placement; MLLM rewards go to contrib | [rewards](rewards/rewards.md) |
| A way to combine rewards, anchors, gates | `rollout_registry` program | [rollouts](training/rollouts.md) |
| A loss on rollout endpoints | `objective_registry` | [trainers](training/trainers.md) |
| Advantages, re-noising, SFT weightings | `advantage_registry`, `train_timesteps_registry`, `timestep_weighting_registry`, `loss_weighting_registry` | [trainers](training/trainers.md) |
| A new trainer | `trainer_registry`, composed from the mixins | [trainers](training/trainers.md) |
| A serving task UI | `task_template_registry` | [serving](scripts/serving.md) |

## Core contracts

Many modules depend on these. Changing one is a discussion with the user before any code, and the chapter changes in the same commit.

- The row: one sample per dict, shared and never cloned; reserved keys; `_` runtime keys ([row](overview/row.md)).
- Adapter conventions (σ time, `ε - x₀` velocity, packed `[B, N, D]` latents) and `predict_velocity_batched` as a collective returning fp32 ([adapter](adapters/adapter.md), [microbatching](adapters/microbatching.md)).
- The variant mechanism; `variant_registry` is internal ([variants](adapters/variants.md)).
- The `Calls` protocol and the collective executor round ([runtime](sampler/runtime.md)).
- The sanctioned collective sites and equal-count invariants ([distributed](training/distributed.md)).
- Reward placement × slot, label naming ([rewards](rewards/rewards.md)); the `SampleContext` / `RewardContext` API and the Runner's threading rules ([rollouts](training/rollouts.md)).
- Plans, `RowStore`, on-disk cache formats ([data](data/data.md), [formats](data/formats.md)); checkpoint keys and run-directory files ([outputs](training/outputs.md)).
- A registry member is live even if no config in the repo selects it ([config-plugins](overview/config-plugins.md)).
