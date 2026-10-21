# Row

- **Row** (`flow_control.data.rows.Row`, a plain `dict[str, Any]`): one sample at one point, at every stage: raw source item, processed or cached row, sampler working row, training point, decoded rollout, reward input, serving request.
- **Microbatch**: a `list[Row]`, what `predict_velocity_batched` takes. Variables are named `rows` or `microbatch`.
- **Batch**: the physical payload an adapter collates from a microbatch (leading dimension `B`, the `*Batch` TypedDicts under `flow_control/adapters/`). It never leaves the adapter; see [microbatching](../adapters/microbatching.md).

## Rules

- One dict is one logical sample at one point. A new point of the same sample is a new dict built from `data(row)`; the K rollouts of one prompt are K rows sharing `key`.
- The row is a shared blackboard. Stages write into it in place and nobody clones it. A consumer reads what it needs from the row, or from the producer's frame (`caller(row, kind, who)`), and checks it with `isinstance` or key tests. Do not thread a new parameter through the layers in between.
- A row object appears in at most one model call per executor round. A concurrent branch gets its own derived row (`derived` + `follow` in `flow_control/samplers/node.py`).
- The dict is open: dataset columns and plugins add fields freely. A field that more than one module reads or writes gets a row in the tables below, in the same commit.
- `_`-prefixed keys are runtime state: frames, generators, derived rows, node history, adapter caches. They are never persisted. `data(row)` drops them, `Runner._decode` moves only `data(row)` to rollout storage, the adapter strips them before casting inputs (`MicrobatchMixin._prepared_batches`, `utils/model_cache.py`), and a remote reward's snapshot holds only its `fields`.

## Shapes, coordinates, precision

- Images are float `[1, C, H, W]` in `[0, 1]`, RGB or RGBA. The exceptions are marked in the tables.
- `image_size`, `model_image_size` and `reference_sizes[i]` are pixel `(H, W)`: not PIL `(W, H)`, not the latent grid.
- Packed latents are `[1, N, D]` in the processor's normalized coordinates (after the VAE's shift and scale), packed `b c (h ph) (w pw) -> b (h w) (c ph pw)`: `N = H/(f·p) · W/(f·p)`, `D = C_z·p²` with `f, p, C_z` = `vae_scale_factor`, `patch_size`, `latent_channels`. Layered tasks concatenate frames or crops along `N`.
- A cached posterior is `[2, N, D]` = (mean, std) in a field listed in `processor.posterior_fields`. `processor.resample` replaces it with one `[1, N, D]` draw, made and kept in fp32 (`sample_posterior`). FLUX.2 is the VAE whose posterior std survives bf16; the other VAEs encode the mode.
- Storage may be bf16. Arithmetic outside the adapter (start, guidance, solver, loss) is fp32. The adapter casts its inputs to the compute dtype and returns fp32 velocity. Initial noise is drawn in the model dtype (`initialize_latents`) and promoted to fp32 by the `Runner`. Detail: [precision](../adapters/precision.md).

## Reserved keys

Defined in `flow_control/data/rows.py`; the data side is in [data](../data/data.md).

| Field | Type | Written by → read by | Meaning |
|---|---|---|---|
| `key` | `str`, `[A-Za-z0-9._-]+` | raw source → seeds, output names, advantage groups | Sample id, unique within a dataset. |
| `cost` | `int` | `processor.get_cost` at preprocess or online preprocess → plans, reports | Token total (latent + text + reference) of one forward; the plan sort key. |
| `image_size` | `(H, W)` | input (optional), then the processor → `initialize_latents`, adapters, shifts, tiling, decode | Target size after the task's resize. |
| `padding` | `bool` | `RowStream.__getitem__` → `is_padding` | Plan-time padding row: forwarded, contributes no loss or output. Never persisted. |

## Inputs

Raw fields survive into the sampling row for inference, validation and RL; into an offline cache only with `save_extra`.

| Field | Type | Task | Meaning |
|---|---|---|---|
| `prompt` | `str` | all | Caption or instruction. Captioning, enhancement and trigger words rewrite it in place. |
| `negative_prompt` | `str` | all built-in tasks | Encoded into `negative` only with `save_negative`. |
| `clean_image` | image | all (training), `qwen_layered` | Training target; the source scene for layer decomposition. Decode overwrites it with the output. |
| `control_image` | image | `t2i_control` | Spatial condition. |
| `inpaint_image` | image | `inpaint` | Inference source; training encodes `clean_image` instead. |
| `inpaint_mask` | image, processed `[1, H, W]` | `inpaint` | Luminance mask, white = editable. |
| `reference_images` | `list` of images | `tie` | Ordered edit sources. |
| `layer_images` | `list` of images | `qwen_layered`, `efficient_layered` | Target layers in training; the decoded layers after sampling. |
| `num_layers` | `int` | `qwen_layered` | Layers generated after the base frame. |
| `tiles` | `list` of prompt dicts | `tiled_t2i` | Row-major per-tile prompts; processed into complete per-tile rows. |
| `layer_boxes`, `layer_prompts` | `list` | `efficient_layered` (contrib) | Pixel boxes `(top, bottom, left, right)` and one caption per box. |
| `tag`, `include`, `exclude` | dataset columns | GenEval prompts | Read by the `geneval` scorer. |

## Processed fields

Written by `prepare_training_row` / `prepare_inference_row` ([processors](../processors/processors.md)), read by the adapter unless noted.

| Field | Shape | Meaning |
|---|---|---|
| `prompt_embeds` | `[1, L, E]` (Krea2 `[1, L, K, E]`) | Text or multimodal embeddings. |
| `pooled_prompt_embeds` | `[1, E]` or `None` | Pooled text embedding; FLUX.1 and SD3 require it. |
| `clean_latents` | `[1, N, D]`, cache `[2, N, D]` | Training target. `Runner._decode` overwrites it with the sampled endpoint. |
| `control_latents` | `[1, N, D]` | Encoded `control_image`. |
| `inpaint_latents` | `[1, N, D]` | The full source image, not a masked one. |
| `inpaint_mask_latents` | `[1, N, p²]` | Mask on the token grid; read by `differential_diffusion`. |
| `reference_latents`, `reference_sizes` | `list` of `[1, Nᵢ, D]`, `list` of `(H, W)` | Encoded references and their pixel sizes. |
| `image_latents` | `[1, N, D]` | Source scene of the layered tasks. |
| `negative` | `dict` | Overrides for the CFG negative branch, overlaid shallowly (`Cfg.uncond_row`, `processor.get_negative_row`). Written only with `save_negative`. |
| `tiling`, `model_image_size` | `TileLayout` dict, `(H, W)` | Tile plan and the per-forward size; read only under a `tiled` prediction and by the shifts. |
| Model-family keys | — | `prompt_embeds_mask` (Krea2, Qwen-Image-2.1), `input_ids` / `pixel_values` / `image_grid_thw` (HiDream-O1; `pixel_values` are not `[0, 1]` images), `txt_ids` / `img_ids` (FLUX, LongCat), `text_lengths` / `block_mask` (`efficient_layered`): contracts between one preset and its adapter. |

## Sampler and runtime keys

Detail in [sampler runtime](../sampler/runtime.md) and [nodes](../sampler/nodes.md).

| Field | Type | Written by → read by | Meaning |
|---|---|---|---|
| `noisy_latents` | `[1, N, D]` fp32 | `Runner._start`, `noised_point` → every node, the adapter | The row's current state x_σ; nodes overwrite it in place. |
| `sigma` | `float` | start (1.0), trajectories, transitions | The row's current σ. |
| `sigma_next` | `float` | `BaseTransition.__call__` → flow-map adapters (FLUX.3 distilled) | Target σ of the enclosing step; present only inside a step. |
| `_frame` | `Frame` | `BaseNode.enter`; the root from the engine | Innermost active node; `caller` walks the chain. |
| `_generator` | `torch.Generator` | `Runner._start`, serving engine → `randn`, `sde_window_grid` | Per-row RNG of every stochastic node. |
| `_uncond:negative`, `_uncond:positive`, `_tiles` | derived rows | `Cfg`, `Tiled` | Concurrent branches, built once and moved to the parent's point by `follow`. |
| `_noise`, `_momentum` | tensor | `differential_diffusion`, contrib `momentum` | Node history kept on the row, so one per branch. |
| `_cache` | `{variant: {name: value}}` | `MicrobatchMixin._prepared_batches` | No-grad adapter intermediates (`_txt_ids`, `_img_ids`, `_kv_cache`) per weight variant. |

## Outputs, rollouts, training points

| Field | Type | Written by → read by | Meaning |
|---|---|---|---|
| `clean_image` | image | `processor.decode_output` → rewards, reports | The generated image, replacing the input value. |
| `base_image`, `layer_images` | image, `list` | `qwen_layered` decode | Reconstructed base frame and generated layers. |
| `epoch` | `int` | `SampleContext.rollouts` → rewards that depend on training progress | RL epoch the rollout was sampled in; a remote reward lists it in `fields`. |
| `reward_raw` | `[C]` fp32, CPU | `rollouts.base.write` (rollout programs, `score_one`) | Raw scores, one per reward label. |
| `reward` | `[C]` fp32, CPU | `write` → `_compute_advantages` | The same components after each scorer's `normalize`, unweighted; labels and weights are `_reward_labels()` / `_reward_weights()`. |
| `trajectory` | `list` of dicts | GRPO `record` observer → `GrpoTrainer._point` | Each stochastic step: `noisy_latents`, `sigma`, `next_latents`, `log_prob`, a tensor-free `_frame` snapshot. |
| `sigmas` | `list[float]` | `keep_sigmas` observer → `train_timesteps.draw` | The executed grid of an endpoint-trainer rollout. |
| `noise`, `advantage` | tensor | `noised_point`, `EndpointTrainer._loss_batched` → objectives | Re-noising ε and the rollout's advantage on a training point. |

The inference report's `reward` (weighted scalar), `reward_raw` and `reward_normalized` (`{label: value}`) come from `Inference._reward_fields`; they are report fields, not these row fields ([formats](../data/formats.md)).

## Condition-image selectors

Samplers and rewards that need one condition image take a selector string (`ConditionImageSpec`) and resolve it with `ConditionImage.parse(spec).image(row)` or `.latents(row)` (`flow_control/utils/condition_image.py`).

| Selector | Pixel field | Latent field |
|---|---|---|
| `reference`, `reference[i]` | `reference_images[i]` | `reference_latents[i]` |
| `control` | `control_image` | `control_latents` |
| `inpaint` | `inpaint_image` | `inpaint_latents` |
| `clean` | `clean_image` | `clean_latents` |
| any other `name`, `name[i]` | `row[name]` | `row[name]` |

- A list field without an index selects entry 0; an index on a non-list field raises. A selector only fetches; it never resizes or re-encodes.
- Consumers: `init_latent.from`, `differential_diffusion.from` (default `inpaint`), `clip_image_similarity.reference` and contrib `pyiqa.reference` (default `reference`).
- `clean` is not a fixed observation: decode overwrites its pixel side and `Runner._decode` its latent side. A task that needs its own observation writes it under a new key and names that key.
