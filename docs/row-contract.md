# Row dictionary contract

Vocabulary used throughout `flow_control`:

- **Row**: any single-sample dictionary, at every stage (raw dataset item, processor output, decoded record, the sampler's working dict, a reward's input, a serving request). Type alias: `flow_control.data.rows.Row`.
- **microbatch**: a `list[Row]`, the chunk an adapter receives in one `predict_velocity_batched` call (variables are named `rows` or `microbatch`).
- **Batch**: only the physical payload an adapter collates from a microbatch (leading dimension `B`; the `Batch` TypedDicts under `flow_control/adapters`). It exists inside the adapter and never leaves it.

This is the field inventory for dataset items, processed samples, model-call
rows and decoded/output records in `flow_control`. These are open dictionaries:
dataset columns and plugins may add fields. The tables cover names interpreted or
produced by the in-tree code, including `contrib`; they are not a universal
`TypedDict` or a requirement that every sample contain every field.

When adding a field, document its stage, shape, units, producer and consumer here.
Processors own representation changes; each algorithm should read only its own
inputs. Sampling execution is described in [sampler-plan-design.md](sampler-plan-design.md).

## Shapes, coordinates and precision

- One dictionary normally represents **one logical sample**. Images are floating
  `[1, C, H, W]` tensors in `[0, 1]`, with RGB/RGBA channels as the task requires.
  Decoder outputs use this nominal range but are not all clamped. The exceptions
  `inpaint_mask`, `pixel_values` and `target_image` are specified below.
- `image_size`, `model_image_size` and entries of `reference_sizes` are **pixel
  `(height, width)`**, not PIL `(width, height)` or latent-grid dimensions.
- Ordinary packed latents are `[1, N, D]`. With VAE spatial factor `f`, patch size
  `p` and latent channels `Cz`, `N = (H / (f*p)) * (W / (f*p))` and `D = Cz*p*p`.
  Packing order is `b c (h ph) (w pw) -> b (h w) (c ph pw)`. Layered models
  concatenate frames/crops along `N`; their stream is not one rectangular image.
- Latents are in the processor's **normalized model coordinates**, not raw VAE
  outputs or necessarily pixels. VAE scale/shift/normalization, `f`, `p` and `Cz`
  live in processor/adapter configuration; they are not row fields.
- For the sampler/training interface, low-precision model computation stays inside
  the adapter: `predict_velocity_batched` casts the call's float tensors to the
  model dtype on entry and returns **fp32 velocity**. Start, predictor, projector,
  solver, tile accumulation and loss arithmetic use fp32. Stored inputs, targets
  and records may have lower precision and are promoted before arithmetic;
  inference and rollout draw the initial noise in the model dtype, and
  `SampleRun.run` casts the final latents back to the dtype of the stored
  `noisy_latents`. VAE/text preprocessing uses its own configured dtype.
- The adapter accepts a microbatch (a list of rows). Per microbatch chunk it
  either collates the declared `dense_batch_fields` into a physical `Batch` with
  leading dimension `B` (list-valued fields become lists of `[B, ...]` tensors)
  or forwards each dictionary sequentially. The dense path builds a new
  dictionary holding only the declared fields and requires non-tensor values to
  agree across the chunk; the fallback decision is synchronized across ranks.
  Callers do not stack dictionary fields themselves.

Sources: [BaseProcessor](../flow_control/processors/base.py) (`encode_latents`,
`_pack_latents`, `decode_output`), [VAE implementations](../flow_control/processors/components/vae.py),
[BaseModelAdapter](../flow_control/adapters/base.py) (`predict_velocity_batched`).

## Lifecycle and retention

| Stage | Contract |
| --- | --- |
| Dataset input | Raw images, prompts and arbitrary columns. Task input coercion accepts supported paths/PIL/arrays/tensors and JSON forms for annotated structured fields; unknown columns pass through. Tensor-file attachments load their native tensor representation, so their shape/range must already be correct. |
| Processor output / offline cache | `prepare_training_row` adds clean targets and conditions; `prepare_inference_row` adds conditions. The offline `ProcessorStage.process` adds `cost`. `save_extra` merges original fields with processed fields taking precedence; `__key__` is carried separately. Processors also rewrite inputs in place (resized `clean_image`, enhanced or generated `prompt`), so retained extras hold the rewritten values. |
| Runtime preprocessing | `DataMixin.prepare_row` moves the row to the device, runs the processor only when the store is an `OnlineStore` (raw source; a cache row is used as is), carries `__key__` / `__padding__` over, adds `cost`, then calls `processor.resample`. Inference and validation keep the raw fields next to the processed ones; SFT training does not. Sampling callers initialize `noisy_latents` in the model dtype; training builds noisy inputs from its targets. |
| Model call | `ModelPrediction` overlays the current request's latents as `noisy_latents` on a fresh shallow copy of the whole working row. Predictor branches and tiles select their own condition rows. The adapter receives a microbatch (`list[Row]`); the physical `Batch` with a leading `B` dimension exists only inside its collate step. Only dense collation narrows the call to declared fields; the sequential path sees every key. |
| Decode | `decode_output(final_latents, row)` returns a new decoded row. Inference and rollout callers merge it into their working row, replacing names such as `clean_image`. |
| Persist / score | Inference scores the merged row; its `save_extra` only decides whether the record holds the decoded fields or the whole merged row, and reward fields are added to it. Raw fields needed for scoring must survive any earlier offline cache as well. |

**Two overwrite rules matter:**

1. `run.row["noisy_latents"]` is the initial/source tensor. The evolving state is
   `run.ctx.latents`, and only the model-call copy receives the current value.
   The one exception is `EndpointTrainer._prepare`, which writes the re-noised
   training input `x_t` into `row["noisy_latents"]` before `make_run` /
   `guided_velocity` re-evaluates the rollout step on it.
2. RL rollout collection writes the sampled endpoint into `row["clean_latents"]`.
   General inference merges decoded images but does **not** write endpoint latents
   there. A retained `clean_latents` may therefore still be an input/cache target.

Sources: [coercion](../flow_control/data/coercion.py),
[offline preprocessing](../flow_control/scripts/preprocess.py) (`ProcessorStage.process`),
[runtime preprocessing](../flow_control/training/mixins/data.py),
[model leaf](../flow_control/samplers/prediction.py) (`ModelPrediction.bind`),
[rollout collection](../flow_control/training/mixins/rollout.py) (`_collect_rollouts`),
[inference](../flow_control/training/inference.py) (`_sample_submitter`, `_write_output`).

### Cached posterior distributions

With `target_posterior` (default `distribution`) or `condition_posterior` (default
`mode`) set to `distribution`, a VAE that
supports this mode stores mean and **standard deviation**, concatenated along the
leading dimension. After packing a logical singleton, `[2, N, D]` means
`[mean, std]`, not two samples and not mean/log-variance. Mode/sample caches are
already `[1, N, D]`; some VAE implementations always use the mode.

`processor.resample` draws `mean + std * noise` for every field named in the
processor's `posterior_fields` whose leading size is 2 (lists element-wise, e.g.
`reference_latents`); it does **not** recurse into `negative` or `tiles` and never
guesses by name. Plugins that cache another posterior field must declare it.

## Identity, geometry and sampling tensors

| Key | Meaning / representation | Producer | Consumers |
| --- | --- | --- | --- |
| `__key__` | String source-sample identifier. K rollouts of one prompt share it. | Dataset readers; offline `reassign_keys` replaces it with the index, and runtime preprocessing copies it from the source item. | Seed derivation, output names, validation/rollout identity, `execute_pairwise_reward` grouping. |
| `__padding__` | `True` on plan-time padding rows (a short tail group repeats rows of the same block). Consumer memory only, never persisted. | `RowStream.__getitem__`. | `is_padding(row)`: SFT / VAE weigh the row 0, inference and validation forward it but write and log nothing. |
| `image_size` | Optional requested size on raw input; actual target pixel `(H, W)` after task resizing. Required for model/sampler use. | Dataset/user, then processor. | Latent initialization, adapter geometry/position IDs, decode, tiling and resolution-dependent shift. |
| `model_image_size` | Optional pixel `(H, W)` seen by one forward, e.g. a tile. Defaults to `image_size`. | `TiledT2IProcessor`. | `BaseShift._get_seq_len`; tiled prediction removes it from leaf conditions. |
| `cost` | Integer token total from `processor.get_cost` (latent + text + reference); the plan sort key. It is not necessarily `noisy_latents.shape[1]`. | Offline `ProcessorStage.process`, runtime `DataMixin.prepare_row`, or an adapter's `cost_test` mock batch. | `RandomCacheWriter` records it in the cache index and grouping sorts on it; `ReportWriter` writes it to `metrics.jsonl`; `SftTrainer.run_cost_test` reads it off the mock batches. Resolution shift computes its own length and does not read this field. |
| `clean_latents` | Normalized packed clean target; `[1, N, D]` at runtime, possibly `[2, N, D]` in a posterior cache. RL collection replaces it with the sampled endpoint. Optional for inference. | Training processor, rollout collector, or an adapter's `cost_test` mock batch. | SFT/AWM/RAM/NFT target calculations; `Start` when selected as a source, including default SDEdit source. |
| `noisy_latents` | Packed noisy state `[1, N, D]`, in the same coordinates as clean targets. Meaning is stage-dependent: initial noise/source in the run row, current state in a model call, corrupted target in training. | `initialize_latents`, trainers, `ModelPrediction`. | `Start`, shift, all adapters; `DifferentialDiffusion` uses the run-row tensor as reference noise. Solver state itself lives in `StepContext`. |
| `negative` | Optional shallow condition override dictionary; see below. | Task processor when `save_negative=True`, or caller/cache. | `BaseProcessor.get_negative_row` resolves it into the separate negative row that CFG/CFG++ and training receive; `TiledT2IProcessor._add_tiles` rewrites it and `TiledPrediction` strips it from leaf conditions. |

Sources: [row contract](../flow_control/data/rows.py), [plan / stream](../flow_control/data/stream.py)
(`RowStream`, `RowCursor`), [grouping](../flow_control/data/grouping.py),
[Start](../flow_control/samplers/sampler.py), [shift](../flow_control/samplers/shift.py),
[reward execution](../flow_control/rewards/__init__.py).

### Negative conditions

`get_negative_row` copies the positive dictionary, removes `negative`, and
overlays its contents. The merge is **shallow**: unchanged tensors/metadata are
shared, and a list such as `negative["tiles"]` replaces the positive list entirely.
The model leaf later injects the same current noisy state into either branch.

Ordinary negative dictionaries contain encoded text fields. Efficient-layered
negatives also replace `text_lengths`; tiled negatives replace `tiles` with
complete per-tile conditions, and are emitted only when the whole-image row
has a `negative`, otherwise per-tile negative prompts are dropped. HiDream editing negatives can also contain
`pixel_values` and `image_grid_thw` when `negative_with_images` is enabled.
These nested dictionaries reuse the field contracts below; they are not
independent samples or recursively merged configuration.

Tasks exposing `negative_prompt` use it, or their configured default, for
inference. Training normally uses the configured default; explicit `tiles`
conditions go through inference preprocessing even during training, so each
tile's supplied negative prompt applies. Efficient-layered always uses its
configured default, including for inference. Encoded negatives are produced only
when `save_negative` is enabled. A CFG training tree needs these conditions even
if the rollout tree does not use CFG; `train_predictor` and rollout guidance are
configured independently.

## Raw inputs and decoded images

Image shapes below follow the `[1, C, H, W]`, nominal `[0, 1]` convention unless
specified. Raw fields survive into the live row for inference, rollout and
validation, which always merge extras, and into records or caches only with
`save_extra`.

| Key | Meaning / representation | Producer | Consumers |
| --- | --- | --- | --- |
| `prompt` | Text instruction/caption. Several tasks caption a missing prompt (training tasks, and Qwen Layered also at inference); prompt enhancement/trigger insertion rewrites it in place. | Dataset/user or processor captioning. | Text encoders; prompt-based rewards and validation logging. |
| `negative_prompt` | Optional raw negative text; encoded only when requested. | Dataset/user. | Task processors exposing this input, with training/tile exceptions described above. |
| `clean_image` | Training target, or source scene for layer decomposition; inpaint training also encodes it as the inpaint source. After decode/merge it is the generated primary image. For layered tasks the decoded value is a **montage**, not the composited scene. | Dataset/user, processor resizing, then `decode_output`. | Target/condition encoding, image rewards, preview/output sinks and VAE training. It is not an immutable observation slot. |
| `control_image` | Spatial condition for `t2i_control`; resized before encoding. Inference writes the resized image back; training may retain the original-size input in extras. | Dataset/user; `generate_control_image` raises unless a subclass overrides it. | `T2IControlProcessor` produces `control_latents`. |
| `inpaint_image` | Inference source image for inpainting; resized before encoding and written back. Training has no such input: the clean target doubles as the source. | Dataset/user, then `InpaintProcessor`. | `InpaintProcessor` produces `inpaint_latents`. |
| `reference_images` | Ordered list of editing source images, trimmed and resized by `TIEProcessor` (HiDream's preset overrides the resize); only entry 0 is resized to the target size, and a missing inference `image_size` is taken from it. | Dataset/user. | TIE VAE/multimodal conditioning, edit annotations, `RationalRewardsEditReward`; the `reference` role below. |
| `inpaint_mask` | Raw mask (file inputs arrive as RGB/RGBA; L/LA only from tensors) becomes pixel-resolution **`[1, H, W]` luminance** after processing. White means editable, black means fixed; alpha is ignored. | Dataset/user, then `InpaintProcessor._prepare_inpaint_mask`. | `Flux1FillAdapter` packs the pixel mask; processor also produces `inpaint_mask_latents` for Differential Diffusion. |
| `layer_images` | Ordered target layers during training; decoded generated layers afterward. Qwen layers are full-frame. Efficient-layered rewrites the training list into RGBA crops aligned to the rescaled boxes and decodes one RGBA crop per box, so entry 0 is the full-frame background. | Dataset/user, `PrismLayersProSource`, layered decoder. | Layered target encoding, annotations/serving and output sinks. |
| `base_image` | Decoded first full-frame reconstruction from Qwen Layered. | `QwenImageLayeredProcessor.decode_output`. | Layered annotations, serving and output sinks. Efficient-layered does not emit it; its background is `layer_images[0]`. |
| `annotated_image` | Optional Efficient-layered RGBA box/label visualization drawn on the resized image. | Input extra, overwritten at inference when `save_annotated_image` is enabled; training never produces it. | Retained extras/output sinks; no model consumer. |
| `target_image` | VAE-training target: fp32 **`[1, 4, H, W]` in `[-1, 1]`**, after resize/crop and optional RGBA background augmentation. | `VaeTargetResample` / `prepare_vae_target_image` (the VAE trainer's `prepare_row`), or a preprepared sample. | `VaeTrainer.train_step` and VAE losses. Separate from diffusion `clean_latents`. |

Rewards and sampler components that need one of these images select it by
[condition-image role](#condition-image-roles) rather than by key.

Sources: processor tasks [t2i](../flow_control/processors/tasks/t2i.py),
[t2i_control](../flow_control/processors/tasks/t2i_control.py),
[inpaint](../flow_control/processors/tasks/inpaint.py), [tie](../flow_control/processors/tasks/tie.py),
[qwen_layered](../flow_control/processors/tasks/qwen_layered.py),
[efficient_layered](../flow_control/contrib/efficient_layered/processor.py);
[row stream](../flow_control/data/stream.py),
[RGBA VAE data helpers](../flow_control/contrib/rgba_vae_training/data.py),
[RGBA VAE trainer](../flow_control/contrib/rgba_vae_training/trainer.py).

## Encoded model conditions

These are model-family contracts. A processor preset and adapter must agree on
embedding layout, latent normalization, geometry and special tokens.

| Key | Meaning / representation | Producer | Consumers |
| --- | --- | --- | --- |
| `prompt_embeds` | Usually `[1, L, E]` text/multimodal embeddings. Krea2 uses **`[1, L, K, E]`** for K selected hidden layers. Efficient-layered concatenates per-layer sequences. | `encode_prompt` / encoder preset; Efficient-layered calls the encoder per layer and never `encode_prompt`. | All current adapter families except HiDream-O1. |
| `pooled_prompt_embeds` | `[1, Epool]` pooled text condition. `encode_prompt` emits `None` without one; Krea2, HiDream and Efficient-layered omit the key entirely. | CLIP-containing prompt encoders/presets. | FLUX.1 and SD3 require the tensor; other adapters ignore it. |
| `prompt_embeds_mask` | Krea2 bool `[1, L]`; true marks real tokens. | Krea2 prompt encoder. | `Krea2Adapter` encoder attention mask. |
| `control_latents` | Packed normalized control image `[1, N, D]`; may be a posterior cache. | `T2IControlProcessor`. | FLUX.1 `d_concat` concatenates features; `n_concat` concatenates tokens. Returned velocity covers only the target. |
| `inpaint_latents` | Packed **full source image** `[1, N, D]`, not a zero-masked image; may be a posterior cache. | `InpaintProcessor`, from `inpaint_image` at inference and `clean_image` in training. | `Flux1FillAdapter` condition; default `DifferentialDiffusion` source via the `inpaint` role. |
| `inpaint_mask_latents` | `[1, N, p*p]` packed mask: resized pixel luminance is bilinearly downsampled to the VAE grid, then packed with the latent patch order. | `InpaintProcessor`. | `DifferentialDiffusion` expands over latent channels to control reference release. It is not the FLUX Fill pixel-mask packing. |
| `reference_latents` | Ordered list of `[1, Ni, D]` encoded references; entries may be posterior caches. HiDream entries are scaled packed pixels. | `TIEProcessor` / preset. | FLUX.1 Kontext, FLUX.2, LongCat Edit, Qwen Edit and HiDream adapters; static reference tokens accompany the noisy target. |
| `reference_sizes` | Ordered list of reference pixel `(H, W)`, matching `reference_latents`. | `TIEProcessor` after resizing. | Reference geometry/position IDs in the same edit adapters. |
| `image_latents` | Packed full source-image condition, `[1, Nimage, D]`; may be a posterior cache. Distinct from the generated layers. | Qwen/Efficient layered processors. | Layered adapters concatenate it with the noisy target stream. |
| `input_ids` | HiDream int64 `[1, L]` chat-template token IDs, including editing placeholders and target/time markers. Resolution-dependent trailing vision tokens are appended by the adapter. | `HiDreamO1Encoder` / preset. | `HiDreamO1Adapter` jointly trained text/vision transformer; `HiDreamO1FullPreset.get_cost`. |
| `pixel_values` | Optional HiDream editing thumbnail patches in the HF processor's native layout/normalization; not a BCHW `[0, 1]` batch image. | HiDream multimodal prompt encoding. | HiDream SigLIP condition path; paired with `image_grid_thw`. |
| `image_grid_thw` | HiDream integer `[K, 3]` thumbnail grids `(T, H, W)`, corresponding to `pixel_values`. These are not target pixel sizes. | HiDream multimodal prompt encoding. | `HiDreamO1Adapter` forwards it with `pixel_values` to the SigLIP tower and uses it in `_build_sequence` positional indexing. |
| `txt_ids` | Optional text positional-coordinate cache: typically `[L, 3]` for FLUX.1/LongCat, `[B, L, 4]` for FLUX.2. Not vocabulary IDs. | Adapter when absent. | The same adapter's positional embedding path. |
| `img_ids` | Optional image positional-coordinate cache: typically `[Ntotal, 3]` for FLUX.1/LongCat, `[B, Ntotal, 4]` for FLUX.2; includes that variant's extra condition tokens. | Adapter when absent. | The same adapter's positional embedding path. |

`txt_ids` and `img_ids` are normally constructed on the adapter's prepared copy,
not persistent sample state. A supplied cache must match the call's actual text,
image and reference geometry.

Sources: [encoder implementations](../flow_control/processors/components/encoder.py),
[presets](../flow_control/processors/presets.py), adapters
[FLUX.1](../flow_control/adapters/flux1/base.py) and its `d_concat`/`n_concat`/`fill`/`kontext` variants,
[FLUX.2](../flow_control/adapters/flux2/base.py), [Z-Image](../flow_control/adapters/zimage/base.py),
[LongCat](../flow_control/adapters/longcat/base.py), [Qwen Edit](../flow_control/adapters/qwen/edit.py),
[Krea2](../flow_control/adapters/krea2/base.py), [SD3](../flow_control/adapters/sd3/base.py),
[HiDream](../flow_control/adapters/hidream/base.py),
[Efficient-layered](../flow_control/contrib/efficient_layered/adapter.py),
[Differential Diffusion](../flow_control/samplers/projectors.py).

### Layered geometry

| Key | Meaning / representation | Producer | Consumers |
| --- | --- | --- | --- |
| `num_layers` | Qwen requested layer count, defaulting to processor configuration for inference and `len(layer_images)` for training. The generated stream contains **one base frame plus this many layers**, all at `image_size`. | Dataset/user, then `QwenImageLayeredProcessor`. | Qwen Layered initialization, adapter and length estimate; decode infers the frame count from the token count. The separate `image_latents` adds another conditioning frame only inside the adapter. |
| `layer_boxes` | Efficient-layered ordered pixel boxes `(top, bottom, left, right)`; bottom/right are exclusive. Input boxes use source-image coordinates; processing rescales them and aligns to the processor's `multiple_of`, which must stay a multiple of the adapter's 16-pixel grid. The full-frame background normally comes first. | Dataset/user, `PrismLayersProSource`, or processor detection. | Efficient-layered crop encoding, latent initialization, length estimate, adapter positional/attention layout and decode. |
| `layer_prompts` | Ordered per-box text captions, including background. | Dataset/user, Prism dataset or processor captioning. | Efficient-layered inference/training preprocessing encodes each prompt; order must match `layer_boxes`. |
| `text_lengths` | Per-layer text-token lengths; sum equals concatenated `prompt_embeds.shape[1]`. Negative conditions carry their own lengths. | Efficient-layered preprocessing / `generate_negative`. | `EfficientLayeredQwenImageAdapter` text positions and attention mask. |
| `block_mask` | Optional `torch.nn.attention.flex_attention.BlockMask` or `None`, specific to layer/image/text geometry. | Efficient-layered adapter when the key is absent; a supplied mask is reused unchecked, so its producer must drop it on any geometry change. | That adapter's flex-attention call (the adapter is not dense-batchable); normally a transient prepared-batch cache. |

Efficient-layered target `N` is the sum of crop token counts, rather than a base
frame followed by a fixed number of full-size frames. Its decoded `layer_images`
must be interpreted together with `layer_boxes`. Ordinary single-image tiling
cannot infer this geometry from the packed tensor.

Sources: [Qwen Layered adapter](../flow_control/adapters/qwen/layered.py),
[Efficient-layered processor](../flow_control/contrib/efficient_layered/processor.py),
[Efficient-layered adapter](../flow_control/contrib/efficient_layered/adapter.py),
[Prism dataset](../flow_control/contrib/efficient_layered/dataset.py).

## Condition image roles

Consumers that need one condition image take a selector string and resolve it
with `ConditionImage` in [utils/condition_image.py](../flow_control/utils/condition_image.py).
A role names both representations of the same image, so one string configures a
reward (pixels) and a sampler component (latents) alike.

| Selector | Pixel field | Latent field |
| --- | --- | --- |
| `reference`, `reference[i]` | `reference_images[i]` | `reference_latents[i]` |
| `control` | `control_image` | `control_latents` |
| `inpaint` | `inpaint_image` | `inpaint_latents` |
| `clean` | `clean_image` | `clean_latents` |
| any other name, optionally `[i]` | `row[name]` | `row[name]` |

A missing index on a list-valued field selects entry 0. Unknown names are literal
row keys: a processor extension that must supply its own condition image for a
stacked sampling trick writes it under a new key and names that key. The selector
only fetches; it never resizes or re-encodes. Latent consumers compare the result
with `noisy_latents` and report the selector in the error; full-reference IQA
compares pixel shapes.

Current selectors and defaults: `sampler.start.source` (`noisy_latents`, or
`clean` once `strength` is set), `DifferentialDiffusion.source` (`inpaint`),
`CLIPImageSimilarityReward.reference` and `PyIQAReward.reference` (`reference`).
`clean` follows the overwrite rules above: after decode its pixel side is the
generated image, and after RL collection its latent side is the sampled endpoint.

## Tiling dictionaries

| Key / nested key | Meaning | Producer | Consumers |
| --- | --- | --- | --- |
| `tiling` | Serialized `TileLayout` dictionary. | `TiledT2IProcessor`. | `TiledPrediction.bind`; adapters and solvers do not interpret it. |
| `tiling.tile_size` | Integer square tile extent in pixels; an axis shorter than this yields one tile of the axis extent. | Processor `TileConfig`. | `TileLayout.token_specs`. |
| `tiling.overlap` | Minimum overlap in pixels, smaller than `tile_size`; tile starts are spread evenly, so actual overlap can be larger. | Processor `TileConfig`. | Tile planning; stitch ramps follow the actual neighbour overlap. |
| `tiling.stride` | Pixels per packed token: `vae_scale_factor * patch_size`. **Not** the distance between tile origins. | Processor geometry. | Conversion between pixel sizes and the token grid. |
| `tiles` | Raw input: optional row-major list of tile prompt dictionaries (`prompt` required, extras allowed; a declared `image_size` must equal the layout tile size; nested `tiles` are rejected). Processed: complete per-tile condition dictionaries with actual tile `image_size`, prepared through inference preprocessing even in training, so tiles carry no `clean_latents`. Count must match the layout. | Dataset/user, then `TiledT2IProcessor._add_tiles`; omitted input repeats whole-image conditions. | `TiledPrediction` binds one child per tile. |
| `negative.tiles` | Complete row-major negative tile conditions, replacing the positive list under shallow negative overlay. | Tiled processor with `save_negative=True`. | Negative tiled predictor branches. |

`tile_size`, `overlap` and the whole-image `image_size` must align to `stride`; origins and blend weights are
derived from the layout and whole-image size, not stored as additional row keys.
The tiled predictor slices the **current request latents**, invokes its children,
and stitches their fp32 velocities. It does not generically crop every tensor in
the condition dictionary: control, reference and mask inputs need a producer that
supplies meaningful tile conditions for that task.

With a layout but no `tiles`, `TiledPrediction` copies the whole-image conditions
for each tile. It removes `tiling`, `tiles`, `model_image_size`, `negative` and
`clean_latents` from leaf conditions and sets the tile's `image_size`. It requires
a single packed image, `[1, N, D]` with `N` equal to the token grid of
`image_size`. With no layout it passes through to its child.
Metadata only takes effect when the selected prediction tree includes `tiled`;
this is independently selectable for rollout and training.

Sources: [tiled processor](../flow_control/processors/tasks/tiled_t2i.py),
[tile configuration](../flow_control/processors/tiles.py),
[TileLayout](../flow_control/utils/tiling.py), [TiledPrediction](../flow_control/samplers/tiling.py).

## Evaluation metadata and output-only fields

| Key / nested key | Meaning / representation | Producer | Consumers |
| --- | --- | --- | --- |
| `tag` | GenEval task category string; e.g. counting selects the counting threshold. | Dataset extra. | `GenevalReward`. This is top-level, not inside a `metadata` wrapper. |
| `include` | Optional list of expected object groups. Each has `class: str`, `count: int`, optional `color: str` and `position: (relation, target_group_index)`. | Dataset extra. | `GenevalReward` object/count/color/position checks. |
| `exclude` | Optional list of forbidden object-count specs, each with `class: str`, `count: int`. | Dataset extra. | GenEval original scoring mode; reward-server mode does not apply exclude penalties. |
| `style_category` | Prism source style/category metadata. | `PrismLayersProSource`. | Retained extras/generic sinks; no current algorithm consumer. |
| `reward` | Aggregate CPU tensor, normally `[1]`, in an inference report record. | `Inference._reward_fields` from `RewardResult.aggregate`. | `ReportWriter` flattens it into `metrics.jsonl`; `records/` rows keep it for downstream filtering. |
| `reward_raw` | Dictionary of component label to raw CPU tensor `[1]`. Labels come from reward configuration. | `Inference._reward_fields`. | `metrics.jsonl` as `reward_raw.<label>`; downstream analysis. |
| `reward_normalized` | Same label mapping, with normalized component values before weighted aggregation. | `Inference._reward_fields`. | `metrics.jsonl` as `reward_normalized.<label>`; downstream analysis. |
| Other dataset/plugin columns | Open-ended names and values, preserved according to `save_extra`. | Dataset or extension. | Only explicitly configured consumers; add any new in-tree contract to this document. |

Image-only rewards read `clean_image`; text-image rewards also read `prompt`.
Reference-based rewards select their reference by
[condition-image role](#condition-image-roles) and declare that role's pixel
field in `_row_fields`. Rewards that overlap with rollout (remote ones, and the
local `PairwiseReward`, `UnifiedReward` and Rational rewards) and every pairwise
scoring path receive a CPU copy filtered to their declared `_row_fields`, so
retaining a dataset column does not by itself make it available to them. Remote
transport additionally casts float tensors to bf16.

`sampler.start.source` is a condition-image selector, so it can also name a
custom tensor field. Its producer must supply the same packed
coordinates/geometry as the model: without `strength` it is the starting state,
while with `strength` it is a clean source to re-noise. Literal keys are one
reason the inventory cannot close the dictionary.

Sources: [GenEval](../flow_control/rewards/geneval.py),
[CLIP image similarity](../flow_control/rewards/clip_image_similarity.py),
[Rational rewards](../flow_control/rewards/rational_rewards.py),
[IQA](../flow_control/contrib/iqa/reward.py), [reward base](../flow_control/rewards/base.py),
[inference output](../flow_control/training/inference.py).

## Values that are not row fields

`ModelCall` carries `timestep` and `variant` as separate arguments. `EvalRequest`
carries the current latents, sigma, next sigma, solver evaluation parameters and
the branch's `variant`, which guidance sets and the model leaf forwards.
RNG, transition position and solver history belong to `StepContext`; guidance
history belongs to the closure created by each predictor binding. Velocities are
return values. The executed plan belongs to `SampleRun`, and collector records
are separate objects.

RL `Rollout` objects own reward results, recorded steps and their own copy of
the rollout `row`/`negative_row`; advantages are computed separately and
passed to the per-algorithm train items, and replay plans live on `ReplayItem`.
None of these are injected into `row`.
The persisted inference `reward*` fields above are a separate output contract.
Likewise, model outputs such as `image_features`, LLM response keys such as
`bbox_2d`, and raw-directory attachment descriptors (`__type__`, `file`, `shape`,
`dtype`, `value`) belong to other protocols, not the runtime sample dictionary.

## DDNM input/output boundary (planned)

`DDNMProjector` and `DDNMPlusSampler` are not implemented yet. The intended split
is a whole-image, post-prediction projector for the clean-estimate correction,
and a dedicated sampler for DDNM+'s compatible transitions, noise adjustment and
time travel. The latter can restrict its solver/parameterization support.

The future observation contract must specify a retained measurement tensor,
its shape and coordinate space, the degradation operator and pseudoinverse, and
the units of measurement noise. Operator behavior belongs in configuration;
sample-dependent measurements belong in the row and should be chosen by a
[condition-image role](#condition-image-roles): `inpaint`, `control`,
`reference[i]` or a processor-supplied custom key. `clean` is not an immutable
observation, since decode overwrites its pixel side and RL collection its latent
side. No degradation-operator fields are standardized today.

For example, HiDream's `IdentityVAE` uses `z = scaling * (2 * image - 1)`, normally
with `scaling = 1/8`. These packed scaled pixels are not `[0, 1]` pixels. A pixel
operator must either be applied after the inverse affine map or have both its
measurement and noise units transformed consistently, including the offset.
The display decoder's clamp is not part of that affine coordinate conversion.
For nonlinear VAEs, a pixel-space linear operator does not automatically define
a linear operator on the latent stream.

The velocity reparameterization and intended extension point are recorded in
[the sampler design](sampler-plan-design.md#planned-ddnm-extensions); this section
reserves the boundary without inventing fields that existing processors cannot
produce or consume.
