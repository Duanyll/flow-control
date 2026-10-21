# Processors

A processor handles the work around the model. It encodes a dataset row into the tensors the adapter reads (text embeddings, VAE latents, sizes), and it decodes sampled latents back into images. It is kept apart from the adapter so that each process loads only the half it uses. A trainer that reads a cache never loads the text encoders.

A processor is built from three axes:

| Axis | Selects | Registry | Owns |
|---|---|---|---|
| Task (`task`) | the row schema | `task_registry` (`processors/tasks/`) | input `TypedDict`s, `prepare_*_row`, `get_cost`, `posterior_fields`, `decode_output` / `annotate_output` |
| Preset (`preset`) | the model family | `preset_registry` (`processors/presets/`) | components, geometry (`vae_scale_factor`, `patch_size`, `latent_channels`), resolutions, prompt templates, `save_negative`, family overrides (`encode_prompt`, `get_cost`, `resize_reference_images`, `enhance_input`) |
| Components | the models a processor calls | `vae_registry`, `encoder_registry`; `LLMClient` | the `vae`, `encoder`, `pooled_encoder` and `llm` fields |

## Config Specs Examples

The same `processor` block goes in two places: the `flow-control preprocess` config ([data](../data/data.md)), and the trainer, inference or serve config that consumes its rows.

### Text to Image

```jsonc
"processor": {
  "task": "t2i",
  "preset": "flux1",
  "default_resolution": [512, 512] // [H, W] of inference rows without image_size
  // "save_negative": false,       // the preset's default; a guided `cfg` needs true
  // "noise_device": "device"      // "cpu": the same initial noise on every GPU model
}
```

### Edit With References

```jsonc
"processor": {
  "task": "tie",
  "preset": "qwen_image_edit" // the negative is encoded together with the references
  // "reference_image_resize_mode": "match_latent", // references take the output size; "multiple_of" / "list" size each one alone
  // "max_reference_images": 3                      // the preset's; extra references are dropped with a warning
},
"dataset": {
  "type": "inline", "attachment_dir": "./examples/assets",
  "data": [{"key": "mug", "prompt": "Fill the mug with coffee", "reference_images": ["mug.png"]}]
  // no image_size: the first reference, after resize_image, sets it
}
```

### Captioning and Prompt Enhancement

```jsonc
"processor": {
  "task": "t2i",
  "preset": "flux2_klein_4b",
  "enable_enhance": true, // rewrite every prompt with t2i_enhance_prompt (tie: tie_enhance_prompt) before encoding
  // "t2i_enhance_prompt": "@flux2_t2i_enhance", // "@key" names a PROMPTS template; any other string is the text itself
  // "caption_prompt": "@default_t2i_caption",   // training rows without a prompt are captioned from clean_image
  // "enhance_json_field": null,                 // take this field of a JSON answer
  "llm": {"base_url": "http://127.0.0.1:8000/v1", "api_key": "EMPTY", "max_concurrency": 16}
  // no llm: a generative text encoder (qwen25vl, mistral3) answers; any other encoder raises
}
// preprocess: "processor_concurrency": 16 keeps that many rows awaiting the LLM per worker
```

### Tiled Inference

```jsonc
"processor": {
  "task": "tiled_t2i",
  "preset": "flux1",
  "tile_size": 1024, // pixels; tile_size and overlap are multiples of patch_size * vae_scale_factor
  "overlap": 128
},
"dataset": {"type": "inline", "data": [{
  "key": "city", "prompt": "...", "image_size": [2048, 3072],
  "tiles": [{"prompt": "..."}, ...] // optional: one prompt per tile, row-major; default: every tile shares the row's conditioning
}]},
"sampler": {"type": "grid", "steps": 28, "transition": {"type": "euler", "prediction": {"type": "tiled"}}}
```

The processor plans the layout and writes it into the row. The sampler's `tiled` prediction only reads it, and a bare `model` leaf raises on a tiled row ([nodes](../sampler/nodes.md)).

## Lifecycle

| Process | Loads | Calls |
|---|---|---|
| `flow-control preprocess` worker (`ProcessorStage`), one per `processor_devices` entry | `load_models("encode")` | `prepare_*_row` (per `processing_mode`), `get_cost` |
| Trainer or inference rank (`DataMixin`) | `load_models("decode")`; also `"encode"` once a raw-source dataset is opened (online preprocessing) | online: `prepare_*_row`, `get_cost`; every fetch: `resample`; through the `Runner`: `initialize_latents`, `decode_output` |
| Serving (`ServingEngine`) | both, on `processor_device` (default `device`) | `prepare_inference_row`, `initialize_latents`, `decode_output`; with `offload_processor: true` the processor's models wait on CPU while the DiT samples |

`load_models(mode)` loads the fields named in `_encoding_components` (`vae`, `encoder`, `pooled_encoder`) or in `_decoding_components` (`vae`). `llm` is an HTTP client and loads nothing. Every rank loads its encoders and VAE whole; they are never sharded ([distributed](../training/distributed.md)).

## Python Spec

### Composition

```python
def parse_processor(conf: dict) -> BaseProcessor:  # the parser behind the `Processor` RegistryUnion
    task_ctor = task_registry.get(conf["task"])
    preset_ctor = preset_registry.get(conf["preset"])  # when a preset is given
    ctor = type(f"{Preset}{Task}", (preset_ctor, task_ctor), {})  # preset first in the MRO: its defaults and methods win
    return ctor(**conf)

Processor = Annotated[BaseProcessor, RegistryUnion(task_registry, "task", parser=parse_processor)]
ProcessorConfig = Annotated[dict[str, Any], RegistryUnion(task_registry, "task", parser=...)]  # validated, kept a dict: crosses into spawned preprocess workers
```

### BaseProcessor

```python
class BaseProcessor[TInput: InputRow, TTrainInput: TrainInputRow, TProcessed: ProcessedRow](BaseModel, ABC):
    task: str
    preset: str = ""

    vae: VAE = Flux1VAE()
    encoder: Encoder = T5TextEncoder()
    pooled_encoder: Encoder | None = None
    llm: LLMClient | None = None
    _encoding_components: list[str] = ["vae", "encoder", "pooled_encoder"]
    _decoding_components: list[str] = ["vae"]

    vae_scale_factor: int = 8  # geometry: equal to the adapter's
    patch_size: int = 2
    latent_channels: int = 16
    initial_noise_scale: float = 1.0
    noise_device: Literal["device", "cpu"] = "device"
    target_posterior: PosteriorMode = "distribution"  # clean_latents: (mean, std) where the VAE implements it, else the mode
    condition_posterior: PosteriorMode = "mode"       # references, controls
    posterior_fields: ClassVar[tuple[str, ...]] = ("clean_latents",)

    def load_models(self, mode: Literal["encode", "decode"], device: torch.device | None = None) -> None: ...

    # preprocessing: async for LLM calls; rewrites input fields in place, returns the encoded fields
    async def prepare_inference_row(self, row: TInput) -> ProcessedRow: ...
    async def prepare_training_row(self, row: TTrainInput) -> ProcessedRow: ...
    def get_cost(self, row: TProcessed) -> int: ...  # latent tokens from image_size; tasks add text and reference tokens
    def resample(self, row: Row, generator: torch.Generator) -> Row: ...  # one fp32 draw per posterior_fields entry, in place

    # sampling
    def initialize_latents(self, row, generator=None, device=None, dtype=torch.float32) -> torch.Tensor: ...  # writes row["noisy_latents"]
    def get_negative_row(self, row: ProcessedRow) -> ProcessedRow | None: ...  # a copy with row["negative"] overlaid
    def decode_output(self, output_latent: torch.Tensor, row: TProcessed) -> DecodedRow: ...  # {"clean_image": [1, C, H, W], task extras}
    def annotate_output(self, decoded: DecodedRow, row: TProcessed) -> torch.Tensor: ...  # one preview image; default clean_image

    # helpers for tasks and presets
    def encode_prompt(self, prompt, images=None, system_prompt=None) -> dict: ...  # {"prompt_embeds", "pooled_prompt_embeds"}
    def encode_latents(self, image, posterior="sample") -> torch.Tensor: ...  # VAE encode, then pack
    def decode_latents(self, latents, size) -> torch.Tensor: ...
    def resize_image(self, image) -> torch.Tensor: ...  # resize_mode "list": preferred_resolutions; "multiple_of": multiple_of, total_pixels, no_upscale
    async def chat_completion(self, prompt, images=None, system_prompt=...) -> str: ...  # llm, else a GenerativeEncoder, else raises
```

| Method | Called by |
|---|---|
| `prepare_*_row`, `get_cost` | `ProcessorStage.process`, `DataMixin._preprocess_online`; `ServingEngine.generate` (no cost) |
| `resample` | `DataMixin.prepare_row` at every fetch; seeding and augmentation are in [data](../data/data.md) |
| `initialize_latents` | `Runner._start`, `ServingEngine._sample_and_decode` |
| `decode_output` | `Runner._decode`, which merges the result into the row; serving |
| `annotate_output` | validation with `validation_annotate_images`, inference with `annotate_output_image` |
| `get_negative_row` | `SftTrainer.train_step` with `cfg_drop_prob` |

### A Task

```python
class T2IInputRow(InputRow):  # what a raw source is coerced to in inference mode
    prompt: str
    negative_prompt: NotRequired[str | None]

class T2ITrainInputRow(TrainInputRow):
    prompt: NotRequired[str | None]
    clean_image: ImageTensor  # paths, PIL images, arrays -> [1, C, H, W] in [0, 1]

class T2IProcessedRow(ProcessedRow):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None

@task_registry.register("t2i")
class T2IProcessor(BaseProcessor[T2IInputRow, T2ITrainInputRow, T2IProcessedRow]):
    task: Literal["t2i"] = "t2i"
    save_negative: bool = False

    async def prepare_training_row(self, row: T2ITrainInputRow) -> T2IProcessedRow:
        row["clean_image"] = clean_image = self.resize_image(row["clean_image"])  # rewritten in place
        if row.get("prompt") is None:
            row["prompt"] = await self.chat_completion(self.caption_prompt, images=[clean_image])
        row["prompt"] = prompt = await self.enhance_prompt(row["prompt"])
        result = T2IProcessedRow(
            image_size=(clean_image.shape[2], clean_image.shape[3]),
            clean_latents=self.encode_latents(clean_image, posterior=self.target_posterior),
            **self.encode_prompt(prompt, system_prompt=self.encoder_prompt),
        )
        if self.save_negative:
            result["negative"] = self.encode_prompt(self.default_negative_prompt, system_prompt=self.encoder_prompt)
        return result

    def get_cost(self, row: T2IProcessedRow) -> int:
        return super().get_cost(row) + row["prompt_embeds"].shape[1]
```

A task that encodes more fields through the VAE lists them in `posterior_fields` (`tie` adds `reference_latents`). A task whose output is not one image overrides `initialize_latents`, `decode_output` and `annotate_output` (`qwen_layered`).

### A Preset

```python
@preset_registry.register("qwen_image")
class QwenImagePreset(BaseModel):  # a plain model placed in front of the task class; not a BaseProcessor
    vae: VAE = QwenImageVAE()
    encoder: Encoder = Qwen25VLEncoder()
    default_resolution: tuple[int, int] = (1328, 1328)
    resize_mode: Literal["list", "multiple_of"] = "list"
    preferred_resolutions: ResolutionList = [(1328, 1328), (1664, 928), ...]
    encoder_prompt: PromptStr = parse_prompt("@qwen_image_encoder")
    save_negative: bool = True  # undistilled: guided with true CFG
```

A preset overrides methods when the family departs from one encoder that returns `prompt_embeds`. `sd35_medium` fuses three text encoders in `encode_prompt` and adds them to `_encoding_components`. `hidream_o1_full` emits token ids instead of embeddings, so it overrides `get_cost` as well, and its pixel-space geometry is `vae_scale_factor=1, patch_size=32, latent_channels=3`.

### Components

| Field | Base | Implements |
|---|---|---|
| `vae` | `BaseVAE` (`components/vae.py`, an `HfModelLoader`) | `_encode(images, posterior)` from `[0, 1]` to normalized latents; `_decode` back to `[0, 1]`. `PosteriorMode`: `"distribution"` stacks (mean, std) as `[2, ...]`; a VAE whose std vanishes in bf16 returns the mode for every mode |
| `encoder`, `pooled_encoder` | `BaseEncoder` (`components/encoder/`) | `encode(prompt, images, system_prompt) -> Tensor`; the `GenerativeEncoder` mixin adds `generate`, the fallback of `chat_completion` |
| `llm` | `LLMClient` (`components/llm.py`) | OpenAI-compatible `/chat/completions`: `generate(user_prompt, images, system_prompt)`, `encode_image` for interleaved content; sessions, retries and `max_concurrency` from `utils/aiohttp_session.py` |

`PromptStr` fields (`encoder_prompt`, `caption_prompt`, `*_enhance_prompt`) accept `"@key"` for an entry of `PROMPTS` in `components/prompts.py`, or the literal text.

### Negative Rows

With `save_negative`, `prepare_*_row` writes `row["negative"] = encode_prompt(negative_prompt or default_negative_prompt)`. Training rows always use `default_negative_prompt`. `tie` encodes the references too when `negative_with_images` is set. Consumers overlay the dict shallowly over the row:

- A `cfg` prediction with `scale != 1` and `negative_condition: "negative"` raises `ValueError` ("set processor.save_negative") on a row without `negative`. `negative_condition: "positive"` (autoguidance) needs no negative.
- SFT `cfg_drop_prob` uses `get_negative_row`. Without a negative it warns once and trains on the conditioned row.

The flag takes effect where `prepare_*_row` runs. A cache built without it holds no negatives, whatever the trainer's processor says.

### Tiles

```python
class TileConfig(BaseModel):  # mixed into TiledT2IProcessor
    tile_size: int = 1024
    overlap: int = 128
    def layout(self, stride: int) -> TileLayout: ...  # raises unless tile_size and overlap are multiples of stride

# TiledT2IProcessor, after the t2i prepare_*_row:
layout = self.layout(self.patch_size * self.vae_scale_factor)  # planned on the packed token grid
result["tiling"] = layout.model_dump()  # read by the `tiled` prediction
result["model_image_size"] = tile_size  # resolution-dependent shifts read this, not image_size
result["tiles"] = per_tile_rows         # prepare_inference_row of each row["tiles"] prompt, or copies of the shared conditioning
result["negative"]["tiles"] = ...       # the negative overlay swaps in the negative tiles
```

`utils/tiling.py` owns the plan (`TileLayout.token_specs`, `plan_tiles`) and the feathered stitch (`stitch_tiles`) that the sampler uses.

## Extension Points and Core Contracts

### Extension Points

- **Task**: `@task_registry.register(tag)` on a `BaseProcessor[Input, TrainInput, Processed]` subclass with a `Literal` `task`. It declares the input `TypedDict`s, writes the prepare logic, and adds its tokens to `get_cost` and its VAE fields to `posterior_fields`. It lives in `processors/tasks/`, or in contrib loaded through `imports` (`contrib/efficient_layered/processor.py`). New row keys go into [row](../overview/row.md).
- **Model family**: a preset in `preset_registry`, plus encoder or VAE components when the family's own are not in tree. The adapter half is in [adapter](../adapters/adapter.md).
- **Encoder / VAE**: a member of `encoder_registry` / `vae_registry`. A VAE implements `"distribution"` only when its posterior std survives bf16.
- **Prompt template**: an entry of `PROMPTS`, named as `"@key"`.
- **Augmentation**: a `resample` override ([data](../data/data.md)).

### Core Contracts

- `vae_scale_factor`, `patch_size` and `latent_channels` equal the adapter's. The processed keys a preset emits (`prompt_embeds`, `input_ids`, `txt_ids`, ...) are a contract between that preset and its adapter.
- `prepare_*_row` rewrite the input row in place: the resized images, the captioned or enhanced `prompt`, and the `image_size` they chose. They return only the encoded fields. Callers rely on this split. With `save_extra` the cache keeps the rewritten inputs, and online inference merges `{**row, **processed}`, so rewards read the prompt the model was conditioned on. Preprocessing runs once per cache, so per-draw randomness belongs to `resample`.
- `decode_output` returns `clean_image` as `[1, C, H, W]` in `[0, 1]`. Rewards and reports read it. Task extras are merged into the row next to it.
- `negative` is a shallow overlay of processed keys.
- A cache records the processor config that wrote it, but no code compares that with the consumer's processor. The trainer's task, preset and geometry must agree with the preprocess config.
