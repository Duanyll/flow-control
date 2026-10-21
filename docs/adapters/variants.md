# Model variants

A **variant** is a named weight set of one adapter. It is one of four kinds: a LoRA, an EMA, a second full transformer, or an alias. Every variant stays resident on the device, and `use_variant(name)` selects one in O(1) by setting a ContextVar that the variant-aware modules read during the forward. Variants exist because RL and guidance alternate between several weight sets on every forward (policy, old policy, reference, EMA, guide). The library implements its own LoRA instead of using peft, because peft switches adapters by walking thousands of modules in Python on every switch. peft remains only in the offline LoRA tools.

## Config examples

### LoRA RL (NFT / RAM)

```jsonc
{
  "model": {
    "arch": "sd3", "type": "base",
    "variants": {
      "default": {"type": "lora", "rank": 32, "alpha": 64, "init": "gaussian"}, // the trained policy
      "ema": {"type": "ema", "of": "default", "decay": 0.9, "warmup": "standard"}, // moved every optimizer step
      "old": {"type": "ema", "of": "default", "decay": 0.5, "every": "epoch",
              "warmup": {"type": "linear_ramp", "ramp_rate": 0.001}},
      "ref": {"type": "alias", "of": "base"} // "base" is built in: the loaded weights
    }
  },
  "validation_variants": ["ema"] // default ["default"]; the first logs as val, the others as val/<name>
}
```

Objectives name the variants they read (`old`, `ref`). Setup fails if `model.variants` lacks one.

### Full Fine-tuning SFT

```jsonc
"model": {
  "arch": "flux2", "type": "base",
  "trainable": true, // the base weights are trainable
  "variants": {
    // "default": {"type": "alias", "of": "base"}, // an undeclared "default" is the base weights
    "ema": {"type": "ema", "of": "default", "decay": 0.999} // an fp32 copy of the transformer
  }
}
```

### LoRA With Extra Trainable Modules

```jsonc
"variants": {
  "default": {
    "type": "lora", "rank": 128, "bias": true,
    "extra_trainable": ["x_embedder", "norm_q", "norm_k"] // cloned from the base, trained whole
    // "target_modules": null, // the adapter's lora_target_modules
  }
}
```

### Ideogram 4

```jsonc
{
  "model": {"arch": "ideogram4", "type": "base"}, // the adapter's default variants: "uncond": {"type": "full", "hf_model": {"subfolder": "unconditional_transformer", ...}}
  // The official 48-step schedule guides at 7.0 for 45 steps, then 3.0: two grids on
  // the same logit-normal curve, split at the sigma of step 45 (0.1669 at 1024², 0.2860 at 2048²).
  "sampler": [
    {
      "type": "grid", "steps": 45, "sigma_end": 0.1669,
      "shift": {"type": "logit_normal", "mean": 0.0, "std": 1.5},
      "transition": {"type": "euler", "prediction": {
        "type": "cfg", "scale": 7.0,
        "negative": {"type": "model", "variant": "uncond"},
        "negative_condition": "positive" // the uncond transformer runs image-only on the row itself: no negative prompt
      }}
    },
    {
      "type": "grid", "steps": 3, "sigma_end": 0.0,
      "shift": {"type": "logit_normal", "mean": 0.0, "std": 1.5},
      "transition": {"type": "euler", "prediction": {
        "type": "cfg", "scale": 3.0,
        "negative": {"type": "model", "variant": "uncond"}, "negative_condition": "positive"
      }}
    }
  ]
}
```

A config's `variants` replaces the adapter's default dict, so a LoRA config declares the second transformer again. The adapter fills in its fp8 loader:

```jsonc
"variants": {
  "default": {"type": "lora", "rank": 64},
  "uncond": {"type": "full", "hf_model": {"subfolder": "unconditional_transformer"}}
}
```

LoRA on a full variant (`{"type": "lora", "of": "uncond"}`) raises `NotImplementedError`, so only the conditional transformer can be trained.

### Autoguidance

```jsonc
"model": {
  "arch": "flux1", "type": "base",
  "variants": {
    "default": {"type": "lora", "rank": 64},
    "guide": {"type": "lora", "rank": 64, "trainable": false,
              "path": "guides/ckpt-5/default.safetensors"} // native file: lora export --native
    // or a weaker checkpoint of the same architecture:
    // "guide": {"type": "full", "hf_model": {"library": "diffusers", "class_name": "...", "pretrained_model_id": "..."}}
  }
},
// in the sampler:
"transition": {"type": "euler", "prediction": {
  "type": "cfg", "scale": 2.0,
  "negative_condition": "positive",
  "negative": {"type": "model", "variant": "guide"}
}}
```

### Signed CFG Against the Base

```jsonc
"prediction": {
  "type": "cfg", "scale": 2.0,
  "negative_condition": "positive",
  "negative": {"type": "model", "variant": "base"}
}
```

### Inference and Serving

```jsonc
"variant": "ema" // the variant for model leaves that name none; null: the adapter's "default"
```

### Exchange LoRAs With Diffusers

```bash
# a lora variant (or an EMA of one) in a DCP checkpoint -> Diffusers LoRA (transformer.-prefixed keys)
flow-control lora export config.jsonc --checkpoint-dir runs/x/ckpt-30 --variant ema --output-dir out/ema
# the same slots -> native file out/ema-native/ema.safetensors for a variant's "path"
flow-control lora export config.jsonc --checkpoint-dir runs/x/ckpt-30 --variant ema --output-dir out/ema-native --native
# Diffusers-compatible LoRA (path or Hub id) -> native file out/guide/lora.safetensors for a variant's "path"
flow-control lora import config.jsonc --lora author/style-lora --output-dir out/guide  # --name lora
# any layout the Diffusers loader reads (ComfyUI, Kohya, ...) -> canonical Diffusers LoRA
flow-control lora convert config.jsonc --lora input.safetensors --output-dir out/normalized
# base transformer + one LoRA -> standalone Diffusers transformer
flow-control lora fuse config.jsonc --lora input.safetensors --scale 0.8 --output-dir out/fused
```

## Python spec

### Variant Types

```python
class BaseVariant(BaseModel):
    type: str
    of: str = "base"                         # the variant it builds on

class LoraVariant(BaseVariant):
    type: Literal["lora"] = "lora"
    rank: int
    alpha: float | None = None               # None: rank (scaling 1)
    bias: bool = False
    target_modules: list[str] | Literal["all-linear"] | None = None  # None: the adapter's lora_target_modules
    extra_trainable: list[str] | None = None # modules cloned from `of`; None: the adapter's lora_extra_trainable
    path: str | None = None                  # initial values: a native LoRA file
    init: Literal["kaiming", "gaussian"] = "kaiming"  # A's init; B starts at zero
    trainable: bool = True

class FullVariant(BaseVariant):              # a module of its own
    type: Literal["full"] = "full"
    hf_model: HfModelLoader | None = None    # separately loaded weights; None: a copy of `of`, taken at setup
    trainable: bool = False

class EmaVariant(BaseVariant):
    type: Literal["ema"] = "ema"
    of: str = "default"
    decay: float
    every: Literal["step", "epoch"] = "step"
    warmup: Warmup = NoWarmup()              # "none" | "standard" | "linear_ramp"

class AliasVariant(BaseVariant):
    type: Literal["alias"] = "alias"
    of: str

Variant = Annotated[BaseVariant, RegistryUnion(variant_registry, "type")]
```

| Variant | Realized as (`_kind`) |
|---|---|
| `"base"` (built in) | the loaded transformer, `adapter.transformer` |
| `lora` of `base` | a `LoraPair` slot in every targeted `LoraLinear`, plus a clone in every `extra_trainable` `VariantModule` |
| `full` | its own module, `transformer.variants[name]` |
| `ema` of a `lora` | fp32 frozen copies of the source's slots and clones |
| `ema` of `base` or a `full` | an fp32 frozen copy of the module, in `transformer.variants` |
| `alias` | nothing; `resolve` follows it |

A `lora` on a full variant and a `full` copy of a `lora` raise `NotImplementedError`.

### Adapter API

```python
class VariantsMixin(BaseModel):
    trainable: bool = False                  # the base weights themselves (full fine-tuning)
    variants: dict[str, Variant] = {}        # replaces a family's default (Ideogram 4: "uncond"), not merged
    lora_target_modules: ClassVar[tuple[str, ...] | Literal["all-linear"]] = "all-linear"
    lora_exclude_modules: ClassVar[tuple[str, ...]] = ()
    lora_extra_trainable: ClassVar[tuple[str, ...]] = ()

    def resolve(self, name: str) -> str: ...  # follows aliases; undeclared "default" is "base"; unknown raises
    def use_variant(self, name: str | None) -> AbstractContextManager[None]: ...  # _active.set(resolve(name)); None keeps it
    active_variant: str                       # property: the resolved active name
    def origin(self, name: str) -> str: ...   # the loaded weights under it: "base", or a full variant with its own hf_model
    def module(self) -> nn.Module:            # what _predict_velocity runs
        return self._full.get(self._active.get(), self.transformer)
    def fp32_copy_active(self) -> bool: ...   # a full-weight EMA: forward under autocast
    def trainable_parameters(self) -> list[nn.Parameter]: ...  # every parameter with requires_grad
    def variant_modules(self) -> set[nn.Module]: ...  # checkpointed even when frozen

    @torch.no_grad()
    def update_variants(self, every: Literal["step", "epoch"]) -> None:
        for name, variant in self.variants.items():  # the EmaVariants scheduled `every`
            count = self.transformer.variant_counts.get_buffer(name)
            decay = variant.warmup.get_decay(int(count), variant.decay)
            if decay < 1:  # (ema, source) cached in _ema_tensors on first use: FSDP replaces the parameters after load
                ema, source = self._weights(name), self._weights(self.resolve(variant.of))
                torch._foreach_lerp_(ema, [t.float() for t in source], 1 - decay)
            count += 1
```

### Realization

```python
self._active = ContextVar("variant", default=self.resolve("default"))  # one per adapter

class LoraLinear(nn.Linear):           # each targeted linear, converted in place once
    lora: nn.ModuleDict                # variant name -> LoraPair (A, B, scaling)

    def forward(self, input):
        output = super().forward(input)
        name = self.active.get()
        return output + self.lora[name](input) if name in self.lora else output  # "base", full variants: no slot

class VariantModule(nn.Module):        # wraps each extra_trainable module once
    base: nn.Module
    clones: nn.ModuleDict              # variant name -> clone of base

    def forward(self, *args, **kwargs):
        name = self.active.get()
        return (self.clones[name] if name in self.clones else self.base)(*args, **kwargs)

def pinned_checkpoint(original, active):  # wraps every _gradient_checkpointing_func
    def checkpoint(function, *args, **kwargs):
        name = active.get()

        def pinned(*inputs, **inner):  # recompute runs under the forward's variant
            token = active.set(name)
            try:
                return function(*inputs, **inner)
            finally:
                active.reset(token)

        return original(pinned, *args, **kwargs)
    return checkpoint
```

- `_realize` runs in `load_transformer` after `_install_modules` ([adapter.md](adapter.md#loading)). It creates the full copies first, so they carry no slots, then the clones, so no clone contains slots, then the slots and the EMA slot copies.
- Full variants and full-weight EMAs are separate `fully_shard` units (`BaseTrainer._shard`). A forward under another variant does not gather them.
- LoRA slots, frozen and EMA slots included, sit inside their block's FSDP unit and are gathered with it on every forward.
- Variant weights and EMA counts are parameters and buffers of the transformer. They are in the seed and in checkpoints, and `model_state_dict` saves the frozen ones through `variant_modules`.
- A LoRA `path` is read when the transformer is loaded with real weights: building the seed, or training without one. A meta-init run takes it from the seed, so the seed must be regenerated after `path` changes.

### Trainer

```python
for name in {self._rollout_variant(), *self._required_variants(), *self.validation_variants}:
    self.model.resolve(name)                    # at setup: an undeclared name fails before loading

# objectives declare their variants: rollout_variant() (NFT / RAM: "old"), required_variants() (NFT: {"old", "ref"})
groups = self.runner.submit(..., variant=self._rollout_variant())  # the RootFrame.variant of every rollout row

# EndpointTrainer._loss_batched: auxiliary velocities inline, no precompute
for name in sorted(self.objective.required_variants()):
    with torch.no_grad(), self.model.use_variant(name):
        aux[name] = self.predict_training(points)
current = self.predict_training(points)         # the active variant: "default"

optimizer_step(model, ...)                      # optimizer.step(); model.update_variants("step")
self.model.update_variants("epoch")             # after each outer epoch's training phase
for name in self.validation_variants:
    self.validate_and_log(..., variant=name)
```

`use_variant` is entered in straight-line engine code, never inside a node generator.

### Sampler

```python
Model(variant=None)     # the engine's current variant
Model(variant="guide")  # an explicit name wins
# A call runs under call.variant, else its row's RootFrame.variant; the name goes to use_variant as given.
# predict_velocity_batched runs once per Executor variant, in one order on every rank:
#   variants(tree) for predict_training and serving; [None, "base", *model.variants] in the Runner.
```

`Ideogram4Adapter` builds different inputs depending on `origin(active_variant)`. A family whose variants differ in their inputs reads that, never the variant's name.

### Offline LoRA Tools

`flow_control/scripts/lora.py` runs as a standalone CPU process. It loads the complete base transformer, which needs host RAM.

- Parsing, PEFT config inference and key conversion are delegated to the architecture's Diffusers pipeline LoRA loader. Loaders are registered for FLUX.1, FLUX.2, Krea 2, Qwen-Image, SD3 and Z-Image. Any other transformer class fails until a plugin calls `register_diffusers_lora_loader(class_name, loader_mixin)`.
- Only transformer LoRA tensors are accepted. Text-encoder keys, DoRA scales and any tensor the loader does not materialize are rejected.
- `export` accepts a `lora` variant or an EMA of one. It rejects variants with `extra_trainable` clones, which Diffusers LoRA cannot represent, and checkpoints missing any slot tensor (`app.transformer.<fqn>.lora.<name>.*`).
- A native file loads into a `lora` variant only if its rank, alpha and set of targeted linears match the config (`load_lora`). Set `rank`, `alpha` and `target_modules` to the imported LoRA's.
- Outputs must not exist yet, and Diffusers LoRA outputs are `.safetensors`. `fuse` takes one LoRA per run. To stack several, point `model.hf_model.pretrained_model_id` at the fused output, with no `subfolder`, and fuse again.

## Extension points and core contracts

**Extension points**

- Declaring variants in a config, and naming them from objectives (`rollout_variant`, `required_variants`), validation (`validation_variants`) and sampler nodes (`Model(variant=...)`).
- A family's default `variants` and its LoRA hints (`lora_target_modules`, `lora_exclude_modules`, `lora_extra_trainable`).
- A new EMA warmup: register a `BaseWarmup` subclass with `get_decay(step, base_decay)` in `warmup_registry`.

**Core contracts**

- `variant_registry` is not an extension point. The variant kinds are implemented inside `VariantsMixin` and are tightly coupled through `_kind`, the `_realize` order, `_weights`, sharding and checkpointing. A new kind is a core change.
- Selection mechanism: `use_variant` sets one ContextVar per adapter, `LoraLinear` / `VariantModule` read it, `module()` returns the full module of the active variant, and `pinned_checkpoint` replays it on recompute. Adapters run `self.module()`.
- `"base"` is built in, and an undeclared `"default"` resolves to it. A config's `variants` replaces the family default.
- Variant weights and counts live in the transformer's state dict, which is the checkpoint and seed format.
- EMA weights are fp32 ([precision.md](precision.md)).
