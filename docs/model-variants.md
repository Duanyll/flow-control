# Model variants

A variant is a named weight set of one model adapter. Every variant is resident on the device, so selecting one costs O(1).

## Config Specs Examples

### LoRA RL (NFT / RAM / SIFR)

```jsonc
{
  "model": {
    "type": "flux2",
    "variants": {
      "default": {"type": "lora", "rank": 64},                 // the trained policy
      "ema": {"type": "ema", "of": "default", "decay": 0.999}, // updated every optimizer step
      "old": {"type": "ema", "of": "default", "decay": 0.9, "every": "epoch", "warmup": "standard"},
      "ref": {"type": "alias", "of": "base"}                   // "base" is built in: no variant weights
    }
  },
  "validation_variants": ["ema", "default"] // replaces validation_non_ema
}
```

Objectives name the variants they read (`old`, `ref`). Setup fails if `model.variants` lacks one.

### Full Fine-tuning SFT

```jsonc
{
  "model": {
    "type": "flux2",
    "trainable": true, // the base weights are trainable
    "variants": {
      "default": {"type": "alias", "of": "base"}, // the trained
      "ema": {"type": "ema", "of": "default", "decay": 0.999}
    }
  }
}
```

### LoRA With Extra Trainable Modules

```jsonc
"variants": {
  "default": {
    "type": "lora", "rank": 128, "bias": true,
    "extra_trainable": ["x_embedder", "norm_q", "norm_k"] // cloned from the base
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

A config's `variants` replaces the adapter's default, so a LoRA config redeclares the second transformer (the adapter fills in its fp8 loader). Training the negative branch too needs LoRA on a full variant, which is not implemented yet:

```jsonc
"variants": {
  "default": {"type": "lora", "rank": 64},
  "uncond": {"type": "full", "hf_model": {"subfolder": "unconditional_transformer"}},
  "uncond_lora": {"type": "lora", "of": "uncond", "rank": 64} // TODO: LoRA on a full variant
}
```

### Autoguidance

```jsonc
{
  "model": {
    "variants": {
      "default": {"type": "lora", "rank": 64},
      "guide": {"type": "lora", "path": "runs/x/ckpt-5/default.safetensors", "trainable": false}
      // or a weaker checkpoint of the same architecture:
      // "guide": {"type": "full", "hf_model": {"pretrained_model_id": "..."}}
    }
  },
  "prediction": {
    "type": "cfg", "scale": 2.0,
    "negative_condition": "positive",
    "negative": {"type": "model", "variant": "guide"}
  }
}
```

### Signed CFG Against the Base

```jsonc
"prediction": {
  "type": "cfg", "scale": 2.0,
  "negative_condition": "positive",
  "negative": {"type": "model", "variant": "base"}
}
```

### Inference

```jsonc
"variant": "ema" // engine-level variant for model leaves that name none
```

### Offline Conversion

Diffusers / PEFT formats are converted only by the CLI, never inside the training process group.

```bash
flow-control lora export config.jsonc --checkpoint-dir runs/x/ckpt-30 --variant ema --output-dir out/ema
flow-control lora import config.jsonc --lora author/style-lora --output-dir guides/style  # native file for "path"
```

## Python Spec

### Variant Types

```python
class LoraVariant(BaseVariant):
    type: Literal["lora"] = "lora"
    of: str = "base"  # the full weights it is applied to
    rank: int
    alpha: float | None = None
    bias: bool = False
    target_modules: list[str] | Literal["all-linear"] = "all-linear"
    extra_trainable: list[str] = []  # modules cloned from `of`; the clones belong to this variant
    path: str | None = None  # initial values: a native file or checkpoint
    trainable: bool = True

class FullVariant(BaseVariant):  # exactly one of `of` / `hf_model`
    type: Literal["full"] = "full"
    of: str | None = None  # a copy of this variant, taken at setup
    hf_model: HfModelLoader | None = None  # separately loaded weights
    trainable: bool = False

class EmaVariant(BaseVariant):
    type: Literal["ema"] = "ema"
    of: str = "default"
    decay: float
    every: Literal["step", "epoch"] = "step"
    warmup: Warmup = NoWarmup()

class AliasVariant(BaseVariant):
    type: Literal["alias"] = "alias"
    of: str

Variant = Annotated[BaseVariant, RegistryUnion(variant_registry, "type")]
```

### Adapter API

```python
class BaseModelAdapter:
    trainable: bool = False  # the base weights themselves (full fine-tuning)
    variants: dict[str, Variant] = {}  # "base" is built in; a preset's default (Ideogram 4: "uncond") is replaced, not merged

    def resolve(self, name: str) -> str: ...  # follows aliases; an unknown name raises, listing the variants

    def use_variant(self, name: str | None) -> AbstractContextManager[None]: ...  # sets self.active; None keeps it

    def update_variants(self, every: Literal["step", "epoch"]) -> None:
        for name, ema in self.emas(every):
            decay = ema.warmup.decay(self.counts[name], ema.decay)
            torch._foreach_lerp_(self.weights(name), self.weights(ema.of), 1 - decay)
            self.counts[name] += 1

    def trainable_parameters(self) -> list[nn.Parameter]: ...  # of every trainable variant
```

### Realization

How each variant is realized is the adapter's business. This is the default.

```python
self.active: ContextVar[str] = ContextVar(f"variant:{id(self)}", default="default")  # one per adapter

class LoraLinear(nn.Linear):  # replaces peft in training
    lora: nn.ModuleDict  # variant name -> (lora_A, lora_B); EMA slots kept in fp32

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        if (pair := self.lora.get(self.active.get())) is None:  # "base", a full variant, or not this layer
            return out
        a, b = pair
        return out + self.dropout(x) @ a.T.to(x.dtype) @ b.T.to(x.dtype) * self.scaling

class VariantModule(nn.Module):  # wraps each `extra_trainable` module
    clones: nn.ModuleDict  # variant name -> clone of the base module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.clones.get(self.active.get(), self.base)(*args, **kwargs)

def module(self) -> nn.Module:  # the full weights that run this forward
    return self.full_modules[self.full_of(self.resolve(self.active.get()))]

def checkpoint(fn, *args):  # wraps _gradient_checkpointing_func once, at load
    name = self.active.get()
    return original(lambda *a: run_under(self.active, name, fn, *a), *args)  # recompute sees the forward's variant
```

- `full` variants are separate `fully_shard` units.
- EMA slots stay out of the forward's `fully_shard` group.
- Variant weights and EMA counts are module parameters and buffers, so they live in the model state dict.

### Trainer

```python
for name in {objective.rollout_variant, *objective.required_variants, *self.validation_variants}:
    model.resolve(name)  # at setup

# The runner carries the variant on each row: Runner.submit(..., variant=) sets its RootFrame.variant.
rows = self.runner.submit(..., variant=objective.rollout_variant)  # "default"; NFT / RAM / SIFR: "old"

for name in objective.required_variants:  # e.g. {"old", "ref"}: computed inline, no precompute
    with torch.no_grad(), model.use_variant(name):
        aux[name] = executor.evaluate([self.train_prediction(p) for p in points])
velocities = executor.evaluate([self.train_prediction(p) for p in points])
loss = objective.compute(points, velocities, aux)

optimizer.step()
model.update_variants("step")
...
model.update_variants("epoch")  # at epoch end

for name in self.validation_variants:
    self.runner.submit(..., variant=name)  # validation rows carry their variant the same way
```

`use_variant` is entered in straight-line engine code, never inside a node generator.

### Sampler

```python
Model(variant=None)     # the engine's current variant
Model(variant="guide")  # an explicit name wins
# Executor groups calls by call.variant or root.variant, the name passed to use_variant as given (None: active).
# variants(sampler) walks the config tree for names; the order is the same on every rank.
```

### Removed

- `EMAOptimizer`, `InitBackupOptimizer`, `apply_ema_maybe`, `apply_init_maybe`
- `reference_model`, `_policy_scope`, `POLICY_ROLE_ORDER`, `use_variant`'s peft traversal, `_base_variant_depth`, `_checkpoint_variant`
- `precompute_aux_model_outputs`, `_fill_caches`, `train_plan` caches, `behavior_scope`, `lookahead_teacher` (rollouts always train against the current policy)
- trainer `ema` / `ema_old`, `validation_non_ema`, `optim_ema_old` / `ema_step_count` checkpoint keys
- adapter `peft_lora_rank`, `peft_lora_config` (now a `lora` variant), `all_trainable` (now `trainable`), `extra_trainable_modules` (now `extra_trainable`); peft stays only in `scripts/lora.py`
- `lora export --weights current|ema|ema_old` becomes `--variant <name>`
