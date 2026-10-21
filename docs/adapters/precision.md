# Model precision

The adapter decides the dtype of every tensor in the transformer: the weights stored on the device, the dtype each forward computes in, trainable parameters, and EMA copies. Low precision stays inside the adapter. On entry, inputs are cast to the compute dtype, and every velocity leaves as fp32, so samplers, solvers and losses never see bf16. Row-level rules (fp32 solver and loss arithmetic, posterior draws, stored dtypes) are in [../overview/row.md](../overview/row.md).

## Config examples

### Default

```jsonc
"model": {"arch": "flux1", "type": "base"}
// hf_model.dtype "bfloat16": load, storage and compute dtype; trainable parameters bfloat16
```

### fp8 storage, bf16 compute

This configuration fits LoRA training of the 12B FLUX.1 on a 24 GB card.

```jsonc
"model": {
  "arch": "flux1", "type": "base",
  "storage_dtype": "float8_e4m3fn", // frozen Linear/Conv weights between forwards; compute stays hf_model.dtype
  "variants": {"default": {"type": "lora", "rank": 64}} // trainable parameters are never downcast
}
```

### fp32 LoRA weights

```jsonc
"model": {
  "arch": "sd3", "type": "base",
  "trainable_dtype": "float32", // LoraPair casts A and B to the input dtype in forward
  "variants": {"default": {"type": "lora", "rank": 32}}
}
```

### fp8 checkpoint

```jsonc
"model": {"arch": "ideogram4", "type": "base"} // weight-only fp8 repo, dequantized to hf_model.dtype at load
```

## Python spec

### Where each dtype comes from

| Tensors | dtype | Set by |
|---|---|---|
| Transformer weights | `hf_model.dtype` (default bf16), which is also `adapter.dtype`, the compute dtype | `HfModelLoader.load_model` |
| Weight-only fp8 checkpoint | dequantized to `hf_model.dtype` (`hf_model.dequantize`, default true) | `HfModelLoader._load_dequantized` |
| Frozen base `nn.Linear` / `nn.Conv*` weights between forwards, and frozen non-EMA full copies | `storage_dtype`, when set and different from `hf_model.dtype` | `apply_layerwise_upcasting` |
| Trainable parameters: LoRA slots, a `trainable` base, `extra_trainable` clones | `trainable_dtype` (default bf16) | `cast_trainable_parameters` |
| EMA of a LoRA variant (slot copies) | fp32 | `VariantsMixin._copy_slots` |
| EMA of full weights (module copy) | fp32, run under `torch.autocast(device, adapter.dtype)` | `VariantsMixin._copy`, `fp32_copy_active` |
| A full variant with its own `hf_model` | that loader's `dtype` | `VariantsMixin._copy` |
| `_keep_in_fp32_modules` of a diffusers model | fp32, also on the meta path; not on the fp8 path | `HfModelLoader` ([adapter.md](adapter.md#meta-init-and-the-seed-checkpoint)) |
| Row inputs (every floating tensor among the data fields) | `adapter.dtype` | `MicrobatchMixin._prepared_batches` |
| `timestep` | `adapter.dtype`, unless `_prepare_timestep` keeps fp32 | `_prepare_timestep` |
| Returned velocity | fp32 | `MicrobatchMixin._checked_velocity` |

```python
# MicrobatchMixin.predict_velocity_batched
prepared = deep_move_to_device(deep_cast_float_dtype(data, self.dtype), self.device)
with torch.autocast(self.device.type, self.dtype, enabled=self.fp32_copy_active()):
    velocity = self._predict_velocity(batch, self._prepare_timestep(timestep))
return velocity.float()

# VariantsMixin.update_variants: EMA accumulates in fp32 whatever the source dtype
torch._foreach_lerp_(ema, [t.float() for t in source], 1 - decay)
```

- `LoraPair.forward` casts A and B to the input's dtype, so fp32 LoRA slots (an EMA, or `trainable_dtype: float32`) compute in `adapter.dtype` without autocast. Other trainable modules (full weights, `extra_trainable` clones) compute in their stored dtype. Autocast is enabled only for a full-weight EMA.
- The timestep reaches the adapter as fp32 `[1]` built from `row["sigma"]`. Near σ = 1, the model time `1 - σ` (or a fine σ grid) is finer than bf16 resolution, and rounding it merges adjacent steps. `Cosmos3Adapter`, `HiDreamO1Adapter` and `Ideogram4Adapter` therefore override `_prepare_timestep` to keep fp32.

### Layerwise upcasting

```python
# BaseModelAdapter.load_transformer, after _realize and cast_trainable_parameters
apply_layerwise_upcasting(
    self.transformer,
    storage_dtype=self.storage_dtype,
    compute_dtype=self.hf_model.dtype,
    # granularity=PYTORCH_LAYER: one hook per nn.Linear / nn.Conv{1,2,3}d / nn.ConvTranspose{1,2,3}d
    skip_modules=self.upcasting_exempt_modules(),  # every LoRA slot and clone, and the fp32 EMA full copies
    # skip_modules_pattern: re.search on the module FQN; default = diffusers' list
    #   ["pos_embed", "patch_embed", "norm", "^proj_in$", "^proj_out$"]
    #   + the model's _skip_layerwise_casting_patterns and _keep_in_fp32_modules
)

class LayerwiseUpcastingHook(ModelHook):
    def init_hook(self, module):          # params and buffers -> storage_dtype
    def pre_forward(self, module, ...):   # -> compute_dtype
    def post_forward(self, module, out):  # -> storage_dtype
    # every cast skips tensors with requires_grad and leaves .grad alone (_module_apply_advanced)
```

- Weights are stored in a low dtype such as fp8 and cast to the compute dtype one layer at a time, for that layer's forward only. The cost is one cast per layer per forward, repeated during gradient-checkpoint recompute.
- diffusers' own layerwise casting is not used because it also casts trainable parameters and their gradients. `utils/upcasting.py` skips both.
- Modules whose name matches a skip pattern keep `hf_model.dtype`. With the default patterns these are the positional and patch embeddings, the input and output projections, and every layer whose name contains `norm`, which includes the AdaLN modulation linears (`norm1.linear`).
- A hook casts only its own layer's weights, not its children's: a `LoraLinear`'s children are variant slots. Variant weights are exempt (`VariantsMixin.upcasting_exempt_modules`): LoRA slots and clones are small, and EMA copies must stay fp32 for `update_variants`. A frozen non-EMA full copy (an autoguidance or uncond transformer) is stored in `storage_dtype` like the base.
- `apply_layerwise_upcasting` warns that fp8 hooks under DDP hit pytorch#162057 on PyTorch 2.8+.

## Extension points and core contracts

**Extension points**

- A family keeps a precision-critical input in fp32 by overriding `_prepare_timestep`, and restores fp32 modules in `_install_modules` ([adapter.md](adapter.md)).
- A checkpoint in another storage format gets an `HfModelLoader` subclass. Its `_convert` maps keys on the dequantizing path.

**Core contracts**

- `predict_velocity_batched` casts inputs to `adapter.dtype` on entry and returns fp32 velocities. Callers never cast for the model.
- EMA weights are fp32, and a full-weight EMA runs under autocast.
- Layerwise upcasting never casts parameters that require grad, or their gradients.
