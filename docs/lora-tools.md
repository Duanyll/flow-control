# Offline LoRA tools

`flow-control lora` converts and fuses transformer LoRAs in a standalone CPU
process. It never installs an external adapter in a training, inference, or
serving model.

The first implementation deliberately supports models whose
`model.hf_model.library` is `diffusers`. Input parsing, PEFT configuration
inference, and compatible non-Diffusers key conversion are delegated to
the architecture's Diffusers pipeline LoRA loader. Flow Control does not
maintain its own key maps. The initial registry covers Flux 1, Flux 2, Krea 2,
Qwen Image, SD3, and Z-Image transformers; an architecture without an official
registered loader fails explicitly. Adapter plugins can register another
official Diffusers pipeline mixin with
`flow_control.scripts.lora.register_diffusers_lora_loader()`.

Only transformer LoRA tensors are accepted. Inputs containing text-encoder or
other component weights are rejected instead of being partially converted.

## DCP to Diffusers

```bash
flow-control lora export config.jsonc \
  --checkpoint-dir runs/example/checkpoints/step_0001000 \
  --weights ema \
  --output-dir export/example-ema
```

`--weights` accepts `current`, `ema`, or `ema_old`. The selected view is saved
as a transformer-only Diffusers pipeline LoRA with `transformer.`-prefixed
keys.

## Diffusers-compatible input to DCP

```bash
flow-control lora import config.jsonc \
  --lora author/model-lora \
  --output-dir checkpoints/model-lora-dcp
```

The output is an ordinary partial DCP checkpoint containing
`app.transformer.*` state. It intentionally omits optimizer, scheduler,
dataloader, RNG, EMA, and step state. Existing DCP loaders keep their initialized
values for absent state, so the checkpoint can initialize or resume a matching
LoRA training configuration. `flow-control lora export` can also consume it to
round-trip back to Diffusers.

Import requires the configured rank, target modules, and LoRA variant to match
the input inferred by Diffusers. Dropout is training behavior rather than part
of the stored delta, so the target config may choose it independently. The DCP
format does not record that configuration separately, so the supplied model
config defines the expected LoRA skeleton when loading it. The converter
verifies that every tensor in that skeleton exists rather than silently
exporting initialized weights.
For standard LoRA, differing alpha/scaling is reparameterized into the B tensor
so the effective delta is preserved; variant scaling that cannot be represented
this way is rejected.

Diffusers pipeline loaders auto-detect the compatible canonical, ComfyUI,
Kohya, or other architecture-specific layouts they support. Unsupported
layouts fail in Diffusers rather than falling back to Flow Control-specific key
rewriting.

## Normalize or fuse

`convert` loads a compatible input and writes canonical transformer-prefixed
Diffusers output:

```bash
flow-control lora convert config.jsonc \
  --lora input.safetensors \
  --output-dir export/normalized
```

`fuse` loads a fresh base transformer on CPU, fuses the adapter, removes PEFT
layers, and writes a standalone Diffusers transformer:

```bash
flow-control lora fuse config.jsonc \
  --lora input.safetensors \
  --scale 0.8 \
  --output-dir export/fused-transformer
```

All commands load the complete base transformer on CPU, which can require
substantial host RAM for large models. They reject an existing output path to
avoid mixing shards or silently overwriting weights; Diffusers LoRA outputs
must use a `.safetensors` file name.

The first version fuses one LoRA per invocation. To bake several LoRAs, point a
new config at the transformer produced by the previous invocation and repeat
`fuse`; no runtime process needs to hold multiple adapters. A fused transformer
is saved directly at the output root, so set `model.hf_model.pretrained_model_id`
to that directory and clear any old `model.hf_model.subfolder`.
