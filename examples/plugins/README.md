# Out-of-tree plugins

`flow_control` extension points (model adapters, processor tasks, rewards,
shifts, solvers, encoders, VAEs, weightings, ...) are **open** discriminated
unions backed by a registry (`flow_control/utils/registry.py`). You can add a new
member to any of them from outside the core package — no edits to the core union
literal, no `model_rebuild`.

A *plugin* is just a Python module that, when imported, runs one or more
`@<registry>.register("<tag>")` decorators. To activate it, list the module in a
config's `imports` key; `flow-control` calls `load_plugins(config["imports"])`
before validating the config, so the new tag is available by the time the field
is parsed.

## Minimal example: `toy.py`

[`toy.py`](./toy.py) registers a trivial `Shift` member — the smallest possible
single-member family — under the tag `toy_constant`:

```python
from typing import Literal

from flow_control.samplers.shift import BaseShift, shift_registry


@shift_registry.register("toy_constant")
class ToyConstantShift(BaseShift):
    type: Literal["toy_constant"] = "toy_constant"
    factor: float = 1.0

    def _calculate_shift_factor(self, seq_len: int, num_steps: int) -> float:
        return self.factor
```

## Using it in a config

Declare the module under `imports`, then reference the registered tag wherever a
`Shift` is accepted (here, `sampler.shift`):

```toml
# my_run.toml
imports = ["examples.plugins.toy"]

[sampler]
steps = 20
cfg_scale = 1.0

[sampler.shift]
type = "toy_constant"
factor = 3.0
```

The same works in the JSONC configs the example scripts ship with:

```jsonc
{
  "imports": ["examples.plugins.toy"],
  "sampler": {
    "steps": 20,
    "shift": { "type": "toy_constant", "factor": 3.0 }
  }
}
```

The module name is an ordinary import path resolved from the working directory
(or the installed environment). Running from the repo root, `examples.plugins.toy`
imports as a namespace package — no `__init__.py` or packaging step required.

## In-repo experimental components: `flow_control.contrib`

The same mechanism powers the in-repo experimental package
`flow_control.contrib`, which is intentionally **not** auto-imported by any core
`__init__.py`. For instance, to use the experimental efficient-layered Qwen-Image
adapter and its processor task:

```toml
imports = ["flow_control.contrib.efficient_layered"]
```

That single import registers both `qwen_efficient_layered` (model adapter) and
`efficient_layered` (processor task) — but not the serving UI, so training and
preprocessing runs stay free of the Gradio stack.

The experimental task's Gradio template lives in the `.serving` submodule. A
**serving** config adds it explicitly (importing `.serving` also pulls in the
adapter + processor via the package `__init__`):

```toml
imports = [
  "flow_control.contrib.efficient_layered",          # adapter + processor
  "flow_control.contrib.efficient_layered.serving",  # + Gradio UI template
]
```
