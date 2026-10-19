# Sampler execution and extensions

## Config Specs Examples

### Simple Flux1 (20 Steps No CFG)

```jsonc
{
  "sampler": {
    "type": "grid",
    "steps": 20,
    "shift": "linear", // Shorthand for {"type": "linear"}
    // "sigma_end": 0.0
    // "transition": "euler" // Shorthand for {"type": "euler"}
  }
}
```

### With True CFG (50 Steps, CFG 2)

```jsonc
{
  "sampler": {
    "type": "grid",
    "steps": 50,
    "shift": "linear", // Shorthand for {"type": "linear"}
    // "sigma_end": 0.0
    "transition": {
      "type": "euler",
      "prediction": {
        "type": "cfg",
        "scale": 2.0,
        // "prediction": "model" // Default to final model call
      }
    }
  }
}
```

### SDEdit

```jsonc
{
  "sampler": [
    { 
      "type": "init_latent",
      "from": "control[0]", // condition_image spec
      "sigma_end": 0.5
    },
    {
      "type": "grid",
      "steps": 20,
      "transition": {
        "type": "euler",
        "prediction": {
          "type": "cfg",
          "scale": 2.0
        }
      }
    }
  ] // Shorthand for {"type": "sequential", "steps": [...]}
}
```

### Time Travel (By JSON Config)

```jsonc
{
  "sampler": [
    { 
      "type": "init_latent",
      "from": "control[0]", // condition_image spec
      "sigma_end": 0.5
    },
    {
      "type": "grid",
      "steps": 5,
      "sigma_end": 0.2
    },
    {
      "type": "renoise",
      "sigma_end": 0.3
    },
    {
      "type": "grid",
      "steps": 5,
      "sigma_end": 0.0
    }
  ]
}
```

### Tiled CFG

```jsonc
{
  "sampler": {
    "type": "grid",
    "steps": 20,
    "shift": "linear", // Shorthand for {"type": "linear"}
    // "sigma_end": 0.0
    "transition": {
      "type": "euler",
      "prediction": {
        "type": "tiled",
        "prediction": {
          "type": "cfg",
          "scale": 2.0
        },
      }
    }
  }
}
```

### Krea2 Piecewise CFG

Dummy example, not real default parameters, just to show how to use piecewise CFG.

```jsonc
{
  "sampler": [
    {
      "type": "grid",
      "steps": 15,
      "sigma_end": 0.2,
      "transition": {
        "type": "euler",
        "prediction": {
          "type": "cfg",
          "scale": 2.0
        }
      }
    },
    {
      "type": "grid",
      "steps": 5,
      "sigma_end": 0.0,
      "transition": {
        "type": "euler",
        "prediction": {
          "type": "cfg",
          "scale": 3.5
        }
      }
    }
  ]
}
```

### Differential Diffusion

Projects between solver steps

```jsonc
{
  "sampler": [
    {
      "type": "grid",
      "steps": 20,
      "transition": {
        "type": "differential_diffusion",
        "from": "inpaint", // condition_image spec
        "transition": {
          "type": "euler",
          "prediction": {
            "type": "cfg",
            "scale": 2.0
          }
        }
      }
    }
  ]
}
```

### SDE and Flow-GRPO SDE Window

```jsonc
{
  "sampler": {
    "type": "grid",
    "steps": 20,
    "transition": {
      "type": "euler",
      "eta": 0.9,
      "prediction": {
        "type": "cfg",
        "scale": 2.0
      }
    }
  }
}
```

The SDE window mechanism selects different eta steps for different rows, so it can not be expressed in a `sequential` sampler. It registers a special `grid` sampler and modifies the `eta` parameter for transition calls.

```jsonc
{
  "sampler": {
    "type": "sde_window_grid",
    "steps": 10,
    "size": 3,
    "window_range": [2, 7],
    "eta": 0.9,
    "transition": {
      "type": "euler",
      "eta": 0.0, // For steps in selected window, sde_window_grid will override this value with the window eta
      "prediction": {
        "type": "cfg",
        "scale": 2.0
      }
    }
  }
}
```

Such mechanism can also be used to generate complex time-travel plans with custom grid.

### DDNM

At velocity:

```jsonc
{
  "sampler": {
    "type": "grid",
    "steps": 20,
    "transition": {
      "type": "euler",
      "prediction": {
        "type": "ddnm",
        "from": "control[0]", // condition_image spec
        "operation": "colorize", // Selects a matrix for projection
        "prediction": {
          "type": "cfg", // So DDNM is applied after CFG
          "scale": 2.0
        }
      }
    }
  }
}
```

At latent, between steps:

```jsonc
{
  "sampler": {
    "type": "grid",
    "steps": 20,
    "transition": {
      "type": "ddnm_latent",
      "from": "control[0]", // condition_image spec
      "operation": "colorize", // Selects a matrix for projection
      "transition": {
        "type": "euler",
        "prediction": {
          "type": "cfg",
          "scale": 2.0
        }
      }
    }
  }
}
```

## Python Spec

Every sampling node is a pydantic config whose `__call__` is a generator. Nodes talk only through the row and the call chain kept on it.

### Row

```python
row["noisy_latents"]  # current x, advanced in place
row["sigma"]          # current σ; the engine starts it at 1.0
row["_generator"]     # per-row RNG for every stochastic node
row["_frame"]         # innermost active Frame, i.e. the call chain
```

A row is one sample at one point. A concurrent branch gets its own derived row, which is built once and kept on the parent.

```python
def derived[T](row: Row, key: str, make: Callable[[], T]) -> T:
    if key not in row:
        row[key] = make()
    return row[key]

def follow(sub: Row, row: Row) -> Row:  # move a derived row to the parent's point and call chain
    sub.update(noisy_latents=row["noisy_latents"], sigma=row["sigma"], _frame=row["_frame"])
    return sub

def data(row: Row) -> Row:  # start of a new point of the same sample: the row without runtime keys
    return {k: v for k, v in row.items() if not k.startswith("_")}
```

### Calls and Executor

```python
@dataclass(slots=True)
class ModelCall:
    row: Row
    variant: str | None = None

type Calls[T] = Generator[list[ModelCall], list[Tensor], T]

executor = Executor(model, variants=variants(sampler))
for row in executor.stream((row, sampler(row)) for row in rows):  # rows in completion order
    ...
# Asserted every round: a row object appears in at most one call.
```

### Node, Frame, caller

```python
class Frame:
    def __init__(self, node: object, parent: Frame | None):
        self.node, self.parent = node, parent
        self.root: RootFrame = parent.root if parent else self

    def snapshot(self) -> Frame: ...  # copy of the chain, tensors dropped; for replay

type Observer = Callable[[Row, Frame], None]

class RootFrame(Frame):  # placed on the row by the engine
    def __init__(
        self,
        engine: object,
        observers: Sequence[tuple[type, Observer]] = (),
        on_call: Callable[[Row], None] | None = None,  # progress; see Progress
    ): ...

class BaseNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    frame_type: ClassVar[type[Frame]] = Frame

    @contextmanager
    def enter(self, row: Row) -> Iterator[Frame]:
        frame = row["_frame"] = self.frame_type(self, row["_frame"])
        try:
            yield frame
        finally:
            row["_frame"] = frame.parent
        for kind, observer in frame.root.observers:  # only on normal return
            if isinstance(self, kind):
                observer(row, frame)

def caller(row: Row, kind: type, who: str) -> Frame:
    """Nearest enclosing frame whose node is a `kind`; otherwise raise, naming `who` and the chain."""
```

### Layers

```python
class BaseTrajectory(BaseNode):  # moves the row from its σ to wherever the node ends
    def __call__(self, row: Row) -> Calls[None]:
        with self.enter(row) as frame:
            yield from self.run(row, frame)

class StepFrame(Frame):
    sigma: float
    sigma_next: float
    latents: Tensor
    next_latents: Tensor
    velocity: Tensor | None  # the step's last velocity; never stored on the row

class BaseTransition(BaseNode):  # one step: row σ -> sigma_next
    frame_type = StepFrame

    def __call__(self, row: Row, sigma_next: float) -> Calls[None]:
        with self.enter(row) as step:
            step.sigma, step.sigma_next = row["sigma"], sigma_next
            step.latents, step.velocity = row["noisy_latents"], None
            yield from self.run(row, step)
            assert row["sigma"] == sigma_next
            step.next_latents = row["noisy_latents"]

class BasePrediction(BaseNode):  # velocity at the row's point; the row does not move
    def __call__(self, row: Row) -> Calls[Tensor]:
        with self.enter(row) as frame:
            return (yield from self.run(row, frame))

Trajectory = Annotated[BaseTrajectory, RegistryUnion(trajectory_registry, "type", list_as=("sequential", "steps"))]
Transition = Annotated[BaseTransition, RegistryUnion(transition_registry, "type")]  # "euler"
Prediction = Annotated[BasePrediction, RegistryUnion(prediction_registry, "type")]  # "model"
```

### Trajectory Nodes

`sigma_end` is the σ the row holds when the node returns. Inversion is a `grid` whose `sigma_end` is above the row's σ.

```python
class GridFrame(Frame):
    sigmas: list[float]
    index: int

@trajectory_registry.register("grid")
class Grid(BaseTrajectory):
    type: Literal["grid"] = "grid"
    steps: int = 50
    sigma_end: float = 0.0
    shift: Shift = ConstantShift()
    custom_sigmas: list[float] | None = None  # skips shift; must run from the row σ to sigma_end
    transition: Transition = Euler()
    frame_type = GridFrame

    def sigmas(self, row: Row) -> list[float]:
        start, end = row["sigma"], self.sigma_end
        if self.custom_sigmas is not None:
            return check_endpoints(self.custom_sigmas, start, end)
        curve = self.shift.curve(row, num_steps=round(self.steps / abs(start - end)))  # full 1 -> 0 steps
        u = torch.linspace(curve.inverse(start), curve.inverse(end), self.steps + 1)
        return [start, *curve(u[1:-1]).tolist(), end]

    def run(self, row: Row, grid: GridFrame) -> Calls[None]:
        grid.sigmas = self.sigmas(row)
        for grid.index in range(self.steps):
            yield from self.step(row, grid)(row, grid.sigmas[grid.index + 1])

    def step(self, row: Row, grid: GridFrame) -> BaseTransition:
        return self.transition

class SdeWindowFrame(GridFrame):
    window: tuple[int, int]

@trajectory_registry.register("sde_window_grid")
class SdeWindowGrid(Grid):
    type: Literal["sde_window_grid"] = "sde_window_grid"
    size: int
    window_range: tuple[int, int] | None = None
    eta: float
    frame_type = SdeWindowFrame

    def run(self, row: Row, grid: SdeWindowFrame) -> Calls[None]:
        grid.window = select_sde_window(self.steps, self.size, self.window_range, row["_generator"])
        yield from super().run(row, grid)

    def step(self, row: Row, grid: SdeWindowFrame) -> BaseTransition:
        start, end = grid.window
        return self._sde if start <= grid.index < end else self.transition

    @cached_property
    def _sde(self) -> BaseTransition:  # setup rejects a transition without `eta`
        return self.transition.model_copy(update={"eta": self.eta})

@trajectory_registry.register("sequential")
class Sequential(BaseTrajectory):
    type: Literal["sequential"] = "sequential"
    steps: list[Trajectory]

    def run(self, row: Row, frame: Frame) -> Calls[None]:
        for child in self.steps:
            yield from child(row)

@trajectory_registry.register("renoise")
class Renoise(BaseTrajectory):  # forward process x_σ -> x_σend for σend >= σ; no model call
    type: Literal["renoise"] = "renoise"
    sigma_end: float

    def run(self, row: Row, frame: Frame) -> Calls[None]:
        a, b = row["sigma"], self.sigma_end
        if b == a:
            return
        c = (1 - b) / (1 - a)
        row["noisy_latents"] = c * row["noisy_latents"] + math.sqrt(b**2 - (c * a) ** 2) * randn(row)
        row["sigma"] = b
        yield from ()

@trajectory_registry.register("init_latent")
class InitLatent(BaseTrajectory):  # x = source at σ = 0, then renoise
    type: Literal["init_latent"] = "init_latent"
    source: ConditionImageSpec = Field(alias="from")
    sigma_end: float = 1.0

    def run(self, row: Row, frame: Frame) -> Calls[None]:
        row["noisy_latents"] = ConditionImage.parse(self.source).latents(row).float()
        row["sigma"] = 0.0
        yield from Renoise(sigma_end=self.sigma_end)(row)
```

Multistep solvers (DPM++, UniPC) are `Grid` subclasses, and their history is a local variable of `run`.

### Shift

```python
class Curve(Protocol):  # σ = F(u); invertible on [0, 1]
    def __call__(self, u: Tensor) -> Tensor: ...
    def inverse(self, sigma: float) -> float: ...

class BaseShift(BaseModel):
    def curve(self, row: Row, num_steps: int) -> Curve: ...
    # F = terminal stretch ∘ (s·u / (1 + (s − 1)·u)); the stretch scale comes from the full num_steps grid
```

Acceptance check: for every shift, a `Grid(steps=n)` from 1 to 0 must produce exactly the sigmas of the current `Sampler.make_sigmas`.

### Transition Nodes

```python
@transition_registry.register("euler")
class Euler(BaseTransition):
    type: Literal["euler"] = "euler"
    eta: float = 0.0
    prediction: Prediction = Model()

    def run(self, row: Row, step: StepFrame) -> Calls[None]:
        step.velocity = yield from self.prediction(row)
        mean, std = euler_moments(row["noisy_latents"], step.velocity, step.sigma, step.sigma_next, self.eta)
        row["noisy_latents"] = mean + std * randn(row) if self.eta else mean
        row["sigma"] = step.sigma_next

@transition_registry.register("differential_diffusion")
class DifferentialDiffusion(BaseTransition):  # project, then step
    type: Literal["differential_diffusion"] = "differential_diffusion"
    source: ConditionImageSpec = Field("inpaint", alias="from")
    transition: Transition = Euler()

    def run(self, row: Row, step: StepFrame) -> Calls[None]:
        row["noisy_latents"] = blend_by_mask(row, self.source, step.sigma)
        yield from self.transition(row, step.sigma_next)
```

### Prediction Nodes

```python
@prediction_registry.register("model")
class Model(BasePrediction):
    type: Literal["model"] = "model"
    variant: str | None = None  # a model variant (model-variants.md); None: the engine's

    def run(self, row: Row, frame: Frame) -> Calls[Tensor]:
        (velocity,) = yield [ModelCall(row, self.variant)]
        return velocity

@prediction_registry.register("cfg")
class Cfg(BasePrediction):
    type: Literal["cfg"] = "cfg"
    scale: float = 1.0
    prediction: Prediction = Model()
    negative: Prediction | None = None  # defaults to `prediction`
    negative_condition: Literal["negative", "positive"] = "negative"

    def run(self, row: Row, frame: Frame) -> Calls[Tensor]:
        if self.scale == 1.0:
            return (yield from self.prediction(row))
        uncond = follow(derived(row, "_uncond", lambda: uncond_row(row, self.negative_condition)), row)
        cond, neg = yield from gather([self.prediction(row), (self.negative or self.prediction)(uncond)])
        return self.guide(row, cond, neg)

    def guide(self, row: Row, cond: Tensor, neg: Tensor) -> Tensor:
        return neg + self.scale * (cond - neg)

@prediction_registry.register("cfg_pp")
class CfgPlusPlus(Cfg):
    def guide(self, row: Row, cond: Tensor, neg: Tensor) -> Tensor:
        step = caller(row, Euler, who="cfg_pp: needs the step's sigma_next and eta")
        return neg + kappa(step.sigma, step.sigma_next, step.node.eta) * self.scale * (cond - neg)

@prediction_registry.register("tiled")
class Tiled(BasePrediction):
    type: Literal["tiled"] = "tiled"
    prediction: Prediction = Model()

    def run(self, row: Row, frame: Frame) -> Calls[Tensor]:
        layout = TileLayout.of(row)
        tiles = derived(row, "_tiles", lambda: [tile_row(row, spec) for spec in layout])
        for tile, spec in zip(tiles, layout, strict=True):
            follow(tile, row)["noisy_latents"] = spec.crop(row["noisy_latents"])
        return layout.blend((yield from gather([self.prediction(tile) for tile in tiles])))

@prediction_registry.register("ddnm")
class Ddnm(BasePrediction):  # replace the range-space part of x̂0
    type: Literal["ddnm"] = "ddnm"
    source: ConditionImageSpec = Field(alias="from")
    operation: DdnmOperation
    prediction: Prediction = Model()

    def run(self, row: Row, frame: Frame) -> Calls[Tensor]:
        velocity = yield from self.prediction(row)
        x0 = row["noisy_latents"] - row["sigma"] * velocity
        return velocity_for_x0(row, self.operation.project(x0, ConditionImage.parse(self.source).latents(row)))
```

### Static Walk

Setup checks and collective requirements are queries over the config tree.

```python
def walk(tree: BaseModel) -> Iterator[BaseModel]: ...  # every node: fields, lists, dicts

def variants(tree: BaseModel) -> list[str | None]:  # same order on every rank
    return list(dict.fromkeys(n.variant for n in walk(tree) if isinstance(n, Model)))

cfg = next(n for n in walk(sampler) if isinstance(n, Cfg))  # serving's guidance slider
```

### Progress

The number of model calls can depend on the row or on runtime choices, so progress is only an estimate. Serving reports it; training does not report per-sample progress.

```python
class BaseNode:
    def estimate_calls(self, row: Row) -> int:  # default: sum over the direct child nodes
        return sum(child.estimate_calls(row) for child in children(self))

class Model:
    def estimate_calls(self, row: Row) -> int:
        return 1

class Grid:
    def estimate_calls(self, row: Row) -> int:
        return self.steps * self.transition.estimate_calls(row)

class Cfg:
    def estimate_calls(self, row: Row) -> int:
        cond = self.prediction.estimate_calls(row)
        return cond if self.scale == 1.0 else cond + (self.negative or self.prediction).estimate_calls(row)

class Tiled:
    def estimate_calls(self, row: Row) -> int:
        return len(TileLayout.of(row)) * self.prediction.estimate_calls(row)

# Executor, after each forward; derived rows share their parent's root
for call in calls:
    if (on_call := call.row["_frame"].root.on_call) is not None:
        on_call(call.row)

# Serving: one counter per request
total, done = sampler.estimate_calls(row), 0

def on_call(_: Row) -> None:
    nonlocal done
    done += 1
    report_progress(min(done / total, 1.0), f"{done}/{total} model calls")

row["_frame"] = RootFrame(self, on_call=on_call)
```

### Consumers

The consumer looks up what it needs and validates it. Nothing is exported specially for it.

```python
# Engine: set the starting point and the chain root, then stream.
for row in rows:
    row.update(noisy_latents=noise(row), sigma=1.0, _generator=generator(row),
               _frame=RootFrame(self, observers=self.observers))

# FLUX.3 distilled (flow map) adapter
step = caller(row, Euler, who="flux3 distilled: a flow map needs the step's target sigma")
if step.node.eta:
    raise ValueError("flux3 distilled is a flow map: deterministic Euler steps only.")
target_timestep = step.sigma_next

# Epoch-dependent reward: the engine wrote row["epoch"] when the rollout was produced.
# Reward components read it from the row; remote rewards get it via _row_fields.

# GRPO: observers = [(Euler, record)] on the rollout RootFrame
def record(row: Row, step: StepFrame) -> None:
    if step.node.eta > 0:
        point = {"noisy_latents": step.latents, "sigma": step.sigma,
                 "next_latents": step.next_latents, "_frame": step.snapshot()}
        point["log_prob"] = step_log_prob(point, step.velocity)
        row.setdefault("trajectory", []).append(point)

def replay(rollout: Row, recorded: Row) -> Calls[Tensor]:  # one new row per replayed point
    point = {**data(rollout), **recorded}
    velocity = yield from point["_frame"].node.prediction(point)
    return step_log_prob(point, velocity)

# Training (SFT / NFT / AWM): a training point is a new row whose chain holds no step.
point = {**data(rollout), "noise": eps, "sigma": t, "_frame": RootFrame(self),
         "noisy_latents": (1 - t) * rollout["clean_latents"] + t * eps}
velocity = yield from self.train_prediction(point)  # cfg_pp and flux3 distilled raise here
```