# Sampler nodes

A sampler is a config tree of nodes. Each node is a `BaseNode` whose call returns a `Calls` generator, driven by the Executor ([runtime.md](runtime.md)). Core's nodes come in three layers:

- A **trajectory** moves the row from its σ to wherever the node ends: a time grid, a renoise, a sequence of trajectories.
- A **transition** takes one step from the row's σ to `sigma_next`.
- A **prediction** assembles the velocity at the row's point (model, CFG, tiles) and does not move the row.

The layers are a convention of core's presets, not of the runtime. A method that couples layers, such as a solver whose next σ depends on the last step, is a single `BaseNode`.

| Config field | Type | Read by |
|---|---|---|
| `sampler` | `Trajectory` | `inference`, `flow-control serve` |
| `validation_sampler` | `Trajectory` | trainers with validation |
| `rollout.sampler` | `Trajectory` | rollout trainers (GRPO, NFT, RAM, AWM) |
| `train_predictor` | `Prediction` | SFT and rollout trainers: the velocity at every training point |

Shorthands accepted by the unions:

| Spelling | Means |
|---|---|
| `"euler"` (any tag) | `{"type": "euler"}` with default fields |
| `[a, b]` as a `Trajectory` | `{"type": "sequential", "steps": [a, b]}` |
| `4.5` as a `Prediction` | `{"type": "cfg", "scale": 4.5}` |
| `3.0` as a `Shift` | `{"type": "constant", "shift_value": 3.0}` |

## Config examples

### Plain grid (FLUX.1)

```jsonc
"sampler": {
  "type": "grid",
  "steps": 20,
  "shift": "linear" // resolution-dependent shift
  // "sigma_end": 0.0,
  // "transition": "euler" // {"type": "euler", "eta": 0.0, "prediction": "model"}
}
```

### True CFG

```jsonc
"sampler": {
  "type": "grid",
  "steps": 28,
  "shift": 3.0, // constant shift factor
  "transition": {
    "type": "euler",
    "prediction": { // shorthand: "prediction": 7.0
      "type": "cfg",
      "scale": 7.0
      // "prediction": "model",
      // "negative_condition": "negative", // the row's "negative" conditioning: needs processor.save_negative
      // "renorm": false
    }
  }
}
```

### CFG++

```jsonc
"transition": {
  "type": "euler", // or "ddim"; cfg_pp raises under any other step
  "prediction": {"type": "cfg_pp", "scale": 0.5}
}
```

### Autoguidance and signed CFG

A `cfg` whose `negative` is a `model` leaf under another variant, with `"negative_condition": "positive"`. See [../adapters/variants.md](../adapters/variants.md).

### Piecewise guidance

```jsonc
"sampler": [ // {"type": "sequential", "steps": [...]}
  {
    "type": "grid", "steps": 15, "shift": 3.0,
    "sigma_end": 0.5, // σ after step 15 of the 20-step grid at shift 3: both grids lie on one curve
    "transition": {"type": "euler", "prediction": 4.0}
  },
  {"type": "grid", "steps": 5, "shift": 3.0, "transition": {"type": "euler", "prediction": 2.0}}
]
```

### SDEdit

```jsonc
"sampler": [
  {"type": "init_latent", "from": "clean", "sigma_end": 0.6}, // the row's clean_latents (training-mode preprocessing), renoised to σ = 0.6
  {"type": "grid", "steps": 12, "transition": {"type": "euler", "prediction": 4.5}} // steps counts the steps taken from 0.6
]
```

`from` is a condition-image selector: a role (`clean`, `reference`, `control`, `inpaint`), optionally indexed (`reference[1]`), or a row key.

### Time travel

```jsonc
"sampler": [
  {"type": "grid", "steps": 10, "sigma_end": 0.3},
  {"type": "renoise", "sigma_end": 0.5}, // forward process back up to σ = 0.5, no model call
  {"type": "grid", "steps": 10}
]
```

### Inversion

```jsonc
"sampler": [
  {"type": "init_latent", "from": "clean", "sigma_end": 0.0},
  {"type": "grid", "steps": 20, "sigma_end": 1.0}, // sigma_end above the row's σ: the ODE runs upward
  {"type": "grid", "steps": 20}
]
```

### Official sigma grid (HiDream-O1)

```jsonc
"sampler": {
  "type": "flow_unipc", // a multistep grid: "prediction" instead of "transition"
  "steps": 50,
  "custom_sigmas": [0.999889, 0.993131, /* 47 more */ 0.057674, 0.0], // steps + 1 values; replaces shift
  "order": 2,
  "prediction": {"type": "cfg", "scale": 5.0}
}
```

### Multistep solvers in rollouts

```jsonc
"rollout": {
  "type": "pointwise",
  "sampler": {"type": "dpm", "steps": 10, "shift": 3.0, "order": 2}, // DPM-Solver++(2M), NFT
  "reward": {"type": "pickscore"}
}
```

```jsonc
"sampler": {
  "type": "sa", // SA-Solver, AWM
  "steps": 14,
  "shift": {"type": "constant", "shift_value": 3.0, "diffusers_grid": 1000} // diffusers' set_timesteps grid
  // "eta": 0.4,
  // "stochasticity_start": 0.3,
  // "stochasticity_end": 0.9
}
```

### SDE sampling

```jsonc
"transition": {"type": "euler", "eta": 0.9, "prediction": 4.5} // stochastic on every step but the last
```

### Flow-GRPO SDE window

The window differs per row, so it cannot be a `sequential` of fixed grids. `sde_window_grid` swaps the transition's `eta` inside a window drawn from the row's generator.

```jsonc
"sampler": {
  "type": "sde_window_grid",
  "steps": 10,
  "size": 3, // consecutive stochastic steps; the start is drawn per row
  // "window_range": null, // the window lies inside (0, steps - 1): every step but the last
  "eta": 0.8, // the transition's eta inside the window
  "shift": "linear",
  "transition": {"type": "cps", "eta": 0.0} // must have an `eta` field
}
```

### Tiled CFG

```jsonc
"processor": {"task": "tiled_t2i"}, // rows carry "tiling" and per-tile conditions in "tiles"
"sampler": {
  "type": "grid",
  "steps": 20,
  "transition": {
    "type": "euler",
    "prediction": {"type": "tiled", "prediction": {"type": "cfg", "scale": 2.0}} // CFG per tile
  }
}
```

A `model` leaf raises on a tiled row unless a `tiled` node encloses it.

### Differential diffusion

```jsonc
"processor": {"task": "inpaint"}, // rows carry inpaint_latents and inpaint_mask_latents
"sampler": {
  "type": "grid", // must start at σ = 1
  "steps": 20,
  "transition": {
    "type": "differential_diffusion",
    // "from": "inpaint",
    "transition": {"type": "euler", "prediction": 2.0}
  }
}
```

### A plugin node

```jsonc
"imports": ["flow_control.contrib.momentum_guidance"],
"sampler": {
  "type": "grid",
  "steps": 28,
  "transition": {
    "type": "euler",
    "prediction": {"type": "momentum", "alpha": 0.5, "beta": 0.25, "prediction": 4.5}
  }
}
```

## Python spec

### Layers

```python
class BaseTrajectory(BaseNode):  # moves the row from its σ to wherever the node ends
    def __call__(self, row: Row) -> Calls[None]:
        with self.enter(row) as frame:
            yield from self.run(row, frame)

    def run(self, row: Row, frame: Any, /) -> Calls[None]: ...  # frame: self.frame_type

class StepFrame(Frame):
    sigma: float
    sigma_next: float
    latents: Tensor           # x at sigma
    next_latents: Tensor      # x at sigma_next, set when the step returns
    velocity: Tensor | None   # the step's last velocity; never stored on the row

class BaseTransition(BaseNode):  # one step: the row's σ -> sigma_next
    frame_type = StepFrame

    def __call__(self, row: Row, sigma_next: float) -> Calls[None]:
        with self.enter(row) as step:
            step.sigma, step.sigma_next = row["sigma"], sigma_next
            step.latents, step.velocity = row["noisy_latents"], None
            row["sigma_next"] = sigma_next  # flow-map adapters read the step's target
            try:
                yield from self.run(row, step)
            finally:
                row.pop("sigma_next", None)
            assert row["sigma"] == sigma_next
            step.next_latents = row["noisy_latents"]

    def run(self, row: Row, step: StepFrame) -> Calls[None]: ...
    def is_stochastic(self, step: StepFrame) -> bool: ...                # GRPO records only stochastic steps; default False
    def log_prob(self, point: Row, velocity: Tensor) -> StepLogProb: ...  # GRPO's step likelihood; raises by default

class BasePrediction(BaseNode):  # the velocity at the row's point; the row does not move
    def __call__(self, row: Row) -> Calls[Tensor]:
        with self.enter(row) as frame:
            return (yield from self.run(row, frame))

Trajectory = Annotated[BaseTrajectory, RegistryUnion(trajectory_registry, "type", list_as=("sequential", "steps"))]
Transition = Annotated[BaseTransition, RegistryUnion(transition_registry, "type")]
Prediction = SerializeAsAny[Annotated[BasePrediction, RegistryUnion(prediction_registry, "type", number_as=("cfg", "scale"))]]
```

### Trajectories

```python
class GridFrame(Frame):
    sigmas: list[float]      # the executed grid, steps + 1 values
    index: int               # the current step
    window: tuple[int, int]  # set by SdeWindowGrid

class BaseGrid(BaseTrajectory):  # `steps` evaluations from the row's σ to sigma_end
    steps: int = 50
    sigma_end: float = 0.0
    shift: Shift = ConstantShift()
    custom_sigmas: list[float] | None = None  # steps + 1 values from the row's σ to sigma_end; skips shift
    frame_type = GridFrame

    def sigmas(self, row: Row) -> list[float]:
        start, end = row["sigma"], self.sigma_end
        curve = self.shift.curve(row, num_steps=round(self.steps / abs(start - end)))  # the full 1 -> 0 grid's count
        u = torch.linspace(curve.inverse(start), curve.inverse(end), self.steps + 1)
        return [start, *curve(u)[1:-1].tolist(), end]

@trajectory_registry.register("grid")
class Grid(BaseGrid):
    type: Literal["grid"] = "grid"
    transition: Transition = Euler()

    def run(self, row: Row, grid: GridFrame, /) -> Calls[None]:
        grid.sigmas = self.sigmas(row)
        for grid.index in range(self.steps):
            yield from self.step(row, grid)(row, grid.sigmas[grid.index + 1])

    def step(self, row: Row, grid: GridFrame, /) -> BaseTransition:  # the per-step hook
        return self.transition

@trajectory_registry.register("sde_window_grid")
class SdeWindowGrid(Grid):
    type: Literal["sde_window_grid"] = "sde_window_grid"
    size: int
    window_range: tuple[int, int] | None = None
    eta: float

    def run(self, row: Row, grid: GridFrame, /) -> Calls[None]:
        grid.window = select_sde_window(self.steps, self.size, self.window_range, row["_generator"])
        yield from super().run(row, grid)

    def step(self, row: Row, grid: GridFrame, /) -> BaseTransition:
        start, end = grid.window
        return self._sde if start <= grid.index < end else self.transition  # _sde: transition with eta = self.eta

@trajectory_registry.register("sequential")
class Sequential(BaseTrajectory):
    type: Literal["sequential"] = "sequential"
    steps: list[Trajectory]  # run in order on the same row

@trajectory_registry.register("renoise")
class Renoise(BaseTrajectory):  # the forward process x_σ -> x_σend for σend >= σ; no model call
    type: Literal["renoise"] = "renoise"
    sigma_end: float

    def run(self, row: Row, frame: Frame, /) -> Calls[None]:
        a, b = row["sigma"], self.sigma_end
        if b != a:
            c = (1 - b) / (1 - a)
            row["noisy_latents"] = c * row["noisy_latents"] + math.sqrt(b**2 - (c * a) ** 2) * randn(row)
            row["sigma"] = b
        yield from gather([])

@trajectory_registry.register("init_latent")
class InitLatent(BaseTrajectory):  # x = the source's latents at σ = 0, then renoise to sigma_end
    type: Literal["init_latent"] = "init_latent"
    source: ConditionImageSpec = Field(alias="from")  # see flow_control/utils/condition_image.py
    sigma_end: float = 1.0
```

- A grid whose `sigma_end` is above the row's σ runs the ODE upward (inversion). Steps going up are deterministic.
- `custom_sigmas` sets the row's σ to its first value, because official grids label pure noise just below 1.
- `sde_window_grid` rejects a transition without an `eta` field at validation.

### Shift

A shift maps a uniform `u` grid to σ through an invertible curve. Through `inverse`, a grid that starts or ends mid-way (SDEdit, piecewise guidance) follows the same curve as the full 1 → 0 grid.

```python
class Curve(Protocol):  # σ = F(u) on [0, 1]
    def __call__(self, u: Tensor) -> Tensor: ...  # over a whole u grid, endpoints included
    def inverse(self, sigma: float) -> float: ...

class BaseShift(BaseModel):
    type: str
    image_seq_len_from: Literal["actual", "image_size"] = "actual"  # tokens of noisy_latents, or image_size / 16²; tiles use model_image_size
    shift_terminal: float | None = None  # stretch a grid ending at 0 so its last positive σ is this
    diffusers_grid: int | None = None    # diffusers' set_timesteps grid with this many training steps; full 1 -> 0 grids only

    def curve(self, row: Row, num_steps: int) -> Curve: ...  # default: f(u) = s·u / (1 + (s − 1)·u), s = self.factor(...)
    def factor(self, seq_len: int, num_steps: int) -> float: ...  # what most shifts implement
```

| `type` | Factor `s` |
|---|---|
| `constant` (default, or a bare number) | `shift_value`; 1.0 is no shift |
| `linear` | `exp` of a line through `(base_image_seq_len, base_shift)` and `(max_image_seq_len, max_shift)` |
| `squared` | `sqrt(seq_len / base_image_seq_len)` |
| `flux2` | FLUX.2's empirical `mu`, which depends on `seq_len` and `num_steps` |
| `logit_normal` | overrides `curve`: a logit-normal quantile whose mean rises by `0.5·ln(pixels / base_pixels)` (Ideogram 4); the base fields do not apply |

A minimal shift plugin is `examples/plugins/toy.py`.

### Transitions

```python
class GaussianTransition(BaseTransition):  # one prediction, then a step whose stochastic form is Gaussian
    eta: float = 0.0
    prediction: Prediction = Model()

    def is_stochastic(self, step: StepFrame) -> bool:
        return self.eta != 0.0 and 0.0 < step.sigma_next < step.sigma  # the step to 0 and upward steps are deterministic

    def run(self, row: Row, step: StepFrame) -> Calls[None]:
        step.velocity = yield from self.prediction(row)
        x = row["noisy_latents"]
        if self.is_stochastic(step):
            mean, scale, _ = self.moments(row, step, x, step.velocity)
            row["noisy_latents"] = mean + scale * self.noise(row)
        else:
            row["noisy_latents"] = self.deterministic(x, step.velocity, step)  # default: x + v·(σ_next − σ)
        row["sigma"] = step.sigma_next

    def moments(self, row, step, x, velocity) -> tuple[Tensor, Tensor, Tensor]: ...  # (mean, noise scale, std_dev for GRPO's KL)
    def noise(self, row: Row) -> Tensor: ...                           # default: randn(row)
    def density(self, sample: Tensor, mean: Tensor, std: Tensor) -> Tensor: ...  # default: normal log-prob
    # log_prob(point, velocity) is built from moments + density: replay needs no code of its own

@transition_registry.register("euler")
class Euler(GaussianTransition):  # a member implements moments() only
    type: Literal["euler"] = "euler"

    def moments(self, row, step, x, velocity):
        mean, std_dev, scale = euler_moments(x, velocity, step.sigma, step.sigma_next, self.eta)
        return mean, scale, std_dev
```

| `type` | Step |
|---|---|
| `euler` | flow SDE-Euler (Flow-GRPO) |
| `ddim` | DDIM; its deterministic form is DDIM too |
| `cps` | coefficient-preserving sampling; its likelihood is the unnormalized residual |
| `dance` | DanceGRPO's SDE |
| `flash` | HiDream-O1 Dev: x̂₀, then fresh clipped noise whose scale ramps over the enclosing grid (`noise_scale_start`, `noise_scale_end`, `noise_clip_std`); `eta` defaults to 1 |
| `differential_diffusion` | a wrapper around another transition, below |

```python
@transition_registry.register("differential_diffusion")
class DifferentialDiffusion(BaseTransition):  # keep masked tokens on the noised reference until released, then step
    type: Literal["differential_diffusion"] = "differential_diffusion"
    source: ConditionImageSpec = Field("inpaint", alias="from")
    transition: Transition = Euler()

    def run(self, row: Row, step: StepFrame) -> Calls[None]:
        grid = caller(row, BaseGrid, who="differential_diffusion releases pixels over the grid")
        noise = derived(row, "_noise", lambda: row["noisy_latents"])  # x at the grid's first step, hence σ = 1
        reference = (1 - step.sigma) * source + step.sigma * noise
        keep = (1 - strength) > grid.index / (len(grid.sigmas) - 1)  # strength: row["inpaint_mask_latents"] per token
        row["noisy_latents"] = torch.where(keep, reference, row["noisy_latents"])
        yield from self.transition(row, step.sigma_next)
```

### Predictions

```python
@prediction_registry.register("model")
class Model(BasePrediction):
    type: Literal["model"] = "model"
    variant: str | None = None  # None: the row's RootFrame.variant

    def run(self, row: Row, frame: Frame) -> Calls[Tensor]:
        (velocity,) = yield [ModelCall(row, self.variant)]  # raises on a row with "tiling": tiling is opt-in
        return velocity

@prediction_registry.register("cfg")
class Cfg(BasePrediction):
    type: Literal["cfg"] = "cfg"
    scale: float = 1.0
    prediction: Prediction = Model()
    negative: Prediction | None = None  # the negative branch; defaults to `prediction`
    negative_condition: Literal["negative", "positive"] = "negative"  # "positive": the row itself under `negative`
    renorm: bool = False  # cap each token's norm at |cond| (renorm_eps, renorm_min, renorm_max)

    def run(self, row: Row, frame: Frame) -> Calls[Tensor]:
        if not self.guided():  # scale == 1: the conditional branch alone
            return (yield from self.prediction(row))
        key = f"_uncond:{self.negative_condition}"
        uncond = follow(derived(row, key, lambda: self.uncond_row(row)), row)
        cond, neg = yield from gather([self.prediction(row), (self.negative or self.prediction)(uncond)])
        return self.guide(row, cond, neg)

    def uncond_row(self, row: Row) -> Row: ...  # data(row); "negative": with row["negative"] merged in, raises without it
    def guide(self, row: Row, cond: Tensor, neg: Tensor) -> Tensor:
        return neg + (cond - neg) * self.scale  # then the optional renorm

@prediction_registry.register("cfg_pp")
class CfgPlusPlus(Cfg):  # first-order CFG++: guided at any scale
    type: Literal["cfg_pp"] = "cfg_pp"

    def guide(self, row: Row, cond: Tensor, neg: Tensor) -> Tensor:
        step = caller(row, (Euler, Ddim), who="cfg_pp needs an enclosing euler or ddim step")
        factor = kappa(step.sigma, step.sigma_next, step.node.eta, isinstance(step.node, Ddim))  # preserves the step mean
        return neg + factor * (super().guide(row, cond, neg) - neg)

@prediction_registry.register("tiled")
class Tiled(BasePrediction):  # each tile on its own, then feather-stitch; a pass-through without row["tiling"]
    type: Literal["tiled"] = "tiled"
    prediction: Prediction = Model()

    def run(self, row: Row, frame: Frame) -> Calls[Tensor]:
        stride, specs = tile_specs(row)
        tiles = derived(row, "_tiles", lambda: tile_rows(row, specs, stride))  # conditions from row["tiles"], else the row
        for tile, crop in zip(tiles, extract_tiles(grid, specs), strict=True):
            follow(tile, row)["noisy_latents"] = crop
        velocities = yield from gather([self.prediction(tile) for tile in tiles])
        return stitch_tiles(velocities, specs, height, width)
```

`cfg_pp` raises outside an Euler or DDIM step: inside a multistep grid, and at a training point.

### Multistep grids

A multistep solver needs the history of earlier steps, so it is a grid with a `prediction` instead of a `transition`. Its history is a local of `run` and therefore per row.

```python
class MultistepGrid(BaseGrid):  # one prediction per step
    prediction: Prediction = Model()

@trajectory_registry.register("dpm")
class DpmGrid(MultistepGrid):
    type: Literal["dpm"] = "dpm"
    order: Literal[1, 2]

    def run(self, row: Row, grid: GridFrame, /) -> Calls[None]:
        grid.sigmas = self.sigmas(row)
        previous = None  # (x̂₀, σ) of the last step
        for grid.index in range(self.steps):
            velocity = yield from self.prediction(row)
            ...  # DPM-Solver++ update from x̂₀ and previous; DDIM on the first and last step
            previous = (x0, sigma)
```

| `type` | Solver |
|---|---|
| `dpm` | multistep DPM-Solver++ in x̂₀ form, `order` 1 or 2 |
| `flow_unipc` | UniPC (`order`, `solver_type` `bh1`/`bh2`, `use_corrector`, `lower_order_final`) |
| `sa` | second-order SA-Solver, predictor-corrector; stochastic with `eta` for σ between `stochasticity_start` and `stochasticity_end` |

A multistep grid pushes no `StepFrame` and sets no `sigma_next`. Under it, `cfg_pp` and flow-map adapters (FLUX.3 distilled) raise, and GRPO records no step. Endpoint trainers (NFT, RAM, AWM) read only the endpoint and the executed grid, which an observer on `BaseGrid` keeps in `row["sigmas"]`, so they work with it.

### A custom node

A prediction wrapper, `flow_control/contrib/momentum_guidance.py`:

```python
@prediction_registry.register("momentum")
class Momentum(BasePrediction):
    type: Literal["momentum"] = "momentum"
    prediction: Prediction = Model()
    alpha: float
    beta: float

    def run(self, row: Row, frame: Frame) -> Calls[torch.Tensor]:
        velocity = yield from self.prediction(row)
        previous = row.get("_momentum", velocity)  # node state on the row: per row, and per branch inside a cfg
        row["_momentum"] = (1 - self.beta) * velocity + self.beta * previous
        return velocity + self.alpha * (velocity - previous)
```

A node that couples trajectory and transition. Its next step size depends on the last step's error, which a `grid` + transition split cannot express:

```python
from typing import Literal

from flow_control.samplers import BaseTrajectory, Calls, Frame, Model, Prediction, Row, trajectory_registry


@trajectory_registry.register("adaptive_heun")
class AdaptiveHeun(BaseTrajectory):
    type: Literal["adaptive_heun"] = "adaptive_heun"
    sigma_end: float = 0.0
    first_step: float = 0.05
    tolerance: float = 1e-3
    prediction: Prediction = Model()

    def run(self, row: Row, frame: Frame, /) -> Calls[None]:
        h = self.first_step  # a local of run: per row
        while row["sigma"] > self.sigma_end:
            x, sigma = row["noisy_latents"], row["sigma"]
            sigma_next = max(sigma - h, self.sigma_end)
            v = yield from self.prediction(row)
            euler = x + (sigma_next - sigma) * v
            row["noisy_latents"], row["sigma"] = euler, sigma_next
            v_next = yield from self.prediction(row)
            row["noisy_latents"] = x + (sigma_next - sigma) * (v + v_next) / 2
            error = (row["noisy_latents"] - euler).abs().mean().item()
            h *= min(2.0, (self.tolerance / max(error, 1e-12)) ** 0.5)

    def estimate_calls(self, row: Row) -> int:  # data-dependent: an estimate for progress
        return 2 * round((row["sigma"] - self.sigma_end) / self.first_step)
```

Rows on one rank may take different numbers of steps; the executor batches whatever calls are pending in each round. The node is neither a `BaseGrid` nor a transition, so GRPO records nothing under it and endpoint trainers find no `row["sigmas"]`; it serves inference and validation. Load it by listing its module in the config's `imports` ([../overview/config-plugins.md](../overview/config-plugins.md)).

## Extension points and core contracts

**Extension points**

- `trajectory_registry` (`BaseTrajectory`): implement `run(row, frame)` and leave the row at the node's end σ. Subclass `BaseGrid` to reuse `steps` / `shift` / `custom_sigmas`, or `Grid` and override `step` to pick the transition per step.
- `transition_registry` (`BaseTransition`): implement `run(row, step)` and end with `row["sigma"] == sigma_next`. A `GaussianTransition` subclass implements `moments` (and optionally `deterministic`, `noise`, `density`) and gets GRPO's `log_prob` for free.
- `prediction_registry` (`BasePrediction`): implement `run(row, frame)` and return a velocity without moving the row. Run concurrent branches on derived rows with `gather`.
- `shift_registry` (`BaseShift`): implement `factor(seq_len, num_steps)`, or `curve(row, num_steps)` for a curve of another family.
- Any `BaseNode` whose call returns `Calls`, outside the three layers.
- A node may replace `noisy_latents` and `sigma`, keep state in generator locals, its frame or its own `_` row keys, read conditions through `ConditionImage`, read enclosing frames with `caller`, and draw noise with `randn(row)`. Override `estimate_calls` when the default sum over child nodes is wrong.
- New nodes live in `flow_control/contrib/` or in any module listed in the config's `imports`.

**Core contracts**

- The layer signatures and their postconditions: a trajectory `(row) -> Calls[None]` ends where it says; a transition `(row, sigma_next) -> Calls[None]` ends at `sigma_next` (asserted); a prediction `(row) -> Calls[Tensor]` leaves the row in place.
- `StepFrame` and `GridFrame` fields. GRPO's `record` and replay, the endpoint trainers' `keep_sigmas`, `cfg_pp`, `flash` and `differential_diffusion` read them.
- `BaseTransition.is_stochastic` and `log_prob` are GRPO's interface to a step.
- `row["sigma_next"]` exists exactly inside a transition. Flow-map adapters depend on it.
- A partial grid follows the full 1 → 0 grid's curve, and step-dependent shifts receive the full grid's step count `round(steps / |σ − sigma_end|)`. `shift_terminal` stretches only a grid ending at 0, from that grid's own last positive σ, and `diffusers_grid` raises on a partial grid.
- The union shorthands: a bare tag, a list as `sequential`, a number as `cfg` scale or `constant` shift.
