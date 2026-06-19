from contextlib import contextmanager
from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict

from flow_control.utils.registry import Registry, RegistryUnion


class BaseWarmup(BaseModel):
    """Base class for EMA decay warmup schedules."""

    model_config = ConfigDict(extra="forbid")
    type: str

    def get_decay(self, step: int, base_decay: float) -> float:
        raise NotImplementedError


warmup_registry: Registry[BaseWarmup] = Registry("warmup", base=BaseWarmup)


@warmup_registry.register("none")
class NoWarmup(BaseWarmup):
    type: Literal["none"] = "none"

    def get_decay(self, step: int, base_decay: float) -> float:
        return base_decay


@warmup_registry.register("standard")
class StandardWarmup(BaseWarmup):
    """Original ``(1 + step) / (10 + step)`` schedule."""

    type: Literal["standard"] = "standard"

    def get_decay(self, step: int, base_decay: float) -> float:
        return min((1 + step) / (10 + step), base_decay)


@warmup_registry.register("linear_ramp")
class LinearRampWarmup(BaseWarmup):
    """NFT-style: 0 for *flat_steps*, then linear ramp up to ``EMAConfig.decay``."""

    type: Literal["linear_ramp"] = "linear_ramp"
    flat_steps: int = 0
    ramp_rate: float = 0.001

    def get_decay(self, step: int, base_decay: float) -> float:
        if step < self.flat_steps:
            return 0.0
        return min((step - self.flat_steps) * self.ramp_rate, base_decay)


Warmup = Annotated[BaseWarmup, RegistryUnion(warmup_registry, "type")]


class EMAConfig(BaseModel):
    decay: float = 1.0
    warmup: Warmup = StandardWarmup()
    interval: int = 1


class EMAOptimizer(torch.optim.Optimizer):
    """Standalone EMA optimizer that shares parameters with the main optimizer.

    Does NOT modify parameters. Only reads current param values and maintains
    EMA buffers in its own optimizer state. Compatible with FSDP and DCP.

    Call ``step()`` after the base optimizer's ``step()`` to update EMA buffers.
    Use :meth:`apply_shadow` / :meth:`restore` or the :func:`apply_ema_maybe`
    context manager to temporarily replace parameters with their EMA values.

    Uses ``torch._foreach_lerp_`` for the EMA update (the hot path, called every
    step), which is significantly faster than per-parameter loops when there are
    many parameters. The infrequent shadow/restore operations use per-parameter
    loops for DTensor (FSDP) compatibility.
    """

    def __init__(self, params, ema_config: EMAConfig):
        defaults: dict[str, Any] = {"decay": ema_config.decay}
        super().__init__(params, defaults)
        self.ema_decay = ema_config.decay
        self.ema_warmup = ema_config.warmup
        self.ema_interval = ema_config.interval
        self.ema_step_count = 0

        # Eagerly initialize EMA buffers so that stepping without prior grad is safe
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["ema_buffer"] = p.clone().to(torch.float32)

    @torch.no_grad()
    def _update_ema(self, decay: float):
        """Update EMA buffers using foreach_lerp for batched computation."""
        ema_buffers: list[torch.Tensor] = []
        param_fp32: list[torch.Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                if "ema_buffer" in self.state[p]:
                    ema_buffers.append(self.state[p]["ema_buffer"])
                    param_fp32.append(p.to(torch.float32))

        if ema_buffers:
            # ema = lerp(ema, param, 1 - decay) = decay * ema + (1 - decay) * param
            torch._foreach_lerp_(ema_buffers, param_fp32, 1 - decay)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Read current params, update EMA buffers. Call AFTER base optimizer.step()."""
        self.ema_step_count += 1
        if self.ema_step_count % self.ema_interval != 0:
            return 0.0
        decay = self.ema_warmup.get_decay(self.ema_step_count, self.ema_decay)
        if decay >= 1.0:
            return 0.0
        self._update_ema(decay)
        return 0.0

    def state_dict(self):
        sd = super().state_dict()
        sd["ema_step_count"] = self.ema_step_count
        return sd

    def load_state_dict(self, state_dict):
        self.ema_step_count = state_dict.pop("ema_step_count", 0)
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def coerce_buffer_dtype(self, dtype: torch.dtype = torch.float32) -> None:
        """Re-cast ``ema_buffer`` entries to ``dtype`` after a DCP load.

        ``torch.distributed.checkpoint.set_optimizer_state_dict`` aligns loaded
        optimizer state tensors with the param dtype (bfloat16 for LoRA
        params). We want the EMA shadow to stay in fp32 for numerical
        precision in ``_update_ema``'s ``lerp`` op, so we re-cast once after
        load.
        """
        for group in self.param_groups:
            for p in group["params"]:
                buf = self.state.get(p, {}).get("ema_buffer")
                if buf is not None and buf.dtype != dtype:
                    self.state[p]["ema_buffer"] = buf.to(dtype=dtype)

    @torch.no_grad()
    def apply_shadow(self):
        """Replace parameters with their EMA values, saving current params as backup."""
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "ema_buffer" in param_state:
                    param_state["ema_backup"] = p.clone()
                    p.copy_(param_state["ema_buffer"].to(p.dtype))

    @torch.no_grad()
    def restore(self):
        """Restore parameters from backup after :meth:`apply_shadow`."""
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "ema_backup" in param_state:
                    p.copy_(param_state["ema_backup"])
                    del param_state["ema_backup"]


class InitBackupOptimizer(torch.optim.Optimizer):
    """Standalone optimizer that saves initial parameter values for reference model access.

    Saves a copy of parameters during initialization. Does not modify parameters.
    Use :meth:`apply_init_backup` / :meth:`restore_from_init_backup` or the
    :func:`apply_init_maybe` context manager to temporarily replace parameters
    with their initial (pre-training) values.
    """

    def __init__(self, params):
        super().__init__(params, defaults={})
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["init_backup"] = p.clone().to(torch.float32)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        return

    @torch.no_grad()
    def apply_init_backup(self):
        """Replace parameters with their initial (pre-training) values."""
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "init_backup" in param_state:
                    param_state["init_swap_backup"] = p.clone()
                    p.copy_(param_state["init_backup"].to(p.dtype))

    @torch.no_grad()
    def restore_from_init_backup(self):
        """Restore parameters after :meth:`apply_init_backup`."""
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "init_swap_backup" in param_state:
                    p.copy_(param_state["init_swap_backup"])
                    del param_state["init_swap_backup"]


@contextmanager
def apply_ema_maybe(ema_optimizer: EMAOptimizer | None):
    """Context manager that temporarily applies EMA shadow weights.

    If ``ema_optimizer`` is not None, applies EMA shadow weights and restores
    original weights on exit. If None, this is a no-op.
    """
    if ema_optimizer is not None:
        ema_optimizer.apply_shadow()
    try:
        yield
    finally:
        if ema_optimizer is not None:
            ema_optimizer.restore()


@contextmanager
def apply_init_maybe(init_optimizer: InitBackupOptimizer | None):
    """Context manager that temporarily restores parameters to their initial (pre-training) values.

    If ``init_optimizer`` is not None, applies initial backup weights and restores
    current weights on exit. If None, this is a no-op.
    """
    if init_optimizer is not None:
        init_optimizer.apply_init_backup()
    try:
        yield
    finally:
        if init_optimizer is not None:
            init_optimizer.restore_from_init_backup()


if __name__ == "__main__":
    from pydantic import TypeAdapter
    from rich import print

    adapter = TypeAdapter(Warmup)

    # Each registered warmup parses from its discriminator tag.
    none = adapter.validate_python({"type": "none"})
    standard = adapter.validate_python({"type": "standard"})
    ramp = adapter.validate_python(
        {"type": "linear_ramp", "flat_steps": 5, "ramp_rate": 0.1}
    )
    assert isinstance(none, NoWarmup)
    assert isinstance(standard, StandardWarmup)
    assert isinstance(ramp, LinearRampWarmup)

    # Decay schedules behave as expected.
    assert none.get_decay(0, 0.99) == 0.99
    assert standard.get_decay(0, 0.99) == min(1 / 10, 0.99)
    assert ramp.get_decay(3, 0.99) == 0.0
    assert ramp.get_decay(10, 0.99) == min(5 * 0.1, 0.99)

    # EMAConfig embeds a warmup; passing through an already-built instance works.
    cfg = EMAConfig(decay=0.5, warmup=LinearRampWarmup(flat_steps=0, ramp_rate=0.001))
    assert isinstance(cfg.warmup, LinearRampWarmup)
    cfg_from_dict = EMAConfig.model_validate(
        {"decay": 0.5, "warmup": {"type": "standard"}}
    )
    assert isinstance(cfg_from_dict.warmup, StandardWarmup)

    print("[green]ema warmup registry smoke test passed[/green]")
