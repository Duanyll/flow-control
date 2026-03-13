from contextlib import contextmanager
from typing import Any

import torch
from pydantic import BaseModel


class EMAConfig(BaseModel):
    decay: float = 1.0
    warmup: bool = True
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

    def get_current_decay(self) -> float:
        if self.ema_warmup:
            return min(
                (1 + self.ema_step_count) / (10 + self.ema_step_count), self.ema_decay
            )
        else:
            return self.ema_decay

    @torch.no_grad()
    def _update_ema(self, decay: float):
        """Update EMA buffers using foreach_lerp for batched computation."""
        # Initialize EMA buffers for new parameters
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "ema_buffer" not in param_state:
                    param_state["ema_buffer"] = p.clone().to(torch.float32)

        # Collect all EMA buffers and corresponding params for batched update
        ema_buffers: list[torch.Tensor] = []
        param_fp32: list[torch.Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                ema_buffers.append(self.state[p]["ema_buffer"])
                param_fp32.append(p.to(torch.float32))

        if ema_buffers:
            # ema = lerp(ema, param, 1 - decay) = decay * ema + (1 - decay) * param
            torch._foreach_lerp_(ema_buffers, param_fp32, 1 - decay)

    @torch.no_grad()
    def step(self, closure: Any = None):
        """Read current params, update EMA buffers. Call AFTER base optimizer.step()."""
        loss = super().step(closure)
        self.ema_step_count += 1
        if self.ema_step_count % self.ema_interval != 0:
            return loss
        decay = self.get_current_decay()
        if decay >= 1.0:
            return loss
        self._update_ema(decay)
        return loss

    def state_dict(self):
        sd = super().state_dict()
        sd["ema_step_count"] = self.ema_step_count
        return sd

    def load_state_dict(self, state_dict):
        self.ema_step_count = state_dict.pop("ema_step_count", 0)
        super().load_state_dict(state_dict)

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

    Saves a copy of parameters on the first ``step()`` call. Does not modify parameters.
    Use :meth:`apply_init_backup` / :meth:`restore_from_init_backup` or the
    :func:`apply_init_maybe` context manager to temporarily replace parameters
    with their initial (pre-training) values.
    """

    def __init__(self, params):
        super().__init__(params, defaults={})
        self._initialized = False

    @torch.no_grad()
    def step(self, closure: Any = None):
        loss = super().step(closure)
        if self._initialized:
            return loss
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self.state[p]["init_backup"] = p.clone().to(torch.float32)
        self._initialized = True
        return loss

    def state_dict(self):
        sd = super().state_dict()
        sd["_initialized"] = self._initialized
        return sd

    def load_state_dict(self, state_dict):
        self._initialized = state_dict.pop("_initialized", False)
        super().load_state_dict(state_dict)

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
