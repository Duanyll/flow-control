from contextlib import contextmanager
from typing import Any

import torch
from pydantic import BaseModel
from torch.optim import Optimizer


@contextmanager
def apply_ema_maybe(optimizer: Optimizer):
    if isinstance(optimizer, EMAOptimizer):
        optimizer.apply_shadow()
    try:
        yield
    finally:
        if isinstance(optimizer, EMAOptimizer):
            optimizer.restore()


@contextmanager
def apply_init_maybe(optimizer: Optimizer):
    """Context manager that temporarily restores parameters to their initial (pre-training) values.

    This is useful for reinforcement learning post-training algorithms where we need
    access to the reference model without storing a full copy.
    Only effective if the optimizer was created with `enable_init_backup=True`.
    """
    if isinstance(optimizer, EMAOptimizer):
        optimizer.apply_init_backup()
    try:
        yield
    finally:
        if isinstance(optimizer, EMAOptimizer):
            optimizer.restore_from_init_backup()


class EMAConfig(BaseModel):
    decay: float = 1.0
    warmup: bool = True
    interval: int = 1


class EMAOptimizer(Optimizer):
    """Optimizer wrapper that maintains EMA of parameters.

    EMA state is stored within the optimizer state, and can be automatically
    sharded with FSDP and saved/loaded via DCP.

    Uses ``torch._foreach_lerp_`` for the EMA update (the hot path, called every
    step), which is significantly faster than per-parameter loops when there are
    many parameters. The infrequent shadow/restore operations use per-parameter
    loops for DTensor (FSDP) compatibility.

    Args:
        params: Iterable of parameters to optimize.
        ema_decay: EMA decay factor. EMA is computed as
            ``ema = ema_decay * ema + (1 - ema_decay) * param``.
        enable_init_backup: If True, the optimizer will save a copy of the
            initial parameter values on the first ``step()`` call. These can
            be restored via :meth:`apply_init_backup` / :meth:`restore_from_init_backup`
            or the :func:`apply_init_maybe` context manager. Useful for RL
            post-training where a reference model is needed.
        **kwargs: Extra keyword arguments forwarded to the base optimizer.
    """

    def __init__(
        self,
        params,
        ema_config: EMAConfig,
        enable_init_backup: bool = False,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self.ema_decay = ema_config.decay
        self.ema_warmup = ema_config.warmup
        self.ema_interval = ema_config.interval

        self.enable_init_backup = enable_init_backup
        self.ema_applied = False
        self.init_backup_applied = False
        self.ema_step_count = 0

    @torch.no_grad()
    def apply_shadow(self):
        """Replace parameters with their EMA values, saving current params as backup."""
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "ema_buffer" in param_state:
                    param_state["ema_backup"] = p.clone()
                    p.copy_(param_state["ema_buffer"].to(p.dtype))
        self.ema_applied = True

    @torch.no_grad()
    def restore(self):
        """Restore parameters from backup after :meth:`apply_shadow`."""
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "ema_backup" in param_state:
                    p.copy_(param_state["ema_backup"])
                    del param_state["ema_backup"]
        self.ema_applied = False

    @torch.no_grad()
    def apply_init_backup(self):
        """Replace parameters with their initial (pre-training) values."""
        if not self.enable_init_backup:
            return
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "init_backup" in param_state:
                    param_state["init_swap_backup"] = p.clone()
                    p.copy_(param_state["init_backup"].to(p.dtype))
        self.init_backup_applied = True

    @torch.no_grad()
    def restore_from_init_backup(self):
        """Restore parameters after :meth:`apply_init_backup`."""
        if not self.enable_init_backup:
            return
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "init_swap_backup" in param_state:
                    p.copy_(param_state["init_swap_backup"])
                    del param_state["init_swap_backup"]
        self.init_backup_applied = False

    @torch.no_grad()
    def _save_init_backup(self):
        """Save current parameter values as the initial backup (called on first step)."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["init_backup"] = p.clone().to(torch.float32)

    @torch.no_grad()
    def update_ema(self, decay: float):
        """Update EMA buffers using foreach_lerp for batched computation."""
        # Initialize EMA buffers for new parameters
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "ema_buffer" not in param_state:
                    param_state["ema_buffer"] = p.clone().to(torch.float32)

        # Collect all EMA buffers and corresponding params
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

    def state_dict(self):
        sd = super().state_dict()
        sd["enable_init_backup"] = self.enable_init_backup
        sd["ema_applied"] = self.ema_applied
        sd["init_backup_applied"] = self.init_backup_applied
        sd["ema_step_count"] = self.ema_step_count
        return sd

    def load_state_dict(self, state_dict):
        self.enable_init_backup = state_dict.pop("enable_init_backup", False)
        self.ema_applied = state_dict.pop("ema_applied", False)
        self.init_backup_applied = state_dict.pop("init_backup_applied", False)
        self.ema_step_count = state_dict.pop("ema_step_count", 0)
        super().load_state_dict(state_dict)

    def get_current_decay(self):
        if self.ema_warmup:
            # TODO: make this configurable
            return min(
                (1 + self.ema_step_count) / (10 + self.ema_step_count), self.ema_decay
            )
        else:
            return self.ema_decay

    def step(self, closure: Any = None):
        if self.ema_applied:
            raise RuntimeError("EMA shadow is applied, cannot step optimizer.")
        if self.init_backup_applied:
            raise RuntimeError("Init backup is applied, cannot step optimizer.")

        if self.ema_step_count == 0 and self.enable_init_backup:
            self._save_init_backup()

        loss = super().step(closure)
        self.ema_step_count += 1
        decay = self.get_current_decay()
        if decay < 1.0 and self.ema_step_count % self.ema_interval == 0:
            self.update_ema(decay)

        return loss


def make_ema_optimizer(optimizer_class):
    return type("EMA" + optimizer_class.__name__, (EMAOptimizer, optimizer_class), {})
