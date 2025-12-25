from contextlib import contextmanager

import torch
from torch.optim import Optimizer


class EMA:
    """
    Conventional implementation of Exponential Moving Average (EMA) for model parameters.
    """

    model: torch.nn.Module
    decay: float
    shadow: dict[str, torch.Tensor]
    backup: dict[str, torch.Tensor]
    active: bool

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.active = False

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().to(torch.float32)

    def update(self):
        if self.active:
            raise RuntimeError("EMA shadow is applied, cannot update.")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                shadow = self.shadow[name]
                param = param.detach().to(torch.float32)
                new_average = (1.0 - self.decay) * param + self.decay * shadow
                self.shadow[name] = new_average

    def apply_shadow(self):
        self.active = True
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param_dtype = param.dtype
                self.backup[name] = param.data
                param.data = self.shadow[name].to(param_dtype)

    def restore(self):
        self.active = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def to(self, device):
        for name, param in self.shadow.items():
            if param.device != device:
                self.shadow[name] = param.to(device)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


@contextmanager
def apply_ema_maybe(optimizer: Optimizer):
    if isinstance(optimizer, EMAOptimizer):
        optimizer.apply_shadow()
    try:
        yield
    finally:
        if isinstance(optimizer, EMAOptimizer):
            optimizer.restore()


class EMAOptimizer(Optimizer):
    """
    Optimizer wrapper that maintains EMA of parameters. EMA state is stored within the
    optimizer state, and can be automatically sharded with FSDP.
    """

    def __init__(self, params, ema_decay, **kwargs):
        super().__init__(params, **kwargs)
        self.ema_decay = ema_decay
        self.ema_applied = False
        self.enable_update = True

    @torch.no_grad()
    def apply_shadow(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "ema_buffer" in param_state:
                    param_state["ema_backup"] = p.clone()
                    p.copy_(param_state["ema_buffer"].to(p.dtype))
        self.ema_applied = True

    @torch.no_grad()
    def restore(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "ema_backup" in param_state:
                    p.copy_(param_state["ema_backup"])
                    del param_state["ema_backup"]
        self.ema_applied = False

    @torch.no_grad()
    def update_ema(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "ema_buffer" not in param_state:
                    param_state["ema_buffer"] = p.clone().to(torch.float32)
                ema_buffer = param_state["ema_buffer"]
                p_fp32 = p.to(torch.float32)
                ema_buffer.mul_(self.ema_decay).add_(p_fp32, alpha=1 - self.ema_decay)

    def step(self, closure=None):
        if self.ema_applied:
            raise RuntimeError("EMA shadow is applied, cannot step optimizer.")
        loss = super().step(closure)
        if self.enable_update:
            self.update_ema()
        return loss


def make_ema_optimizer(optimizer_class):
    return type("EMA" + optimizer_class.__name__, (EMAOptimizer, optimizer_class), {})
