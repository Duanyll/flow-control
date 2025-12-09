import torch


class EMA:
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
