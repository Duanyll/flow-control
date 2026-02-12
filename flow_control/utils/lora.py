import contextlib
import contextvars
import math

import torch
import torch.nn as nn

_active_lora_scope: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_active_lora_scope", default=None
)


@contextlib.contextmanager
def lora_scope(scope: str):
    token = _active_lora_scope.set(scope)
    try:
        yield
    finally:
        _active_lora_scope.reset(token)


class LoraLinear(nn.Linear):
    rank: int
    scaling: float
    drouput: nn.Module
    lora_A: nn.Parameter | nn.ParameterDict
    lora_B: nn.Parameter | nn.ParameterDict
    lora_bias: nn.ParameterDict | None
    default_scope: str | None = None

    def init(
        self,
        rank: int,
        scopes: list[str] | None = None,
        lora_alpha: float | None = None,
        lora_bias: bool = False,
        dropout: float = 0.0,
    ):
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scaling = lora_alpha / rank if lora_alpha is not None else 1.0

        if scopes is None:
            self.lora_A = nn.Parameter(torch.zeros((rank, self.in_features)))
            self.lora_B = nn.Parameter(torch.zeros((self.out_features, rank)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            if lora_bias:
                self.bias.requires_grad_(True)

        else:
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            if lora_bias:
                self.lora_bias = nn.ParameterDict()
            else:
                self.lora_bias = None

            self.default_scope = scopes[0]

            for scope in scopes:
                lora_A = nn.Parameter(torch.zeros((rank, self.in_features)))
                lora_B = nn.Parameter(torch.zeros((self.out_features, rank)))
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
                nn.init.zeros_(lora_B)
                self.lora_A[scope] = lora_A
                self.lora_B[scope] = lora_B

                if self.lora_bias is not None:
                    lora_b = nn.Parameter(torch.zeros(self.out_features))
                    self.lora_bias[scope] = lora_b

    def forward(self, input: torch.Tensor, scope: str | None = None) -> torch.Tensor:
        result = super().forward(input)

        active_scope = scope or _active_lora_scope.get() or self.default_scope

        if isinstance(self.lora_A, nn.ParameterDict):
            if active_scope is None:
                raise ValueError(
                    "No active LoRA scope set for LoRA layer with multiple scopes."
                )
            assert isinstance(self.lora_B, nn.ParameterDict)
            assert isinstance(self.lora_bias, (nn.ParameterDict, type(None)))
            lora_A = self.lora_A[active_scope]
            lora_B = self.lora_B[active_scope]
            lora_b = (
                self.lora_bias[active_scope] if self.lora_bias is not None else None
            )
        else:
            lora_A = self.lora_A
            lora_B = self.lora_B
            lora_b = None

        lora_update = self.dropout(input) @ lora_A.T @ lora_B.T * self.scaling

        if lora_b is not None:
            lora_update += lora_b

        return result + lora_update
