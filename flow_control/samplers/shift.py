import math
from typing import Literal

import torch

from flow_control.adapters import Batch, ModelAdapter

from .simple_euler import SimpleEulerSampler


class ShiftedEulerSampler(SimpleEulerSampler):
    """
    Shifted Euler Sampler for Flow Matching Models.
    """

    # --- 基础配置 ---
    steps: int = 28
    shift_strategy: Literal["linear", "squared", "constant", "none"] = "linear"
    """
    Shift strategy to use:
    - "linear": Shift factor increases linearly with sequence length (Flux.1, Qwen-Image).
    - "squared": Shift factor increases with the square root of sequence length (Qwen-Image-Layered).
    - "constant": Use a constant shift factor defined by `shift_value`. (SD3)
    - "none": No shift applied.
    """
    latent_length_from: Literal["actual", "image_size"] = "actual"
    """
    Method to determine latent sequence length:
    - "actual": Use the actual latent sequence length from the input tensor.
    - "image_size": Calculate sequence length based on the original image size and patch size.
    """

    # --- 策略参数 (Flux / Qwen 常用) ---
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096  # 8192 for Qwen-Image
    base_shift: float = 0.5
    max_shift: float = 1.15  # 0.9 for Qwen-Image

    # --- 策略参数 (Constant 常用) ---
    shift_value: float = 1.0
    """
    Constant shift value when using 'constant' shift strategy.
    """

    # --- 额外的高级参数 ---
    # 是否将 sigma 的终点拉伸到特定值 (Diffusers 中的 shift_terminal)
    shift_terminal: float | None = None
    # 0.02 for Qwen-Image

    def sample(
        self,
        model: ModelAdapter,
        batch: Batch,
        negative_batch: Batch | None = None,
        t_start: float = 1.0,
        t_end: float = 0.0,
    ) -> torch.Tensor:
        # 1. 获取序列长度 (H * W / P^2)
        if self.latent_length_from == "actual":
            seq_len = batch["noisy_latents"].shape[1]
        else:
            h, w = batch["image_size"]
            seq_len = h * w // 256  # assuming patch size 16x16

        # 2. 生成并处理 Sigmas
        sigmas = self._make_shifted_sigmas(seq_len, t_start, t_end)

        # 3. 调用父类的采样循环
        return self._euler_sample(model, batch, sigmas, negative_batch)

    def _make_shifted_sigmas(
        self, seq_len: int, t_start: float, t_end: float
    ) -> torch.Tensor:
        # 生成基础的时间步 (线性)
        t = torch.linspace(t_start, t_end, self.steps + 1)

        # 计算 Shift Factor (S = exp(mu))
        shift_factor = self._calculate_shift_factor(seq_len)

        # 应用 Shift 变换
        # 公式: t_new = (S * t) / (1 + (S - 1) * t)
        # 这等价于 diffusers 的 time_shift(mu, 1.0, t) 或 standard shift
        if shift_factor != 1.0:
            t = (shift_factor * t) / (1 + (shift_factor - 1) * t)

        # (可选) Stretch to terminal
        # 对应 diffusers 的 stretch_shift_to_terminal
        if self.shift_terminal is not None:
            one_minus_z = 1 - t
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            t = 1 - (one_minus_z / scale_factor)

        return t

    def _calculate_shift_factor(self, seq_len: int) -> float:
        """根据策略计算 shift factor (S)"""

        if self.shift_strategy == "none":
            return 1.0

        elif self.shift_strategy == "constant":
            return self.shift_value

        elif self.shift_strategy == "linear":
            m = (self.max_shift - self.base_shift) / (
                self.max_image_seq_len - self.base_image_seq_len
            )
            b = self.base_shift - m * self.base_image_seq_len
            mu = seq_len * m + b
            return math.exp(mu)

        elif self.shift_strategy == "squared":
            mu = (seq_len / self.base_image_seq_len) ** 0.5
            return mu

        else:
            raise ValueError(f"Unknown shift strategy: {self.shift_strategy}")
