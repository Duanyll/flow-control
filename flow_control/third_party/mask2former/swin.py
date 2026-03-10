# Standalone SwinTransformer backbone for Mask2Former inference.
# Rewritten from mmdet/models/backbones/swin.py as clean nn.Module classes.
# Based on mmdet code, Copyright (c) OpenMMLab. All rights reserved.

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _to_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return x


# ---------------------------------------------------------------------------
# Adaptive padding and Patch operations (from mmdet transformer utils)
# ---------------------------------------------------------------------------


class AdaptivePadding(nn.Module):
    """Pads input so it can be fully covered by the convolution kernel."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = 1,
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        padding: str = "corner",
    ) -> None:
        super().__init__()
        self.padding = padding
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.dilation = _to_2tuple(dilation)

    def get_pad_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max(
            (output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h,
            0,
        )
        pad_w = max(
            (output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w,
            0,
        )
        return pad_h, pad_w

    def forward(self, x: Tensor) -> Tensor:
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h])
            else:
                x = F.pad(
                    x,
                    [
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                    ],
                )
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding using Conv2d."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 768,
        kernel_size: int = 16,
        stride: int = 16,
        padding: str = "corner",
        dilation: int = 1,
        norm_layer: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        kernel_size_t = _to_2tuple(kernel_size)
        stride_t = _to_2tuple(stride)
        dilation_t = _to_2tuple(dilation)

        self.adap_padding: AdaptivePadding | None
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size_t,
                stride=stride_t,
                dilation=dilation_t,
                padding=padding,
            )
            pad = 0
        else:
            self.adap_padding = None
            pad = padding

        self.projection = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size_t,
            stride=stride_t,
            padding=_to_2tuple(pad),
            dilation=dilation_t,
        )
        self.norm = nn.LayerNorm(embed_dims) if norm_layer else None

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        if self.adap_padding:
            x = self.adap_padding(x)
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PatchMerging(nn.Module):
    """Merge patch feature map using nn.Unfold."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int | None = None,
        padding: str = "corner",
        dilation: int = 1,
        norm_layer: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride is None:
            stride = kernel_size

        kernel_size_t = _to_2tuple(kernel_size)
        stride_t = _to_2tuple(stride)
        dilation_t = _to_2tuple(dilation)

        self.adap_padding: AdaptivePadding | None
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size_t,
                stride=stride_t,
                dilation=dilation_t,
                padding=padding,
            )
            pad = 0
        else:
            self.adap_padding = None
            pad = padding

        self.sampler = nn.Unfold(
            kernel_size=kernel_size_t,
            dilation=dilation_t,
            padding=_to_2tuple(pad),
            stride=stride_t,
        )
        sample_dim = kernel_size_t[0] * kernel_size_t[1] * in_channels
        self.norm = nn.LayerNorm(sample_dim) if norm_layer else None
        self.reduction = nn.Linear(sample_dim, out_channels, bias=False)

    def forward(
        self, x: Tensor, input_size: tuple[int, int]
    ) -> tuple[Tensor, tuple[int, int]]:
        b, _l, c = x.shape
        h, w = input_size
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)

        if self.adap_padding:
            x = self.adap_padding(x)
            h, w = x.shape[-2:]

        x = self.sampler(x)
        out_h = (
            h
            + 2 * self.sampler.padding[0]  # type: ignore[index]
            - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1)  # type: ignore[index]
            - 1
        ) // self.sampler.stride[0] + 1  # type: ignore[index]
        out_w = (
            w
            + 2 * self.sampler.padding[1]  # type: ignore[index]
            - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1)  # type: ignore[index]
            - 1
        ) // self.sampler.stride[1] + 1  # type: ignore[index]

        x = x.transpose(1, 2)  # (B, H/2*W/2, 4*C)
        if self.norm:
            x = self.norm(x)
        x = self.reduction(x)
        return x, (out_h, out_w)


# ---------------------------------------------------------------------------
# DropPath (stochastic depth)
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# ---------------------------------------------------------------------------
# Window-based Multi-head Self-Attention
# ---------------------------------------------------------------------------


class WindowMSA(nn.Module):
    """Window based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: tuple[int, int],
        qkv_bias: bool = True,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = head_embed_dims**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        wh, ww = self.window_size
        rel_index_coords = self._double_step_seq(2 * ww - 1, wh, 1, ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer("relative_position_index", rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _double_step_seq(step1: int, len1: int, step2: int, len2: int) -> Tensor:
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ShiftWindowMSA(nn.Module):
    """Shifted Window Multi-head Self-Attention."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        qkv_bias: bool = True,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=_to_2tuple(window_size),
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
        )
        self.drop = DropPath(drop_path_rate)

    def forward(self, query: Tensor, hw_shape: tuple[int, int]) -> Tensor:
        b, l, c = query.shape
        h, w = hw_shape
        assert l == h * w
        query = query.view(b, h, w, c)

        # pad to multiples of window size
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        h_pad, w_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            img_mask = torch.zeros((1, h_pad, w_pad, 1), device=query.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h_s in h_slices:
                for w_s in w_slices:
                    img_mask[:, h_s, w_s, :] = cnt
                    cnt += 1
            mask_windows = self._window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            shifted_query = query
            attn_mask = None

        query_windows = self._window_partition(shifted_query)
        query_windows = query_windows.view(-1, self.window_size**2, c)

        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)

        shifted_x = self._window_reverse(attn_windows, h_pad, w_pad)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.view(b, h * w, c)
        x = self.drop(x)
        return x

    def _window_partition(self, x: Tensor) -> Tensor:
        """Partition into non-overlapping windows. (B, H, W, C) -> (num_windows*B, ws, ws, C)"""
        b, h, w, c = x.shape
        ws = self.window_size
        x = x.view(b, h // ws, ws, w // ws, ws, c)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, c)

    def _window_reverse(self, windows: Tensor, h: int, w: int) -> Tensor:
        """Reverse window partition. (num_windows*B, ws, ws, C) -> (B, H, W, C)"""
        ws = self.window_size
        b = int(windows.shape[0] / (h * w / ws / ws))
        x = windows.view(b, h // ws, w // ws, ws, ws, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


# ---------------------------------------------------------------------------
# Swin Transformer blocks
# ---------------------------------------------------------------------------


class SwinBlock(nn.Module):
    """A single Swin Transformer block."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        window_size: int = 7,
        shift: bool = False,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        # FFN with GELU and DropPath
        self.ffn_linear1 = nn.Linear(embed_dims, feedforward_channels)
        self.ffn_act = nn.GELU()
        self.ffn_drop1 = nn.Dropout(drop_rate)
        self.ffn_linear2 = nn.Linear(feedforward_channels, embed_dims)
        self.ffn_drop2 = nn.Dropout(drop_rate)
        self.ffn_drop_path = DropPath(drop_path_rate)

    def forward(self, x: Tensor, hw_shape: tuple[int, int]) -> Tensor:
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn_linear1(x)
        x = self.ffn_act(x)
        x = self.ffn_drop1(x)
        x = self.ffn_linear2(x)
        x = self.ffn_drop2(x)
        x = identity + self.ffn_drop_path(x)
        return x


class SwinBlockSequence(nn.Module):
    """One stage in Swin Transformer."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        depth: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float | list[float] = 0.0,
        downsample: PatchMerging | None = None,
    ) -> None:
        super().__init__()
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=i % 2 == 1,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
            )
            self.blocks.append(block)
        self.downsample = downsample

    def forward(
        self, x: Tensor, hw_shape: tuple[int, int]
    ) -> tuple[Tensor, tuple[int, int], Tensor, tuple[int, int]]:
        for block in self.blocks:
            x = block(x, hw_shape)
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        return x, hw_shape, x, hw_shape


# ---------------------------------------------------------------------------
# Top-level SwinTransformer backbone
# ---------------------------------------------------------------------------


class SwinTransformer(nn.Module):
    """Swin Transformer backbone.

    Hardcoded for Swin-S config by default:
        embed_dims=96, depths=[2,2,18,2], num_heads=[3,6,12,24], window_size=7
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 96,
        patch_size: int = 4,
        window_size: int = 7,
        mlp_ratio: int = 4,
        depths: tuple[int, ...] = (2, 2, 18, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        strides: tuple[int, ...] = (4, 2, 2, 2),
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        qkv_bias: bool = True,
        patch_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.3,
    ) -> None:
        super().__init__()
        num_layers = len(depths)
        self.out_indices = out_indices

        assert strides[0] == patch_size

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=strides[0],
            padding="corner",
            norm_layer=patch_norm,
        )

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth decay
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = nn.ModuleList()
        in_ch = embed_dims
        for i in range(num_layers):
            downsample: PatchMerging | None
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_ch,
                    out_channels=2 * in_ch,
                    kernel_size=2,
                    stride=strides[i + 1],
                    norm_layer=patch_norm,
                )
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_ch,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_ch,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=downsample,
            )
            self.stages.append(stage)
            if downsample:
                in_ch = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # normalization layer for each output
        for i in out_indices:
            layer = nn.LayerNorm(self.num_features[i])
            self.add_module(f"norm{i}", layer)

    def forward(self, x: Tensor) -> list[Tensor]:
        x, hw_shape = self.patch_embed(x)
        x = self.drop_after_pos(x)

        outs: list[Tensor] = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer: nn.LayerNorm = getattr(self, f"norm{i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, self.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)
        return outs
