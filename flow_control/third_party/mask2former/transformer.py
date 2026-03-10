# Standalone transformer building blocks for Mask2Former.
# Replaces mmcv FFN, MultiheadAttention, MultiScaleDeformableAttention
# with pure PyTorch implementations.
# Based on mmcv and mmdet code, Copyright (c) OpenMMLab. All rights reserved.

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FFN(nn.Module):
    """Feed-forward network with identity connection.

    Equivalent to mmcv's FFN with add_identity=True.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        act_cfg: str = "relu",
        ffn_drop: float = 0.0,
        add_identity: bool = True,
    ) -> None:
        super().__init__()
        assert num_fcs >= 2
        self.embed_dims = embed_dims
        self.add_identity = add_identity

        layers: list[nn.Module] = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.extend(
                [
                    nn.Linear(in_channels, feedforward_channels),
                    nn.ReLU(inplace=True) if act_cfg == "relu" else nn.GELU(),
                    nn.Dropout(ffn_drop),
                ]
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, identity: Tensor | None = None) -> Tensor:
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


class MultiheadAttention(nn.Module):
    """Wrapper around ``nn.MultiheadAttention`` with positional encoding
    and identity connection.

    Equivalent to mmcv's MultiheadAttention with batch_first=True.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dropout: float = 0.0,
        batch_first: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        # mmcv compatibility: 'dropout' kwarg maps to attn_drop
        if dropout > 0.0 and attn_drop == 0.0:
            attn_drop = dropout

        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=attn_drop, batch_first=False
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        identity: Tensor | None = None,
        query_pos: Tensor | None = None,
        key_pos: Tensor | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:  # noqa: SIM102
            if query_pos is not None and query_pos.shape == key.shape:
                key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # nn.MultiheadAttention expects (seq, batch, dim) when batch_first=False
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.proj_drop(out)


def multi_scale_deformable_attn_pytorch(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Pure PyTorch implementation of multi-scale deformable attention.

    Uses F.grid_sample instead of CUDA kernels.

    Args:
        value: (bs, num_keys, num_heads, embed_dims // num_heads)
        value_spatial_shapes: (num_levels, 2), each row is (h, w)
        sampling_locations: (bs, num_queries, num_heads, num_levels, num_points, 2)
        attention_weights: (bs, num_queries, num_heads, num_levels, num_points)

    Returns:
        Tensor: (bs, num_queries, embed_dims)
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split([int(h * w) for h, w in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h_, w_) in enumerate(value_spatial_shapes):
        h_ = int(h_)
        w_ = int(w_)
        # (bs, h_*w_, num_heads, embed_dims) -> (bs*num_heads, embed_dims, h_, w_)
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, h_, w_)
        )
        # (bs, num_queries, num_heads, num_points, 2) -> (bs*num_heads, num_queries, num_points, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module.

    Always uses the pure PyTorch path (no CUDA kernel needed).
    Equivalent to mmcv's MultiScaleDeformableAttention with batch_first=True.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        batch_first: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads

        def _is_power_of_2(n: int) -> bool:
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims to make the dimension of each "
                "attention head a power of 2 for better efficiency.",
                stacklevel=2,
            )

        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.dropout = nn.Dropout(dropout)

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.batch_first = batch_first
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        identity: Tensor | None = None,
        query_pos: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        reference_points: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        level_start_index: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert spatial_shapes is not None
        assert reference_points is not None
        assert level_start_index is not None
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, "
                f"but got {reference_points.shape[-1]}"
            )

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
