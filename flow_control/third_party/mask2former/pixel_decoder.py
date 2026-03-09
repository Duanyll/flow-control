# MSDeformAttnPixelDecoder for Mask2Former.
# Rewritten from mmdet/models/layers/msdeformattn_pixel_decoder.py.
# Based on mmdet code, Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .transformer import FFN, MultiScaleDeformableAttention

# ---------------------------------------------------------------------------
# Sine Positional Encoding
# ---------------------------------------------------------------------------


class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions."""

    def __init__(
        self,
        num_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: Tensor) -> Tensor:
        """Generate positional encoding from mask.

        Args:
            mask: (bs, h, w) bool tensor. Non-zero = ignored positions.

        Returns:
            pos: (bs, num_feats*2, h, w)
        """
        b, h, w = mask.size()
        not_mask = 1 - mask.to(torch.int)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            y_embed = (
                (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            )
            x_embed = (
                (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
            )

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).view(b, h, w, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).view(b, h, w, -1)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Encoder layers
# ---------------------------------------------------------------------------


class DeformableDetrTransformerEncoderLayer(nn.Module):
    """Single encoder layer with deformable attention."""

    def __init__(
        self,
        self_attn_cfg: dict | None = None,
        ffn_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        if self_attn_cfg is None:
            self_attn_cfg = dict(
                embed_dims=256,
                num_heads=8,
                num_levels=4,
                num_points=4,
                dropout=0.0,
                batch_first=True,
            )
        if ffn_cfg is None:
            ffn_cfg = dict(
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                act_cfg="relu",
                ffn_drop=0.0,
            )

        self.self_attn = MultiScaleDeformableAttention(**self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**ffn_cfg)
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(self.embed_dims),
                nn.LayerNorm(self.embed_dims),
            ]
        )

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key_padding_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reference_points: Tensor,
        **kwargs: object,
    ) -> Tensor:
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points,
        )
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)
        return query


class Mask2FormerTransformerEncoder(nn.Module):
    """Transformer encoder for the pixel decoder."""

    def __init__(
        self,
        num_layers: int = 6,
        layer_cfg: dict | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        if layer_cfg is None:
            layer_cfg = {}
        self.layers = nn.ModuleList(
            [
                DeformableDetrTransformerEncoderLayer(**layer_cfg)
                for _ in range(num_layers)
            ]
        )
        self.embed_dims = self.layers[0].embed_dims

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key_padding_mask: Tensor,
        spatial_shapes: Tensor,
        reference_points: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        **kwargs: object,
    ) -> Tensor:
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
            )
        return query


# ---------------------------------------------------------------------------
# ConvModule replacement: Conv2d + GroupNorm + optional ReLU
# ---------------------------------------------------------------------------


class ConvModule(nn.Module):
    """Conv2d + GroupNorm + optional ReLU, replacing mmcv's ConvModule."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        num_groups: int = 32,
        act: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.gn = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.gn(x)
        if self.act is not None:
            x = self.act(x)
        return x


# ---------------------------------------------------------------------------
# MSDeformAttnPixelDecoder
# ---------------------------------------------------------------------------


class MSDeformAttnPixelDecoder(nn.Module):
    """Pixel decoder with multi-scale deformable attention.

    Args:
        in_channels: Number of channels per backbone level.
        strides: Output strides of each backbone level.
        feat_channels: Internal feature channels.
        out_channels: Output channels (for mask_feature).
        num_outs: Number of output multi-scale features.
        num_encoder_levels: Number of levels processed by deformable encoder.
        encoder_cfg: Config dict for the transformer encoder.
    """

    def __init__(
        self,
        in_channels: list[int] = [96, 192, 384, 768],  # noqa: B006
        strides: list[int] = [4, 8, 16, 32],  # noqa: B006
        feat_channels: int = 256,
        out_channels: int = 256,
        num_outs: int = 3,
        num_encoder_levels: int = 3,
        encoder_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.strides = strides
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = num_encoder_levels

        # Input projection convs for encoder levels (from high to low resolution)
        input_conv_list: list[nn.Module] = []
        for i in range(
            self.num_input_levels - 1,
            self.num_input_levels - self.num_encoder_levels - 1,
            -1,
        ):
            input_conv = ConvModule(
                in_channels[i], feat_channels, kernel_size=1, bias=True
            )
            input_conv_list.append(input_conv)
        self.input_convs = nn.ModuleList(input_conv_list)

        if encoder_cfg is None:
            encoder_cfg = dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=feat_channels,
                        num_heads=8,
                        num_levels=num_encoder_levels,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True,
                    ),
                    ffn_cfg=dict(
                        embed_dims=feat_channels,
                        feedforward_channels=1024,
                        num_fcs=2,
                        act_cfg="relu",
                        ffn_drop=0.0,
                    ),
                ),
            )
        self.encoder = Mask2FormerTransformerEncoder(**encoder_cfg)
        self.postional_encoding = SinePositionalEncoding(
            num_feats=feat_channels // 2, normalize=True
        )
        self.level_encoding = nn.Embedding(self.num_encoder_levels, feat_channels)

        # FPN for lower-level features
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1):
            lateral_conv = ConvModule(
                in_channels[i], feat_channels, kernel_size=1, bias=False
            )
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                act=True,
            )
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.mask_feature = nn.Conv2d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.num_outs = num_outs

    def forward(self, feats: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
        """Forward pass.

        Args:
            feats: Multi-scale features from backbone, each (B, C, H, W).

        Returns:
            mask_feature: (B, out_channels, H, W) at highest resolution.
            multi_scale_features: list of (B, feat_channels, H_i, W_i).
        """
        batch_size = feats[0].shape[0]
        encoder_input_list: list[Tensor] = []
        padding_mask_list: list[Tensor] = []
        level_positional_encoding_list: list[Tensor] = []
        spatial_shapes: list[Tensor] = []
        reference_points_list: list[Tensor] = []

        for i in range(self.num_encoder_levels):
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            feat_projected = self.input_convs[i](feat)
            h, w = feat.shape[-2:]

            # no padding for inference
            padding_mask = feat.new_zeros((batch_size, h, w), dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed

            # reference points (grid priors)
            shift_x = (torch.arange(0, w, device=feat.device) + 0.5) * self.strides[
                level_idx
            ]
            shift_y = (torch.arange(0, h, device=feat.device) + 0.5) * self.strides[
                level_idx
            ]
            shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
            reference_points = torch.stack([shift_xx, shift_yy], dim=-1).reshape(-1, 2)

            # normalize reference points
            factor_x = w * self.strides[level_idx]
            factor_y = h * self.strides[level_idx]
            reference_points[:, 0] /= factor_x
            reference_points[:, 1] /= factor_y

            # flatten spatial dims
            feat_projected = feat_projected.flatten(2).permute(0, 2, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(0, 2, 1)
            padding_mask_flat = padding_mask.flatten(1)

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_flat)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(
                torch.tensor([h, w], device=feat.device, dtype=torch.long)
            )
            reference_points_list.append(reference_points)

        padding_masks = torch.cat(padding_mask_list, dim=1)
        encoder_inputs = torch.cat(encoder_input_list, dim=1)
        level_positional_encodings = torch.cat(level_positional_encoding_list, dim=1)

        spatial_shapes_tensor = torch.stack(spatial_shapes)  # (num_levels, 2)
        num_queries_per_level = [s[0] * s[1] for s in spatial_shapes]

        level_start_index = torch.cat(
            [
                spatial_shapes_tensor.new_zeros((1,)),
                spatial_shapes_tensor.prod(1).cumsum(0)[:-1],
            ]
        )

        reference_points_cat = torch.cat(reference_points_list, dim=0)
        reference_points_cat = reference_points_cat[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1
        )
        valid_ratios = reference_points_cat.new_ones(
            (batch_size, self.num_encoder_levels, 2)
        )

        memory = self.encoder(
            query=encoder_inputs,
            query_pos=level_positional_encodings,
            key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes_tensor,
            reference_points=reference_points_cat,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        memory = memory.permute(0, 2, 1)  # (B, C, total_queries)

        # split back per level (from low resolution to high resolution)
        outs_list = list(
            torch.split(memory, [int(n) for n in num_queries_per_level], dim=-1)
        )
        outs: list[Tensor] = [
            x.reshape(batch_size, -1, spatial_shapes[i][0], spatial_shapes[i][1])
            for i, x in enumerate(outs_list)
        ]

        # FPN for remaining (lower) levels
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            y = cur_feat + F.interpolate(
                outs[-1], size=cur_feat.shape[-2:], mode="bilinear", align_corners=False
            )
            y = self.output_convs[i](y)
            outs.append(y)

        multi_scale_features = outs[: self.num_outs]
        mask_feature = self.mask_feature(outs[-1])
        return mask_feature, multi_scale_features
