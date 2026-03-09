# Mask2FormerHead - inference only.
# Rewritten from mmdet/models/dense_heads/mask2former_head.py.
# Based on mmdet code, Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .pixel_decoder import MSDeformAttnPixelDecoder, SinePositionalEncoding
from .transformer import FFN, MultiheadAttention

# ---------------------------------------------------------------------------
# Transformer decoder layers for Mask2Former
# ---------------------------------------------------------------------------


class Mask2FormerTransformerDecoderLayer(nn.Module):
    """Single decoder layer: cross-attention -> self-attention -> FFN.

    Note: Mask2Former reverses the standard DETR order
    (cross-attn first, then self-attn).
    """

    def __init__(
        self,
        self_attn_cfg: dict | None = None,
        cross_attn_cfg: dict | None = None,
        ffn_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        if self_attn_cfg is None:
            self_attn_cfg = dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                batch_first=True,
            )
        if cross_attn_cfg is None:
            cross_attn_cfg = dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                batch_first=True,
            )
        if ffn_cfg is None:
            ffn_cfg = dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                act_cfg="relu",
                ffn_drop=0.0,
            )

        self.cross_attn = MultiheadAttention(**cross_attn_cfg)
        self.self_attn = MultiheadAttention(**self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**ffn_cfg)
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.LayerNorm(self.embed_dims),
            ]
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        query_pos: Tensor | None = None,
        key_pos: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        # cross-attention first (masked cross-attention in Mask2Former)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
        )
        query = self.norms[0](query)
        # self-attention
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
        )
        query = self.norms[1](query)
        # FFN
        query = self.ffn(query)
        query = self.norms[2](query)
        return query


class Mask2FormerTransformerDecoder(nn.Module):
    """Mask2Former transformer decoder."""

    def __init__(
        self,
        num_layers: int = 9,
        layer_cfg: dict | None = None,
        post_norm_cfg: str = "LN",
        return_intermediate: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__()
        if layer_cfg is None:
            layer_cfg = {}
        self.layers = nn.ModuleList(
            [Mask2FormerTransformerDecoderLayer(**layer_cfg) for _ in range(num_layers)]
        )
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = nn.LayerNorm(self.embed_dims)
        self.return_intermediate = return_intermediate


# ---------------------------------------------------------------------------
# Mask2FormerHead (inference only)
# ---------------------------------------------------------------------------


class Mask2FormerHead(nn.Module):
    """Mask2Former head for inference.

    Takes multi-scale backbone features and produces classification logits
    and mask predictions through iterative masked cross-attention.
    """

    def __init__(
        self,
        in_channels: list[int] = [96, 192, 384, 768],  # noqa: B006
        feat_channels: int = 256,
        out_channels: int = 256,
        num_things_classes: int = 80,
        num_stuff_classes: int = 53,
        num_queries: int = 100,
        num_transformer_feat_level: int = 3,
        pixel_decoder_cfg: dict | None = None,
        transformer_decoder_cfg: dict | None = None,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_things_classes + num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = num_heads

        # Build pixel decoder
        if pixel_decoder_cfg is None:
            pixel_decoder_cfg = dict(
                in_channels=in_channels,
                strides=[4, 8, 16, 32],
                feat_channels=feat_channels,
                out_channels=out_channels,
                num_outs=num_transformer_feat_level,
                num_encoder_levels=num_transformer_feat_level,
            )
        self.pixel_decoder = MSDeformAttnPixelDecoder(**pixel_decoder_cfg)

        # Build transformer decoder
        if transformer_decoder_cfg is None:
            transformer_decoder_cfg = dict(
                num_layers=9,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=feat_channels,
                        num_heads=num_heads,
                        attn_drop=0.0,
                        proj_drop=0.0,
                        batch_first=True,
                    ),
                    cross_attn_cfg=dict(
                        embed_dims=feat_channels,
                        num_heads=num_heads,
                        attn_drop=0.0,
                        proj_drop=0.0,
                        batch_first=True,
                    ),
                    ffn_cfg=dict(
                        embed_dims=feat_channels,
                        feedforward_channels=2048,
                        num_fcs=2,
                        act_cfg="relu",
                        ffn_drop=0.0,
                    ),
                ),
            )
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            **transformer_decoder_cfg
        )
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        self.num_transformer_decoder_layers = len(self.transformer_decoder.layers)

        # Input projections for decoder
        self.decoder_input_projs = nn.ModuleList()
        for _ in range(num_transformer_feat_level):
            if self.decoder_embed_dims != feat_channels:
                self.decoder_input_projs.append(
                    nn.Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1)
                )
            else:
                self.decoder_input_projs.append(nn.Identity())

        self.decoder_positional_encoding = SinePositionalEncoding(
            num_feats=feat_channels // 2, normalize=True
        )

        self.query_embed = nn.Embedding(num_queries, feat_channels)
        self.query_feat = nn.Embedding(num_queries, feat_channels)
        self.level_embed = nn.Embedding(num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels),
        )

    def _forward_head(
        self,
        decoder_out: Tensor,
        mask_feature: Tensor,
        attn_mask_target_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward head after each decoder layer.

        Args:
            decoder_out: (B, num_queries, C)
            mask_feature: (B, C, H, W)
            attn_mask_target_size: (H, W) for attention mask

        Returns:
            cls_pred: (B, num_queries, num_classes+1)
            mask_pred: (B, num_queries, H, W)
            attn_mask: (B*num_heads, num_queries, H*W)
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        cls_pred = self.cls_embed(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)

        attn_mask = F.interpolate(
            mask_pred, attn_mask_target_size, mode="bilinear", align_corners=False
        )
        attn_mask = (
            attn_mask.flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
        )
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward(self, x: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Forward pass.

        Args:
            x: Multi-scale features from backbone, each (B, C, H, W).

        Returns:
            cls_pred_list: Classification logits per decoder layer.
            mask_pred_list: Mask logits per decoder layer.
        """
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)

        # Prepare decoder inputs
        decoder_inputs: list[Tensor] = []
        decoder_positional_encodings: list[Tensor] = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool,
            )
            decoder_pos = self.decoder_positional_encoding(mask)
            decoder_pos = decoder_pos.flatten(2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_pos)

        # Query embeddings
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        cls_pred_list: list[Tensor] = []
        mask_pred_list: list[Tensor] = []

        # Initial prediction
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        # Iterative decoder layers
        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # If mask is all True (all background), set it all False
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum

            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                key_padding_mask=None,
            )

            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
            )
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list
