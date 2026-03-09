# Instance segmentation post-processing for Mask2Former.
# Based on mmdet code, Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class InstanceResult:
    """Container for instance segmentation results."""

    scores: Tensor = field(default_factory=lambda: torch.empty(0))
    labels: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    bboxes: Tensor = field(default_factory=lambda: torch.empty(0, 4))
    masks: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.bool))


def mask2bbox(masks: Tensor) -> Tensor:
    """Compute tight bounding boxes from binary masks.

    Args:
        masks: (N, H, W) binary mask tensor.

    Returns:
        bboxes: (N, 4) as (x1, y1, x2, y2).
    """
    n = masks.shape[0]
    bboxes = masks.new_zeros((n, 4), dtype=torch.float32)
    x_any = torch.any(masks, dim=1)  # (N, W)
    y_any = torch.any(masks, dim=2)  # (N, H)
    for i in range(n):
        x = torch.where(x_any[i])[0]
        y = torch.where(y_any[i])[0]
        if len(x) > 0 and len(y) > 0:
            bboxes[i] = bboxes.new_tensor([x[0], y[0], x[-1] + 1, y[-1] + 1])
    return bboxes


def instance_postprocess(
    mask_cls: Tensor,
    mask_pred: Tensor,
    num_classes: int = 133,
    num_things_classes: int = 80,
    max_per_image: int = 100,
) -> InstanceResult:
    """Instance segmentation post-processing.

    Args:
        mask_cls: (num_queries, cls_out_channels) classification logits.
        mask_pred: (num_queries, H, W) mask logits.
        num_classes: Total number of classes (things + stuff).
        num_things_classes: Number of "thing" classes.
        max_per_image: Maximum number of detections.

    Returns:
        InstanceResult with scores, labels, bboxes, masks.
    """
    num_queries = mask_cls.shape[0]

    # softmax over classes, drop background column
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]

    # create label indices
    labels = (
        torch.arange(num_classes, device=mask_cls.device)
        .unsqueeze(0)
        .repeat(num_queries, 1)
        .flatten(0, 1)
    )

    scores_per_image, top_indices = scores.flatten(0, 1).topk(
        max_per_image, sorted=False
    )
    labels_per_image = labels[top_indices]

    query_indices = top_indices // num_classes
    mask_pred = mask_pred[query_indices]

    # extract things only
    is_thing = labels_per_image < num_things_classes
    scores_per_image = scores_per_image[is_thing]
    labels_per_image = labels_per_image[is_thing]
    mask_pred = mask_pred[is_thing]

    mask_pred_binary = (mask_pred > 0).float()
    mask_scores_per_image = (mask_pred.sigmoid() * mask_pred_binary).flatten(1).sum(
        1
    ) / (mask_pred_binary.flatten(1).sum(1) + 1e-6)
    det_scores = scores_per_image * mask_scores_per_image
    mask_pred_binary = mask_pred_binary.bool()
    bboxes = mask2bbox(mask_pred_binary)

    return InstanceResult(
        scores=det_scores,
        labels=labels_per_image,
        bboxes=bboxes,
        masks=mask_pred_binary,
    )
