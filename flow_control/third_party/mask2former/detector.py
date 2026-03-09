# Top-level Mask2Former detector for inference.
# Combines SwinTransformer backbone, Mask2FormerHead, and post-processing.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .head import Mask2FormerHead
from .postprocess import InstanceResult, instance_postprocess
from .swin import SwinTransformer

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# fmt: off
COCO_THING_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
# fmt: on


class Mask2FormerDetector(nn.Module):
    """Mask2Former with Swin-S backbone for instance segmentation inference.

    Input: BCHW tensor in [0, 1] range (RGB).
    Output: list of InstanceResult per image.
    """

    def __init__(
        self,
        # Swin-S defaults
        embed_dims: int = 96,
        depths: tuple[int, ...] = (2, 2, 18, 2),
        num_heads_backbone: tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        drop_path_rate: float = 0.3,
        # Head defaults
        feat_channels: int = 256,
        out_channels: int = 256,
        num_queries: int = 100,
        num_things_classes: int = 80,
        num_stuff_classes: int = 53,
        num_transformer_feat_level: int = 3,
        num_heads_decoder: int = 8,
        num_decoder_layers: int = 9,
        # Post-processing
        max_per_image: int = 100,
        # Inference settings
        divisor: int = 32,
    ) -> None:
        super().__init__()
        self.divisor = divisor
        self.max_per_image = max_per_image
        self.num_things_classes = num_things_classes
        self.num_classes = num_things_classes + num_stuff_classes

        in_channels = [int(embed_dims * 2**i) for i in range(len(depths))]

        self.backbone = SwinTransformer(
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads_backbone,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
        )

        self.head = Mask2FormerHead(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes,
            num_queries=num_queries,
            num_transformer_feat_level=num_transformer_feat_level,
            num_heads=num_heads_decoder,
            transformer_decoder_cfg=dict(
                num_layers=num_decoder_layers,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=feat_channels,
                        num_heads=num_heads_decoder,
                        attn_drop=0.0,
                        proj_drop=0.0,
                        batch_first=True,
                    ),
                    cross_attn_cfg=dict(
                        embed_dims=feat_channels,
                        num_heads=num_heads_decoder,
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
            ),
        )

        # Register ImageNet normalization constants as buffers
        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
        )

    def _preprocess(self, images: Tensor) -> tuple[Tensor, list[tuple[int, int]]]:
        """Normalize and pad images to divisible size.

        Args:
            images: (B, 3, H, W) in [0, 1] range.

        Returns:
            images: preprocessed tensor.
            original_sizes: list of (H, W) per image.
        """
        original_sizes = [(images.shape[2], images.shape[3])] * images.shape[0]

        # ImageNet normalization
        images = (images - self.pixel_mean) / self.pixel_std

        # Pad to divisor
        h, w = images.shape[2], images.shape[3]
        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))

        return images, original_sizes

    @torch.no_grad()
    def forward(self, images: Tensor) -> list[InstanceResult]:
        """Run inference.

        Args:
            images: (B, 3, H, W) tensor in [0, 1] range (RGB).

        Returns:
            List of InstanceResult, one per image.
        """
        images, original_sizes = self._preprocess(images)

        # Backbone
        feats = self.backbone(images)

        # Head
        cls_pred_list, mask_pred_list = self.head(feats)

        # Use last decoder layer output
        cls_preds = cls_pred_list[-1]  # (B, num_queries, num_classes+1)
        mask_preds = mask_pred_list[-1]  # (B, num_queries, H/4, W/4)

        results: list[InstanceResult] = []
        for i in range(images.shape[0]):
            orig_h, orig_w = original_sizes[i]

            # Resize masks to original image size
            mask_pred = mask_preds[i]  # (num_queries, H/4, W/4)
            mask_pred = F.interpolate(
                mask_pred.unsqueeze(0),
                size=(images.shape[2], images.shape[3]),
                mode="bilinear",
                align_corners=False,
            )[0]
            # Remove padding
            mask_pred = mask_pred[:, :orig_h, :orig_w]

            result = instance_postprocess(
                mask_cls=cls_preds[i],
                mask_pred=mask_pred,
                num_classes=self.num_classes,
                num_things_classes=self.num_things_classes,
                max_per_image=self.max_per_image,
            )
            results.append(result)

        return results


def _convert_checkpoint(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert mmdet Mask2Former checkpoint to our module format.

    mmdet hierarchy:
        backbone.X                    -> backbone.X
        panoptic_head.X               -> head.X
        panoptic_fusion_head.X        -> (skip, no learnable params needed)
        data_preprocessor.X           -> (skip)

    Key differences:
        - Swin backbone FFN: mmcv uses nested Sequential
          ffn.layers.0.0.{w,b} (1st Linear) -> ffn_linear1.{w,b}
          ffn.layers.1.{w,b}   (2nd Linear) -> ffn_linear2.{w,b}
        - Head/decoder FFN: mmcv uses nested Sequential
          ffn.layers.0.0.{w,b} -> ffn.layers.0.{w,b}
          ffn.layers.1.{w,b}   -> ffn.layers.3.{w,b}
    """
    new_state_dict: dict[str, Tensor] = {}

    for key, value in state_dict.items():
        new_key = key

        # Skip training-only and preprocessing keys
        if any(
            skip in key
            for skip in [
                "loss",
                "assigner",
                "sampler",
                "class_weight",
                "data_preprocessor",
            ]
        ):
            continue

        # panoptic_head.X -> head.X
        if new_key.startswith("panoptic_head."):
            new_key = "head." + new_key[len("panoptic_head.") :]

        # Skip fusion head (no learnable params we need)
        if new_key.startswith("panoptic_fusion_head."):
            continue

        # Swin backbone FFN key remapping
        if "backbone." in new_key and ".ffn.layers.0.0." in new_key:
            new_key = new_key.replace(".ffn.layers.0.0.", ".ffn_linear1.")
        elif "backbone." in new_key and ".ffn.layers.1." in new_key:
            new_key = new_key.replace(".ffn.layers.1.", ".ffn_linear2.")

        # Head/decoder FFN key remapping (mmcv nested Sequential -> flat Sequential)
        # mmcv: ffn.layers.0.0.{w,b} -> ours: ffn.layers.0.{w,b}
        # mmcv: ffn.layers.1.{w,b}   -> ours: ffn.layers.3.{w,b}
        if "backbone." not in new_key and ".ffn.layers." in new_key:
            new_key = new_key.replace(".ffn.layers.0.0.", ".ffn.layers.0.")
            new_key = new_key.replace(".ffn.layers.1.", ".ffn.layers.3.")

        new_state_dict[new_key] = value

    return new_state_dict


def load_mask2former_swin_s(
    checkpoint_path: str,
    device: str | torch.device = "cuda",
    num_things_classes: int = 80,
    num_stuff_classes: int = 0,
) -> Mask2FormerDetector:
    """Load Mask2Former with Swin-S backbone from mmdet checkpoint.

    Args:
        checkpoint_path: Path to mmdet checkpoint file.
        device: Device to load model on.
        num_things_classes: Number of thing classes. Default 80 (COCO instance seg).
        num_stuff_classes: Number of stuff classes. Default 0 (COCO instance seg).
            Set to 53 for COCO panoptic segmentation checkpoints.

    Returns:
        Mask2FormerDetector ready for inference.
    """
    model = Mask2FormerDetector(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    converted = _convert_checkpoint(state_dict)

    # Load with strict=False to allow missing training-only keys
    missing, unexpected = model.load_state_dict(converted, strict=False)

    if missing:
        # Filter out expected missing keys
        truly_missing = [
            k
            for k in missing
            if not any(
                skip in k
                for skip in ["loss", "assigner", "sampler", "pixel_mean", "pixel_std"]
            )
        ]
        if truly_missing:
            from flow_control.utils.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(
                "Missing keys when loading checkpoint: %s", truly_missing[:20]
            )
    if unexpected:
        from flow_control.utils.logging import get_logger

        logger = get_logger(__name__)
        logger.warning("Unexpected keys in checkpoint: %s", unexpected[:20])

    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    import sys

    from rich import print as rprint

    if len(sys.argv) < 3:
        rprint(
            "[bold red]Usage:[/] python -m flow_control.third_party.mask2former.detector "
            "<checkpoint_path> <image_path>"
        )
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    image_path = sys.argv[2]

    from PIL import Image

    from flow_control.utils.common import pil_to_tensor, tensor_to_pil
    from flow_control.utils.draw import draw_bbox_on_image

    rprint(f"[bold]Loading model from[/] {checkpoint_path}")
    model = load_mask2former_swin_s(checkpoint_path, device="cuda")

    rprint(f"[bold]Loading image from[/] {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_tensor = pil_to_tensor(image).cuda()

    rprint("[bold]Running inference...[/]")
    results = model(image_tensor)

    result = results[0]
    rprint(f"[bold]Detected {len(result.scores)} instances[/]")

    # Print top detections
    sorted_indices = result.scores.argsort(descending=True)
    for idx in sorted_indices[:20]:
        label_id = result.labels[idx].item()
        score = result.scores[idx].item()
        bbox = result.bboxes[idx].tolist()
        label_name = (
            COCO_THING_CLASSES[label_id]
            if label_id < len(COCO_THING_CLASSES)
            else f"class_{label_id}"
        )
        rprint(
            f"  {label_name}: {score:.3f} "
            f"bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
        )

    # Draw bboxes on image
    boxes = [
        tuple(int(c) for c in result.bboxes[i].tolist()) for i in sorted_indices[:20]
    ]
    labels = [
        f"{COCO_THING_CLASSES[result.labels[i].item()] if result.labels[i].item() < len(COCO_THING_CLASSES) else 'cls'} "
        f"{result.scores[i].item():.2f}"
        for i in sorted_indices[:20]
    ]
    drawn = draw_bbox_on_image(image_tensor[0].cpu(), boxes, labels)  # type: ignore[arg-type]
    output_path = image_path.rsplit(".", 1)[0] + "_mask2former.png"
    tensor_to_pil(drawn).save(output_path)
    rprint(f"[bold green]Saved visualization to[/] {output_path}")
