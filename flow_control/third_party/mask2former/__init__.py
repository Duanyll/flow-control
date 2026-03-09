# Standalone Mask2Former implementation for inference.
# No mmcv/mmdet/mmengine dependencies required.

from .detector import (
    COCO_THING_CLASSES,
    Mask2FormerDetector,
    load_mask2former_swin_s,
)
from .postprocess import InstanceResult

__all__ = [
    "Mask2FormerDetector",
    "load_mask2former_swin_s",
    "InstanceResult",
    "COCO_THING_CLASSES",
]
