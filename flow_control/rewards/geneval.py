"""GenEval reward: object detection + color/position evaluation.

Uses a standalone Mask2Former detector (no mmdet dependency) and
open_clip for color classification. Evaluates generated images against
structured metadata specifying expected objects, counts, colors, and
relative positions.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageOps
from pydantic import ConfigDict, PrivateAttr

from flow_control.utils.common import tensor_to_pil
from flow_control.utils.logging import get_logger

from .base import BaseReward

logger = get_logger(__name__)

COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
]


class GenevalReward(BaseReward):
    """GenEval reward based on Mask2Former object detection and CLIP color
    classification.

    Expects ``batch["clean_image"]`` ([1, C, H, W] in [0, 1]) and
    ``batch["metadata"]`` (geneval metadata dict with ``tag``, ``include``,
    ``exclude``, ``prompt`` fields).
    """

    type: Literal["geneval"] = "geneval"
    checkpoint_path: str
    clip_arch: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    threshold: float = 0.3
    counting_threshold: float = 0.9
    max_objects: int = 16
    nms_threshold: float = 1.0
    position_threshold: float = 0.1

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _detector: Any = PrivateAttr(default=None)
    _clip_model: Any = PrivateAttr(default=None)
    _clip_transform: Any = PrivateAttr(default=None)
    _clip_tokenizer: Any = PrivateAttr(default=None)
    _device: torch.device | None = PrivateAttr(default=None)
    _classnames: list[str] = PrivateAttr(default_factory=list)
    _color_classifiers: dict[str, Any] = PrivateAttr(default_factory=dict)

    def load_model(self, device: torch.device) -> None:
        import open_clip

        from flow_control.third_party.mask2former import (
            COCO_THING_CLASSES,
            load_mask2former_swin_s,
        )

        self._device = device
        self._detector = load_mask2former_swin_s(self.checkpoint_path, device=device)
        self._classnames = list(COCO_THING_CLASSES)

        clip_model, _, clip_transform = open_clip.create_model_and_transforms(
            self.clip_arch,
            pretrained=self.clip_pretrained,
            device=str(device),
        )
        clip_model.eval()
        self._clip_model = clip_model
        self._clip_transform = clip_transform
        self._clip_tokenizer = open_clip.get_tokenizer(self.clip_arch)
        self._color_classifiers = {}

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _postprocess_detections(
        self,
        result: Any,
        metadata: dict[str, Any],
    ) -> dict[str, list[tuple[np.ndarray, np.ndarray | None]]]:
        """Convert InstanceResult into per-class detection dict.

        Returns:
            ``{classname: [(bbox_5d, mask_or_None), ...]}`` sorted by
            confidence (descending) with thresholding and NMS applied.
        """
        scores = result.scores.cpu().numpy()
        labels = result.labels.cpu().numpy()
        bboxes = result.bboxes.cpu().numpy()  # (N, 4)
        masks = result.masks.cpu().numpy() if result.masks.numel() > 0 else None

        confidence_threshold = (
            self.counting_threshold
            if metadata.get("tag") == "counting"
            else self.threshold
        )

        detected: dict[str, list[tuple[np.ndarray, np.ndarray | None]]] = {}

        for class_idx, classname in enumerate(self._classnames):
            # Find all detections of this class
            class_mask = labels == class_idx
            if not class_mask.any():
                continue

            class_scores = scores[class_mask]
            class_bboxes = bboxes[class_mask]
            class_masks = masks[class_mask] if masks is not None else None

            # Sort by confidence (descending)
            ordering = np.argsort(class_scores)[::-1]
            # Apply confidence threshold
            ordering = ordering[class_scores[ordering] > confidence_threshold]
            # Limit number of objects
            ordering = ordering[: self.max_objects].tolist()

            objs: list[tuple[np.ndarray, np.ndarray | None]] = []
            while ordering:
                max_idx = ordering.pop(0)
                bbox_5d = np.concatenate(
                    [class_bboxes[max_idx], [class_scores[max_idx]]]
                )
                mask_arr = class_masks[max_idx] if class_masks is not None else None
                objs.append((bbox_5d, mask_arr))
                # NMS
                if self.nms_threshold < 1.0:
                    ordering = [
                        idx
                        for idx in ordering
                        if _compute_iou(class_bboxes[max_idx], class_bboxes[idx])
                        < self.nms_threshold
                    ]

            if objs:
                detected[classname] = objs

        return detected

    # ------------------------------------------------------------------
    # Color classification
    # ------------------------------------------------------------------

    def _color_classification(
        self,
        image: Image.Image,
        bboxes: list[tuple[np.ndarray, np.ndarray | None]],
        classname: str,
    ) -> list[str]:
        from clip_benchmark.metrics import zeroshot_classification as zsc

        if classname not in self._color_classifiers:
            self._color_classifiers[classname] = zsc.zero_shot_classifier(
                self._clip_model,
                self._clip_tokenizer,
                COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    "a photo of a {c} object",
                ],
                str(self._device),
            )
        clf = self._color_classifiers[classname]
        dataloader = torch.utils.data.DataLoader(
            _ImageCrops(image, bboxes, self._clip_transform),
            batch_size=16,
            num_workers=0,
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(
                self._clip_model, clf, dataloader, str(self._device)
            )
            return [COLORS[int(index.item())] for index in pred.argmax(1)]

    # ------------------------------------------------------------------
    # Evaluation logic (ported from gen_eval.py)
    # ------------------------------------------------------------------

    def _evaluate_include_req(
        self,
        image: Image.Image,
        detected: dict[str, list[tuple[np.ndarray, np.ndarray | None]]],
        req: dict[str, Any],
        matched_groups: list[list[tuple[np.ndarray, np.ndarray | None]] | None],
    ) -> tuple[bool, list[float], list[str]]:
        """Evaluate a single 'include' requirement.

        Returns:
            (matched, reward_components, reason_parts)
        """
        classname = req["class"]
        matched = True
        rewards: list[float] = []
        reason: list[str] = []
        found_objects = detected.get(classname, [])
        rewards.append(1 - abs(req["count"] - len(found_objects)) / req["count"])

        if len(found_objects) != req["count"]:
            matched = False
            reason.append(
                f"expected {classname}=={req['count']}, found {len(found_objects)}"
            )
            if "color" in req or "position" in req:
                rewards.append(0.0)
            return matched, rewards, reason

        if "color" in req:
            colors = self._color_classification(image, found_objects, classname)
            rewards.append(
                1 - abs(req["count"] - colors.count(req["color"])) / req["count"]
            )
            if colors.count(req["color"]) != req["count"]:
                matched = False
                reason.append(
                    f"expected {req['color']} {classname}"
                    f">={req['count']}, found "
                    f"{colors.count(req['color'])} {req['color']}; and "
                    + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                )

        if "position" in req and matched:
            expected_rel, target_group = req["position"]
            target = matched_groups[target_group]
            if target is None:
                matched = False
                reason.append(f"no target for {classname} to be {expected_rel}")
                rewards.append(0.0)
            else:
                position_ok = _check_position(
                    found_objects, target, expected_rel, self.position_threshold
                )
                if not position_ok:
                    matched = False
                    reason.append(f"{classname} not {expected_rel} target")
                rewards.append(1.0 if position_ok else 0.0)

        return matched, rewards, reason

    def _evaluate_reward(
        self,
        image: Image.Image,
        detected: dict[str, list[tuple[np.ndarray, np.ndarray | None]]],
        metadata: dict[str, Any],
    ) -> tuple[bool, float, str]:
        """Evaluate image against metadata specification.

        Returns:
            (strict_correct, score, reason)
        """
        correct = True
        reason: list[str] = []
        rewards: list[float] = []
        matched_groups: list[list[tuple[np.ndarray, np.ndarray | None]] | None] = []

        for req in metadata.get("include", []):
            matched, req_rewards, req_reason = self._evaluate_include_req(
                image, detected, req, matched_groups
            )
            rewards.extend(req_rewards)
            reason.extend(req_reason)
            if not matched:
                correct = False
            matched_groups.append(detected.get(req["class"], []) if matched else None)

        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(detected.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(
                    f"expected {classname}<{req['count']}, "
                    f"found {len(detected[classname])}"
                )

        score = sum(rewards) / len(rewards) if rewards else 0.0
        return correct, score, "\n".join(reason)

    # ------------------------------------------------------------------
    # Main score interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(self, batch: dict[str, Any]) -> torch.Tensor:
        image = batch["clean_image"]  # [1, C, H, W] in [0, 1]
        metadata = batch["metadata"]

        # Run detection
        results = self._detector(image.to(self._device))
        result = results[0]

        # Group detections by class
        detected = self._postprocess_detections(result, metadata)

        # Convert to PIL for color classification
        image_pil = ImageOps.exif_transpose(tensor_to_pil(image[0]))

        # Evaluate
        _, score, _ = self._evaluate_reward(image_pil, detected, metadata)

        return torch.tensor(score, device=image.device, dtype=image.dtype)

    def unload_model(self) -> None:
        import gc

        del self._detector, self._clip_model, self._clip_transform
        del self._clip_tokenizer
        self._detector = None
        self._clip_model = None
        self._clip_transform = None
        self._clip_tokenizer = None
        self._color_classifiers = {}
        gc.collect()
        torch.cuda.empty_cache()


# ======================================================================
# Module-level helpers
# ======================================================================


class _ImageCrops(torch.utils.data.Dataset[tuple[Any, int]]):
    """Dataset that yields CLIP-transformed crops of detected objects."""

    def __init__(
        self,
        image: Image.Image,
        objects: list[tuple[np.ndarray, np.ndarray | None]],
        transform: Any,
    ) -> None:
        self._image = image.convert("RGB")
        self._blank = Image.new("RGB", image.size, color="#999")
        self._objects = objects
        self._transform = transform

    def __len__(self) -> int:
        return len(self._objects)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (
                index,
                self._image.size[::-1],
                mask.shape,
            )
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        image = image.crop(box[:4].astype(int).tolist())
        return (self._transform(image), 0)


def _check_position(
    found_objects: list[tuple[np.ndarray, np.ndarray | None]],
    target_objects: list[tuple[np.ndarray, np.ndarray | None]],
    expected_rel: str,
    position_threshold: float,
) -> bool:
    """Check if all found objects have expected relation to target objects."""
    for obj in found_objects:
        for target_obj in target_objects:
            true_rels = _relative_position(obj, target_obj, position_threshold)
            if expected_rel not in true_rels:
                return False
    return True


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    i_area = max(x2 - x1 + 1, 0) * max(y2 - y1 + 1, 0)
    area_a = max(box_a[2] - box_a[0] + 1, 0) * max(box_a[3] - box_a[1] + 1, 0)
    area_b = max(box_b[2] - box_b[0] + 1, 0) * max(box_b[3] - box_b[1] + 1, 0)
    u_area = area_a + area_b - i_area
    return float(i_area / u_area) if u_area else 0.0


def _relative_position(
    obj_a: tuple[np.ndarray, np.ndarray | None],
    obj_b: tuple[np.ndarray, np.ndarray | None],
    position_threshold: float,
) -> set[str]:
    """Give position of A relative to B, factoring in object dimensions."""
    boxes = np.array([obj_a[0][:4], obj_b[0][:4]]).reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b

    revised_offset = np.maximum(
        np.abs(offset) - position_threshold * (dim_a + dim_b), 0
    ) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()

    dx, dy = revised_offset / np.linalg.norm(offset)
    relations: set[str] = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations
