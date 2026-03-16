"""GenEval reward: object detection + color/position evaluation.

Uses a standalone Mask2Former detector (no mmdet dependency) and
open_clip for color classification. Evaluates generated images against
structured metadata specifying expected objects, counts, colors, and
relative positions.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageOps
from pydantic import ConfigDict, PrivateAttr

from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import tensor_to_pil

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


GenevalIncludeSpec = TypedDict(
    "GenevalIncludeSpec",
    {
        "class": str,
        "count": int,
        "color": NotRequired[str],
        "position": NotRequired[tuple[str, int]],  # (relation, target_group_index)
    },
)

GenevalExcludeSpec = TypedDict(
    "GenevalExcludeSpec",
    {
        "class": str,
        "count": int,
    },
)


class GenEvalMetadata(TypedDict):
    tag: str
    include: NotRequired[list[GenevalIncludeSpec]]
    exclude: NotRequired[list[GenevalExcludeSpec]]


class GenevalReward(BaseReward):
    """GenEval reward based on Mask2Former object detection and CLIP color
    classification.

    Expects ``batch["clean_image"]`` ([1, C, H, W] in [0, 1]) and
    ``batch["metadata"]`` (geneval metadata dict with ``tag``, ``include``,
    ``exclude``, ``prompt`` fields).
    """

    type: Literal["geneval"] = "geneval"
    checkpoint_path: str
    """
    Download the checkpoint from https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth
    """
    scoring_mode: Literal["reward_server", "original"] = "reward_server"
    """Scoring variant to use for soft reward computation.

    - ``"reward_server"``: Reproduces the behaviour of
      `yifan123/reward-server <https://github.com/yifan123/reward-server>`_.
      Strict count matching (``!=``), no exclude penalty in score, and the
      original position double-append quirk.  More widely used in practice.
    - ``"original"``: Aligns with the official GenEval binary evaluation
      logic, extended with a soft score.  Lenient count matching (only
      penalises too few), exclude contributes to score, and position reward
      is cleanly conditional.
    """
    clip_arch: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    threshold: float = 0.3
    counting_threshold: float = 0.9
    max_objects: int = 16
    nms_threshold: float = 1.0
    position_threshold: float = 0.1

    model_config = ConfigDict(extra="forbid")

    _detector: Any = PrivateAttr(default=None)
    _clip_model: Any = PrivateAttr(default=None)
    _clip_transform: Any = PrivateAttr(default=None)
    _clip_tokenizer: Any = PrivateAttr(default=None)
    _device: torch.device | None = PrivateAttr(default=None)
    _classnames: list[str] = PrivateAttr(default_factory=list)
    _color_classifiers: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def _batch_fields(self) -> set[str]:
        return {"clean_image", "tag", "include", "exclude"}

    def _load_model(self, device: torch.device) -> None:
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

        from clip_benchmark.metrics import zeroshot_classification as zsc

        zsc.tqdm = lambda x, **kwargs: x  # disable tqdm in color classification

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _postprocess_detections(
        self,
        result: Any,
        metadata: GenEvalMetadata,
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
        req: GenevalIncludeSpec,
        matched_groups: list[list[tuple[np.ndarray, np.ndarray | None]] | None],
    ) -> tuple[bool, list[float], list[str]]:
        """Evaluate a single 'include' requirement.

        Returns:
            (matched, reward_components, reason_parts)
        """
        if self.scoring_mode == "reward_server":
            return self._evaluate_include_req_reward_server(
                image, detected, req, matched_groups
            )
        return self._evaluate_include_req_original(image, detected, req, matched_groups)

    def _evaluate_include_req_reward_server(
        self,
        image: Image.Image,
        detected: dict[str, list[tuple[np.ndarray, np.ndarray | None]]],
        req: GenevalIncludeSpec,
        matched_groups: list[list[tuple[np.ndarray, np.ndarray | None]] | None],
    ) -> tuple[bool, list[float], list[str]]:
        """reward-server variant: strict count (``!=``), unclamped reward,
        position double-append quirk."""
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
        else:
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
                        + ", ".join(
                            f"{colors.count(c)} {c}" for c in COLORS if c in colors
                        )
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
                        rewards.append(0.0)
                    # Reproduce reward-server quirk: always append 1.0 after
                    # the position loop, regardless of pass/fail.
                    rewards.append(1.0)

        return matched, rewards, reason

    def _evaluate_include_req_original(
        self,
        image: Image.Image,
        detected: dict[str, list[tuple[np.ndarray, np.ndarray | None]]],
        req: GenevalIncludeSpec,
        matched_groups: list[list[tuple[np.ndarray, np.ndarray | None]] | None],
    ) -> tuple[bool, list[float], list[str]]:
        """Original GenEval variant: lenient count (only penalises too few),
        clamped reward, clean position scoring."""
        classname = req["class"]
        matched = True
        rewards: list[float] = []
        reason: list[str] = []
        all_objects = detected.get(classname, [])
        # Original geneval: take top-N by confidence, only fail if too few
        found_objects = all_objects[: req["count"]]
        rewards.append(
            max(0.0, 1 - abs(req["count"] - len(all_objects)) / req["count"])
        )

        if len(found_objects) < req["count"]:
            matched = False
            reason.append(
                f"expected {classname}>={req['count']}, found {len(found_objects)}"
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
        metadata: GenEvalMetadata,
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
            found_count = len(detected.get(classname, []))
            if found_count >= req["count"]:
                correct = False
                reason.append(
                    f"expected {classname}<{req['count']}, found {found_count}"
                )
                # reward-server ignores exclude in score; original penalises it
                if self.scoring_mode == "original":
                    rewards.append(0.0)
            elif self.scoring_mode == "original":
                rewards.append(1.0)

        score = sum(rewards) / len(rewards) if rewards else 0.0
        return correct, score, "\n".join(reason)

    # ------------------------------------------------------------------
    # Main score interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
        image = batch["clean_image"]  # [1, C, H, W] in [0, 1]

        # Run detection
        results = self._detector(image.to(self._device))
        result = results[0]

        # Group detections by class
        detected = self._postprocess_detections(result, batch)  # type: ignore

        # Convert to PIL for color classification
        image_pil = ImageOps.exif_transpose(tensor_to_pil(image[0]))

        # Evaluate
        _, score, _ = self._evaluate_reward(image_pil, detected, batch)  # type: ignore

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        return torch.tensor(score, device=image.device, dtype=image.dtype)

    def _unload_model(self) -> None:
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


if __name__ == "__main__":
    import urllib.request
    from pathlib import Path

    from rich import print as rprint

    from flow_control.third_party.mask2former import (
        COCO_THING_CLASSES,
        load_mask2former_swin_s,
    )
    from flow_control.utils.draw import draw_bbox_on_image
    from flow_control.utils.tensor import pil_to_tensor, tensor_to_pil

    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download checkpoint if needed
    # ------------------------------------------------------------------
    ckpt_url = (
        "https://download.openmmlab.com/mmdetection/v3.0/mask2former/"
        "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/"
        "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth"
    )
    ckpt_path = data_dir / "mask2former_swin-s-p4-w7-224_coco.pth"
    if not ckpt_path.exists():
        rprint(f"[bold]Downloading checkpoint to[/] {ckpt_path} ...")
        urllib.request.urlretrieve(ckpt_url, ckpt_path)
        rprint("[bold green]Done.[/]")
    else:
        rprint(f"[bold]Checkpoint already exists at[/] {ckpt_path}")

    # ------------------------------------------------------------------
    # 2. Download test image (COCO val2017: 2 cats + 2 remotes on a couch)
    # ------------------------------------------------------------------
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_path = data_dir / "000000039769.jpg"
    if not image_path.exists():
        rprint(f"[bold]Downloading test image to[/] {image_path} ...")
        urllib.request.urlretrieve(image_url, image_path)
        rprint("[bold green]Done.[/]")

    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = pil_to_tensor(image_pil)  # [1, C, H, W] in [0, 1]
    rprint(f"[bold]Image size:[/] {image_pil.size}, tensor shape: {image_tensor.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================================================================
    # Part A: Verify Mask2Former detection directly
    # ==================================================================
    rprint("\n[bold cyan]===== Part A: Mask2Former Detection =====[/]")
    detector = load_mask2former_swin_s(str(ckpt_path), device=device)
    results = detector(image_tensor.to(device))
    result = results[0]

    rprint(f"[bold]Detected {len(result.scores)} instances[/]")
    sorted_indices = result.scores.argsort(descending=True)
    for idx in sorted_indices[:15]:
        label_id = result.labels[idx].item()
        score = result.scores[idx].item()
        bbox = result.bboxes[idx].tolist()
        name = (
            COCO_THING_CLASSES[label_id]
            if label_id < len(COCO_THING_CLASSES)
            else f"class_{label_id}"
        )
        rprint(
            f"  {name}: {score:.3f}  "
            f"bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
        )

    # Draw detections and save visualization
    top_indices = sorted_indices[:15]
    # draw_bbox_on_image expects (top, bottom, left, right);
    # detector outputs (x1, y1, x2, y2) = (left, top, right, bottom)
    draw_boxes: list[tuple[int, int, int, int]] = [
        (
            int(result.bboxes[i][1].item()),  # top = y1
            int(result.bboxes[i][3].item()),  # bottom = y2
            int(result.bboxes[i][0].item()),  # left = x1
            int(result.bboxes[i][2].item()),  # right = x2
        )
        for i in top_indices
    ]
    draw_labels = [
        f"{COCO_THING_CLASSES[result.labels[i].item()]} {result.scores[i].item():.2f}"
        for i in top_indices
    ]
    vis = draw_bbox_on_image(image_tensor[0].cpu(), draw_boxes, draw_labels)
    vis_path = data_dir / "geneval_detection_vis.png"
    tensor_to_pil(vis).save(vis_path)
    rprint(f"[bold green]Saved detection visualization to[/] {vis_path}")

    del detector
    torch.cuda.empty_cache()

    # ==================================================================
    # Part B: Verify GenevalReward scoring (both modes)
    # ==================================================================

    # Test cases with different metadata scenarios
    test_cases: list[tuple[str, GenEvalMetadata]] = [
        (
            "2 cats (correct count)",
            GenEvalMetadata(
                tag="single_object",
                include=[{"class": "cat", "count": 2}],
            ),
        ),
        (
            "3 cats (too many expected)",
            GenEvalMetadata(
                tag="single_object",
                include=[{"class": "cat", "count": 3}],
            ),
        ),
        (
            "1 cat (too few expected)",
            GenEvalMetadata(
                tag="single_object",
                include=[{"class": "cat", "count": 1}],
            ),
        ),
        (
            "2 tv remotes (correct)",
            GenEvalMetadata(
                tag="single_object",
                include=[{"class": "tv remote", "count": 2}],
            ),
        ),
        (
            "2 cats + 2 remotes",
            GenEvalMetadata(
                tag="two_object",
                include=[
                    {"class": "cat", "count": 2},
                    {"class": "tv remote", "count": 2},
                ],
            ),
        ),
        (
            "1 dog (absent)",
            GenEvalMetadata(
                tag="single_object",
                include=[{"class": "dog", "count": 1}],
            ),
        ),
        (
            "2 cats, excl dog (pass)",
            GenEvalMetadata(
                tag="single_object",
                include=[{"class": "cat", "count": 2}],
                exclude=[{"class": "dog", "count": 1}],
            ),
        ),
        (
            "2 remotes, excl cat (fail)",
            GenEvalMetadata(
                tag="single_object",
                include=[{"class": "tv remote", "count": 2}],
                exclude=[{"class": "cat", "count": 1}],
            ),
        ),
    ]

    for mode in ("reward_server", "original"):
        rprint(f"\n[bold cyan]===== Part B: scoring_mode={mode!r} =====[/]")
        reward = GenevalReward(
            checkpoint_path=str(ckpt_path),
            scoring_mode=mode,
        )
        reward.load_model(device)
        for desc, metadata in test_cases:
            batch: dict[str, Any] = {
                "clean_image": image_tensor.to(device),
                **metadata,
            }
            score = reward._score(batch)
            rprint(f"  {desc:<32s} → score = {score.item():.4f}")
        reward.unload_model()

    rprint("\n[bold green]All tests completed.[/]")
