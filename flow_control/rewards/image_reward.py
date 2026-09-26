"""ImageReward (THUDM) reward wrapper.

Wraps the vendored inference-only ImageReward implementation in
``flow_control.third_party.imagereward`` as a :class:`BaseReward`.

The underlying model returns a score that is already mean/std-normalized by
the original training pipeline (``mean=0.16717``, ``std=1.0333``), typically
in roughly ``[-2, 2]``.  ``_score`` returns this raw value as a ``[1]``
tensor; downstream :attr:`BaseReward.normalize` (e.g. ``SigmoidNormalize``)
can squash it into a bounded range when desired.
"""

from typing import Any, Literal

import torch
from huggingface_hub import hf_hub_download
from pydantic import ConfigDict, PrivateAttr

from flow_control.third_party.imagereward import ImageReward as _VendoredImageReward
from flow_control.utils import device as devutil
from flow_control.utils.logging import get_logger
from flow_control.utils.tensor import tensor_to_pil

from .base import BaseReward, reward_registry

logger = get_logger(__name__)

_IMAGEREWARD_REPO = "THUDM/ImageReward"
_IMAGEREWARD_CHECKPOINT = "ImageReward.pt"
_IMAGEREWARD_MED_CONFIG = "med_config.json"


@reward_registry.register("image_reward")
class ImageRewardReward(BaseReward):
    """ImageReward-based reward (THUDM / Jiazheng Xu et al.)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image_reward"] = "image_reward"

    checkpoint_path: str | None = None
    """Path to ``ImageReward.pt``.  ``None`` means download from HF hub."""

    med_config_path: str | None = None
    """Path to ``med_config.json``.  ``None`` means download from HF hub."""

    _model: _VendoredImageReward | None = PrivateAttr(default=None)
    _device: torch.device | None = PrivateAttr(default=None)

    @property
    def _row_fields(self) -> set[str]:
        return {"clean_image", "prompt"}

    def _load_model(self, device: torch.device) -> None:
        self._device = device

        med_config_path = self.med_config_path or hf_hub_download(
            repo_id=_IMAGEREWARD_REPO, filename=_IMAGEREWARD_MED_CONFIG
        )
        checkpoint_path = self.checkpoint_path or hf_hub_download(
            repo_id=_IMAGEREWARD_REPO, filename=_IMAGEREWARD_CHECKPOINT
        )
        logger.info(
            "Loading ImageReward: checkpoint=%s med_config=%s",
            checkpoint_path,
            med_config_path,
        )

        model = _VendoredImageReward(med_config=med_config_path, device=str(device))
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(
                "ImageReward checkpoint missing keys (first 10): %s",
                missing[:10],
            )
        if unexpected:
            logger.warning(
                "ImageReward checkpoint unexpected keys (first 10): %s",
                unexpected[:10],
            )

        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        # ``self.device`` inside the vendored model was set from the constructor
        # arg (``str(device)`` above) and is used by ``score`` for tensor moves.
        self._model = model

    @torch.no_grad()
    def _score(self, row: dict[str, Any]) -> torch.Tensor:
        """Compute ImageReward score for a single sample.

        Expects ``row["clean_image"]`` ([1, C, H, W] in [0, 1]) and
        ``row["prompt"]`` (str).  Returns a raw, un-normalized ``[1]``
        tensor on the image's device.
        """
        if self._model is None:
            raise RuntimeError("ImageRewardReward._load_model has not been called.")

        image = row["clean_image"]
        prompt = row["prompt"]

        pil_image = tensor_to_pil(image)
        # ``score(prompt, [pil_image])`` follows the ``list`` branch
        # (``inference_rank``) which returns ``(indices, rewards)``;
        # ``score(prompt, pil_image)`` returns a Python float.  We use the
        # scalar form to avoid having to disambiguate single-vs-batch.
        value = self._model.score(prompt, pil_image)
        # value is a Python float (single-image path).
        return torch.tensor([float(value)], dtype=torch.float32, device=image.device)

    def _unload_model(self) -> None:
        self._model = None
        self._device = None
        devutil.empty_cache()


if __name__ == "__main__":
    from PIL import Image
    from rich import print as rprint

    from flow_control.utils.tensor import pil_to_tensor

    image = Image.open("examples/assets/image1.png").convert("RGB")
    image_tensor = pil_to_tensor(image)  # [1, C, H, W] in [0, 1]

    reward = ImageRewardReward()
    device = devutil.default_device()
    reward.load_model(device)

    def score(prompt: str) -> float:
        row = {"clean_image": image_tensor.to(device), "prompt": prompt}
        return reward.score(row).raw.item()

    dinner = "Roast dinner with potatoes, brussels sprouts and carrots, red wine"
    car = "A picture of a sports car on a race track"
    score_dinner, score_car = score(dinner), score(car)
    rprint(f"[bold]ImageReward:[/] dinner {score_dinner:.4f}, car {score_car:.4f}")
    assert score_dinner > score_car, (score_dinner, score_car)

    # Regression anchor: under transformers 4.57, before any 5.x shim was
    # needed, this code scores the asset at -0.57376 ("a photo").
    anchor = score("a photo")
    assert abs(anchor + 0.57376) < 1e-3, anchor
    rprint("[bold green]Ordering holds and the anchor matches transformers 4.57[/]")
