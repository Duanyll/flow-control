"""Target preparation for the RGBA VAE trainer: its resample stage (design §9.5).

The VAE has no processor, so ``VaeTargetResample`` stands in for
``processor.resample``: random crop, random background blend and the
``[0, 1] -> [-1, 1]`` mapping, all driven by the per-row generator the trainer
seeds from ``(epoch, key)``.
"""

import torch

from flow_control.data import KEY, Row
from flow_control.utils.resize import (
    resize_short_side_and_random_crop,
    resize_to_multiple_of,
)
from flow_control.utils.tensor import ensure_alpha_channel

TARGET_IMAGE = "target_image"


def _maybe_blend_rgba_to_bg(
    target: torch.Tensor, blend_prob: float, generator: torch.Generator
) -> torch.Tensor:
    if blend_prob <= 0:
        return target
    draw = torch.rand(1, generator=generator, device=generator.device).item()
    if draw >= blend_prob:
        return target

    rgb = target[:, :3]
    alpha = target[:, 3:4]
    bg = torch.randint(
        0, 3, (1, 3, 1, 1), generator=generator, device=target.device
    ).to(target.dtype)
    bg = bg / 2.0
    blended_rgb = rgb * alpha + bg * (1 - alpha)
    return torch.cat([blended_rgb, torch.ones_like(alpha)], dim=1)


def prepare_vae_target_image(
    clean_image: torch.Tensor,
    *,
    key: str = "unknown",
    random_crop_size: int | None,
    resize_multiple: int,
    resize_pixels: int,
    blend_prob: float,
    generator: torch.Generator,
) -> torch.Tensor:
    target = clean_image.float()
    if target.shape[1] not in {3, 4}:
        raise ValueError(
            f"Expected clean_image to have 3 or 4 channels, got shape {tuple(target.shape)} "
            f"for sample {key!r}."
        )

    target = ensure_alpha_channel(target)

    if random_crop_size is not None:
        target = resize_short_side_and_random_crop(
            target,
            crop_size=random_crop_size,
            multiple=resize_multiple,
            generator=generator,
        )
    else:
        target = resize_to_multiple_of(target, resize_multiple, pixels=resize_pixels)

    target = _maybe_blend_rgba_to_bg(target, blend_prob, generator)
    return target * 2 - 1  # [0, 1] -> [-1, 1]


class VaeTargetResample:
    """Writes ``row["target_image"]`` from ``row["clean_image"]``; the generator
    must live on the row's device (call after ``deep_move_to_device``)."""

    def __init__(
        self,
        *,
        random_crop_size: int | None,
        resize_multiple: int,
        resize_pixels: int,
        blend_prob: float,
    ):
        self.random_crop_size = random_crop_size
        self.resize_multiple = resize_multiple
        self.resize_pixels = resize_pixels
        self.blend_prob = blend_prob

    def __call__(self, row: Row, generator: torch.Generator) -> Row:
        row[TARGET_IMAGE] = prepare_vae_target_image(
            row["clean_image"],
            key=row.get(KEY, "unknown"),
            random_crop_size=self.random_crop_size,
            resize_multiple=self.resize_multiple,
            resize_pixels=self.resize_pixels,
            blend_prob=self.blend_prob,
            generator=generator,
        )
        return row


if __name__ == "__main__":
    from rich import print

    resample = VaeTargetResample(
        random_crop_size=16, resize_multiple=8, resize_pixels=0, blend_prob=1.0
    )
    image = torch.rand(1, 4, 40, 24)
    image[:, 3] = 0.5
    outputs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(3)
        outputs.append(
            resample({KEY: "a", "clean_image": image.clone()}, g)["target_image"]
        )
    print(outputs[0].shape, outputs[0].min().item(), outputs[0].max().item())
    assert outputs[0].shape == (1, 4, 16, 16)
    assert torch.equal(outputs[0], outputs[1])  # same seed, same crop and blend
    assert torch.all(outputs[0][:, 3] == 1.0)  # blended onto an opaque background
    assert outputs[0].min() >= -1 and outputs[0].max() <= 1
    other = resample(
        {KEY: "a", "clean_image": image.clone()}, torch.Generator().manual_seed(4)
    )
    assert not torch.equal(other["target_image"], outputs[0])
