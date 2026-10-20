import importlib
import importlib.util
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from torch import Tensor

from flow_control.rewards.base import BaseScorer, LocalReward, scorer_registry
from flow_control.samplers.node import Row
from flow_control.utils.condition_image import ConditionImage, ConditionImageSpec


def _import_pyiqa(module: str = "pyiqa") -> Any:
    """Import the optional IQA backend with an actionable error."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name == "pyiqa":
            raise ModuleNotFoundError(
                "PyIQAScorer requires the optional iqa dependency group; "
                "install it with `uv sync --group iqa`."
            ) from error
        raise


def _default_config(name: str) -> dict[str, Any]:
    """Look up a metric's pyiqa default config, with a clear error for typos."""
    default_configs = _import_pyiqa("pyiqa.default_model_configs").DEFAULT_CONFIGS

    try:
        return default_configs[name]
    except KeyError:
        raise ValueError(
            f"Unknown pyiqa metric {name!r}. "
            f"See `pyiqa.list_models()` for available metric names."
        ) from None


class PyIQAMetricSpec(BaseModel):
    """One pyiqa metric inside a :class:`PyIQAScorer` battery."""

    name: str
    """pyiqa metric id, e.g. ``"psnr"``, ``"lpips"``, ``"musiq"``."""
    label: str | None = None
    """Unique output label. Defaults to :attr:`name`; set it when configuring
    the same metric more than once with different parameters."""
    weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    """Aggregation weight (magnitude only; direction is handled by
    :attr:`PyIQAScorer.flip_lower_better`)."""
    params: dict[str, Any] = Field(default_factory=dict)
    """Extra kwargs forwarded to ``pyiqa.create_metric``."""

    model_config = ConfigDict(extra="forbid")


@scorer_registry.register("pyiqa")
class PyIQAScorer(BaseScorer, LocalReward[Row, Tensor]):
    """Multi-component IQA battery backed by pyiqa.

    Each configured metric becomes one component (labelled ``<name>/<metric>``),
    so a single scorer can report a full quality table (PSNR, SSIM, LPIPS,
    MUSIQ, ...) per sample.

    Row contract: ``clean_image`` is the image under evaluation as
    ``[1, C, H, W]`` in ``[0, 1]``; full-reference (FR) metrics additionally
    read the :attr:`reference` condition image in the same format and size.
    No-reference (NR) metrics ignore the reference, so a battery of only NR
    metrics can score images without ground truth.

    Raw scores stay model-native (PSNR in dB, LPIPS distance, ...), so a
    multi-metric aggregate is only meaningful when the caller deliberately
    chooses compatible normalization and weights. With
    :attr:`flip_lower_better` (default), lower-is-better metrics get negated
    weights so the weighted sum is monotonically higher-is-better;
    per-component values are unaffected.
    """

    type: Literal["pyiqa"] = "pyiqa"
    metrics: list[PyIQAMetricSpec] = Field(min_length=1)
    flip_lower_better: bool = True
    reference: ConditionImageSpec = "reference"
    """Condition image for full-reference metrics, e.g. ``"reference[0]"`` or
    ``"control"``; it must have the generated image's shape."""

    model_config = ConfigDict(extra="forbid")

    _models: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_metrics(self) -> "PyIQAScorer":
        labels = self.labels
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(
                "PyIQAScorer metric labels must be unique; set an explicit label "
                f"for repeated metrics: {duplicates}"
            )
        for spec in self.metrics:
            _default_config(spec.name)
        return self

    @property
    def labels(self) -> list[str]:
        return [spec.label or spec.name for spec in self.metrics]

    @property
    def weights(self) -> list[float]:
        weights = []
        for spec in self.metrics:
            lower_better = _default_config(spec.name).get("lower_better", False)
            flip = -1.0 if lower_better and self.flip_lower_better else 1.0
            weights.append(spec.weight * flip)
        return weights

    @property
    def fields(self) -> set[str]:
        fields = {"clean_image"}
        if any(
            _default_config(spec.name)["metric_mode"] == "FR" for spec in self.metrics
        ):
            fields.add(ConditionImage.parse(self.reference).image_field)
        return fields

    def load(self, device: torch.device) -> None:
        pyiqa = _import_pyiqa()

        self._device = device
        self._models = [
            pyiqa.create_metric(spec.name, device=device, **spec.params)
            for spec in self.metrics
        ]

    @torch.no_grad()
    def run_batch(self, rows: list[Row]) -> list[Tensor]:
        return [self._score_row(row) for row in rows]

    def _score_row(self, row: Row) -> Tensor:
        image = row["clean_image"].to(device=self._device, dtype=torch.float32)
        reference: torch.Tensor | None = None

        scores = []
        for spec, model in zip(self.metrics, self._models, strict=True):
            if model.metric_mode == "FR":
                if reference is None:
                    reference = (
                        ConditionImage.parse(self.reference)
                        .image(row)
                        .to(device=self._device, dtype=torch.float32)
                    )
                if reference.shape != image.shape:
                    raise ValueError(
                        f"FR metric {spec.name!r} requires matching shapes, got "
                        f"image {tuple(image.shape)} vs reference {tuple(reference.shape)}"
                    )
                value = model(image, reference)
            else:
                value = model(image)
            scores.append(value.reshape(-1).float())
        return torch.cat(scores)  # [C]: one value per metric of the [1, C, H, W] image


if __name__ == "__main__":
    from rich import print

    from flow_control.utils import device as devutil

    if importlib.util.find_spec("pyiqa") is None:
        print("[yellow]pyiqa is not installed; skipping IQA self-test.[/yellow]")
        raise SystemExit(0)

    device = devutil.default_device()
    generator = torch.Generator().manual_seed(0)
    clean = torch.rand(1, 3, 256, 256, generator=generator)
    slightly_noisy = (
        clean + 0.05 * torch.randn(clean.shape, generator=generator)
    ).clamp(0, 1)
    very_noisy = (clean + 0.30 * torch.randn(clean.shape, generator=generator)).clamp(
        0, 1
    )

    reward = PyIQAScorer(
        metrics=[
            PyIQAMetricSpec(name="psnr"),
            PyIQAMetricSpec(name="lpips"),
            PyIQAMetricSpec(name="musiq"),
        ]
    )
    print(f"[bold]fields:[/] {reward.fields}")
    print(f"[bold]weights:[/] {reward.weights}")
    reward.load(device)

    good, bad = reward.run_batch(
        [
            {"clean_image": slightly_noisy, "reference_images": [clean]},
            {"clean_image": very_noisy, "reference_images": [clean]},
        ]
    )
    weights = torch.tensor(reward.weights)
    for label, raw in [("slightly noisy", good), ("very noisy", bad)]:
        row = {k: f"{v:.4f}" for k, v in zip(reward.labels, raw.tolist(), strict=True)}
        print(f"[bold]{label}:[/] {row} weighted={(raw.cpu() * weights).sum():.4f}")

    psnr_good, lpips_good = good[0], good[1]
    psnr_bad, lpips_bad = bad[0], bad[1]
    assert psnr_good > psnr_bad, (
        f"PSNR should prefer mild noise: {psnr_good} vs {psnr_bad}"
    )
    assert lpips_good < lpips_bad, (
        f"LPIPS should prefer mild noise: {lpips_good} vs {lpips_bad}"
    )
    assert (good.cpu() * weights).sum() > (bad.cpu() * weights).sum(), (
        "the weighted sum should be higher-is-better with flip_lower_better"
    )

    nr_only = PyIQAScorer(metrics=[PyIQAMetricSpec(name="niqe")])
    assert nr_only.fields == {"clean_image"}
    nr_only.load(device)
    (nr_score,) = nr_only.run_batch([{"clean_image": very_noisy}])
    print(f"[bold]NR-only niqe:[/] {nr_score.item():.4f}")
    print("[bold green]Self-test passed.[/]")
