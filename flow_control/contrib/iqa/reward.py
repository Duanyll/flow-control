import importlib
import importlib.util
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from flow_control.rewards.base import BaseReward, reward_registry
from flow_control.utils import device as devutil


def _import_pyiqa(module: str = "pyiqa") -> Any:
    """Import the optional IQA backend with an actionable error."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name == "pyiqa":
            raise ModuleNotFoundError(
                "PyIQAReward requires the optional iqa dependency group; "
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
    """One pyiqa metric inside a :class:`PyIQAReward` battery."""

    name: str
    """pyiqa metric id, e.g. ``"psnr"``, ``"lpips"``, ``"musiq"``."""
    label: str | None = None
    """Unique output label. Defaults to :attr:`name`; set it when configuring
    the same metric more than once with different parameters."""
    weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    """Aggregation weight (magnitude only; direction is handled by
    :attr:`PyIQAReward.flip_lower_better`)."""
    params: dict[str, Any] = Field(default_factory=dict)
    """Extra kwargs forwarded to ``pyiqa.create_metric``."""

    model_config = ConfigDict(extra="forbid")


@reward_registry.register("pyiqa")
class PyIQAReward(BaseReward):
    """Multi-component IQA battery backed by pyiqa.

    Each configured metric becomes one component of the :class:`RewardResult`,
    so a single reward instance can report a full quality table (PSNR, SSIM,
    LPIPS, MUSIQ, ...) per sample.

    Batch contract: ``clean_image`` is the image under evaluation as
    ``[1, C, H, W]`` in ``[0, 1]``; full-reference (FR) metrics additionally
    read ``reference_image`` (the ground truth) in the same format and size.
    No-reference (NR) metrics ignore the reference, so a battery of only NR
    metrics can score images without ground truth.

    Raw scores stay model-native (PSNR in dB, LPIPS distance, ...), so a
    multi-metric aggregate is only meaningful when the caller deliberately
    chooses compatible normalization and weights. With
    :attr:`flip_lower_better` (default), lower-is-better metrics get negated
    aggregation weights so ``aggregate()`` is monotonically higher-is-better;
    per-component values are unaffected.
    """

    type: Literal["pyiqa"] = "pyiqa"
    metrics: list[PyIQAMetricSpec] = Field(min_length=1)
    flip_lower_better: bool = True

    model_config = ConfigDict(extra="forbid")

    _models: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_metrics(self) -> "PyIQAReward":
        labels = [spec.label or spec.name for spec in self.metrics]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(
                "PyIQAReward metric labels must be unique; set an explicit label "
                f"for repeated metrics: {duplicates}"
            )
        for spec in self.metrics:
            _default_config(spec.name)
        return self

    @property
    def component_weights(self) -> list[float]:
        weights = []
        for spec in self.metrics:
            lower_better = _default_config(spec.name).get("lower_better", False)
            flip = -1.0 if lower_better and self.flip_lower_better else 1.0
            weights.append(spec.weight * flip)
        return weights

    @property
    def component_labels(self) -> list[str]:
        return [spec.label or spec.name for spec in self.metrics]

    @property
    def _batch_fields(self) -> set[str]:
        fields = {"clean_image"}
        if any(
            _default_config(spec.name)["metric_mode"] == "FR" for spec in self.metrics
        ):
            fields.add("reference_image")
        return fields

    def _load_model(self, device: torch.device) -> None:
        pyiqa = _import_pyiqa()

        self._device = device
        self._models = [
            pyiqa.create_metric(spec.name, device=device, **spec.params)
            for spec in self.metrics
        ]

    @torch.no_grad()
    def _score(self, batch: dict[str, Any]) -> torch.Tensor:
        image = batch["clean_image"].to(device=self._device, dtype=torch.float32)
        reference = batch.get("reference_image")
        if reference is not None:
            reference = reference.to(device=self._device, dtype=torch.float32)

        scores = []
        for spec, model in zip(self.metrics, self._models, strict=True):
            if model.metric_mode == "FR":
                if reference is None:
                    raise ValueError(
                        f"FR metric {spec.name!r} requires batch['reference_image']"
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
        return torch.stack(scores, dim=-1)  # [N, C]

    def _unload_model(self) -> None:
        import gc

        self._models = None
        gc.collect()
        devutil.empty_cache()


if __name__ == "__main__":
    from rich import print

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

    reward = PyIQAReward(
        metrics=[
            PyIQAMetricSpec(name="psnr"),
            PyIQAMetricSpec(name="lpips"),
            PyIQAMetricSpec(name="musiq"),
        ]
    )
    print(f"[bold]batch fields:[/] {reward._batch_fields}")
    print(f"[bold]component weights:[/] {reward.component_weights}")
    reward.load_model(device)

    good = reward.score({"clean_image": slightly_noisy, "reference_image": clean})
    bad = reward.score({"clean_image": very_noisy, "reference_image": clean})
    for label, result in [("slightly noisy", good), ("very noisy", bad)]:
        row = {
            k: f"{v:.4f}"
            for k, v in zip(result.labels, result.raw[0].tolist(), strict=True)
        }
        print(f"[bold]{label}:[/] {row} aggregate={result.aggregate().item():.4f}")

    psnr_good, lpips_good = good.raw[0, 0], good.raw[0, 1]
    psnr_bad, lpips_bad = bad.raw[0, 0], bad.raw[0, 1]
    assert psnr_good > psnr_bad, (
        f"PSNR should prefer mild noise: {psnr_good} vs {psnr_bad}"
    )
    assert lpips_good < lpips_bad, (
        f"LPIPS should prefer mild noise: {lpips_good} vs {lpips_bad}"
    )
    assert good.aggregate().item() > bad.aggregate().item(), (
        "aggregate() should be higher-is-better with flip_lower_better"
    )

    nr_only = PyIQAReward(metrics=[PyIQAMetricSpec(name="niqe")])
    assert nr_only._batch_fields == {"clean_image"}
    nr_only.load_model(device)
    nr_score = nr_only.score({"clean_image": very_noisy})
    print(f"[bold]NR-only niqe:[/] {nr_score.raw[0, 0].item():.4f}")

    reward.unload_model()
    nr_only.unload_model()
    print("[bold green]Self-test passed.[/]")
