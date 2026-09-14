"""Select a condition image from a row by role.

Tasks store condition images under their own names, split between the pixel
image and its packed latents, and references form ordered lists. Samplers and
rewards that need one condition image are configured with a selector such as
``"reference[1]"`` and resolve it through :class:`ConditionImage` instead of
naming row keys.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Any

import torch
from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError

_ROLES: dict[str, tuple[str, str]] = {
    "reference": ("reference_images", "reference_latents"),
    "control": ("control_image", "control_latents"),
    "inpaint": ("inpaint_image", "inpaint_latents"),
    "clean": ("clean_image", "clean_latents"),
}
_SPEC = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?")


class ConditionImage(BaseModel):
    """A parsed condition-image selector.

    Known roles map to a task's pixel field and packed-latent field; any other
    name is a literal row key holding whichever representation is requested.
    An index picks one entry of a list-valued field and defaults to ``0``.
    """

    model_config = ConfigDict(frozen=True)
    role: str
    index: int | None = None

    @classmethod
    def parse(cls, spec: str) -> ConditionImage:
        match = _SPEC.fullmatch(spec.strip())
        if match is None:
            raise ValueError(
                f"Condition image {spec!r} must be a role or row key, optionally "
                f"indexed like 'reference[1]'; roles: {sorted(_ROLES)}."
            )
        role, index = match.groups()
        return cls(role=role, index=None if index is None else int(index))

    def __str__(self) -> str:
        return self.role if self.index is None else f"{self.role}[{self.index}]"

    @property
    def image_field(self) -> str:
        """Top-level row key holding the ``[1, C, H, W]`` pixel image."""
        return _ROLES.get(self.role, (self.role, self.role))[0]

    @property
    def latents_field(self) -> str:
        """Top-level row key holding the packed ``[1, N, D]`` latents."""
        return _ROLES.get(self.role, (self.role, self.role))[1]

    def image(self, row: Mapping[str, Any]) -> torch.Tensor:
        return self._select(row, self.image_field)

    def latents(self, row: Mapping[str, Any]) -> torch.Tensor:
        return self._select(row, self.latents_field)

    def _select(self, row: Mapping[str, Any], field: str) -> torch.Tensor:
        value = row.get(field)
        if isinstance(value, list):
            index = self.index or 0
            if index >= len(value):
                raise IndexError(
                    f"Condition image {self} indexes {field!r}, which has {len(value)} entries."
                )
            value = value[index]
        elif self.index is not None:
            raise TypeError(
                f"Condition image {self} indexes {field!r}, which is not a list in the row."
            )
        if not isinstance(value, torch.Tensor):
            raise KeyError(
                f"Condition image {self} requires tensor field {field!r}; "
                f"available fields: {sorted(row)}."
            )
        return value


def _validate_spec(spec: str) -> str:
    ConditionImage.parse(spec)
    return spec


ConditionImageSpec = Annotated[str, AfterValidator(_validate_spec)]
"""Config field type: a selector string such as ``"reference[0]"`` or ``"control"``,
checked at validation time and parsed with :meth:`ConditionImage.parse` at use."""


if __name__ == "__main__":
    from rich import print

    row = {
        "reference_images": [torch.zeros(1, 3, 8, 8), torch.ones(1, 3, 4, 4)],
        "reference_latents": [torch.zeros(1, 4, 16), torch.ones(1, 1, 16)],
        "control_image": torch.zeros(1, 3, 8, 8),
        "custom_latents": torch.zeros(1, 4, 16),
    }
    assert ConditionImage.parse("reference").image(row).shape == (1, 3, 8, 8)
    assert ConditionImage.parse("reference[1]").latents(row).shape == (1, 1, 16)
    assert ConditionImage.parse("control").image_field == "control_image"
    assert ConditionImage.parse("custom_latents").latents(row).shape == (1, 4, 16)
    for spec, error in (
        ("reference[2]", IndexError),
        ("control[0]", TypeError),
        ("inpaint", KeyError),
        ("bad spec", ValueError),
    ):
        try:
            ConditionImage.parse(spec).image(row)
        except error as e:
            print(f"[green]{spec!r}[/green]: {e}")
        else:
            raise AssertionError(f"{spec!r} should raise {error.__name__}")

    class Config(BaseModel):
        source: ConditionImageSpec = "inpaint"

    assert Config(source="reference[1]").model_dump() == {"source": "reference[1]"}
    try:
        Config.model_validate({"source": "bad spec"})
    except ValidationError as e:
        print(f"[green]config[/green]: {e.errors()[0]['msg']}")
    else:
        raise AssertionError("invalid selector should fail validation")
    schema = Config.model_json_schema()["properties"]["source"]
    assert (schema["type"], schema["default"]) == ("string", "inpaint"), schema
    print("[bold green]ConditionImage OK[/bold green]")
