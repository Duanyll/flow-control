"""T2I preprocessing with a cached tile layout and per-tile conditions."""

from typing import Annotated, Any, ClassVar, Literal, NotRequired

from flow_control.data.coercion import JsonBeforeValidator

from ..base import task_registry
from ..tiles import TileConfig
from .t2i import T2IInputRow, T2IProcessedRow, T2IProcessor, T2ITrainInputRow


class TiledT2IInputRow(T2IInputRow):
    tiles: NotRequired[Annotated[list[T2IInputRow], JsonBeforeValidator]]


class TiledT2ITrainInputRow(T2ITrainInputRow):
    tiles: NotRequired[Annotated[list[T2IInputRow], JsonBeforeValidator]]


@task_registry.register("tiled_t2i")
class TiledT2IProcessor(T2IProcessor, TileConfig):
    task: Literal["tiled_t2i"] = "tiled_t2i"
    inference_input_type: ClassVar[type] = TiledT2IInputRow
    training_input_type: ClassVar[type] = TiledT2ITrainInputRow

    async def _add_tiles(
        self, result: T2IProcessedRow, inputs: list[T2IInputRow] | None
    ) -> T2IProcessedRow:
        layout = self.layout(self.patch_size * self.vae_scale_factor)
        specs = layout.token_specs(result["image_size"])
        size = specs[0].height * layout.stride, specs[0].width * layout.stride
        tiles: list[Any] = []
        if inputs is None:
            # Each tile carries the shared conditioning, including its negative.
            shared: Any = {k: v for k, v in result.items() if k != "clean_latents"}
            shared["image_size"] = size
            tiles = [shared.copy() for _ in specs]
        else:
            if len(inputs) != len(specs):
                raise ValueError(
                    f"Expected {len(specs)} row-major tile prompts, got {len(inputs)}."
                )
            for index, source in enumerate(inputs):
                if "tiles" in source:
                    raise ValueError("Nested tile conditioning is not supported.")
                declared = source.get("image_size")
                if declared is not None and tuple(declared) != size:
                    raise ValueError(
                        f"Tile {index} image_size must match the layout size {size}."
                    )
                tiles.append(
                    await super().prepare_inference_row({**source, "image_size": size})
                )
        result["tiling"] = layout.model_dump()
        result["model_image_size"] = size
        result["tiles"] = tiles
        negative: Any = result.get("negative")
        if negative is not None:
            # The whole-image negative overlay also swaps in the negative tiles.
            result["negative"] = {
                **negative,
                "tiles": [{**tile, **tile["negative"]} for tile in tiles],
            }
        return result

    async def prepare_inference_row(
        self, row: T2IInputRow | TiledT2IInputRow
    ) -> T2IProcessedRow:
        return await self._add_tiles(
            await super().prepare_inference_row(row), row.get("tiles")
        )

    async def prepare_training_row(
        self, row: T2ITrainInputRow | TiledT2ITrainInputRow
    ) -> T2IProcessedRow:
        return await self._add_tiles(
            await super().prepare_training_row(row), row.get("tiles")
        )
