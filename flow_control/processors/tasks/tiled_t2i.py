"""T2I preprocessing with a cached tile layout and per-tile conditions."""

from typing import Annotated, Any, ClassVar, Literal, NotRequired

from flow_control.data.coercion import JsonBeforeValidator

from ..base import ProcessedRow, task_registry
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
        negative: Any = result.get("negative")
        tiles: list[ProcessedRow] = []
        negative_tiles: list[ProcessedRow] = []
        if inputs is None:
            shared: Any = {
                key: value
                for key, value in result.items()
                if key not in ("clean_latents", "negative")
            }
            shared["image_size"] = size
            tiles = [shared.copy() for _ in specs]
            if negative is not None:
                negative_tiles = [{**shared, **negative} for _ in specs]
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
                tile_input = source.copy()
                tile_input["image_size"] = size
                tile: Any = await super().prepare_inference_row(tile_input)
                tile_negative = tile.pop("negative", None)
                tiles.append(tile)
                if tile_negative is not None:
                    negative_tiles.append({**tile, **tile_negative})
        result["tiling"] = layout.model_dump()
        result["model_image_size"] = size
        result["tiles"] = tiles
        if negative is not None:
            # ``get_negative_row`` overlays this dict, replacing ``tiles`` too.
            result["negative"] = {**negative, "tiles": negative_tiles}
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
