"""T2I preprocessing with a cached tile layout and per-tile conditions."""

from typing import Annotated, Any, ClassVar, Literal, NotRequired

from flow_control.data.coercion import JsonBeforeValidator

from ..base import ProcessedBatch, task_registry
from ..tiles import TileConfig
from .t2i import T2IInputBatch, T2IProcessedBatch, T2IProcessor, T2ITrainInputBatch


class TiledT2IInputBatch(T2IInputBatch):
    tiles: NotRequired[Annotated[list[T2IInputBatch], JsonBeforeValidator]]


class TiledT2ITrainInputBatch(T2ITrainInputBatch):
    tiles: NotRequired[Annotated[list[T2IInputBatch], JsonBeforeValidator]]


@task_registry.register("tiled_t2i")
class TiledT2IProcessor(T2IProcessor, TileConfig):
    task: Literal["tiled_t2i"] = "tiled_t2i"
    inference_input_type: ClassVar[type] = TiledT2IInputBatch
    training_input_type: ClassVar[type] = TiledT2ITrainInputBatch

    async def _add_tiles(
        self, result: T2IProcessedBatch, inputs: list[T2IInputBatch] | None
    ) -> T2IProcessedBatch:
        layout = self.layout(self.patch_size * self.vae_scale_factor)
        specs = layout.token_specs(result["image_size"])
        size = specs[0].height * layout.stride, specs[0].width * layout.stride
        negative: Any = result.get("negative")
        tiles: list[ProcessedBatch] = []
        negative_tiles: list[ProcessedBatch] = []
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
                tile: Any = await super().prepare_inference_batch(tile_input)
                tile_negative = tile.pop("negative", None)
                tiles.append(tile)
                if tile_negative is not None:
                    negative_tiles.append({**tile, **tile_negative})
        result["tiling"] = layout.model_dump()
        result["model_image_size"] = size
        result["tiles"] = tiles
        if negative is not None:
            # ``get_negative_batch`` overlays this dict, replacing ``tiles`` too.
            result["negative"] = {**negative, "tiles": negative_tiles}
        return result

    async def prepare_inference_batch(
        self, batch: T2IInputBatch | TiledT2IInputBatch
    ) -> T2IProcessedBatch:
        return await self._add_tiles(
            await super().prepare_inference_batch(batch), batch.get("tiles")
        )

    async def prepare_training_batch(
        self, batch: T2ITrainInputBatch | TiledT2ITrainInputBatch
    ) -> T2IProcessedBatch:
        return await self._add_tiles(
            await super().prepare_training_batch(batch), batch.get("tiles")
        )
