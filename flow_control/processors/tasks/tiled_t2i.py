"""T2I preprocessing with a cached tile layout and per-tile conditions."""

from typing import Annotated, ClassVar, Literal, NotRequired

from flow_control.datasets.coercion import JsonBeforeValidator

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
        stride = self.patch_size * self.vae_scale_factor
        self.validate_stride(stride)
        image_size = result["image_size"]
        if any(length % stride for length in image_size):
            raise ValueError(
                f"Tiled image_size {image_size} must be aligned to the packed pixel stride ({stride})."
            )
        size = self.size_for(image_size)
        count = len(self.origins(image_size))
        tiles: list[ProcessedBatch] = []
        if inputs is None:
            shared = result.copy()
            shared.pop("clean_latents", None)
            shared["image_size"] = size
            tiles = [shared.copy() for _ in range(count)]
        else:
            if len(inputs) != count:
                raise ValueError(
                    f"Expected {count} row-major tile prompts, got {len(inputs)}."
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
                tiles.append(await super().prepare_inference_batch(tile_input))
        result["tiling"] = self.model_dump(include=set(TileConfig.model_fields))
        result["tiles"] = tiles
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
