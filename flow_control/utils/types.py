import sys
from typing import Annotated, Any

import torch
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, WithJsonSchema
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema


class _TorchDTypePydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def validate_from_str(value: str) -> torch.dtype:
            try:
                dtype_obj = getattr(torch, value)
                if isinstance(dtype_obj, torch.dtype):
                    return dtype_obj
            except AttributeError:
                pass
            raise ValueError(f"Invalid torch dtype: {value}")

        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(validate_from_str),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(torch.dtype),
                    from_str_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: str(v).split(".")[-1]
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {
            "type": "string",
            "description": "PyTorch dtype (e.g. float32, bfloat16)",
        }


class _TorchDevicePydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def validate_from_str(value: str) -> torch.device:
            return torch.device(value)

        def validate_from_int(value: int) -> torch.device:
            return torch.device(value)

        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(validate_from_str),
            ]
        )
        from_int_schema = core_schema.chain_schema(
            [
                core_schema.int_schema(),
                core_schema.no_info_plain_validator_function(validate_from_int),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(torch.device),
                    from_str_schema,
                    from_int_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: str(v)
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {
            "type": "string",
            "description": "PyTorch device (e.g. cuda:0, cpu)",
        }


TorchDType = Annotated[torch.dtype, _TorchDTypePydanticAnnotation]
TorchDevice = Annotated[torch.device, _TorchDevicePydanticAnnotation]


OptimizerConfig = Annotated[
    dict[str, Any],
    WithJsonSchema(
        {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "Optimizer class name (e.g. AdamW, Prodigy)",
                },
                "lr": {"type": "number", "description": "Learning rate"},
            },
            "required": ["class_name"],
            "additionalProperties": True,
        }
    ),
]


def parse_optimizer(
    conf: OptimizerConfig,
    parameters,
) -> torch.optim.Optimizer:
    conf = conf.copy()
    class_name = conf.pop("class_name")
    if class_name == "Prodigy":
        from prodigyopt import Prodigy

        ctor = Prodigy
    elif class_name.endswith("8bit") or class_name.startswith("Paged"):
        if sys.platform == "darwin":
            raise ImportError(
                f"bitsandbytes optimizers are not supported on macOS: {class_name}"
            )
        import bitsandbytes as bnb

        ctor = getattr(bnb.optim, class_name)
    else:
        ctor = getattr(torch.optim, class_name)
    return ctor(parameters, **conf)


SchedulerConfig = Annotated[
    dict[str, Any],
    WithJsonSchema(
        {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "LR scheduler class name (e.g. ConstantLR, CosineAnnealingLR), or 'diffusers' for Diffusers schedulers",
                },
            },
            "required": ["class_name"],
            "additionalProperties": True,
        }
    ),
]


def parse_scheduler(conf: SchedulerConfig, optimizer: torch.optim.Optimizer):
    conf = conf.copy()
    class_name = conf.pop("class_name")
    if class_name == "diffusers":
        from diffusers.optimization import get_scheduler

        name = conf.pop("name", None)
        return get_scheduler(
            name=name,
            optimizer=optimizer,
            **conf,
        )
    else:
        ctor = getattr(torch.optim.lr_scheduler, class_name)
        return ctor(optimizer, **conf)
