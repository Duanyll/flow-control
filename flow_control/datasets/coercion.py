"""Compatibility shell: the coercion helpers moved to ``flow_control.data.coercion``.

Kept only so the old ``flow_control.datasets`` imports keep resolving until that
package is deleted; new code imports ``flow_control.data.coercion`` directly.
"""

from flow_control.data.coercion import (
    ImageTensor,
    ImageTensorList,
    JsonBeforeValidator,
    JsonStrList,
    _coerce_to_image_tensor,
    _load_attachment,
    _resolve_path,
    _validate_image_tensor,
    _validate_image_tensor_list,
    _validate_json,
    _validate_json_str_list,
    build_type_adapter,
    coerce_record,
)

__all__ = [
    "ImageTensor",
    "ImageTensorList",
    "JsonBeforeValidator",
    "JsonStrList",
    "_coerce_to_image_tensor",
    "_load_attachment",
    "_resolve_path",
    "_validate_image_tensor",
    "_validate_image_tensor_list",
    "_validate_json",
    "_validate_json_str_list",
    "build_type_adapter",
    "coerce_record",
]
