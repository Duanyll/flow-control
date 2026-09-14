"""``flow-control pack <config.jsonc>``: random cache -> packed tar shards."""

from flow_control.data.pack import PackConfig, pack


def run(config_data: dict) -> None:
    pack(PackConfig(**config_data))
