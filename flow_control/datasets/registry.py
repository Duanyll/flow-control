"""Dataset / datasink registries.

Members are duck-typed torch ``Dataset`` / pipeline ``DataSink`` subclasses (not
Pydantic models), so they don't use ``RegistryUnion``; ``parse_dataset`` /
``parse_datasink`` instantiate the looked-up class with the config kwargs. The
registries live here (datasets have no shared base class to host them) so each
member module can ``@dataset_registry.register(...)`` at its definition site
without importing the package ``__init__``. ``base=None`` skips the nominal
subclass guard (members don't share a project base class).
"""

from typing import Any

from flow_control.utils.registry import Registry

dataset_registry: Registry[Any] = Registry("dataset")
datasink_registry: Registry[Any] = Registry("datasink")
