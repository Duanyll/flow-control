"""Out-of-tree-style experimental components that ship in the repo but are
deliberately NOT imported by any core ``__init__.py``.

Each submodule self-registers its members into the relevant registry on import
(via ``@<registry>.register(...)`` decorators at the definition site), exactly
like an external plugin would. To activate a ``contrib`` component, list its
module in a config's ``imports`` key, e.g.::

    imports = ["flow_control.contrib.efficient_layered"]

This package keeps no auto-imports on purpose: importing ``flow_control.contrib``
must have zero side effects, so the core unions stay free of experimental tags
unless a config explicitly opts in.
"""
