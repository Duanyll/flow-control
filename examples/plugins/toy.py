"""Canonical toy plugin: the smallest possible out-of-tree registry member.

This file is the teaching template for writing your own ``flow_control`` plugin.
A plugin is just a module that, when imported, registers one or more members into
a core registry via the registry's ``@register(...)`` decorator. Nothing else is
required: no entry-point metadata, no base-class edits, no ``model_rebuild``.

To activate it, list this module in a config's ``imports`` key, e.g. (TOML)::

    imports = ["examples.plugins.toy"]

    [sampler.shift]
    type = "toy_constant"
    factor = 3.0

``flow-control`` calls ``load_plugins(config["imports"])`` before validating the
config, so the ``"toy_constant"`` tag is registered into the open ``Shift`` union
by the time the ``sampler.shift`` field is parsed.

We pick ``Shift`` because it is the simplest single-member family: subclass
``BaseShift``, set a ``Literal`` ``type`` matching the registered tag, and
implement one method.
"""

from typing import Literal

from flow_control.samplers.shift import BaseShift, shift_registry


@shift_registry.register("toy_constant")
class ToyConstantShift(BaseShift):
    """A trivial shift that always returns the same constant factor.

    Behaves exactly like the built-in ``constant`` shift; it exists only to show
    the minimal shape of a plugin member.
    """

    type: Literal["toy_constant"] = "toy_constant"
    factor: float = 1.0

    def _calculate_shift_factor(self, seq_len: int, num_steps: int) -> float:
        return self.factor
