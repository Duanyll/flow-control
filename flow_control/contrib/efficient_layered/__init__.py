"""Experimental ``efficient_layered`` components (adapter, processor task, dataset).

Relocated out of core into ``flow_control.contrib`` so the experimental
``efficient_layered`` tag never appears in the core model-adapter / processor-task
unions unless a config opts in via ``imports``. Importing this package
self-registers:

- ``EfficientLayeredQwenImageAdapter`` under ``adapter_registry`` tag
  ``"qwen_efficient_layered"`` (the ``f"{arch}_{type}"`` discriminator for
  ``arch="qwen"``, ``type="efficient_layered"``).
- ``EfficientLayeredProcessor`` under ``task_registry`` tag ``"efficient_layered"``.
- ``PrismLayersProDataset`` under ``dataset_registry`` tag ``"prism_layers_pro"``.

Activate for training / preprocessing with, e.g.::

    imports = ["flow_control.contrib.efficient_layered"]

The Gradio UI template lives in the ``.serving`` submodule and is NOT imported
here, so training runs stay free of the serving stack. Serving configs add it
explicitly::

    imports = ["flow_control.contrib.efficient_layered.serving"]
"""

from . import adapter, dataset, processor

__all__ = ["adapter", "dataset", "processor"]
