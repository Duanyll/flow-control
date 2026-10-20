"""pyiqa-backed image quality assessment rewards.

Wraps `pyiqa <https://github.com/chaofengc/IQA-PyTorch>`_ metrics (PSNR, SSIM,
LPIPS, DISTS, NIQE, MUSIQ, ...) as a single multi-component scorer, so IQA
batteries can be used anywhere a scorer is accepted (inference eval, scorer
lists, the reward server).

Activate from a config via::

    imports = ["flow_control.contrib.iqa"]

Requires the ``iqa`` dependency group (``pyiqa``).
"""

from .reward import PyIQAMetricSpec, PyIQAScorer

__all__ = ["PyIQAMetricSpec", "PyIQAScorer"]
