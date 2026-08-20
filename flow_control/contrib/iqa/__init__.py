"""pyiqa-backed image quality assessment rewards.

Wraps `pyiqa <https://github.com/chaofengc/IQA-PyTorch>`_ metrics (PSNR, SSIM,
LPIPS, DISTS, NIQE, MUSIQ, ...) as a single multi-component reward, so IQA
batteries can be used anywhere the reward framework is accepted (inference
eval, composite rewards, remote offload).

Activate from a config via::

    imports = ["flow_control.contrib.iqa"]

Requires the ``iqa`` dependency group (``pyiqa``).
"""

from .reward import PyIQAMetricSpec, PyIQAReward

__all__ = ["PyIQAMetricSpec", "PyIQAReward"]
