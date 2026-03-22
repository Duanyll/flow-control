# Vendored from taming-transformers
# Source: https://github.com/CompVis/taming-transformers
# File: taming/modules/losses/lpips.py
# Original: https://github.com/richzhang/PerceptualSimilarity/tree/master/models
# Modifications:
#   - Require explicit weight file paths instead of auto-downloading.
#   - Load VGG16 backbone from a local checkpoint instead of torchvision hub.
#
# Required weight files (download manually before use):
#   1. VGG16 ImageNet weights:
#      URL:  https://download.pytorch.org/models/vgg16-397923af.pth
#      Default path: data/vgg16-397923af.pth
#   2. LPIPS learned linear weights:
#      URL:  https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1
#      Default path: data/vgg_lpips.pth

import logging
from collections import namedtuple

import torch
import torch.nn as nn
from torchvision.models import vgg16 as _torchvision_vgg16

logger = logging.getLogger(__name__)


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity metric using VGG16 features.

    Args:
        vgg16_weights_path: Path to VGG16 ImageNet checkpoint
            (``vgg16-397923af.pth``).
        lpips_weights_path: Path to LPIPS learned linear weights
            (``vgg_lpips.pth``).
        use_dropout: Whether to use dropout in learned linear layers.
    """

    def __init__(
        self,
        vgg16_weights_path: str,
        lpips_weights_path: str,
        use_dropout: bool = True,
    ):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vgg16 feature channels
        self.net = VGG16Features(vgg16_weights_path, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self._load_lpips_weights(lpips_weights_path)
        for param in self.parameters():
            param.requires_grad = False

    def _load_lpips_weights(self, path: str) -> None:
        self.load_state_dict(
            torch.load(path, map_location=torch.device("cpu"), weights_only=True),
            strict=False,
        )
        logger.info("Loaded pretrained LPIPS weights from %s", path)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        in0_input, in1_input = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = (
                normalize_tensor(outs0[kk]),
                normalize_tensor(outs1[kk]),
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    """Normalize input with ImageNet mean/std."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv."""

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False):
        super().__init__()
        layers: list[nn.Module] = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class VGG16Features(nn.Module):
    """VGG16 feature extractor sliced into 5 stages.

    Loads weights from a local checkpoint file instead of using
    ``torchvision.models.vgg16(weights=...)``, which internally calls
    ``torch.hub.load_state_dict_from_url`` (unsafe under multiprocessing).

    Args:
        weights_path: Path to ``vgg16-397923af.pth``.
        requires_grad: Whether VGG parameters require gradients.
    """

    def __init__(self, weights_path: str, requires_grad: bool = False):
        super().__init__()
        # Build VGG16 without pretrained weights, then load from file
        backbone = _torchvision_vgg16(weights=None)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state_dict)
        logger.info("Loaded VGG16 backbone weights from %s", weights_path)

        vgg_features = backbone.features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X: torch.Tensor):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)
