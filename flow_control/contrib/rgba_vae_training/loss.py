"""Multi-objective loss for RGBA VAE training.

Ported from AlphaVAE (https://github.com/TODO/AlphaVAE), with the following
loss components:

- **Alpha-aware reconstruction loss**: L2 that accounts for alpha-premultiplied
  RGB using the Eb/Eb2 background-statistics coefficients.
- **Perceptual loss (LPIPS)**: Computed on black- and white-background composites
  of the RGBA image, then averaged.
- **KL divergence**: Standard KL vs N(0, I) or against a reference posterior.
- **PatchGAN discriminator**: Hinge-loss adversarial training with adaptive
  gradient-based weighting of the generator term.
"""

import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from flow_control.third_party.taming.discriminator import (
    NLayerDiscriminator,
    hinge_d_loss,
    weights_init,
)
from flow_control.third_party.taming.lpips import LPIPS

# Default Eb / Eb2 values from AlphaVAE — per-channel background statistics
# that balance RGB and alpha reconstruction errors.
_DEFAULT_EB = [-0.0357, -0.0811, -0.1797]
_DEFAULT_EB2 = [0.3163, 0.3060, 0.3634]


class RGBAVAELoss(nn.Module):
    """Multi-objective RGBA VAE loss module.

    Contains frozen LPIPS and a trainable PatchGAN discriminator as
    sub-modules (both optional).  The ``reconstruction_loss``,
    ``perceptual_loss``, ``kl_loss``, ``generator_loss``, and
    ``discriminator_loss`` methods are designed to be called individually so
    that the trainer can compose them with configurable scale factors.
    """

    Eb: torch.Tensor
    Eb2: torch.Tensor

    def __init__(
        self,
        *,
        use_lpips: bool = True,
        use_patchgan: bool = True,
        use_naive_mse: bool = False,
        vgg16_weights_path: str = "",
        lpips_weights_path: str = "",
        disc_in_channels: int = 4,
        disc_n_layers: int = 3,
        custom_Eb: list[float] | None = None,
        custom_Eb2: list[float] | None = None,
    ) -> None:
        super().__init__()
        eb = custom_Eb if custom_Eb is not None else _DEFAULT_EB
        eb2 = custom_Eb2 if custom_Eb2 is not None else _DEFAULT_EB2
        self.register_buffer(
            "Eb", torch.tensor(eb).reshape(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "Eb2", torch.tensor(eb2).reshape(1, 3, 1, 1), persistent=False
        )

        self.use_naive_mse = use_naive_mse
        self.use_lpips = use_lpips
        self.use_patchgan = use_patchgan

        if self.use_lpips:
            if not vgg16_weights_path or not lpips_weights_path:
                raise ValueError(
                    "vgg16_weights_path and lpips_weights_path are required when use_lpips=True. "
                    "Download from:\n"
                    "  VGG16:  https://download.pytorch.org/models/vgg16-397923af.pth\n"
                    "  LPIPS:  https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
                )
            self.lpips = LPIPS(vgg16_weights_path, lpips_weights_path).eval()

        if self.use_patchgan:
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_n_layers,
                use_actnorm=False,
            ).apply(weights_init)

    # --------------------------------------------------------------------- #
    # Reduction helper
    # --------------------------------------------------------------------- #

    @staticmethod
    def _reduce(value: torch.Tensor) -> torch.Tensor:
        """sum / batch_size (matches AlphaVAE default ``reduce_mean=False``)."""
        return torch.sum(value) / value.shape[0]

    # --------------------------------------------------------------------- #
    # Loss components
    # --------------------------------------------------------------------- #

    def reconstruction_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Alpha-aware L2 reconstruction loss.

        Both *pred* and *target* are 4-channel BCHW tensors in [-1, 1].
        """
        if self.use_naive_mse:
            return self._reduce((pred - target) ** 2)

        target_rgb = target[:, :3]
        target_a = (target[:, 3:] + 1) / 2  # [-1,1] -> [0,1]
        pred_rgb = pred[:, :3]
        pred_a = (pred[:, 3:] + 1) / 2

        rgba_diff = target_rgb * target_a - pred_rgb * pred_a
        alpha_diff = target_a - pred_a
        loss = (
            rgba_diff**2
            - 2 * self.Eb * rgba_diff * alpha_diff
            + self.Eb2 * alpha_diff**2
        )
        return self._reduce(loss)

    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """LPIPS on black- and white-background composites, averaged.

        Both *pred* and *target* are 4-channel BCHW tensors in [-1, 1].
        """
        target_rgb = target[:, :3]
        target_a = (target[:, 3:] + 1) / 2
        pred_rgb = pred[:, :3]
        pred_a = (pred[:, 3:] + 1) / 2

        # Composite over black background: rgb * alpha
        loss_black = self.lpips(target_rgb * target_a, pred_rgb * pred_a)
        # Composite over white background: rgb * alpha + (1 - alpha)
        loss_white = self.lpips(
            target_rgb * target_a + (1 - target_a),
            pred_rgb * pred_a + (1 - pred_a),
        )
        loss = (loss_black + loss_white) / 2
        return self._reduce(loss)

    def kl_loss(
        self,
        pred_posterior: DiagonalGaussianDistribution,
        ref: DiagonalGaussianDistribution | None = None,
    ) -> torch.Tensor:
        """KL divergence loss — vs N(0, I) when *ref* is ``None``."""
        loss = pred_posterior.kl(ref)  # type: ignore[arg-type]  # diffusers kl() accepts None
        return self._reduce(loss)

    # --------------------------------------------------------------------- #
    # GAN components
    # --------------------------------------------------------------------- #

    def calculate_adaptive_weight(
        self,
        nll_loss: torch.Tensor,
        g_loss: torch.Tensor,
        last_layer: torch.Tensor,
    ) -> torch.Tensor:
        """Adaptive weight that balances reconstruction vs. adversarial loss.

        Uses ``torch.autograd.grad`` with ``retain_graph=True``.
        """
        nll_grads = torch.autograd.grad(
            nll_loss.float(), last_layer, retain_graph=True
        )[0]
        g_grads = torch.autograd.grad(g_loss.float(), last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def generator_loss(
        self,
        rec_loss: torch.Tensor,
        reconstructions: torch.Tensor,
        last_layer: torch.Tensor,
    ) -> torch.Tensor:
        """Adversarial generator loss with adaptive weighting.

        The discriminator's gradients are disabled during this call.
        """
        self.discriminator.requires_grad_(False)
        logits_fake = self.discriminator(reconstructions)
        g_loss = -torch.mean(logits_fake).float()
        d_weight = self.calculate_adaptive_weight(
            rec_loss, g_loss, last_layer=last_layer
        )
        return (g_loss * d_weight).float()

    def discriminator_loss(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
    ) -> torch.Tensor:
        """Hinge discriminator loss on detached real / fake images."""
        self.discriminator.requires_grad_(True)
        logits_real = self.discriminator(inputs.contiguous().detach())
        logits_fake = self.discriminator(reconstructions.contiguous().detach())
        return hinge_d_loss(logits_real, logits_fake).float()
