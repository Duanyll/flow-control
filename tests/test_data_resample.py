"""Cross-module lock for the resample stage (design §8) and the ``cost`` field (D4).

``test_processor_resample_and_cost``: ``BaseProcessor.resample`` draws one sample
from every ``posterior_fields`` tensor cached as a ``[2, ...]`` (mean, std)
posterior -- tensors become ``[1, ...]``, TIE's ``reference_latents`` list is
sampled element-wise -- using only the caller's generator (same seed, same draw;
independent of the global RNG), leaves rows cached with
``target_posterior="mode"`` untouched, and never touches a field that is not
declared (no name guessing: an undeclared ``foo_latents`` keeps its ``[2, ...]``).
``get_cost`` is the token total ``ProcessorStage`` writes for the cache index:
latent + text for T2I, plus every reference for TIE.
"""

import unittest
from typing import Any

import torch

from flow_control.processors import parse_processor


def posterior(g: torch.Generator, n: int, d: int) -> torch.Tensor:
    return torch.cat(
        [torch.randn(1, n, d, generator=g), torch.rand(1, n, d, generator=g)]
    )


class ResampleTest(unittest.TestCase):
    def test_processor_resample_and_cost(self) -> None:
        t2i = parse_processor({"task": "t2i", "preset": "flux1"})
        tie = parse_processor({"task": "tie", "preset": "flux1"})
        g = torch.Generator().manual_seed(0)
        clean = posterior(g, 4, 8)
        mean, std = clean[0:1], clean[1:2]

        row: Any = {
            "__key__": "a",
            "image_size": (64, 64),
            "clean_latents": clean.clone(),
            "prompt_embeds": torch.zeros(1, 3, 8),
            "foo_latents": torch.zeros(2, 4, 8),
        }
        torch.manual_seed(123)
        out = t2i.resample(row, torch.Generator().manual_seed(7))
        self.assertIs(out, row, "resample modifies the row in place and returns it")
        self.assertEqual(tuple(out["clean_latents"].shape), (1, 4, 8))
        eps = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(7))
        torch.testing.assert_close(out["clean_latents"], mean + std * eps)
        self.assertEqual(
            tuple(out["foo_latents"].shape), (2, 4, 8), "undeclared field untouched"
        )

        # Determinism comes from the generator alone: the global RNG state differs
        # here, the draw does not; a different seed gives a different draw.
        torch.manual_seed(456)
        again = t2i.resample(
            {"clean_latents": clean.clone()}, torch.Generator().manual_seed(7)
        )
        torch.testing.assert_close(again["clean_latents"], out["clean_latents"])
        other = t2i.resample(
            {"clean_latents": clean.clone()}, torch.Generator().manual_seed(8)
        )
        self.assertFalse(torch.allclose(other["clean_latents"], out["clean_latents"]))

        # target_posterior="mode" caches a [1, ...] sample: nothing to do.
        sample = torch.randn(1, 4, 8, generator=g)
        mode_row: Any = {"clean_latents": sample}
        self.assertIs(
            t2i.resample(mode_row, torch.Generator().manual_seed(7))["clean_latents"],
            sample,
        )

        # TIE: list field element-wise, [1, ...] entries pass through by identity;
        # T2I does not declare reference_latents at all.
        refs = [posterior(g, 2, 8), torch.randn(1, 3, 8, generator=g)]
        tie_row: Any = {"clean_latents": clean.clone(), "reference_latents": list(refs)}
        tie.resample(tie_row, torch.Generator().manual_seed(1))
        self.assertEqual(
            [tuple(r.shape) for r in tie_row["reference_latents"]],
            [(1, 2, 8), (1, 3, 8)],
        )
        self.assertIs(tie_row["reference_latents"][1], refs[1])
        self.assertEqual(tuple(tie_row["clean_latents"].shape), (1, 4, 8))
        self.assertEqual(tie.posterior_fields, ("clean_latents", "reference_latents"))
        self.assertEqual(t2i.posterior_fields, ("clean_latents",))

        # cost: flux1 packs 16x16 pixels per token, so 64x64 -> 16 latent tokens.
        t2i_batch: Any = {"image_size": (64, 64), "prompt_embeds": torch.zeros(1, 3, 8)}
        self.assertEqual(t2i.get_cost(t2i_batch), 16 + 3)
        tie_batch: Any = {**t2i_batch, "reference_sizes": [(64, 64), (32, 32)]}
        self.assertEqual(tie.get_cost(tie_batch), 16 + 3 + 16 + 4)


if __name__ == "__main__":
    unittest.main()
