"""Consumer-side data contract: how the trainers treat plan-time padding rows.

Locks the cross-module behaviour between ``flow_control.data`` (``__padding__``
marker set by ``RowStream``) and ``SftTrainer.train_step``: a padding row must
still be forwarded (its rank's FSDP collectives have to line up with the other
ranks) but must weigh nothing, with the microbatch loss normalized by the real
rows. Before the data rework, padding rows were detected by the
``__key__ == "__padding__"`` sentinel and SFT averaged over every row, so a
padded tail microbatch double-counted the duplicated samples.
"""

import unittest
from typing import Any
from unittest.mock import patch

import torch
from pydantic import BaseModel, Field
from test_microbatching import FakeSamplerModel

from flow_control.data import PADDING
from flow_control.processors import parse_processor
from flow_control.samplers import Sampler
from flow_control.training.sft import SftTrainer
from flow_control.training.weighting import LogitNormalTimestepWeighting


class _ProbeOverrides(BaseModel):
    """Defaults for the heavy required trainer fields (config-only tests)."""

    model: Any = None
    processor: Any = None
    dataset: Any = None
    launch: Any = None
    checkpoint_root: str = ""
    experiment_name: str = "probe"
    seed_checkpoint_dir: str = ""
    validation_sampler: Sampler = Field(default_factory=Sampler)

    @property
    def device(self):
        return torch.device("cpu")


class _SftProbe(_ProbeOverrides, SftTrainer):
    pass


def _row(value: float, *, padding: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {
        "__key__": "row",
        "image_size": (2, 2),
        "clean_latents": torch.full((1, 1, 1), value),
        "noisy_latents": torch.zeros(1, 1, 1),
    }
    if padding:
        row[PADDING] = True
    return row


class SftPaddingRowsTest(unittest.TestCase):
    def test_padding_rows_are_forwarded_with_zero_loss_weight(self) -> None:
        trainer = _SftProbe.model_validate({"train_predictor": "model"})
        trainer.processor = parse_processor({"task": "t2i", "preset": "flux1"})
        model = FakeSamplerModel()
        trainer.model = model

        def loss(rows: list[dict[str, Any]]) -> torch.Tensor:
            # Fixed timestep and noise: the loss depends on the rows alone.
            with (
                patch.object(
                    LogitNormalTimestepWeighting,
                    "sample_timesteps",
                    return_value=torch.tensor([0.5]),
                ),
                patch("torch.randn_like", return_value=torch.ones(1, 1, 1)),
            ):
                return trainer.train_step([dict(row) for row in rows])

        real_only = loss([_row(3.0)])
        with_padding = loss([_row(3.0), _row(-7.0, padding=True)])
        # Padding is real work for the model (collectives), not for the loss.
        self.assertEqual(model.forward_batch_sizes, [1, 2])
        torch.testing.assert_close(with_padding, real_only)
        self.assertGreater(real_only.item(), 0.0)

        # Two real rows average; a padding row in the same microbatch changes nothing.
        two_real = loss([_row(3.0), _row(5.0)])
        two_real_padded = loss([_row(3.0), _row(5.0), _row(9.0, padding=True)])
        torch.testing.assert_close(two_real_padded, two_real)
        torch.testing.assert_close(
            two_real, (loss([_row(3.0)]) + loss([_row(5.0)])) / 2
        )

        # An all-padding microbatch (a rank whose whole tail is padding) is 0.
        torch.testing.assert_close(loss([_row(4.0, padding=True)]), torch.tensor(0.0))


if __name__ == "__main__":
    unittest.main()
