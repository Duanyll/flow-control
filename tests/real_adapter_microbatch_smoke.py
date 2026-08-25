import argparse
import time
from dataclasses import dataclass
from typing import cast

import torch

from flow_control.adapters.base import BaseModelAdapter, Batch
from flow_control.adapters.flux1 import Flux1Adapter
from flow_control.adapters.sd3 import SD3Adapter


@dataclass(slots=True)
class BenchmarkResult:
    adapter: str
    singleton_seconds: float
    batched_seconds: float
    peak_memory_gib: float
    max_difference: float
    relative_l2_difference: float
    batched_gradient_norm: float


def make_batch(
    adapter_name: str,
    adapter: BaseModelAdapter,
    device: torch.device,
    seed: int,
) -> Batch:
    generator = torch.Generator(device=device).manual_seed(seed)
    image_size = (256, 256)
    dtype = adapter.dtype
    batch: dict[str, object]
    if adapter_name == "flux1":
        latents = torch.randn(
            1, 256, 64, generator=generator, device=device, dtype=dtype
        )
        batch = {
            "image_size": image_size,
            "clean_latents": torch.zeros_like(latents),
            "noisy_latents": latents,
            "pooled_prompt_embeds": torch.randn(
                1, 768, generator=generator, device=device, dtype=dtype
            ),
            "prompt_embeds": torch.randn(
                1, 32, 4096, generator=generator, device=device, dtype=dtype
            ),
        }
    else:
        latents = torch.randn(
            1, 256, 64, generator=generator, device=device, dtype=dtype
        )
        batch = {
            "image_size": image_size,
            "clean_latents": torch.zeros_like(latents),
            "noisy_latents": latents,
            "pooled_prompt_embeds": torch.randn(
                1, 2048, generator=generator, device=device, dtype=dtype
            ),
            "prompt_embeds": torch.randn(
                1, 32, 4096, generator=generator, device=device, dtype=dtype
            ),
        }
    batch["prompt"] = f"ignored metadata {seed}"
    batch["__key__"] = str(seed)
    return cast(Batch, batch)


def timed_forward(
    adapter: BaseModelAdapter,
    batches: list[Batch],
    repeats: int,
) -> float:
    timesteps = [torch.tensor([0.75], device=adapter.device) for _ in batches]
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        adapter.predict_velocity_batched(batches, timesteps)
    torch.cuda.synchronize()
    return time.perf_counter() - started


def run(adapter_name: str, repeats: int) -> BenchmarkResult:
    device = torch.device("cuda", 0)
    adapter: BaseModelAdapter = (
        Flux1Adapter() if adapter_name == "flux1" else SD3Adapter()
    )
    adapter.load_transformer(device)
    adapter.transformer.eval()
    batches = [make_batch(adapter_name, adapter, device, seed) for seed in (11, 29)]
    timesteps = [torch.tensor([0.75], device=device) for _ in batches]

    with torch.inference_mode():
        singleton_outputs = [
            adapter.predict_velocity_batched([batch], [timestep])[0]
            for batch, timestep in zip(batches, timesteps, strict=True)
        ]
        batched_outputs = adapter.predict_velocity_batched(batches, timesteps)
        max_difference = max(
            (singleton - batched).abs().max().item()
            for singleton, batched in zip(
                singleton_outputs, batched_outputs, strict=True
            )
        )
        difference_norm = torch.sqrt(
            torch.stack(
                [
                    (singleton - batched).float().square().sum()
                    for singleton, batched in zip(
                        singleton_outputs, batched_outputs, strict=True
                    )
                ]
            ).sum()
        )
        reference_norm = torch.sqrt(
            torch.stack(
                [output.float().square().sum() for output in singleton_outputs]
            ).sum()
        )
        relative_l2_difference = (difference_norm / reference_norm).item()
        if not torch.isfinite(difference_norm) or relative_l2_difference > 0.05:
            raise AssertionError(
                "Dense and singleton predictions diverged: "
                f"relative L2 difference={relative_l2_difference:.6f}."
            )

        torch.cuda.reset_peak_memory_stats()
        singleton_seconds = timed_forward(adapter, batches[:1], repeats * 2)
        batched_seconds = timed_forward(adapter, batches, repeats)
        peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30

    trainable_name, trainable_parameter = next(
        reversed(list(adapter.transformer.named_parameters()))
    )
    trainable_parameter.requires_grad_(True)
    training_outputs = adapter.predict_velocity_batched(batches, timesteps)
    training_loss = torch.stack(
        [prediction.float().square().mean() for prediction in training_outputs]
    ).mean()
    training_loss.backward()
    if (
        trainable_parameter.grad is None
        or not torch.isfinite(trainable_parameter.grad).all()
    ):
        raise AssertionError(f"Invalid batched gradient for {trainable_name}.")
    batched_gradient_norm = trainable_parameter.grad.float().norm().item()
    trainable_parameter.grad = None
    trainable_parameter.requires_grad_(False)

    adapter.hf_model.unload_model()
    return BenchmarkResult(
        adapter=adapter_name,
        singleton_seconds=singleton_seconds,
        batched_seconds=batched_seconds,
        peak_memory_gib=peak_memory_gib,
        max_difference=max_difference,
        relative_l2_difference=relative_l2_difference,
        batched_gradient_norm=batched_gradient_norm,
    )


def main() -> BenchmarkResult:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", choices=("flux1", "sd3"))
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    return run(args.adapter, args.repeats)


if __name__ == "__main__":
    from rich import print

    print(main())
