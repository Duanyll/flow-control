"""Run official prompts through the repository's processors, adapters and Executor.

Use a GPU job: uv run tests/kv_inference_smoke.py qwen21 --output demo/qwen21
The editing run also recomputes KV every step to report numeric/timing differences.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, cast

import torch

from flow_control.adapters import BaseModelAdapter
from flow_control.adapters.base import Batch
from flow_control.adapters.flux2.kv import Flux2KVAdapter
from flow_control.adapters.qwen21 import QwenImage21Adapter
from flow_control.processors import BaseProcessor, parse_processor
from flow_control.samplers import Sampler, SampleRequest
from flow_control.samplers.shift import Flux2Shift, LinearShift
from flow_control.utils.logging import console
from flow_control.utils.model_cache import cache_fields
from flow_control.utils.tensor import tensor_to_pil


def sample(
    adapter: BaseModelAdapter, sampler: Sampler, row: dict[str, Any]
) -> tuple[torch.Tensor, float, float]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    completed = list(sampler.sample(adapter, [SampleRequest(cast(Batch, row))]))[0]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    assert not cache_fields(completed.row)
    assert torch.isfinite(completed.ctx.latents).all()
    return completed.ctx.latents, seconds, torch.cuda.max_memory_allocated() / 2**30


async def run(model: str, size: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    qwen = model == "qwen21"
    adapter = QwenImage21Adapter() if qwen else Flux2KVAdapter()
    preset = "qwen21" if qwen else "flux2_klein_9b_kv"
    processor: BaseProcessor = parse_processor({"task": "tie", "preset": preset})
    if qwen:
        # Match output_resolution: it also sets the official reference area.
        processor.total_pixels = size**2
    device = torch.device("cuda")
    sampler = Sampler(
        steps=40 if qwen else 4,
        seed=42 if qwen else 0,
        shift=LinearShift(
            base_shift=0.5, max_shift=0.9, max_image_seq_len=8192, shift_terminal=0.02
        )
        if qwen
        else Flux2Shift(),
    )
    # Exact prompts and seeds from the official model cards.
    prompts = (
        [
            (
                "neon",
                'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement',
            ),
            (
                "transparent",
                "This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent.",
            ),
            ("edit", "Change the background to a sunset beach"),
        ]
        if qwen
        else [
            ("cat", "A cat holding a sign that says hello world"),
            ("edit", "A cat dressed like a wizard"),
        ]
    )
    metrics: list[dict[str, Any]] = []
    reference = None
    for name, prompt in prompts:
        console.print(f"{model}: preparing {name}")
        processor.load_models("encode", device)
        raw: Any = {
            "prompt": prompt,
            "image_size": (size, size),
            "reference_images": [reference] if name == "edit" else [],
        }
        row: Any = await processor.prepare_inference_row(raw)
        processor.encoder.unload_model()
        generator = torch.Generator(device=device).manual_seed(sampler.seed)
        processor.initialize_latents(
            row, generator=generator, device=device, dtype=torch.bfloat16
        )
        adapter.load_transformer(device)
        adapter.transformer.eval()
        console.print(f"{model}: sampling {name} ({sampler.steps} steps, KV enabled)")
        adapter.use_cache = True
        latents, seconds, peak_gib = sample(adapter, sampler, row)
        image = processor.decode_latents(latents, (size, size))
        tensor_to_pil(image).save(output / f"{name}.png")
        if name != "edit":
            # Re-read the same 8-bit pixels a subsequent user invocation receives.
            # Qwen edits the dragon sticker; Flux edits the cat.
            from flow_control.utils.tensor import pil_to_tensor

            reference = pil_to_tensor(tensor_to_pil(image)).to(device)
        result: dict[str, Any] = {
            "name": name,
            "prompt": prompt,
            "seed": sampler.seed,
            "steps": sampler.steps,
            "size": size,
            "cached_seconds": seconds,
            "peak_gib": peak_gib,
        }
        if image.shape[1] == 4:
            result["alpha_min_max"] = [
                image[:, 3].min().item(),
                image[:, 3].max().item(),
            ]
        if name == "edit":
            console.print(f"{model}: sampling edit with KV recomputed each step")
            adapter.use_cache = False
            recomputed, plain_seconds, plain_peak = sample(adapter, sampler, row)
            difference = (
                latents.float() - recomputed.float()
            ).norm() / recomputed.float().norm()
            result.update(
                uncached_seconds=plain_seconds,
                uncached_peak_gib=plain_peak,
                relative_l2=difference.item(),
            )
            tensor_to_pil(processor.decode_latents(recomputed, (size, size))).save(
                output / "edit_uncached.png"
            )
            # Qwen upstream documents visible bf16 rounding differences between
            # cached/full sequence attention. Float32 equivalence is covered by
            # test_microbatching; do not claim identical images here.
            assert torch.isfinite(recomputed).all()
        metrics.append(result)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        console.print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["qwen21", "flux2_kv"])
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with torch.inference_mode():
        asyncio.run(run(args.model, args.size, args.output))
