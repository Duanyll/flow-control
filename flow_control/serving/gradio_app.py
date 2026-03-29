"""Gradio Blocks application for interactive model serving."""

from __future__ import annotations

from typing import Any

import gradio as gr
import json5
from PIL import Image

from flow_control.utils.logging import get_logger

from .engine import ServingEngine
from .tasks import get_task_template

logger = get_logger(__name__)


def create_gradio_app(engine: ServingEngine) -> gr.Blocks:
    """Build and return the Gradio Blocks UI wired to the given engine."""
    task_name: str = engine.processor.task
    template = get_task_template(task_name)

    with gr.Blocks(title=f"flow-control: {task_name}") as app:
        gr.Markdown(f"# flow-control Serving &mdash; `{task_name}`")

        with gr.Row():
            # ---- Left column: inputs ---- #
            with gr.Column(scale=1):
                task_components = template.render()

                with gr.Accordion("Sampler Settings", open=True):
                    steps_slider = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=engine.sampler.steps,
                    )
                    cfg_slider = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.1,
                        value=engine.sampler.cfg_scale,
                    )
                    seed_input = gr.Number(
                        label="Seed",
                        value=engine.sampler.seed,
                        precision=0,
                    )

                with gr.Accordion("Checkpoints", open=False):
                    seed_ckpt_input = gr.Textbox(
                        label="Seed Checkpoint Dir",
                        placeholder="Path to DCP seed checkpoint (optional)",
                        value=engine._seed_checkpoint_dir or "",
                    )
                    train_ckpt_input = gr.Textbox(
                        label="Training Checkpoint Dir",
                        placeholder="Path to DCP training checkpoint (optional)",
                        value=engine._checkpoint_dir or "",
                    )
                    reload_btn = gr.Button("Reload Model")
                    reload_status = gr.Textbox(
                        label="Status", interactive=False, max_lines=3
                    )

                with gr.Accordion("Config Editor (JSONC)", open=False):
                    jsonc_box = gr.Code(
                        label="Full Config",
                        language="json",
                        value=json5.dumps(
                            engine.config_dump(), indent=2, ensure_ascii=False
                        ),
                    )
                    apply_btn = gr.Button("Apply Config")
                    override_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        max_lines=5,
                    )

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            # ---- Right column: output ---- #
            with gr.Column(scale=1):
                output_image = gr.Image(label="Output", type="pil")
                status_text = gr.Textbox(label="Generation Info", interactive=False)

        # ---- Event handlers ---- #

        async def on_generate(
            *args: Any,
        ) -> tuple[Image.Image | None, str]:
            # args = [*task_components, steps, cfg_scale, seed,
            #         seed_ckpt, train_ckpt]
            n_task = len(task_components)
            task_args = args[:n_task]
            steps = int(args[n_task])
            cfg_scale = float(args[n_task + 1])
            seed = int(args[n_task + 2])
            seed_ckpt = str(args[n_task + 3]).strip()
            train_ckpt = str(args[n_task + 4]).strip()

            try:
                input_batch = template.coerce(*task_args)
            except gr.Error:
                raise
            except Exception as e:
                raise gr.Error(f"Input error: {e}") from e

            # Auto-reload transformer if checkpoint paths changed
            engine.update_checkpoints(seed_ckpt or None, train_ckpt or None)

            # Apply sampler overrides for this run
            engine.sampler.steps = steps
            engine.sampler.cfg_scale = cfg_scale
            engine.sampler.seed = seed

            try:
                image, status = await engine.generate(input_batch, seed=seed)
                return image, status
            except Exception as e:
                logger.exception("Generation failed")
                raise gr.Error(f"Generation failed: {e}") from e

        generate_btn.click(
            fn=on_generate,
            inputs=[
                *task_components,
                steps_slider,
                cfg_slider,
                seed_input,
                seed_ckpt_input,
                train_ckpt_input,
            ],
            outputs=[output_image, status_text],
        )

        async def on_reload(seed_ckpt: str, train_ckpt: str) -> str:
            try:
                messages = engine.update_checkpoints(
                    seed_ckpt.strip() or None,
                    train_ckpt.strip() or None,
                )
                result = "\n".join(messages)
                logger.info(f"Checkpoint reload: {result}")
                return result
            except Exception as e:
                logger.exception("Failed to reload checkpoints")
                raise gr.Error(f"Reload failed: {e}") from e

        reload_btn.click(
            fn=on_reload,
            inputs=[seed_ckpt_input, train_ckpt_input],
            outputs=[reload_status],
        )

        async def on_apply_config(jsonc_text: str) -> tuple[str, str]:
            try:
                messages = engine.apply_config(jsonc_text)
                result = "\n".join(messages)
                logger.info(f"Config applied: {result}")
                new_dump = json5.dumps(
                    engine.config_dump(), indent=2, ensure_ascii=False
                )
                return new_dump, result
            except Exception as e:
                logger.exception("Failed to apply config")
                raise gr.Error(f"Failed to apply config: {e}") from e

        apply_btn.click(
            fn=on_apply_config,
            inputs=[jsonc_box],
            outputs=[jsonc_box, override_status],
        )

    return app
