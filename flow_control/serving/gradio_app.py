"""Gradio Blocks application for interactive model serving."""

from __future__ import annotations

from typing import Any

import gradio as gr
import json5
from PIL import Image

from flow_control.utils.logging import get_logger
from flow_control.utils.progress import progress_var

from .engine import ServingEngine
from .tasks import get_task_template

logger = get_logger(__name__)


def _render_checkpoint_input(engine: ServingEngine) -> gr.Dropdown | gr.Textbox:
    """Create the training checkpoint input — dropdown when root is set, textbox otherwise.

    When ``checkpoint_root`` is configured, also renders a refresh button and
    wires its click event.
    """
    if engine._checkpoint_root:
        with gr.Row():
            dropdown = gr.Dropdown(
                label="Training Checkpoint Dir",
                choices=[""] + engine.list_checkpoints(),
                value=engine._checkpoint_dir or "",
                allow_custom_value=True,
            )
            refresh_btn = gr.Button("🔄", scale=0, min_width=40)

        def on_refresh() -> gr.update:  # type: ignore[type-arg]
            return gr.update(choices=[""] + engine.list_checkpoints())

        refresh_btn.click(fn=on_refresh, inputs=[], outputs=[dropdown])
        return dropdown
    return gr.Textbox(
        label="Training Checkpoint Dir",
        placeholder="Path to DCP training checkpoint (optional)",
        value=engine._checkpoint_dir or "",
    )


def create_gradio_app(engine: ServingEngine) -> gr.Blocks:
    """Build and return the Gradio Blocks UI wired to the given engine."""
    task_name: str = engine.processor.task
    template = get_task_template(task_name)

    initial_config_jsonc = json5.dumps(
        engine.config_dump(), indent=2, ensure_ascii=False
    )

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

                with gr.Accordion("Configuration", open=False):
                    gr.Markdown(
                        "修改 `processor.task` 不受支持。"
                        "修改可能影响权重装载的模型配置后，请先点击 `Reload Models`。"
                    )
                    jsonc_box = gr.Code(
                        label="Full Config (JSONC)",
                        language="json",
                        value=initial_config_jsonc,
                    )
                    seed_ckpt_input = gr.Textbox(
                        label="Seed Checkpoint Dir",
                        placeholder="Path to DCP seed checkpoint (optional)",
                        value=engine._seed_checkpoint_dir or "",
                    )
                    train_ckpt_input = _render_checkpoint_input(engine)
                    use_ema_checkbox = gr.Checkbox(
                        label="Use EMA weights",
                        value=engine._use_ema,
                    )

                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary", size="lg")
                    reload_btn = gr.Button("Reload Models", variant="secondary")

            # ---- Right column: output ---- #
            with gr.Column(scale=1):
                output_image = gr.Image(label="Output", type="pil")
                status_text = gr.Textbox(label="Generation Info", interactive=False)

        # ---- Event handler ---- #

        async def on_generate(
            *args: Any,
            progress: gr.Progress = gr.Progress(),  # noqa: B008
        ) -> tuple[Image.Image | None, str]:
            # args = [*task_components, steps, cfg_scale, seed,
            #         seed_ckpt, train_ckpt, config_jsonc, use_ema]
            n_task = len(task_components)
            task_args = args[:n_task]
            steps = int(args[n_task])
            cfg_scale = float(args[n_task + 1])
            seed = int(args[n_task + 2])
            seed_ckpt = str(args[n_task + 3]).strip()
            train_ckpt = str(args[n_task + 4]).strip()
            config_jsonc = str(args[n_task + 5])
            use_ema = bool(args[n_task + 6])

            try:
                input_batch = template.coerce(*task_args)
            except gr.Error:
                raise
            except Exception as e:
                raise gr.Error(f"Input error: {e}") from e

            def _report(frac: float, desc: str) -> None:
                progress(frac, desc=desc)

            token = progress_var.set(_report)
            try:
                image, status = await engine.generate(
                    input_batch,
                    seed=seed,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    config_jsonc=config_jsonc,
                    seed_checkpoint_dir=seed_ckpt or None,
                    checkpoint_dir=train_ckpt or None,
                    use_ema=use_ema,
                )
                return image, status
            except Exception as e:
                logger.exception("Generation failed")
                raise gr.Error(f"Generation failed: {e}") from e
            finally:
                progress_var.reset(token)

        async def on_reload(
            seed_ckpt: str,
            train_ckpt: str,
            config_jsonc: str,
            use_ema: bool,
            progress: gr.Progress = gr.Progress(),  # noqa: B008
        ) -> str:
            def _report(frac: float, desc: str) -> None:
                progress(frac, desc=desc)

            token = progress_var.set(_report)
            try:
                return await engine.reload(
                    config_jsonc=config_jsonc,
                    seed_checkpoint_dir=seed_ckpt.strip() or None,
                    checkpoint_dir=train_ckpt.strip() or None,
                    use_ema=use_ema,
                )
            except Exception as e:
                logger.exception("Model reload failed")
                raise gr.Error(f"Model reload failed: {e}") from e
            finally:
                progress_var.reset(token)

        generate_btn.click(
            fn=on_generate,
            inputs=[
                *task_components,
                steps_slider,
                cfg_slider,
                seed_input,
                seed_ckpt_input,
                train_ckpt_input,
                jsonc_box,
                use_ema_checkbox,
            ],
            outputs=[output_image, status_text],
        )
        reload_btn.click(
            fn=on_reload,
            inputs=[
                seed_ckpt_input,
                train_ckpt_input,
                jsonc_box,
                use_ema_checkbox,
            ],
            outputs=[status_text],
        )

    return app
