"""Gradio Blocks application for interactive model serving."""

from __future__ import annotations

from typing import Any

import gradio as gr
import json5
import torch
from PIL import Image

from flow_control.utils.logging import get_logger
from flow_control.utils.progress import progress_var
from flow_control.utils.tensor import (
    BlendBackground,
    pil_to_tensor,
    remove_alpha_channel,
    tensor_to_pil,
)

from .engine import ServingEngine
from .tasks import get_task_template

logger = get_logger(__name__)

_BG_CHOICES = ["Raw RGBA", "Checkerboard", "White", "Black", "Auto"]


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _apply_background(image: Image.Image, bg: str) -> Image.Image:
    """Composite an RGBA image over the chosen background."""
    if image.mode != "RGBA" or bg == "Raw RGBA":
        return image
    tensor = pil_to_tensor(image)
    blend: BlendBackground
    if bg == "White":
        blend = (1.0, 1.0, 1.0)
    elif bg == "Black":
        blend = (0.0, 0.0, 0.0)
    elif bg == "Auto":
        blend = "auto"
    else:
        blend = "checkerboard"
    return tensor_to_pil(remove_alpha_channel(tensor, blend))


def _extract_images(result: dict[str, Any]) -> list[tuple[str, torch.Tensor]]:
    """Extract ``(label, tensor)`` pairs from a decoded-batch dict."""
    images: list[tuple[str, torch.Tensor]] = []
    if "clean_image" in result:
        images.append(("Output", result["clean_image"]))
    if "base_image" in result:
        images.append(("Base", result["base_image"]))
    for i, layer in enumerate(result.get("layer_images", [])):
        images.append((f"Layer {i}", layer))
    return images


def _build_gallery(
    raw_images: list[tuple[str, Image.Image]], bg: str
) -> list[tuple[Image.Image, str]]:
    """Build a Gradio gallery list, applying background compositing."""
    return [(_apply_background(img, bg), label) for label, img in raw_images]


def _format_info(info: dict[str, Any]) -> str:
    """Serialize info dict to pretty JSON for display."""
    return json5.dumps(info, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
#  Checkpoint input                                                            #
# --------------------------------------------------------------------------- #


def _render_checkpoint_input(engine: ServingEngine) -> gr.Dropdown | gr.Textbox:
    """Create the training checkpoint input — dropdown when root is set, textbox otherwise."""
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


# --------------------------------------------------------------------------- #
#  Static event handlers (extracted for C901)                                  #
# --------------------------------------------------------------------------- #


def _wire_static_events(
    engine: ServingEngine,
    *,
    bg_dropdown: gr.Dropdown,
    raw_images_state: gr.State,
    output_gallery: gr.Gallery,
    reload_btn: gr.Button,
    seed_ckpt_input: gr.Textbox,
    train_ckpt_input: gr.Dropdown | gr.Textbox,
    jsonc_box: gr.Code,
    use_ema_checkbox: gr.Checkbox,
    info_box: gr.Code,
    task_state: gr.State,
) -> None:
    """Wire event handlers that live outside ``@gr.render``."""

    def on_bg_change(
        bg_mode: str,
        raw_images: list[tuple[str, Image.Image]] | None,
    ) -> list[tuple[Image.Image, str]] | None:
        if not raw_images:
            return None
        return _build_gallery(raw_images, bg_mode)

    bg_dropdown.change(
        fn=on_bg_change,
        inputs=[bg_dropdown, raw_images_state],
        outputs=[output_gallery],
    )

    async def on_reload(
        seed_ckpt: str,
        train_ckpt: str,
        config_jsonc: str,
        use_ema: bool,
        progress: gr.Progress = gr.Progress(),  # noqa: B008
    ) -> tuple[str, str]:
        def _report(frac: float, desc: str) -> None:
            progress(frac, desc=desc)

        token = progress_var.set(_report)
        try:
            result = await engine.reload(
                config_jsonc=config_jsonc,
                seed_checkpoint_dir=seed_ckpt.strip() or None,
                checkpoint_dir=train_ckpt.strip() or None,
                use_ema=use_ema,
            )
            return result, engine.processor.task
        except Exception as e:
            logger.exception("Model reload failed")
            raise gr.Error(f"Model reload failed: {e}") from e
        finally:
            progress_var.reset(token)

    reload_btn.click(
        fn=on_reload,
        inputs=[seed_ckpt_input, train_ckpt_input, jsonc_box, use_ema_checkbox],
        outputs=[info_box, task_state],
    )


# --------------------------------------------------------------------------- #
#  App builder                                                                 #
# --------------------------------------------------------------------------- #


def create_gradio_app(engine: ServingEngine) -> gr.Blocks:
    """Build and return the Gradio Blocks UI wired to the given engine."""

    initial_config_jsonc = json5.dumps(
        engine.config_dump(), indent=2, ensure_ascii=False
    )

    with gr.Blocks(title="flow-control Serving") as app:
        gr.Markdown("# flow-control Serving")
        task_state = gr.State(engine.processor.task)
        raw_images_state: gr.State = gr.State(None)

        with gr.Row():
            # ---- Left column: inputs ---- #
            with gr.Column(scale=1):
                # Dynamic task components — re-rendered when task_state changes
                @gr.render(inputs=[task_state])
                def render_task_inputs(  # noqa: ANN202
                    current_task: str,
                ) -> None:
                    template = get_task_template(current_task)
                    task_components = template.render()

                    async def on_generate(
                        *args: Any,
                        progress: gr.Progress = gr.Progress(),  # noqa: B008
                    ) -> tuple[
                        list[tuple[Image.Image, str]],
                        list[tuple[str, Image.Image]],
                        gr.update,  # type: ignore[type-arg]
                        str,
                    ]:
                        n_task = len(task_components)
                        task_args = args[:n_task]
                        steps = int(args[n_task])
                        cfg_scale = float(args[n_task + 1])
                        seed = int(args[n_task + 2])
                        seed_ckpt = str(args[n_task + 3]).strip()
                        train_ckpt = str(args[n_task + 4]).strip()
                        config_jsonc = str(args[n_task + 5])
                        use_ema = bool(args[n_task + 6])
                        bg_mode = str(args[n_task + 7])

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
                            result, info = await engine.generate(
                                input_batch,
                                seed=seed,
                                steps=steps,
                                cfg_scale=cfg_scale,
                                config_jsonc=config_jsonc,
                                seed_checkpoint_dir=seed_ckpt or None,
                                checkpoint_dir=train_ckpt or None,
                                use_ema=use_ema,
                            )
                        except Exception as e:
                            logger.exception("Generation failed")
                            raise gr.Error(f"Generation failed: {e}") from e
                        finally:
                            progress_var.reset(token)

                        # Extract images and build gallery
                        raw_images = [
                            (label, tensor_to_pil(t))
                            for label, t in _extract_images(result)
                        ]
                        has_alpha = any(img.mode == "RGBA" for _, img in raw_images)
                        gallery = _build_gallery(raw_images, bg_mode)
                        return (
                            gallery,
                            raw_images,
                            gr.update(visible=has_alpha),
                            _format_info(info),
                        )

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
                            bg_dropdown,
                        ],
                        outputs=[
                            output_gallery,
                            raw_images_state,
                            bg_dropdown,
                            info_box,
                        ],
                    )

                # ---- Static settings ---- #

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
                        "You may need to click `Reload Models` after editing"
                        " the config below to apply changes."
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
                output_gallery = gr.Gallery(label="Output", columns=2, height=512)
                bg_dropdown = gr.Dropdown(
                    label="Background",
                    choices=_BG_CHOICES,
                    value="Checkerboard",
                    visible=False,
                )
                info_box = gr.Code(
                    label="Generation Info",
                    language="json",
                    interactive=False,
                )

        # ---- Static event handlers ---- #
        _wire_static_events(
            engine,
            bg_dropdown=bg_dropdown,
            raw_images_state=raw_images_state,
            output_gallery=output_gallery,
            reload_btn=reload_btn,
            seed_ckpt_input=seed_ckpt_input,
            train_ckpt_input=train_ckpt_input,
            jsonc_box=jsonc_box,
            use_ema_checkbox=use_ema_checkbox,
            info_box=info_box,
            task_state=task_state,
        )

    return app
