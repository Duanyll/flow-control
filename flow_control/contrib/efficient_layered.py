"""Experimental ``efficient_layered`` components (adapter + processor task).

Relocated out of core into ``flow_control.contrib`` so the experimental
``efficient_layered`` tag never appears in the core model-adapter / processor-task
unions unless a config opts in via ``imports``. Importing this module self-registers:

- ``EfficientLayeredQwenImageAdapter`` under ``adapter_registry`` tag
  ``"qwen_efficient_layered"`` (the ``f"{arch}_{type}"`` discriminator for
  ``arch="qwen"``, ``type="efficient_layered"``).
- ``EfficientLayeredProcessor`` under ``task_registry`` tag ``"efficient_layered"``.

Activate with, e.g.::

    imports = ["flow_control.contrib.efficient_layered"]
"""

import asyncio
import math
from typing import Annotated, Any, Literal, NotRequired, cast

import torch
import torch.nn as nn
import torch.nn.attention.flex_attention as flex_attention
from einops import rearrange, repeat

from flow_control.adapters.base import BaseModelAdapter, adapter_registry
from flow_control.adapters.qwen.base import QwenImageAdapter, QwenImageBatch
from flow_control.datasets.coercion import (
    ImageTensor,
    ImageTensorList,
    JsonBeforeValidator,
    JsonStrList,
)
from flow_control.processors import task_registry
from flow_control.processors.base import (
    BaseProcessor,
    DecodedBatch,
    InputBatch,
    ProcessedBatch,
    TrainInputBatch,
)
from flow_control.processors.components.llm import parse_llm_json_output
from flow_control.processors.components.prompts import PromptStr, parse_prompt
from flow_control.utils.draw import draw_bbox_on_image
from flow_control.utils.logging import get_logger, warn_once
from flow_control.utils.merge_images import merge_images
from flow_control.utils.resize import (
    resize_to_closest_resolution,
    resize_to_multiple_of,
    resize_to_resolution,
)
from flow_control.utils.tensor import (
    ensure_alpha_channel,
    ensure_compiled_flex_attention,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Model adapter
# --------------------------------------------------------------------------- #


class EfficientLayeredQwenEmbedRope(nn.Module):
    inv_freq_t: torch.Tensor
    inv_freq_h: torch.Tensor
    inv_freq_w: torch.Tensor

    def __init__(
        self,
        theta: int,
        axes_dim: list[int],
        scale_rope: bool = False,
        base_image_index: int = 0,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope
        self.base_image_index = base_image_index

        inv_freqs = []
        for dim in axes_dim:
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            inv_freqs.append(inv_freq)
        self.register_buffer("inv_freq_t", inv_freqs[0])
        self.register_buffer("inv_freq_h", inv_freqs[1])
        self.register_buffer("inv_freq_w", inv_freqs[2])

    def _cal_freqs(self, indices: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
        freqs = torch.outer(indices.to(inv_freq.dtype), inv_freq)
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        video_fhw: Any,
        txt_seq_lens: list[int] | None = None,
        device: torch.device | None = None,
        max_txt_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param self: Description
        :param video_fhw: Originally list[list[tuple[int, int, int]]]. QwenTransformer2DModel directly passes its
            `img_shapes` argument here, so we utilize it to carry additional `txt_seq_lens` information. In `video_fhw`,
            a three element tuple `(frame, height, width)` indicates the full frame, while a five element tuple
            `(frame, top, bottom, left, right)` indicates a cropped frame. Since the first image is always the base image,
            we can infer `all_height` and `all_width` from it to center the cropped frames if `scale_rope` is True.
        :type video_fhw: Any
        :param txt_seq_lens: Unfortunately, this parameter is marked as deprecated in newer QwenTransformer2DModel,
            and thus will always be None. We have to squeeze this information from `video_fhw` instead.
        :type txt_seq_lens: list[int] | None
        :param device: Not quite useful for the EfficientLayered patch, kept for API compatibility.
        :type device: torch.device | None
        :param max_txt_seq_len: Not quite useful for the EfficientLayered patch, kept for API compatibility.
        :type max_txt_seq_len: int | None
        :return: Description
        :rtype: tuple[Tensor, Tensor]
        """
        if isinstance(video_fhw, tuple):
            video_fhw, txt_seq_lens = video_fhw
        if txt_seq_lens is None:
            txt_seq_lens = [max_txt_seq_len]  # type: ignore

        fhws = video_fhw[0]

        if device is None:
            device = self.inv_freq_t.device

        vid_freqs_list = []
        max_vid_index = 0

        _, all_height, all_width = fhws[0]

        for idx, spec in enumerate(fhws):
            if len(spec) == 3:
                frame, height, width = spec
                t_idx = torch.full(
                    (frame,), float(self.base_image_index), device=device
                )

                if self.scale_rope:
                    h_idx = torch.arange(
                        -(height - height // 2), height // 2, device=device
                    )
                    w_idx = torch.arange(
                        -(width - width // 2), width // 2, device=device
                    )
                    max_vid_index = max(max_vid_index, height // 2, width // 2)
                else:
                    h_idx = torch.arange(height, device=device)
                    w_idx = torch.arange(width, device=device)
                    max_vid_index = max(max_vid_index, height, width)
            else:
                frame, top, bottom, left, right = spec
                if self.scale_rope:
                    top -= all_height // 2
                    bottom -= all_height // 2
                    left -= all_width // 2
                    right -= all_width // 2
                t_idx = torch.arange(frame, device=device) + idx
                h_idx = torch.arange(top, bottom, device=device)
                w_idx = torch.arange(left, right, device=device)
                max_vid_index = max(
                    max_vid_index, abs(top), abs(bottom), abs(left), abs(right)
                )
                height = bottom - top
                width = right - left

            freq_t = self._cal_freqs(t_idx, self.inv_freq_t)
            freq_h = self._cal_freqs(h_idx, self.inv_freq_h)
            freq_w = self._cal_freqs(w_idx, self.inv_freq_w)

            video_freq_3d = torch.cat(
                [
                    repeat(freq_t, "f d -> f h w d", h=height, w=width),
                    repeat(freq_h, "h d -> f h w d", f=frame, w=width),
                    repeat(freq_w, "w d -> f h w d", f=frame, h=height),
                ],
                dim=-1,
            )

            vid_freqs_list.append(rearrange(video_freq_3d, "f h w d -> (f h w) d"))

        vid_freqs = torch.cat(vid_freqs_list, dim=0)

        txt_idx = torch.cat(
            [
                torch.arange(max_vid_index, max_vid_index + length, device=device)
                for length in txt_seq_lens  # type: ignore
            ]
        )
        txt_freqs = torch.cat(
            [
                self._cal_freqs(txt_idx, self.inv_freq_t),
                self._cal_freqs(txt_idx, self.inv_freq_h),
                self._cal_freqs(txt_idx, self.inv_freq_w),
            ],
            dim=1,
        )

        return vid_freqs, txt_freqs


class EfficientLayeredQwenImageBatch(QwenImageBatch):
    image_latents: torch.Tensor
    """
        `[B, N, D]` Tensor representing input image latents.
        """
    layer_boxes: list[tuple[int, int, int, int]]
    """
        `(top, bottom, left, right)` in pixels for each layer in the image. Should be aligned with
        multiples of 16. `top` and `left` are inclusive, `bottom` and `right` are exclusive.
        """
    text_lengths: list[int]
    """
        Lengths of prompts corresponding to each layer in the image.
        """
    prompt_embeds: torch.Tensor
    """
        `[B, N, D]` Multimodal embeddings per layer from Qwen2.5-VL-7B, already concatenated
        along the sequence dimension.
        """
    noisy_latents: torch.Tensor
    """
        `[B, N, D]` The noisy latents to denoise, each layer is already concatenated along N dimension.
        """
    block_mask: NotRequired[flex_attention.BlockMask | None]


@adapter_registry.register("qwen_efficient_layered")
class EfficientLayeredQwenImageAdapter(
    QwenImageAdapter[EfficientLayeredQwenImageBatch]
):
    type: Literal["efficient_layered"] = "efficient_layered"

    attn_mask_mode: Literal["full", "text-only", "per-layer", "per-layer-strict"] = (
        "text-only"
    )
    """
    Attention mask mode for the adapter. Options are:
    - "full": No attention masking, full attention.
    - "text-only": All layers can attend to each other, and the base image, but text prompts
      can only attend to their corresponding layers.
    - "per-layer": Each layer attends only to themselves, their corresponding text prompt,
      and the base image. They do not attend to other layers.
    - "per-layer-strict": Similar to "per-layer", but the base image does not attend to
      any layers, ensuring the decoupling of layers from the base image.
    """
    attn_block_size: int = 128
    base_image_index: Literal[0, -1] = 0

    def load_transformer(self, device: torch.device) -> None:
        BaseModelAdapter.load_transformer(self, device=device)
        # Must use flex attention backend for block masks
        ensure_compiled_flex_attention()
        self.transformer.set_attention_backend("flex")
        orig_module = self.transformer.pos_embed
        self.transformer.pos_embed = EfficientLayeredQwenEmbedRope(  # type: ignore
            theta=orig_module.theta,
            axes_dim=orig_module.axes_dim,
            scale_rope=orig_module.scale_rope,
        )

    def make_block_mask(
        self, base_len: int, layer_lens: list[int], txt_lens: list[int]
    ) -> flex_attention.BlockMask | None:
        if self.attn_mask_mode == "full":
            return None
        elif self.attn_mask_mode in ["text-only", "per-layer", "per-layer-strict"]:
            # 1. 计算总长度
            txt_total_len = sum(txt_lens)
            total_len = txt_total_len + base_len + sum(layer_lens)

            # 2. 初始化 ID Tensor
            layer_ids = torch.zeros(total_len, dtype=torch.long, device=self.device)

            # 3. 定义各部分的起始位置
            # 现在的顺序是: Text -> Base Image -> Layer Images
            # Base Image 在 Text 之后
            base_img_begin = txt_total_len
            # Layer Images 在 Base Image 之后
            layer_img_begin = base_img_begin + base_len

            # 4. 填充 layer_ids
            # 指针初始化
            current_loc_txt = 0  # Text 从 0 开始
            current_loc_layer = layer_img_begin  # Layer 从 Base 之后开始

            # Base Image 部分默认为 0，不需要显式填充，保持为 0 即可

            for i, (layer_size, txt_size) in enumerate(
                zip(layer_lens, txt_lens, strict=True)
            ):
                # 填充 Text ID (i+1)
                layer_ids[current_loc_txt : current_loc_txt + txt_size] = i + 1
                current_loc_txt += txt_size

                # 填充 Layer Image ID (i+1)
                layer_ids[current_loc_layer : current_loc_layer + layer_size] = i + 1
                current_loc_layer += layer_size

            # 定义 Mask 函数
            # 注意：现在的 Image 部分是从 base_img_begin 开始直到结束

            def text_only_mask_fn(b, h, q_idx, kv_idx):
                return (
                    (layer_ids[q_idx] == layer_ids[kv_idx])
                    # 原逻辑是 "Both in Image"，现在 Image 是指 index >= base_img_begin
                    | ((q_idx >= base_img_begin) & (kv_idx >= base_img_begin))
                    # Allow attention to base image (id 0)
                    # | (layer_ids[kv_idx] == 0)
                )

            def per_layer_mask_fn(b, h, q_idx, kv_idx):
                return (
                    (layer_ids[q_idx] == layer_ids[kv_idx])
                    | (layer_ids[q_idx] == 0)
                    | (layer_ids[kv_idx] == 0)
                )

            def per_layer_strict_mask_fn(b, h, q_idx, kv_idx):
                return (layer_ids[q_idx] == layer_ids[kv_idx]) | (
                    layer_ids[kv_idx] == 0
                )

            mask_fn_dict = {
                "text-only": text_only_mask_fn,
                "per-layer": per_layer_mask_fn,
                "per-layer-strict": per_layer_strict_mask_fn,
            }

            mask_fn = mask_fn_dict[self.attn_mask_mode]

            block_mask = flex_attention.create_block_mask(
                mask_fn,
                1,
                1,
                total_len,
                total_len,
                BLOCK_SIZE=self.attn_block_size,
                device=self.device,
            )
            return block_mask
        else:
            raise ValueError(f"Unknown attn_mask_mode: {self.attn_mask_mode}")

    def _predict_velocity(self, batch: EfficientLayeredQwenImageBatch, timestep):
        b, n, d = batch["noisy_latents"].shape
        h, w = batch["image_size"]

        input_latents = torch.cat(
            [batch["image_latents"], batch["noisy_latents"]], dim=1
        )

        img_shapes = [
            [(1, h // 16, w // 16)]
            + [
                (1, top // 16, bottom // 16, left // 16, right // 16)
                for (top, bottom, left, right) in batch["layer_boxes"]
            ]
        ] * b

        if "block_mask" not in batch:
            batch["block_mask"] = self.make_block_mask(
                base_len=batch["image_latents"].shape[1],
                layer_lens=[
                    (bottom - top) * (right - left) // 256
                    for (top, bottom, left, right) in batch["layer_boxes"]
                ],
                txt_lens=batch["text_lengths"],
            )

        is_rgb = torch.tensor([0] * b).to(device=self.device, dtype=torch.long)

        model_pred = self.transformer(
            hidden_states=input_latents,
            timestep=timestep,
            encoder_hidden_states=batch["prompt_embeds"],
            img_shapes=(img_shapes, batch["text_lengths"]),
            attention_kwargs={
                "attention_mask": batch["block_mask"],
            },
            return_dict=False,
            additional_t_cond=is_rgb,
        )[0]

        return model_pred[:, batch["image_latents"].shape[1] :, :]


# --------------------------------------------------------------------------- #
# Processor task
# --------------------------------------------------------------------------- #


class EfficientLayeredInputBatch(InputBatch):
    clean_image: ImageTensor
    layer_boxes: NotRequired[
        Annotated[list[tuple[int, int, int, int]], JsonBeforeValidator] | None
    ]
    layer_prompts: NotRequired[JsonStrList | None]

    annotated_image: NotRequired[ImageTensor]


class EfficientLayeredTrainInputBatch(TrainInputBatch):
    clean_image: ImageTensor
    layer_boxes: Annotated[list[tuple[int, int, int, int]], JsonBeforeValidator]
    layer_images: ImageTensorList
    layer_prompts: NotRequired[JsonStrList | None]


class EfficientLayeredProcessedBatch(ProcessedBatch):
    prompt_embeds: torch.Tensor
    layer_boxes: list[tuple[int, int, int, int]]
    image_latents: torch.Tensor
    text_lengths: list[int]


class EfficientLayeredDecodedBatch(DecodedBatch):
    layer_images: list[torch.Tensor]


@task_registry.register("efficient_layered")
class EfficientLayeredProcessor(
    BaseProcessor[
        EfficientLayeredInputBatch,
        EfficientLayeredTrainInputBatch,
        EfficientLayeredProcessedBatch,
    ]
):
    task: Literal["efficient_layered"] = "efficient_layered"
    encoder_prompt: PromptStr = ""
    fg_caption_prompt: PromptStr = parse_prompt("@efficient_layered_caption_fg_en")
    bg_caption_prompt: PromptStr = parse_prompt("@efficient_layered_caption_bg_en")
    default_negative_prompt: str = " "
    save_negative: bool = False

    detection_prompt: PromptStr = parse_prompt("@efficient_layered_detection_en")
    detection_coord_type: Literal["qwen25vl", "qwen3vl"] = "qwen3vl"
    save_annotated_image: bool = False

    def resize_image(self, image: torch.Tensor) -> torch.Tensor:
        # Cropping is disabled to make resizing layer box calculation easier
        if self.resize_mode == "list":
            return resize_to_closest_resolution(
                image,
                self.preferred_resolutions,
                crop=False,
            )
        else:
            return resize_to_multiple_of(
                image, self.multiple_of, crop=False, pixels=self.total_pixels
            )

    def _scale_and_align_layer_boxes(self, layer_boxes, orig_size, new_size):
        orig_h, orig_w = orig_size
        new_h, new_w = new_size
        scale_h = new_h / orig_h
        scale_w = new_w / orig_w
        align = self.multiple_of
        return [
            (
                int(math.floor(top * scale_h / align) * align),
                int(math.ceil(bottom * scale_h / align) * align),
                int(math.floor(left * scale_w / align) * align),
                int(math.ceil(right * scale_w / align) * align),
            )
            for (top, bottom, left, right) in layer_boxes
        ]

    def _stack_all_images(self, whole_image, layer_boxes, layer_images):
        n_layers = len(layer_boxes)
        h, w = whole_image.shape[-2:]
        stacked_images = torch.zeros((n_layers + 1, 4, h, w), device=whole_image.device)
        stacked_images[0] = ensure_alpha_channel(whole_image)[0]
        for i in range(n_layers):
            top, bottom, left, right = layer_boxes[i]
            layer_img = layer_images[i]
            if layer_img.shape[-2:] != (bottom - top, right - left):
                layer_img = resize_to_resolution(
                    layer_img,
                    (bottom - top, right - left),
                    crop=False,
                )
            if layer_img.shape[1] == 3:
                # Insert alpha channel
                alpha_channel = torch.ones(
                    (1, 1, layer_img.shape[2], layer_img.shape[3]),
                    device=layer_img.device,
                )
                layer_img = torch.cat([layer_img, alpha_channel], dim=1)
            stacked_images[i + 1, :, top:bottom, left:right] = layer_img[0]
        return stacked_images

    def _crop_stacked_images(self, stacked_images, layer_boxes):
        cropped_layers = []
        for i in range(1, stacked_images.shape[0]):
            top, bottom, left, right = layer_boxes[i - 1]
            cropped_layer = stacked_images[i : i + 1, :, top:bottom, left:right]
            cropped_layers.append(cropped_layer)
        return cropped_layers

    async def genearte_layer_boxes_prompts(
        self, image: torch.Tensor
    ) -> tuple[list[tuple[int, int, int, int]], list[str]]:
        h, w = image.shape[2], image.shape[3]
        result_text = await self.chat_completion(self.detection_prompt, [image])
        try:
            result_json = parse_llm_json_output(result_text)
            layer_boxes: list[tuple[int, int, int, int]] = [(0, h, 0, w)]
            layer_prompts: list[str] = [result_json["background"]]
            for item in result_json["foreground"]:
                xmin, ymin, xmax, ymax = item["bbox_2d"]
                if self.detection_coord_type == "qwen3vl":
                    # Qwen3-VL use normized coordinates [0, 1000]
                    xmin = int(xmin / 1000 * w)
                    xmax = int(xmax / 1000 * w)
                    ymin = int(ymin / 1000 * h)
                    ymax = int(ymax / 1000 * h)
                layer_boxes.append((ymin, ymax, xmin, xmax))
                layer_prompts.append(item["label"])
            return layer_boxes, layer_prompts
        except Exception:
            logger.error(
                "Failed to parse detection output, this is likely due to LLM returning invalid JSON."
            )
            logger.debug("Dumping raw LLM response:")
            logger.debug(result_text)
            raise

    def generate_negative(self, num_layers: int):
        negative_prompt_embeds = self.encoder.encode(
            self.default_negative_prompt,
            system_prompt=self.encoder_prompt,
        )
        text_lengths = [negative_prompt_embeds.shape[1]] * num_layers
        prompt_embeds = repeat(
            negative_prompt_embeds, "b l d -> b (n l) d", n=num_layers
        )
        return {
            "prompt_embeds": prompt_embeds,
            "text_lengths": text_lengths,
        }

    async def prepare_inference_batch(
        self, batch: EfficientLayeredInputBatch
    ) -> EfficientLayeredProcessedBatch:
        orig_size = batch["clean_image"].shape[2], batch["clean_image"].shape[3]
        batch["clean_image"] = clean_image = self.resize_image(
            ensure_alpha_channel(batch["clean_image"])
        )
        image_size = clean_image.shape[2], clean_image.shape[3]
        image_latents = self.encode_latents(
            clean_image, posterior=self.condition_posterior
        )

        layer_boxes = batch.get("layer_boxes", None)
        layer_prompts = batch.get("layer_prompts", None)
        if (layer_boxes is None) ^ (layer_prompts is None):
            warn_once(
                logger,
                "Either both or neither of layer_boxes and layer_prompts should be provided. Ignoring both.",
            )
        if layer_boxes is None or layer_prompts is None:
            batch["layer_boxes"], batch["layer_prompts"] = (
                layer_boxes,
                layer_prompts,
            ) = await self.genearte_layer_boxes_prompts(clean_image)
            # No need to scale layer boxes as image is already resized
            batch["layer_boxes"] = layer_boxes = self._scale_and_align_layer_boxes(
                layer_boxes, image_size, image_size
            )
        else:
            batch["layer_boxes"] = layer_boxes = self._scale_and_align_layer_boxes(
                layer_boxes, orig_size, image_size
            )
        if self.save_annotated_image:
            batch["annotated_image"] = draw_bbox_on_image(
                clean_image,
                layer_boxes[1:],
                [str(i) for i in range(1, len(layer_boxes))],
            )

        prompt_embeds_list = [
            self.encoder.encode(prompt, system_prompt=self.encoder_prompt)
            for prompt in layer_prompts
        ]
        prompt_embeds = torch.cat(prompt_embeds_list, dim=1)
        text_lengths = [embed.shape[1] for embed in prompt_embeds_list]

        result = EfficientLayeredProcessedBatch(
            image_size=image_size,
            image_latents=image_latents,
            prompt_embeds=prompt_embeds,
            layer_boxes=layer_boxes,
            text_lengths=text_lengths,
        )

        if self.save_negative:
            result["negative"] = self.generate_negative(len(layer_boxes))

        return result

    async def prepare_training_batch(
        self, batch: EfficientLayeredTrainInputBatch
    ) -> EfficientLayeredProcessedBatch:
        if (layer_prompts := batch.get("layer_prompts", None)) is None:
            batch["layer_prompts"] = layer_prompts = await asyncio.gather(
                *(
                    [
                        self.chat_completion(
                            self.bg_caption_prompt,
                            [batch["layer_images"][0]],
                        )
                    ]
                    + [
                        self.chat_completion(self.fg_caption_prompt, [img])
                        for img in batch["layer_images"][1:]
                    ]
                )
            )
        stacked_images = self._stack_all_images(
            batch["clean_image"], batch["layer_boxes"], batch["layer_images"]
        )
        orig_size = batch["clean_image"].shape[2], batch["clean_image"].shape
        resized_images = self.resize_image(stacked_images)
        new_size = resized_images.shape[2], resized_images.shape[3]
        batch["clean_image"] = clean_image = resized_images[0:1]
        image_latents = self.encode_latents(
            clean_image, posterior=self.condition_posterior
        )
        batch["layer_boxes"] = layer_boxes = self._scale_and_align_layer_boxes(
            batch["layer_boxes"], orig_size, new_size
        )
        batch["layer_images"] = layer_images = self._crop_stacked_images(
            resized_images, layer_boxes
        )
        clean_latents = torch.cat(
            [
                self.encode_latents(img, posterior=self.target_posterior)
                for img in layer_images
            ],
            dim=1,
        )
        prompt_embeds_list = [
            self.encoder.encode(prompt, system_prompt=self.encoder_prompt)
            for prompt in layer_prompts
        ]
        prompt_embeds = torch.cat(prompt_embeds_list, dim=1)
        text_lengths = [embed.shape[1] for embed in prompt_embeds_list]

        result = EfficientLayeredProcessedBatch(
            image_size=new_size,
            image_latents=image_latents,
            prompt_embeds=prompt_embeds,
            layer_boxes=layer_boxes,
            text_lengths=text_lengths,
            clean_latents=clean_latents,
        )

        if self.save_negative:
            result["negative"] = self.generate_negative(len(layer_boxes))

        return result

    def get_latent_length(self, batch: EfficientLayeredProcessedBatch):
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        return (
            super().get_latent_length(batch)
            + batch["prompt_embeds"].shape[1]
            + sum(
                (bottom - top) * (right - left) // ratio
                for (top, bottom, left, right) in batch["layer_boxes"]
            )
        )

    def decode_output(
        self,
        output_latent: torch.Tensor,
        batch: EfficientLayeredProcessedBatch,
    ) -> EfficientLayeredDecodedBatch:
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        latent_len_per_image = [
            (bottom - top) * (right - left) // ratio
            for (top, bottom, left, right) in batch["layer_boxes"]
        ]
        split_latents = torch.split(
            output_latent,
            latent_len_per_image,
            dim=1,
        )
        decoded_layers: list[torch.Tensor] = []
        for i, latents in enumerate(split_latents):
            layer_size = batch["layer_boxes"][i]
            decoded_layer = self.decode_latents(
                latents, (layer_size[1] - layer_size[0], layer_size[3] - layer_size[2])
            )
            decoded_layers.append(decoded_layer)
        return EfficientLayeredDecodedBatch(
            clean_image=merge_images(decoded_layers),
            layer_images=decoded_layers,
        )

    def annotate_output(
        self,
        decoded: DecodedBatch,
        batch: EfficientLayeredProcessedBatch,
    ) -> torch.Tensor:
        layer_images = cast(EfficientLayeredDecodedBatch, decoded)["layer_images"]
        return merge_images(layer_images, border_width=4, draw_labels=True)

    def initialize_latents(
        self,
        batch: EfficientLayeredProcessedBatch,
        generator=None,
        device=None,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        ratio = (self.vae_scale_factor * self.patch_size) ** 2
        latent_len_per_image = [
            (bottom - top) * (right - left) // ratio
            for (top, bottom, left, right) in batch["layer_boxes"]
        ]
        total_latent_len = sum(latent_len_per_image)
        latents = torch.randn(
            (1, total_latent_len, 64),
            generator=generator,
            device=device or self.device,
            dtype=dtype,
        )
        batch["noisy_latents"] = latents
        return latents
