from typing import Any, Literal, NotRequired

import torch
from einops import rearrange
from pydantic import PrivateAttr

from flow_control.utils.llm import LLMClient
from flow_control.utils.loaders import HfModelLoader
from flow_control.utils.resize import resize_to_resolution

from .base import BaseProcessor
from .qwen import QwenImageProcessor

_DEFAULT_IMAGE_CAPTION_PROMPT_CN = """
# 图像标注器
你是一个专业的图像标注器。请基于输入图像，撰写图注:
1. 使用自然、描述性的语言撰写图注，不要使用结构化形式或富文本形式。
2. 通过加入以下内容，丰富图注细节：
 - 对象的属性：如数量、颜色、形状、大小、位置、材质、状态、动作等
 - 对象间的视觉关系：如空间关系、功能关系、动作关系、从属关系、比较关系、因果关系等
 - 环境细节：例如天气、光照、颜色、纹理、气氛等
 - 文字内容：识别图像中清晰可见的文字，不做翻译和解释，用引号在图注中强调
3. 保持真实性与准确性：
 - 不要使用笼统的描述
 - 描述图像中所有可见的信息，但不要加入没有在图像中出现的内容
"""

_DEFAULT_IMAGE_CAPTION_PROMPT_EN = """
# Image Annotator
You are a professional image annotator. Please write an image caption based on the input image:
1. Write the caption using natural, descriptive language without structured formats or rich text.
2. Enrich caption details by including: 
 - Object attributes, such as quantity, color, shape, size, material, state, position, actions, and so on
 - Vision Relations between objects, such as spatial relations, functional relations, possessive relations, attachment relations, action relations, comparative relations, causal relations, and so on
 - Environmental details, such as weather, lighting, colors, textures, atmosphere, and so on
 - Identify the text clearly visible in the image, without translation or explanation, and highlight it in the caption with quotation marks
3. Maintain authenticity and accuracy:
 - Avoid generalizations
 - Describe all visible information in the image, while do not add information not explicitly shown in the image
"""


class QwenImageLayeredProcessor(QwenImageProcessor):
    class BatchType(BaseProcessor.BatchType):  # type: ignore
        clean_image: torch.Tensor
        """
        [B, C, H, W] Tensor representing the clean input image. This is NOT training target,
        but input image used for reference.
        """
        layer_images: NotRequired[list[torch.Tensor]]
        """
        List of [B, C, H, W] Tensors representing individual layer images. This is the 
        training target for layered image generation. `clean_latents` will only be generated
        if `layer_images` is provided.
        """

        prompt: NotRequired[str]
        negative_prompt: NotRequired[str]
        num_layers: NotRequired[int]

        prompt_embeds: NotRequired[torch.Tensor]
        clean_latents: NotRequired[torch.Tensor]
        image_latents: NotRequired[torch.Tensor]

    _loading_preset = {
        "vae": ["encode", "decode", "preview", "allow_remote_preview"],
        "text_encoder": ["encode"],
        "tokenizer": ["encode"],
        "vl_processor": ["encode"],
    }

    vae: HfModelLoader = HfModelLoader(
        type="diffusers",
        class_name="AutoencoderKLQwenImage",
        pretrained_model_id="Qwen/Qwen-Image-Layered",
        subfolder="vae",
        dtype=torch.bfloat16,
    )
    _vae: Any = PrivateAttr()

    vl_processor: HfModelLoader = HfModelLoader(
        type="transformers",
        class_name="Qwen2VLProcessor",
        pretrained_model_id="Qwen/Qwen-Image-Layered",
        subfolder="processor",
    )
    _vl_processor: Any = PrivateAttr()

    default_resolution: tuple[int, int] = (640, 640)
    resize_mode: Literal["multiple_of", "list"] = "multiple_of"
    multiple_of: int = 32
    pixels: int = 640 * 640
    default_num_layers: int = 4

    llm: LLMClient | None = None
    chat_template: str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user{}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
    image_caption_prompt: str = "cn"  # or "en", or custom string

    async def generate_caption(self, image: torch.Tensor):
        prompt = (
            _DEFAULT_IMAGE_CAPTION_PROMPT_CN
            if self.image_caption_prompt == "cn"
            else _DEFAULT_IMAGE_CAPTION_PROMPT_EN
            if self.image_caption_prompt == "en"
            else self.image_caption_prompt
        )

        if self.llm:
            content, _ = await self.llm.generate(user_prompt=prompt, images=[image])
            return content.strip()
        else:
            text_input = self.chat_template.format(prompt)
            with torch.no_grad():
                model_inputs = self._vl_processor(
                    text=text_input,
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                generated_ids = self._text_encoder.generate(
                    **model_inputs, max_new_tokens=512
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(
                        model_inputs.input_ids, generated_ids, strict=True
                    )
                ]
                output_text = self._vl_processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            return output_text.strip()

    @torch.no_grad()
    def encode_latents(self, images: torch.Tensor | list[torch.Tensor]):
        if not isinstance(images, list):
            images = [images]

        layer_images = []
        for image in images:
            if image.shape[1] == 3:
                # Add dummy alpha channel
                image = torch.cat([image, torch.ones_like(image[:, :1, :, :])], dim=1)
            layer_images.append(image)

        all_images = torch.stack(layer_images, dim=2)  # b, c, f, h, w
        latents = self._vae.encode(all_images).latent_dist.sample()
        latents_mean = (
            torch.tensor(self._vae.config.latents_mean)
            .view(1, self._vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self._vae.config.latents_std).view(
            1, self._vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std
        latents = self._pack_latents(latents)
        return latents

    def _pack_latents_layered(self, latents: torch.Tensor) -> torch.Tensor:
        return rearrange(
            latents,
            "b c f (h ph) (w pw) -> b (f h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, size: tuple[int, int]):
        latents = self._unpack_latents_layered(latents, size)
        latents_mean = (
            torch.tensor(self._vae.config.latents_mean)
            .view(1, self._vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self._vae.config.latents_std).view(
            1, self._vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        images = self._vae.decode(latents).sample
        images = (images + 1) / 2
        base_image = images[:, :, 0, :, :]
        layer_images = [images[:, :, i + 1, :, :] for i in range(images.shape[2] - 1)]
        return base_image, layer_images

    def _unpack_latents_layered(
        self, latents: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        h, w = size
        h = h // self.vae_scale_factor
        w = w // self.vae_scale_factor
        return rearrange(
            latents,
            "b (f h w) (c ph pw) -> b c f (h ph) (w pw)",
            f=self.default_num_layers + 1,
            h=h // self.patch_size,
            w=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )

    async def preprocess_batch(self, batch: BatchType) -> BatchType:
        if "pooled_prompt_embeds" not in batch or "prompt_embeds" not in batch:
            if "prompt" not in batch:
                batch["prompt"] = await self.generate_caption(batch["clean_image"])
            prompt_embeds = self.encode_prompt(batch["prompt"])
            batch["prompt_embeds"] = prompt_embeds

        if "reference_latents" not in batch:
            # Add dummy alpha channel if needed
            if batch["clean_image"].shape[1] == 3:
                batch["clean_image"] = torch.cat(
                    [
                        batch["clean_image"],
                        torch.ones_like(batch["clean_image"][:, :1, :, :]),
                    ],
                    dim=1,
                )  # b, 4, h, w
            batch["clean_image"] = self.resize_image(batch["clean_image"])
            batch["image_latents"] = self.encode_latents(batch["clean_image"])

        if "image_size" not in batch:
            batch["image_size"] = (
                batch["clean_image"].shape[2],
                batch["clean_image"].shape[3],
            )

        if "layer_images" in batch and "clean_latents" not in batch:
            for i in range(len(batch["layer_images"])):
                batch["layer_images"][i] = resize_to_resolution(
                    batch["layer_images"][i], batch["image_size"]
                )
            batch["clean_latents"] = self.encode_latents(
                [batch["clean_image"], *batch["layer_images"]]
            )

        if "num_layers" not in batch:
            if "layer_images" in batch:
                batch["num_layers"] = len(batch["layer_images"])
            else:
                batch["num_layers"] = self.default_num_layers

        latent_h = batch["image_size"][0] // self.vae_scale_factor // self.patch_size
        latent_w = batch["image_size"][1] // self.vae_scale_factor // self.patch_size
        batch["latent_length"] = batch["prompt_embeds"].shape[
            1
        ] + latent_h * latent_w * (batch["num_layers"] + 2)

        return batch

    def decode_output(
        self, output_latent: torch.Tensor, batch: BatchType
    ) -> torch.Tensor:
        base_image, layer_images = self.decode_latents(
            output_latent,
            batch["image_size"],  # type: ignore
        )
        batch["layer_images"] = layer_images
        return base_image

    def initialize_latents(self, batch, generator=None, device=None):
        if device is None:
            device = self.device
        if "image_size" in batch:
            h, w = batch["image_size"]
        else:
            h, w = self.default_resolution
        c = self.latent_channels
        h = h // self.vae_scale_factor
        w = w // self.vae_scale_factor
        f = batch.get("num_layers", 0) + 1
        latents = torch.randn((1, c, f, h, w), generator=generator, device=device)
        batch["noisy_latents"] = self._pack_latents_layered(latents)
        return batch["noisy_latents"]
