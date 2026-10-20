import math
import re
from typing import Any, Literal, cast

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
)

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.resize import resize_to_multiple_of, resize_to_resolution
from flow_control.utils.tensor import remove_alpha_channel, tensor_to_pil
from flow_control.utils.types import TorchDType

from .base import BaseEncoder, GenerativeEncoder, encoder_registry


@encoder_registry.register("qwen25vl")
class Qwen25VLEncoder(
    BaseEncoder[Qwen2_5_VLForConditionalGeneration], GenerativeEncoder
):
    type: Literal["qwen25vl"] = "qwen25vl"
    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "Qwen2_5_VLForConditionalGeneration"
    pretrained_model_id: str = "Qwen/Qwen-Image"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[Qwen2Tokenizer] = HfModelLoader(
        library="transformers",
        class_name="Qwen2Tokenizer",
        pretrained_model_id="Qwen/Qwen-Image",
        subfolder="tokenizer",
    )

    vl_processor: HfModelLoader[Qwen2VLProcessor] = HfModelLoader(
        library="transformers",
        class_name="Qwen2VLProcessor",
        pretrained_model_id="Qwen/Qwen-Image-Edit",
        subfolder="processor",
        # This processor subfolder also lacks the model config AutoTokenizer
        # requires in Transformers 5.17.
        extra_from_pretrained_kwargs={"tokenizer_type": "qwen2"},
    )

    chat_template: str = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    image_template: str = "Picture {index}: <|vision_start|><|image_pad|><|vision_end|>"
    split_quotation: bool = False
    quote_pairs: list[tuple[str, str]] = [
        ("“", "”"),
        ("‘", "’"),
        ('"', '"'),
        ("'", "'"),
    ]
    drop_suffix_tokens: bool = False
    tokenizer_max_length: int = 1024
    keep_padding_tokens: bool = False
    generate_max_new_tokens: int = 512

    resize_mode: Literal["none", "scale", "pixels"] = "pixels"
    image_pixels: int = 384 * 384
    image_multiple: int = 32
    image_scale: int = 2

    def _resize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Legacy bilinear, center-cropping resize, unlike the official preprocessing.
        ``encode`` no longer calls it; kept for out-of-tree subclasses."""
        if self.resize_mode == "none":
            return image
        elif self.resize_mode == "scale":
            new_size = (
                image.shape[2] // self.image_scale,
                image.shape[3] // self.image_scale,
            )
            return resize_to_resolution(image, new_size)
        elif self.resize_mode == "pixels":
            return resize_to_multiple_of(
                image, self.image_multiple, pixels=self.image_pixels
            )
        else:
            raise ValueError(f"Invalid resize mode: {self.resize_mode}")

    def _split_quotation(self, prompt: str):
        patterns = []
        for q1, q2 in self.quote_pairs:
            e_q1 = re.escape(q1)
            e_q2 = re.escape(q2)
            content_pattern = r".*?"
            if q1 == "'":
                pattern = f"(?<![a-zA-Z]){e_q1}{content_pattern}{e_q2}"
            else:
                pattern = f"{e_q1}{content_pattern}{e_q2}"
            patterns.append(pattern)

        full_pattern = f"({'|'.join(patterns)})"
        parts = re.split(full_pattern, prompt)

        result = []
        for part in parts:
            if not part:
                continue
            is_quoted = False
            if len(part) >= 2:
                for q1, q2 in self.quote_pairs:
                    if part.startswith(q1) and part.endswith(q2):
                        is_quoted = True
                        break
            result.append((part, is_quoted))
        return result

    def _vl_image(self, image: torch.Tensor) -> Image.Image:
        """The VL model's view of a [0, 1] BCHW reference, resized with PIL Lanczos
        like the official pipelines. The processor rescales its input by 1/255, so
        it must get 8-bit images, not [0, 1] floats."""
        pil = tensor_to_pil(remove_alpha_channel(image))
        if self.resize_mode == "none":
            return pil
        if self.resize_mode == "scale":
            size = (pil.width // self.image_scale, pil.height // self.image_scale)
        else:  # diffusers' calculate_dimensions, with image_multiple for its 32
            ratio = pil.width / pil.height
            width = math.sqrt(self.image_pixels * ratio)
            m = self.image_multiple
            size = (round(width / m) * m, round(width / ratio / m) * m)
        return pil.resize(size, Image.Resampling.LANCZOS)

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        self.vl_processor.load_model(device, frozen)
        # Patch to a no-op for _check_special_mm_tokens
        self.vl_processor.model._check_special_mm_tokens = (  # ty: ignore[invalid-assignment]
            lambda *args, **kwargs: None
        )
        return super().load_model(device)

    def encode(self, prompt, images=None, system_prompt: str | None = None):
        prefix, suffix = self.chat_template.split("{user}")
        prefix = prefix.format(system=system_prompt or "")

        words = []
        if self.split_quotation:
            for part, is_quoted in self._split_quotation(prompt):
                if is_quoted:
                    # Each character in the quoted part is treated as a separate token
                    words.extend(part)
                else:
                    words.append(part)
        else:
            words.append(prompt)

        vl_processor = self.vl_processor.model
        model = self.model
        max_length = self.tokenizer_max_length
        text_kwargs = {"is_split_into_words": True, "return_tensors": "pt"}
        prefix_inputs = vl_processor(text=prefix, text_kwargs={"return_tensors": "pt"})
        vision = (
            vl_processor(
                images=[self._vl_image(image) for image in images],
                text=[
                    self.image_template.format(index=i + 1) for i in range(len(images))
                ],
                text_kwargs=text_kwargs,
                images_kwargs={"return_tensors": "pt"},
            )
            if images
            else None
        )
        # The token budget covers the prompt text alone, as in LongCat's pipelines.
        prompt_inputs = vl_processor(
            text=words,
            text_kwargs=text_kwargs
            | {
                "padding": "max_length"
                if self.keep_padding_tokens and max_length > 0
                else False,
                "truncation": max_length > 0,
                "max_length": max_length if max_length > 0 else None,
            },
        )
        suffix_inputs = vl_processor(text=suffix, text_kwargs={"return_tensors": "pt"})
        parts = [
            prefix_inputs,
            *([vision] if vision else []),
            prompt_inputs,
            suffix_inputs,
        ]

        def joined(key: str) -> torch.Tensor:
            return torch.cat([part[key] for part in parts], dim=1).to(model.device)

        input_ids = joined("input_ids")
        attention_mask = joined("attention_mask")
        image_grid_thw = vision.image_grid_thw.to(model.device) if vision else None
        # The mRoPE positions of Transformers 4.x, which the checkpoints were used
        # with. 5.x needs mm_token_type_ids for 3-D image positions, counts padding
        # in text-only input, and puts padding at position 0 (4.x: 1).
        position_ids, _ = model.model.get_rope_index(
            cast(torch.LongTensor, input_ids),
            cast(torch.IntTensor, joined("mm_token_type_ids")),
            image_grid_thw,
            attention_mask=attention_mask,
        )
        hidden_states = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids.masked_fill(attention_mask == 0, 1),
            pixel_values=vision.pixel_values.to(model.device) if vision else None,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        ).hidden_states[-1]
        prefix_len = prefix_inputs["input_ids"].shape[-1]
        if self.drop_suffix_tokens:
            suffix_len = suffix_inputs["input_ids"].shape[-1]
            return hidden_states[:, prefix_len:-suffix_len, :]
        return hidden_states[:, prefix_len:, :]

    def generate(
        self, prompt, images=None, system_prompt: str = "You are a helpful assistant."
    ) -> str:
        vl_processor = self.vl_processor.model
        model = self.model

        if images:
            images = [tensor_to_pil(remove_alpha_channel(image)) for image in images]
        text_input = self._format_prompt(prompt, images, system_prompt)
        model_inputs = vl_processor(
            text=text_input,
            images=images,
            text_kwargs={"padding": True, "return_tensors": "pt"},
            images_kwargs={"return_tensors": "pt"},
        ).to(model.device)
        generated_ids = cast(Any, model).generate(
            **model_inputs, max_new_tokens=self.generate_max_new_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(
                model_inputs.input_ids, generated_ids, strict=True
            )
        ]
        output_text = vl_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()


if __name__ == "__main__":
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        QwenImageEditPlusPipeline,
        calculate_dimensions,
    )
    from rich import print

    from flow_control.processors.components.prompts import parse_prompt
    from flow_control.utils import device as devutil
    from flow_control.utils.tensor import pil_to_tensor

    # Qwen25VLEncoder vs the official diffusers encode_prompt (T2I, and Edit-Plus with
    # two example images), bitwise. Diffusers omits the mm_token_type_ids that
    # Transformers 5.x needs for 3-D mRoPE, so a hook supplies them. It also counts
    # padding on 5.x, so LongCat-style padding is checked against the 4.x rule.
    device = devutil.default_device()
    encoder = Qwen25VLEncoder()
    encoder.load_model(device)
    text_encoder = encoder.model
    # Diffusers accepts absent components at runtime; annotations omit None.
    t2i_pipe = QwenImagePipeline(
        scheduler=None,
        vae=None,  # ty: ignore[invalid-argument-type]
        text_encoder=text_encoder,
        tokenizer=encoder.tokenizer.model,
        transformer=None,  # ty: ignore[invalid-argument-type]
    )
    edit_pipe = QwenImageEditPlusPipeline(
        scheduler=None,
        vae=None,  # ty: ignore[invalid-argument-type]
        text_encoder=text_encoder,
        tokenizer=encoder.tokenizer.model,
        processor=encoder.vl_processor.model,
        transformer=None,  # ty: ignore[invalid-argument-type]
    )

    def mark_image_tokens(module, args, kwargs):
        image_token_id = module.config.image_token_id
        kwargs["mm_token_type_ids"] = (kwargs["input_ids"] == image_token_id).int()
        return args, kwargs

    text_encoder.register_forward_pre_hook(mark_image_tokens, with_kwargs=True)
    images = [
        Image.open(f"examples/assets/{name}").convert("RGB")
        for name in ("image9.png", "image10.png")
    ]
    conditions = []
    for image in images:
        width, height = calculate_dimensions(
            CONDITION_IMAGE_SIZE, image.width / image.height
        )
        conditions.append(edit_pipe.image_processor.resize(image, height, width))
    prompt = "Dress the man in Picture 1 in the blue polo shirt from Picture 2."
    with torch.no_grad():
        results = {
            "t2i": (
                encoder.encode(
                    prompt, system_prompt=parse_prompt("@qwen_image_encoder")
                ),
                t2i_pipe.encode_prompt(prompt, device=device)[0],
            ),
            "edit": (
                encoder.encode(
                    prompt,
                    [pil_to_tensor(image) for image in images],
                    system_prompt=parse_prompt("@qwen_image_edit_encoder"),
                ),
                edit_pipe.encode_prompt(
                    prompt,
                    # The pipeline passes PIL images; the annotation says tensor.
                    image=conditions,  # ty: ignore[invalid-argument-type]
                    device=device,
                )[0],
            ),
        }
    for name, (ours, theirs) in results.items():
        print(f"{name}: {tuple(ours.shape)} vs diffusers {tuple(theirs.shape)}")
        assert torch.equal(ours, theirs), f"max |diff| {(ours - theirs).abs().max()}"

    # Padding in the middle of the template (LongCat): real tokens keep consecutive
    # positions across it, and padding sits at position 1.
    padded = Qwen25VLEncoder(tokenizer_max_length=64, keep_padding_tokens=True)
    padded.load_model(device)
    seen = {}
    text_encoder.register_forward_pre_hook(
        lambda module, args, kwargs: seen.update(kwargs), with_kwargs=True
    )
    with torch.no_grad():
        padded.encode(prompt)
    real = seen["attention_mask"][0].bool()
    positions = seen["position_ids"][:, 0]
    assert (positions[:, ~real] == 1).all() and torch.equal(
        positions[:, real], torch.arange(int(real.sum()), device=device).expand(3, -1)
    )
    print(f"Bitwise equal to diffusers; {int((~real).sum())} padding tokens skipped.")
