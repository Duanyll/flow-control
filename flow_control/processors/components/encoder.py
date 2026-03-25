import re
from typing import Annotated, Any, Literal

import torch
from pydantic import Discriminator, Tag
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    Mistral3ForConditionalGeneration,
    PixtralProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
    Qwen3ForCausalLM,
    T5EncoderModel,
    T5Tokenizer,
)

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger, warn_once
from flow_control.utils.resize import resize_to_multiple_of, resize_to_resolution
from flow_control.utils.tensor import remove_alpha_channel
from flow_control.utils.types import TorchDType

logger = get_logger(__name__)


class BaseEncoder[T](HfModelLoader[T]):
    chat_template: str = "{user}"
    image_template: str = ""

    def _format_prompt(
        self, prompt: str, images: Any | None = None, system_prompt: str | None = None
    ) -> Any:
        user_prompt = prompt
        for i in range(len(images or [])):
            user_prompt += self.image_template.format(index=i + 1)
        user_prompt += prompt
        formatted_prompt = self.chat_template.format(
            system=system_prompt, user=user_prompt
        )
        return formatted_prompt

    def encode(
        self,
        prompt: str,
        images: list[torch.Tensor] | None = None,
        system_prompt: str | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Encode method must be implemented by subclasses.")


def warn_no_image_support(func):
    def wrapper(self, prompt, images=None, system_prompt=None):
        if images:
            warn_once(
                logger,
                f"{self.__class__.__name__} does not support image inputs. Ignoring provided images.",
            )
        return func(self, prompt, images=None, system_prompt=system_prompt)

    return wrapper


class GenerativeEncoder:
    def generate(
        self,
        prompt: str,
        images: list[torch.Tensor] | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        raise NotImplementedError("Generate method must be implemented by subclasses.")


class T5TextEncoder(BaseEncoder[T5EncoderModel]):
    type: Literal["t5"] = "t5"

    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "T5EncoderModel"
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    subfolder: str | None = "text_encoder_2"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[T5Tokenizer] = HfModelLoader(
        library="transformers",
        class_name="T5Tokenizer",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="tokenizer_2",
    )

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        return super().load_model(device, frozen)

    @warn_no_image_support
    def encode(self, prompt, images=None, system_prompt: str | None = None):
        tokenizer = self.tokenizer.model
        model = self.model

        prompt = self._format_prompt(prompt, images, system_prompt)

        t5_inputs = tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        t5_input_ids = t5_inputs.input_ids
        prompt_embeds = model(
            t5_input_ids.to(model.device), output_hidden_states=False
        )[0]

        return prompt_embeds


class ClipTextEncoder(BaseEncoder[CLIPTextModel]):
    type: Literal["clip"] = "clip"

    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "CLIPTextModel"
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[CLIPTokenizer] = HfModelLoader(
        library="transformers",
        class_name="CLIPTokenizer",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="tokenizer",
    )

    max_length: int = 77

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        return super().load_model(device, frozen)

    @warn_no_image_support
    def encode(self, prompt, images=None, system_prompt: str | None = None):
        tokenizer = self.tokenizer.model
        model = self.model

        prompt = self._format_prompt(prompt, images, system_prompt)

        clip_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        clip_input_ids = clip_inputs.input_ids
        pooled_prompt_embeds = model(
            clip_input_ids.to(model.device), output_hidden_states=False
        ).pooler_output

        return pooled_prompt_embeds


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

    def _get_image_pad_len(self, images: list[torch.Tensor]) -> int:
        if not images:
            return 0
        processor: Any = self.vl_processor.model
        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        merge_length = processor.image_processor.merge_size**2
        return sum(i.prod() // merge_length for i in image_inputs["image_grid_thw"])

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        self.vl_processor.load_model(device, frozen)
        # Patch to a no-op for _check_special_mm_tokens
        self.vl_processor.model._check_special_mm_tokens = lambda *args, **kwargs: None
        return super().load_model(device)

    def encode(self, prompt, images=None, system_prompt: str | None = None):
        prefix, suffix = self.chat_template.split("{user}")
        prefix = prefix.format(system=system_prompt or "")

        pretokenized_inputs = [
            self.image_template.format(index=i + 1) for i in range(len(images or []))
        ]
        if self.split_quotation:
            for part, is_quoted in self._split_quotation(prompt):
                if is_quoted:
                    # Each character in the quoted part is treated as a separate token
                    pretokenized_inputs.extend(part)
                else:
                    pretokenized_inputs.append(part)
        else:
            pretokenized_inputs.append(prompt)

        max_length = self.tokenizer_max_length
        if images:
            images = [
                remove_alpha_channel(self._resize_image(image)) for image in images
            ]

            # FIXME: There is something wrong with LongCat-Image-Edit when caculating
            # the required number of image padding tokens. The behavior is strange in
            # the original codebase as well.

            if self.keep_padding_tokens and self.tokenizer_max_length > 0:
                max_length += self._get_image_pad_len(images) - len(images)

        vl_processor = self.vl_processor.model
        model = self.model
        prefix_inputs = vl_processor(
            text=prefix, text_kwargs={"return_tensors": "pt"}
        ).to(model.device)
        prompt_inputs = vl_processor(
            images=images,
            text=pretokenized_inputs,
            text_kwargs={
                "padding": "max_length"
                if self.keep_padding_tokens and max_length > 0
                else False,
                "is_split_into_words": True,
                "max_length": max_length if max_length > 0 else None,
                "return_tensors": "pt",
            },
            images_kwargs={
                "return_tensors": "pt",
            },
        ).to(model.device)
        suffix_inputs = vl_processor(
            text=suffix, text_kwargs={"return_tensors": "pt"}
        ).to(model.device)

        # When is_split_into_words=True is passed, returned inputs_ids and attention_mask
        # will not have batch dimension. However, pixel_values and image_grid_thw will
        # always have batch dimension.

        prefix_len = prefix_inputs.input_ids.shape[1]
        suffix_len = suffix_inputs.input_ids.shape[1]

        input_ids = torch.cat(
            [
                prefix_inputs.input_ids,
                prompt_inputs.input_ids.unsqueeze(0),
                suffix_inputs.input_ids,
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                prefix_inputs.attention_mask,
                prompt_inputs.attention_mask.unsqueeze(0),
                suffix_inputs.attention_mask,
            ],
            dim=1,
        )
        pixel_values = prompt_inputs.pixel_values if images else None
        image_grid_thw = prompt_inputs.image_grid_thw if images else None
        encoder_hidden_states = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        if self.drop_suffix_tokens:
            return hidden_states[:, prefix_len:-suffix_len, :]
        else:
            return hidden_states[:, prefix_len:, :]

    def generate(
        self, prompt, images=None, system_prompt: str = "You are a helpful assistant."
    ) -> str:
        vl_processor = self.vl_processor.model
        model = self.model

        if images:
            images = [remove_alpha_channel(image) for image in images]
        text_input = self._format_prompt(prompt, images, system_prompt)
        model_inputs = vl_processor(
            text=text_input,
            images=images,
            text_kwargs={"padding": True, "return_tensors": "pt"},
            images_kwargs={"return_tensors": "pt"},
        ).to(model.device)
        generated_ids = model.generate(
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


class Qwen3Encoder(BaseEncoder[Qwen3ForCausalLM]):
    type: Literal["qwen3"] = "qwen3"
    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "Qwen3ForCausalLM"
    pretrained_model_id: str = "Tongyi-MAI/Z-Image"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[Qwen2Tokenizer] = HfModelLoader(
        library="transformers",
        class_name="Qwen2Tokenizer",
        pretrained_model_id="Tongyi-MAI/Z-Image",
        subfolder="tokenizer",
    )

    max_sequence_length: int = 512
    hidden_state_layers: list[int] = [-2]
    enable_thinking: bool = True
    keep_padding_tokens: bool = False

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        return super().load_model(device, frozen)

    @warn_no_image_support
    def encode(self, prompt, images=None, system_prompt=None):
        messages = [{"role": "user", "content": prompt}]
        formated_prompt = self.tokenizer.model.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        assert isinstance(formated_prompt, str)
        text_inputs = self.tokenizer.model(
            [formated_prompt],
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.model.device)
        prompt_masks = text_inputs.attention_mask.to(self.model.device).bool()
        hidden_states = self.model(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states
        prompt_embeds = torch.cat(
            [hidden_states[i] for i in self.hidden_state_layers], dim=2
        )
        if not self.keep_padding_tokens:
            prompt_embeds = prompt_embeds[prompt_masks].unsqueeze(0)
        return prompt_embeds


class Mistral3Encoder(BaseEncoder[Mistral3ForConditionalGeneration], GenerativeEncoder):
    type: Literal["mistral3"] = "mistral3"
    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "Mistral3ForConditionalGeneration"
    pretrained_model_id: str = "black-forest-labs/FLUX.2-dev"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[PixtralProcessor] = HfModelLoader(
        library="transformers",
        class_name="PixtralProcessor",
        pretrained_model_id="black-forest-labs/FLUX.2-dev",
        subfolder="tokenizer",
    )

    encode_with_images: bool = False
    max_sequence_length: int = 512
    hidden_state_layers: list[int] = [10, 20, 30]
    temperature: float = 0.7

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        return super().load_model(device, frozen)

    def format_prompt(
        self,
        prompt: str,
        images: list[torch.Tensor] | None = None,
        system_prompt: str | None = None,
    ) -> Any:
        cleaned_prompt = prompt.replace("[IMG]", "")
        if images:
            images = [remove_alpha_channel(image) for image in images]
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt or ""},
                ],
            },
            {
                "role": "user",
                "content": [
                    *(
                        [{"type": "image", "image": image} for image in images]
                        if images
                        else []
                    ),
                    {"type": "text", "text": cleaned_prompt},
                ],
            },
        ]
        return messages

    def encode(self, prompt, images=None, system_prompt=None):
        # Ignore input images, Flux.2 does not use them for encoding.
        messages = self.format_prompt(
            prompt,
            images=images if self.encode_with_images else None,
            system_prompt=system_prompt,
        )
        tokenizer: Any = self.tokenizer.model
        # The PixtralProcessor's apply_chat_template is badly typed
        inputs = tokenizer.apply_chat_template(
            [messages],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
        )

        device = self.model.device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden_states = torch.cat(
            [output.hidden_states[i] for i in self.hidden_state_layers], dim=2
        )
        return hidden_states

    def generate(
        self, prompt, images=None, system_prompt="You are a helpful assistant."
    ):
        messages = self.format_prompt(prompt, images, system_prompt)
        tokenizer: Any = self.tokenizer.model
        inputs = tokenizer.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2048,
        )

        device = self.model.device
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                device=device, dtype=self.model.dtype
            )

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.temperature,
            use_cache=True,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated_ids[:, input_length:]

        result = tokenizer.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return result[0].strip()


Encoder = Annotated[
    Annotated[T5TextEncoder, Tag("t5")]
    | Annotated[ClipTextEncoder, Tag("clip")]
    | Annotated[Qwen25VLEncoder, Tag("qwen25vl")]
    | Annotated[Qwen3Encoder, Tag("qwen3")]
    | Annotated[Mistral3Encoder, Tag("mistral3")],
    Discriminator("type"),
]
