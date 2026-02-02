import re
from typing import Annotated, Any, Literal

import torch
from pydantic import PlainValidator

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger
from flow_control.utils.types import TorchDType

logger = get_logger(__name__)


class BaseEncoder(HfModelLoader):
    type: str

    chat_template: str = "{user}"
    image_template: str = ""

    def _format_prompt(
        self, prompt: str, images: Any | None = None, system_prompt: str | None = None
    ) -> str:
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


class T5TextEncoder(BaseEncoder):
    type: str = "t5"

    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "T5EncoderModel"
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    subfolder: str | None = "text_encoder_2"
    dtype: TorchDType | Literal["auto"] = torch.bfloat16

    tokenizer: HfModelLoader = HfModelLoader(
        library="transformers",
        class_name="T5Tokenizer",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="tokenizer_2",
    )

    def load_model(self, device):
        self.tokenizer.load_model(device)
        return super().load_model(device)

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


class ClipTextEncoder(BaseEncoder):
    type: str = "clip"

    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "CLIPTextModel"
    pretrained_model_id: str = "black-forest-labs/FLUX.1-dev"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType | Literal["auto"] = torch.bfloat16

    tokenizer: HfModelLoader = HfModelLoader(
        library="transformers",
        class_name="CLIPTokenizer",
        pretrained_model_id="black-forest-labs/FLUX.1-dev",
        subfolder="tokenizer",
    )

    max_length: int = 77

    def load_model(self, device):
        self.tokenizer.load_model(device)
        return super().load_model(device)

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


class Qwen25VLEncoder(BaseEncoder):
    type: str = "qwen_2_5_vl"
    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "Qwen2_5_VLForConditionalGeneration"
    pretrained_model_id: str = "Qwen/Qwen-Image"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType | Literal["auto"] = torch.bfloat16

    tokenizer: HfModelLoader = HfModelLoader(
        library="transformers",
        class_name="Qwen2Tokenizer",
        pretrained_model_id="Qwen/Qwen-Image",
        subfolder="tokenizer",
    )

    vl_processor: HfModelLoader = HfModelLoader(
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
    generate_max_new_tokens: int = 512

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

    def load_model(self, device):
        self.tokenizer.load_model(device)
        self.vl_processor.load_model(device)
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
                    pretokenized_inputs.append(part)
                else:
                    pretokenized_inputs.extend(part)
        else:
            pretokenized_inputs.append(prompt)

        vl_processor = self.vl_processor.model
        model = self.model
        prefix_inputs = vl_processor(text=prefix, return_tensors="pt").to(model.device)
        prompt_inputs = vl_processor(
            images=images,
            text=pretokenized_inputs,
            padding=False,
            is_split_into_words=True,
            return_tensors="pt",
        ).to(model.device)
        suffix_inputs = vl_processor(text=suffix, return_tensors="pt").to(model.device)

        prefix_len = prefix_inputs.input_ids.shape[0]
        suffix_len = suffix_inputs.input_ids.shape[0]

        input_ids = torch.cat(
            [
                prefix_inputs.input_ids,
                prompt_inputs.input_ids,
                suffix_inputs.input_ids,
            ],
            dim=0,
        ).unsqueeze(0)
        attention_mask = torch.cat(
            [
                prefix_inputs.attention_mask,
                prompt_inputs.attention_mask,
                suffix_inputs.attention_mask,
            ],
            dim=0,
        ).unsqueeze(0)
        pixel_values = prompt_inputs.pixel_values if images else None
        image_grid_thw = prompt_inputs.image_grid_thw if images else None
        encoder_hidden_states = model.model.encoder(
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

        text_input = self._format_prompt(prompt, images, system_prompt)
        model_inputs = vl_processor(
            text=text_input,
            images=images,
            padding=True,
            return_tensors="pt",
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


ENCODER_REGISTRY = {
    "t5": T5TextEncoder,
    "clip": ClipTextEncoder,
    "qwen25vl": Qwen25VLEncoder,
}


def parse_encoder(config: dict[str, Any]) -> BaseEncoder:
    encoder_type = config["type"]
    encoder_class = ENCODER_REGISTRY.get(encoder_type)
    if encoder_class is None:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
    return encoder_class(**config)


Encoder = Annotated[BaseEncoder, PlainValidator(parse_encoder)]
