from typing import Any, Literal, cast

import torch
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.tensor import remove_alpha_channel, tensor_to_pil
from flow_control.utils.types import TorchDType

from .base import BaseEncoder, GenerativeEncoder, encoder_registry


@encoder_registry.register("mistral3")
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
        # PIL, as the FLUX.2 pipeline passes: the Pixtral image processor rescales
        # by 1/255 unconditionally, so [0, 1] tensors would arrive near-black.
        pil_images = [
            tensor_to_pil(remove_alpha_channel(image)) for image in images or []
        ]
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
                    *({"type": "image", "image": image} for image in pil_images),
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

        generated_ids = cast(Any, self.model).generate(
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


if __name__ == "__main__":
    from PIL import Image
    from rich import print

    from flow_control.utils.tensor import pil_to_tensor

    # FLUX.2: a [0, 1] tensor image reaches the Pixtral processor exactly like
    # the PIL image the FLUX.2 pipeline passes.
    mistral = Mistral3Encoder()
    mistral.tokenizer.load_model(torch.device("cpu"))
    pixtral: Any = mistral.tokenizer.model
    pil = Image.open("examples/assets/image1.png").convert("RGB").resize((256, 256))

    def pixel_values(content: list[dict[str, Any]]) -> torch.Tensor:
        messages = [{"role": "user", "content": content}]
        inputs = pixtral.apply_chat_template(
            [messages], tokenize=True, return_dict=True, return_tensors="pt"
        )
        return inputs["pixel_values"]

    ours = pixel_values(mistral.format_prompt("x", [pil_to_tensor(pil)])[1]["content"])
    reference = pixel_values(
        [{"type": "image", "image": pil}, {"type": "text", "text": "x"}]
    )
    assert torch.equal(ours, reference), "Mistral3 image path differs from PIL input"
    print("[green]Mistral3: tensor images match the PIL path[/]")
