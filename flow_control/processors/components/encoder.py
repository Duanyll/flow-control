import math
import re
from typing import Annotated, Any, Literal, cast

import torch
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
from PIL import Image
from transformers import (
    BatchEncoding,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    Mistral3ForConditionalGeneration,
    PixtralProcessor,
    PreTrainedTokenizerBase,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
    Qwen3ForCausalLM,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLProcessor,
    T5EncoderModel,
    T5Tokenizer,
)

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger, warn_once
from flow_control.utils.registry import Registry, RegistryUnion
from flow_control.utils.resize import resize_to_multiple_of, resize_to_resolution
from flow_control.utils.tensor import remove_alpha_channel, tensor_to_pil
from flow_control.utils.types import TorchDType

logger = get_logger(__name__)


class BaseEncoder[T](HfModelLoader[T]):
    chat_template: str = "{user}"
    image_template: str = ""

    def _format_prompt(
        self, prompt: str, images: Any | None = None, system_prompt: str | None = None
    ) -> Any:
        user_prompt = ""
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


encoder_registry: Registry[BaseEncoder] = Registry("encoder", base=BaseEncoder)


@encoder_registry.register("qwen21")
class QwenImage21Encoder(BaseEncoder[Qwen3VLForConditionalGeneration]):
    type: Literal["qwen21"] = "qwen21"
    library: Literal["transformers"] = "transformers"
    class_name: str = "Qwen3VLForConditionalGeneration"
    pretrained_model_id: str = "Qwen/Qwen-Image-2.1"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16
    processor: HfModelLoader[Qwen3VLProcessor] = HfModelLoader(
        library="transformers",
        class_name="Qwen3VLProcessor",
        pretrained_model_id="Qwen/Qwen-Image-2.1",
        subfolder="processor",
        # Transformers 5.17 AutoTokenizer otherwise requires config.json in
        # this tokenizer-only subfolder before reading tokenizer_config.json.
        extra_from_pretrained_kwargs={"tokenizer_type": "qwen2"},
    )

    def load_model(self, device: torch.device, frozen: bool = True) -> bool:
        self.processor.load_model(device)
        return super().load_model(device, frozen)

    def encode_condition(
        self, prompt: str, images: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        # Reuse the checkpoint's raw template, image-slot masks, and pre-final-
        # RMSNorm extraction. This lightweight pipeline loads no extra models.
        pipe = QwenImage21Pipeline(
            text_encoder=self.model,
            processor=self.processor.model,
            # Diffusers accepts absent components at runtime; annotations omit None.
            vae=None,  # ty: ignore[invalid-argument-type]
            transformer=None,  # ty: ignore[invalid-argument-type]
            scheduler=None,
        )
        return pipe.encode_prompt(
            prompt,
            image=[tensor_to_pil(img) for img in images] if images else None,
            device=self.model.device,
        )


def warn_no_image_support(func):
    def wrapper(self, prompt, images=None, system_prompt=None):
        if images:
            warn_once(
                logger,
                f"{self.__class__.__name__} does not support image inputs. Ignoring provided images.",
            )
        return func(self, prompt, images=None, system_prompt=system_prompt)

    return wrapper


@encoder_registry.register("cosmos3")
class Cosmos3Encoder(BaseEncoder[PreTrainedTokenizerBase]):
    """Tokenizer-only "encoder" for Cosmos3.

    Cosmos3 is a Mixture-of-Transformers with no separate text encoder: token
    IDs go straight into the transformer, whose understanding tower does the
    encoding. So this component only applies the chat template and returns IDs,
    replicating ``Cosmos3OmniPipeline.tokenize_prompt`` (that method cannot be
    reused directly -- constructing the pipeline requires a VAE).

    One deliberate deviation: upstream also appends ``"This image is of HxW
    resolution."`` (``add_resolution_template``, default on). ``encode`` has no
    per-row size to fill in, and the JSON-upsampled captions the model expects
    already carry a ``resolution`` field, so the sentence is dropped.

    ``encode`` returns an ``int64`` ``[1, L]`` tensor, not embeddings.
    """

    type: Literal["cosmos3"] = "cosmos3"
    library: Literal["transformers"] = "transformers"
    # AutoTokenizer (transformers 5.x) first looks up a config.json, which this
    # tokenizer-only subfolder lacks, and fails under HF_HUB_OFFLINE=1. The fast
    # base class loads tokenizer.json as is, matching AutoTokenizer on Nano,
    # Edge and Super; Qwen2Tokenizer would rebuild Edge's pre-tokenizer.
    class_name: str = "PreTrainedTokenizerFast"
    pretrained_model_id: str = "nvidia/Cosmos3-Nano"
    subfolder: str | None = "text_tokenizer"

    default_system_prompt: str = (
        # Upstream string, typo included; the checkpoints were trained with it.
        "You are a helpful assistant who will generate images from a give prompt."
    )
    use_system_prompt: bool = True
    """Cosmos3-Edge's ``model_index.json`` sets ``default_use_system_prompt`` to
    false; Nano and Super default to true."""
    start_of_generation_token: str = "<|vision_start|>"

    @warn_no_image_support
    def encode(
        self,
        prompt: str,
        images: list[torch.Tensor] | None = None,
        system_prompt: str | None = None,
    ) -> torch.Tensor:
        conversations = []
        if self.use_system_prompt:
            conversations.append(
                {
                    "role": "system",
                    "content": system_prompt or self.default_system_prompt,
                }
            )
        conversations.append({"role": "user", "content": prompt})
        # `return_dict=True` always yields a BatchEncoding; the annotation is a
        # union over every `tokenize`/`return_dict` combination.
        encoded = cast(
            BatchEncoding,
            self.model.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                add_vision_id=False,
                return_dict=True,
            ),
        )
        input_ids = list(encoded.input_ids) + [
            self.model.eos_token_id,
            self.model.convert_tokens_to_ids(self.start_of_generation_token),
        ]
        return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)


class GenerativeEncoder:
    def generate(
        self,
        prompt: str,
        images: list[torch.Tensor] | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        raise NotImplementedError("Generate method must be implemented by subclasses.")


@encoder_registry.register("t5")
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

    max_length: int | None = None
    """Cap the padded sequence length. When ``None`` the tokenizer's own
    ``model_max_length`` is used (FLUX behaviour). SD3 sets this to 256."""

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        fresh = super().load_model(device, frozen)
        # Transformers 5.17 made SDPA the default for T5. In bf16 it moves the
        # FLUX/SD3 T5-XXL embeddings by up to 20% (relative L2) and lands further
        # from an fp32 run than eager does; eager matches transformers <= 5.16
        # bitwise, which every existing T5 embedding was computed with.
        self.model.set_attn_implementation("eager")
        return fresh

    @warn_no_image_support
    def encode(self, prompt, images=None, system_prompt: str | None = None):
        tokenizer = self.tokenizer.model
        model = self.model

        prompt = self._format_prompt(prompt, images, system_prompt)

        t5_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.max_length,
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


@encoder_registry.register("clip")
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


@encoder_registry.register("sd3_clip")
class Sd3ClipEncoder(BaseEncoder[CLIPTextModelWithProjection]):
    """CLIP text encoder for SD3-family models.

    Unlike :class:`ClipTextEncoder` (a plain ``CLIPTextModel`` returning only
    ``pooler_output``), SD3 needs **both** the penultimate hidden states (used as
    part of the sequence conditioning) and the **projected** pooled embedding
    (``text_embeds``) from a ``CLIPTextModelWithProjection``.
    :meth:`encode_seq_pooled` returns both; :meth:`encode` returns just the sequence
    part to satisfy the single-tensor ``Encoder`` interface.
    """

    type: Literal["sd3_clip"] = "sd3_clip"

    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "CLIPTextModelWithProjection"
    pretrained_model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[CLIPTokenizer] = HfModelLoader(
        library="transformers",
        class_name="CLIPTokenizer",
        pretrained_model_id="stabilityai/stable-diffusion-3.5-medium",
        subfolder="tokenizer",
    )

    max_length: int = 77
    hidden_state_layer: int = -2

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        return super().load_model(device, frozen)

    @warn_no_image_support
    def encode_seq_pooled(
        self, prompt, images=None, system_prompt: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        outputs = model(
            clip_inputs.input_ids.to(model.device), output_hidden_states=True
        )
        seq = outputs.hidden_states[self.hidden_state_layer]
        pooled = outputs.text_embeds
        return seq, pooled

    def encode(self, prompt, images=None, system_prompt: str | None = None):
        return self.encode_seq_pooled(prompt, images, system_prompt)[0]


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


@encoder_registry.register("qwen3")
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


@encoder_registry.register("qwen3vl")
class Qwen3VLEncoder(BaseEncoder[Qwen3VLModel]):
    """Krea 2 text encoder.

    Krea 2 taps 12 intermediate ``Qwen3-VL`` decoder layers and **stacks** them into a
    4D ``(B, seq, num_layers, hidden)`` tensor (NOT concatenated along the feature dim
    like :class:`Qwen3Encoder`); the transformer's internal ``Krea2TextFusion`` collapses
    the layer axis. ``encode_seq_mask`` also returns the bool attention mask, since Krea
    keeps padding tokens in the sequence and passes the mask to the transformer.

    This replicates ``Krea2Pipeline.get_text_hidden_states`` exactly: the Qwen-Image chat
    template with a fixed describe-image system prompt, mid-template padding
    (``[prefix | prompt | PAD | suffix]``), cumulative-valid-token mRoPE positions (so the
    suffix keeps its trained phase), and dropping the system-prefix tokens from the output.
    """

    type: Literal["qwen3vl"] = "qwen3vl"
    library: Literal["diffusers", "transformers"] = "transformers"
    class_name: str = "Qwen3VLModel"
    pretrained_model_id: str = "krea/Krea-2-Raw"
    subfolder: str | None = "text_encoder"
    dtype: TorchDType = torch.bfloat16

    tokenizer: HfModelLoader[Qwen2Tokenizer] = HfModelLoader(
        library="transformers",
        class_name="Qwen2Tokenizer",
        pretrained_model_id="krea/Krea-2-Raw",
        subfolder="tokenizer",
    )

    max_sequence_length: int = 512
    select_layers: list[int] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    prompt_template_encode_prefix: str = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n"
    )
    prompt_template_encode_suffix: str = "<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx: int = 34
    prompt_template_encode_num_suffix_tokens: int = 5

    def load_model(self, device, frozen: bool = True):
        self.tokenizer.load_model(device)
        return super().load_model(device, frozen)

    def encode_seq_mask(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(hidden_states, attention_mask)`` of shapes
        ``(1, seq, num_layers, hidden)`` and ``(1, seq)`` (bool)."""
        tokenizer = self.tokenizer.model
        model = self.model
        device = model.device
        prefix_idx = self.prompt_template_encode_start_idx

        text = [self.prompt_template_encode_prefix + prompt]
        text_tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_sequence_length
            + prefix_idx
            - self.prompt_template_encode_num_suffix_tokens,
            return_tensors="pt",
        ).to(device)
        suffix_tokens = tokenizer(
            [self.prompt_template_encode_suffix], return_tensors="pt"
        ).to(device)

        input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
        attention_mask = torch.cat(
            [text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1
        ).bool()

        # Krea 2 pads in the middle of the template ([prefix | prompt | PAD | suffix]), so
        # positions must count only real tokens; broadcast across the 3 mRoPE axes
        # (T/H/W are equal for text) as Qwen3-VL expects position_ids of shape (3, B, N).
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = torch.stack(
            [outputs.hidden_states[i] for i in self.select_layers], dim=2
        )

        # Drop the system-prefix tokens from both the features and the mask.
        hidden_states = hidden_states[:, prefix_idx:]
        attention_mask = attention_mask[:, prefix_idx:]
        return hidden_states, attention_mask

    @warn_no_image_support
    def encode(self, prompt, images=None, system_prompt=None):
        return self.encode_seq_mask(prompt)[0]


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


@encoder_registry.register("hidream_o1")
class HiDreamO1Encoder(BaseEncoder[Qwen3VLProcessor]):
    """Tokenizer-only "encoder" for HiDream-O1 (pixel-space unified transformer).

    HiDream-O1 has no separate text encoder: prompt token ids are consumed by the
    same trainable transformer that denoises pixels, so encoding a prompt is pure
    CPU tokenization. ``self.model`` is the HF ``Qwen3VLProcessor`` (tokenizer +
    image processor); no GPU model is ever loaded.

    - :meth:`encode_ids` reproduces the official ``build_t2i_text_sample`` text
      part: ``chat_template(prompt) + <|boi_token|> + <|tms_token|>*n``. The
      trailing target-image vision tokens are resolution-dependent and appended
      by the adapter at forward time.
    - :meth:`encode_ids_with_images` reproduces the editing/personalization
      conditioning: reference thumbnails go through the model's own SigLIP tower
      via ``pixel_values``/``image_grid_thw`` with image placeholders in the chat
      template (official ``CONDITION_IMAGE_SIZE`` K-count heuristic).
    """

    type: Literal["hidream_o1"] = "hidream_o1"
    library: Literal["diffusers", "transformers", "custom"] = "transformers"
    class_name: str = "Qwen3VLProcessor"
    pretrained_model_id: str = "HiDream-ai/HiDream-O1-Image"
    subfolder: str | None = None
    dtype: TorchDType = torch.bfloat16

    boi_token: str = "<|boi_token|>"
    tms_token: str = "<|tms_token|>"
    num_timestep_tokens: int = 1
    condition_image_size: int = 384
    """Base thumbnail size for the SigLIP condition path; scaled down with the
    reference count K like the official pipeline (K<=4: 384, K<=8: 288, else 192)."""

    def _apply_chat_template(
        self, content: Any, system_prompt: str | None = None
    ) -> str:
        processor: Any = self.model
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        template = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return template + self.boi_token + self.tms_token * self.num_timestep_tokens

    def encode_ids(self, prompt: str, system_prompt: str | None = None) -> torch.Tensor:
        """Tokenize a text-only prompt; returns ``input_ids`` of shape ``[1, L]``."""
        template = self._apply_chat_template(prompt, system_prompt)
        tokenizer: Any = self.model.tokenizer
        return tokenizer.encode(template, return_tensors="pt", add_special_tokens=False)

    def _thumbnail_size(self, num_images: int) -> int:
        if num_images <= 4:
            return self.condition_image_size
        if num_images <= 8:
            return self.condition_image_size * 48 // 64
        return self.condition_image_size // 2

    def encode_ids_with_images(
        self,
        prompt: str,
        images: list[torch.Tensor],
        system_prompt: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a prompt with K reference-image placeholders and preprocess
        the SigLIP-condition thumbnails.

        Returns ``{"input_ids": [1, L], "pixel_values", "image_grid_thw"}``.
        Input images are BCHW in ``[0, 1]``; ``do_rescale=False`` avoids the
        processor's unconditional ``1/255`` rescale (it would double-rescale).
        """
        from flow_control.third_party.hidream_o1 import calculate_dimensions

        cond_size = self._thumbnail_size(len(images))
        thumbnails = []
        for image in images:
            image = remove_alpha_channel(image)
            width, height = calculate_dimensions(
                cond_size, image.shape[3] / image.shape[2]
            )
            thumbnails.append(resize_to_resolution(image, (height, width)))

        content: list[dict[str, Any]] = [{"type": "image"} for _ in thumbnails]
        content.append({"type": "text", "text": prompt})
        template = self._apply_chat_template(content, system_prompt)

        processor: Any = self.model
        proc = processor(
            text=[template],
            images=thumbnails,
            text_kwargs={"padding": "longest", "return_tensors": "pt"},
            images_kwargs={"do_rescale": False, "return_tensors": "pt"},
        )
        return {
            "input_ids": proc.input_ids,
            "pixel_values": proc.pixel_values,
            "image_grid_thw": proc.image_grid_thw,
        }

    def encode(self, prompt, images=None, system_prompt=None):
        if images:
            return self.encode_ids_with_images(prompt, images, system_prompt)[
                "input_ids"
            ]
        return self.encode_ids(prompt, system_prompt)


Encoder = Annotated[BaseEncoder, RegistryUnion(encoder_registry, "type")]


if __name__ == "__main__":
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        QwenImageEditPlusPipeline,
        calculate_dimensions,
    )
    from rich import print
    from transformers import AttentionInterface

    from flow_control.processors.components.prompts import parse_prompt
    from flow_control.utils import device as devutil
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

    # T5 (FLUX.1, SD3): bitwise equal to the attention math of transformers
    # <= 5.16 (bf16 scores + position bias, fp32 softmax).
    def legacy_t5_attention(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: None,  # unregistered with the mask interface
        position_bias: torch.Tensor,
        **kwargs,
    ):
        scores = torch.matmul(query, key.transpose(3, 2))
        scores += position_bias
        weights = torch.softmax(scores.float(), dim=-1).type_as(scores)
        return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights

    AttentionInterface.register("t5_legacy", legacy_t5_attention)
    t5 = T5TextEncoder()
    t5.load_model(devutil.default_device())
    prompt = 'A neon sign that says "OPEN 24/7" above a rainy street'
    ours = t5.encode(prompt)
    t5.model.set_attn_implementation("t5_legacy")
    reference = t5.encode(prompt)
    assert torch.equal(ours, reference), "T5 no longer matches transformers <= 5.16"
    print("[green]T5: bitwise equal to the transformers <= 5.16 attention[/]")
    t5.unload_model()

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
