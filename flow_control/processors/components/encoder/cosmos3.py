from typing import Literal, cast

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from .base import BaseEncoder, encoder_registry, warn_no_image_support


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
