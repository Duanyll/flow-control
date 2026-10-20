from typing import Annotated, Any

import torch

from flow_control.utils.hf_model import HfModelLoader
from flow_control.utils.logging import get_logger, warn_once
from flow_control.utils.registry import Registry, RegistryUnion

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


Encoder = Annotated[BaseEncoder, RegistryUnion(encoder_registry, "type")]
