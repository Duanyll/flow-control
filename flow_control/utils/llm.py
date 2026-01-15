import asyncio
import base64
import io
import json
import re
from typing import Any, Literal, TypedDict

import aiohttp
import json5
import torch
from pydantic import BaseModel, PrivateAttr

from .common import tensor_to_pil
from .logging import get_logger

logger = get_logger(__name__)


Role = Literal["system", "user", "assistant"]


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrl(TypedDict):
    url: str


class ImageContent(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl


class Message(TypedDict):
    role: Role
    content: str | list[TextContent | ImageContent]


class LLMClient(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str
    timeout: int = 120
    model: str = "auto"
    max_tokens: int = 2048
    temperature: float = 1.0  # Recent LLMs work well with temperature=1.0
    max_concurrency: int = 0  # 0 means no limit

    _session: aiohttp.ClientSession | None = PrivateAttr(default=None)
    _semaphore: asyncio.Semaphore | None = PrivateAttr(default=None)
    _model_lock: asyncio.Lock | None = PrivateAttr(default=None)

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            # 显式创建一个 Connector
            # force_close=True: 每次请求后强制关闭连接，禁止复用
            # limit=self.max_concurrency: 限制连接池大小，配合 semaphore 避免过载
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrency if self.max_concurrency > 0 else 64,
                force_close=True,
            )

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            # 将 connector 传递给 session
            self._session = aiohttp.ClientSession(
                timeout=timeout, headers=headers, connector=connector
            )
        return self._session

    def _get_semaphore(self) -> asyncio.Semaphore | None:
        if self.max_concurrency <= 0:
            return None
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    def _get_model_lock(self) -> asyncio.Lock:
        if self._model_lock is None:
            self._model_lock = asyncio.Lock()
        return self._model_lock

    async def get_model_name(self) -> str:
        # Fast path: if model is already set, return immediately
        if self.model != "auto":
            return self.model

        # Slow path: need to fetch model from API with lock protection
        lock = self._get_model_lock()
        async with lock:
            # Double-check: another coroutine might have fetched it while we waited
            if self.model != "auto":
                return self.model

            # Fetch first available model from the API
            session = self._get_session()
            async with session.get(f"{self.base_url}/models") as resp:
                resp.raise_for_status()
                data = await resp.json()
                models = data.get("data", [])
                if not models:
                    raise ValueError("No models available from LLM API")
                model_name = models[0].get("id", "unknown-model")
                logger.info(f"Auto-selected model: {model_name}")
                if len(models) > 1:
                    logger.warning(
                        f"Multiple models available, using the first one: {model_name}"
                    )
                self.model = model_name
                return model_name

    def _tensor_to_base64url(self, tensor: torch.Tensor) -> str:
        pil_image = tensor_to_pil(tensor)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{img_b64}"

    async def generate(
        self,
        user_prompt: str,
        images: list[torch.Tensor] | None = None,
        system_prompt: str = "You are a helpful assistant.",
        context: list[Message] | None = None,
    ) -> tuple[str, list[Message]]:
        semaphore = self._get_semaphore()

        async def _generate_impl():
            session = self._get_session()
            model_name = await self.get_model_name()
            url = f"{self.base_url}/chat/completions"

            messages: list[Message] = []
            if context:
                messages.extend(context)
            else:
                messages.append({"role": "system", "content": system_prompt})

            if images:
                image_contents: list[ImageContent] = []
                for img in images:
                    img_url = self._tensor_to_base64url(img)
                    image_contents.append(
                        {"type": "image_url", "image_url": {"url": img_url}}
                    )
                text_contents: list[TextContent] = [
                    {"type": "text", "text": user_prompt}
                ]
                messages.append(
                    {"role": "user", "content": image_contents + text_contents}
                )
            else:
                messages.append({"role": "user", "content": user_prompt})

            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise ValueError("No choices returned from LLM API")
                message = choices[0].get("message", {})
                content = message.get("content", "")
                messages.append({"role": "assistant", "content": content})
                return content, messages

        if semaphore is not None:
            async with semaphore:
                return await _generate_impl()
        else:
            return await _generate_impl()

    def __del__(self):
        if self._session is None:
            return
        try:
            asyncio.get_running_loop().create_task(
                self._session.close()
            )
        except RuntimeError:
            asyncio.run(self._session.close())


def is_chinese_text(text: str) -> bool:
    """Check if the text contains Chinese characters."""
    return any("一" <= char <= "\u9fff" for char in text)


def _extract_last_json_block(text: str) -> str:
    """
    从文本中提取最后一个被 markdown 包围的 JSON 代码块。
    支持 ```json ... ``` 和 ``` ... ``` 格式。
    """
    # 正则表达式查找 markdown 代码块
    # (?:json)? 表示 "json" 这个词是可选的
    # \s* 匹配任何空白字符（包括换行符）
    # ([\s\S]*?) 非贪婪地匹配所有字符，直到下一个 ```
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text)

    if matches:
        # 如果找到，返回最后一个匹配项
        return matches[-1]

    # 如果没有找到 markdown 块，则返回原始文本
    return text


def _find_last_json_object(text: str) -> str:
    """
    从字符串末尾开始查找第一个完整匹配的 JSON 对象或数组。
    """
    text = text.strip()

    # 查找最后一个 '}' 或 ']'
    last_brace = text.rfind("}")
    last_bracket = text.rfind("]")

    if last_brace == -1 and last_bracket == -1:
        return ""

    last_end_char_index = max(last_brace, last_bracket)
    end_char = text[last_end_char_index]
    start_char = "{" if end_char == "}" else "["

    level = 0
    # 从后向前遍历，寻找匹配的起始符号
    for i in range(last_end_char_index, -1, -1):
        char = text[i]
        if char == end_char:
            level += 1
        elif char == start_char:
            level -= 1
            if level == 0:
                # 找到了匹配的起始位置
                return text[i : last_end_char_index + 1]

    return ""  # 没有找到完整的 JSON 对象


def parse_llm_json_output(llm_output: str) -> Any:
    """
    从 LLM 的输出中稳健地解析最后一个 JSON 对象。

    处理逻辑:
    1. 优先尝试从 Markdown 代码块 (```json ... ```) 中提取 JSON。
    2. 如果没有代码块，则从整个字符串的末尾寻找最后一个完整的 JSON 对象/数组。
    3. 使用更宽松的 `json5` 库进行解析，以处理注释、末尾逗号等情况。
    4. 如果解析失败，则返回 None。

    :param llm_output: LLM 返回的可能包含 JSON 的字符串。
    :return: 解析后的 Python 字典或列表，如果失败则返回 None。
    """
    # 1. 优先从 markdown 块提取
    potential_json_str = _extract_last_json_block(llm_output)

    try:
        # 尝试直接用标准 json 库解析
        return json5.loads(potential_json_str)  # type: ignore
    except json.JSONDecodeError:
        pass  # 如果失败，继续后续步骤

    # 2. 从提取的字符串（或原始字符串）中定位最后一个 JSON 对象
    json_str = _find_last_json_object(potential_json_str)

    if not json_str:
        # 如果 markdown 提取和定位都失败了，最后尝试对原始文本进行一次定位
        json_str = _find_last_json_object(llm_output)
        if not json_str:
            return None

    # 3. 使用 json5 进行解析
    try:
        # 使用 json5.loads，它可以处理注释、末尾逗号等
        return json5.loads(json_str)  # type: ignore
    except Exception as e:
        print(f"Failed to parse JSON with json5. Error: {e}")
        # 可以选择在这里增加一个使用标准库 json 的回退尝试
        try:
            # 去掉常见的注释（一个简单的实现）
            no_comments = re.sub(r"//.*", "", json_str)
            no_comments = re.sub(r"/\*[\s\S]*?\*/", "", no_comments, flags=re.MULTILINE)
            return json.loads(no_comments)
        except json.JSONDecodeError as final_e:
            print(
                f"Failed to parse with standard json library after cleaning. Error: {final_e}"
            )
            return None
