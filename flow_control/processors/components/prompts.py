from typing import Annotated, Any

from pydantic import BeforeValidator

QWEN_LAYERED_CAPTION_CN = """
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

QWEN_LAYERED_CAPTION_EN = """
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

PROMPTS: dict[str, str] = {
    "qwen_image_encoder": "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:",
    "qwen_image_edit_encoder": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
    "qwen_layered_caption_cn": QWEN_LAYERED_CAPTION_CN,
    "qwen_layered_caption_en": QWEN_LAYERED_CAPTION_EN,
    "efficient_layered_caption_fg_cn": "请你给我给出的图片生成一句话的描述。你要描述的图片是从平面设计作品中提取出的部分设计元素，你只用关注图片的前景部分。如果图片中包含文字，你必须先描述文字的样式，再用双引号完整地给出图片中的文字内容。直接输出最终结果，不要加额外的解释。",
    "efficient_layered_caption_fg_en": 'Task: Describe the image in exactly one sentence. Context: The image is a specific design element extracted from a larger graphic design work. Requirements: 1. Focus exclusively on the foreground. 2. If text is present, first describe the style of the text, then include the text content verbatim inside "double quotes". 3. Output ONLY the description string. Do not include introductory or concluding remarks.',
    "efficient_layered_caption_bg_cn": "请你给我给出的图片生成一句话的描述。你要描述的图片是从平面设计作品中提取出的背景部分，它可能是纯色背景，也可能有一些图案。直接输出最终结果，不要加额外的解释。",
    "efficient_layered_caption_bg_en": "Task: Describe the image in exactly one sentence. Context: The image is the background layer extracted from a graphic design work. Requirements: 1. Analyze the visual style, noting whether it is a solid color, a gradient, a texture, or contains specific patterns. 2. Output ONLY the description string. Do not include introductory or concluding remarks.",
    "longcat_image_encoder": "As an image captioning expert, generate a descriptive text prompt based on an image content, suitable for input to a text-to-image model.",
    "longcat_image_edit_encoder": "As an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).",
    "default_t2i_caption": "As an image captioning expert, generate a descriptive text prompt based on an image content, suitable for input to a text-to-image model. Output ONLY the description string. Do not include introductory or concluding remarks.",
}


def parse_prompt(input: Any):
    if not isinstance(input, str):
        raise ValueError("Prompt must be a string.")
    if input.startswith("@"):
        key = input[1:].lower()
        if key in PROMPTS:
            return PROMPTS[key]
        else:
            raise ValueError(f"Unknown prompt template key: {key}")
    return input


PromptStr = Annotated[str, BeforeValidator(parse_prompt)]
