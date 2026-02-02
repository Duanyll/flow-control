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

EFFICIENT_LAYERED_DETECTION_CN = r"""
请分析提供的图片，识别并定位所有前景中的可见对象及设计元素，并将结果输出为指定的 JSON 格式。识别每一个显著的前景元素，并为每个元素提供以下信息：
- 2D边界框（bbox_2d）：该对象的二维边界框坐标。
- 标签（label）：一句描述该对象外观特征（颜色、形状、材质等）的句子。如果对象中包含可识别的文字，必须先描述文字的视觉样式（字体、颜色、特效），然后在一个双引号内完整写出文字内容
最后，用一句话描述排除了前景对象后的背景环境或底色。

输出格式示例：

```
{
    "foreground": [
        { 
            "bbox_2d": [0, 54, 473, 999], 
            "label": "一个绿色的礼物盒，顶部系着红色的蝴蝶结丝带。" 
        },
        { 
            "bbox_2d": [218, 53, 478, 567], 
            "label": "带有黑色轮廓和细微阴影效果的白色粗体文字，用现代无衬线字体展示了短语 \"Time For The\"。" 
        }
    ],
    "background": "纯深蓝色背景，带有微妙的渐变效果，从边缘的深蓝向中心略微变浅。"
}
```
"""

EFFICIENT_LAYERED_DETECTION_EN = r"""
Please analyze the provided image to identify and locate all visible objects and design elements in the foreground, outputting the results in the specified JSON format. For each prominent foreground element, provide the following information:
- 2D Bounding Box (bbox_2d): The 2D bounding box coordinates of the object.
- Label (label): A sentence describing the object's visual characteristics (color, shape, material, etc.). If the object contains recognizable text, first describe the visual style of the text (font, color, effects), then write the text content verbatim within "double quotes".
Finally, provide a one-sentence description of the background environment or base color, excluding the foreground objects.

Output format example:

```
{
    "foreground": [
        { 
            "bbox_2d": [0, 54, 473, 999], 
            "label": "A green gift box with a red ribbon tied in a bow on top." 
        },
        { 
            "bbox_2d": [218, 53, 478, 567], 
            "label": "bold, white text with a black outline and a subtle shadow effect, displaying the phrase \"Time For The\" in a modern sans-serif font." 
        }
    ],
    "background": "a solid dark blue background with a subtle gradient effect, transitioning slightly from a deeper blue at the edges to a slightly lighter blue towards the center."
}
```
"""

PROMPTS: dict[str, str] = {
    "qwen_image_encoder": "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:",
    "qwen_image_edit_encoder": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
    "qwen_layered_caption_cn": QWEN_LAYERED_CAPTION_CN,
    "qwen_layered_caption_en": QWEN_LAYERED_CAPTION_EN,
    "efficient_layered_caption_fg_cn": "请你给我给出的图片生成一句话的描述。你要描述的图片是从平面设计作品中提取出的部分设计元素，你只用关注图片的前景部分，不要描述背景。如果图片中包含文字，你必须先描述文字的样式，再用双引号完整地给出图片中的文字内容。直接输出最终结果，不要加额外的解释。",
    "efficient_layered_caption_fg_en": 'Task: Describe the image in exactly one sentence. Context: The image is a specific design element extracted from a larger graphic design work. Requirements: 1. Focus exclusively on the foreground, do not describe the background. 2. If text is present, first describe the style of the text, then include the text content verbatim inside "double quotes". 3. Output ONLY the description string. Do not include introductory or concluding remarks.',
    "efficient_layered_caption_bg_cn": "请你给我给出的图片生成一句话的描述。你要描述的图片是从平面设计作品中提取出的背景部分，它可能是纯色背景，也可能有一些图案。直接输出最终结果，不要加额外的解释。",
    "efficient_layered_caption_bg_en": "Task: Describe the image in exactly one sentence. Context: The image is the background layer extracted from a graphic design work. Requirements: 1. Analyze the visual style, noting whether it is a solid color, a gradient, a texture, or contains specific patterns. 2. Output ONLY the description string. Do not include introductory or concluding remarks.",
    "efficient_layered_detection_cn": EFFICIENT_LAYERED_DETECTION_CN,
    "efficient_layered_detection_en": EFFICIENT_LAYERED_DETECTION_EN,
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
