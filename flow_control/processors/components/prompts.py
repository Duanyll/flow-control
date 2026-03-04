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
请分析提供的图片，首先输出一段文本，描述图片中的场景，然后再识别并定位所有前景中的可见对象及设计元素，并将结果输出为指定的 JSON 格式。确保你识别出的元素没有重复，没有遗漏，如果某元素是另一个元素的一部份，你只需要识别整体而不需要识别单独的部分。比如，如果有一个人，则整个人是一个对象，无需再把人的头和手作为单独的对象；如果有一群人，则每个人都是独立的对象。

为每个元素提供以下信息：

- 2D边界框（bbox_2d）：该对象的二维边界框坐标。
- 标签（label）：一句描述该对象外观特征（颜色、形状、材质等）的句子。如果对象中包含可识别的文字，必须先描述文字的视觉样式（字体、颜色、特效），然后在一个双引号内完整写出文字内容

最后，用一句话描述排除了前景对象后的背景环境或底色。

请先输出对场景的描述，再把 JSON 放在代码块里。JSON 输出格式示例：

```
{
    "foreground": [
        { "bbox_2d": [0, 54, 473, 999], "label": "一个绿色的礼物盒，顶部系着红色的蝴蝶结丝带。" },
        { "bbox_2d": [218, 53, 478, 567], "label": "带有黑色轮廓和细微阴影效果的白色粗体文字，用现代无衬线字体展示了短语 \"Time For The\"。" }
    ],
    "background": "纯深蓝色背景，带有微妙的渐变效果，从边缘的深蓝向中心略微变浅。"
}
```
"""

EFFICIENT_LAYERED_DETECTION_EN = r"""
Please analyze the provided image, first output a paragraph of text describing the scene in the image, then identify and locate all visible objects and design elements in the foreground, and output the results in the specified JSON format. Ensure that the elements you identify are not duplicated or omitted. If an element is part of another element, you only need to identify the whole without identifying the individual parts. For example, if there is a person, the whole person is one object, and there is no need to identify the person's head and hands as separate objects; if there is a group of people, each person is an independent object.

For each element, provide the following information:

- 2D Bounding Box (bbox_2d): The 2D bounding box coordinates of the object.
- Label (label): A sentence describing the visual features (color, shape, material, etc.) of the object. If the object contains recognizable text, you must first describe the visual style of the text (font, color, special effects), and then write the text content in full within double quotes.

Finally, provide a one-sentence description of the background environment or base color after excluding the foreground objects.

Please first output the description of the scene, then put the JSON in a code block. Example of JSON output format:

```
{
    "foreground": [
        { "bbox_2d": [0, 54, 473, 999], "label": "A green gift box with a red bow ribbon on top." },
        { "bbox_2d": [218, 53, 478, 567], "label": "White bold text with black outline and subtle shadow effect, displaying the phrase \"Time For The\" in a modern sans-serif font." }
    ],
    "background": "Solid deep blue background with a subtle gradient effect, slightly lighter towards the center from the edges."
}
```
"""

LONGCAT_T2I_ENHANCE_EN = """
You are a prompt engineering expert for text-to-image models. Since text-to-image models have limited capabilities in
understanding user prompts, you need to identify the core theme and intent of the user's input and improve the model's
understanding accuracy and generation quality through optimization and rewriting. The rewrite must strictly retain all
information from the user's original prompt without deleting or distorting any details. Specific requirements are as
follows:
1. The rewrite must not affect any information expressed in the user's original prompt; the rewritten prompt should use
   coherent natural language, avoid low-information redundant descriptions, and keep the rewritten prompt length as
   concise as possible.
2. Ensure consistency between input and output languages: Chinese input yields Chinese output, and English input yields
   English output. The rewritten token count should not exceed 512.
3. The rewritten description should further refine subject characteristics and aesthetic techniques appearing in the
   original prompt, such as lighting and textures.
4. If the original prompt does not specify an image style, ensure the rewritten prompt uses a **realistic photography
   style**. If the user specifies a style, retain the user's style.
5. When the original prompt requires reasoning to clarify user intent, use logical reasoning based on world knowledge
   to convert vague abstract descriptions into specific tangible objects (e.g., convert "the tallest animal" to "a
   giraffe").
6. When the original prompt requires text generation, please use double quotes to enclose the text part (e.g., `"50%
   OFF"`).
7. When the original prompt requires generating text-heavy scenes like webpages, logos, UIs, or posters, and no
   specific text content is specified, you need to infer appropriate text content and enclose it in double quotes. For
   example, if the user inputs: "A tourism flyer with a grassland theme," it should be rewritten as: "A tourism flyer
   with the image title 'Grassland'."
8. When negative words exist in the original prompt, ensure the rewritten prompt does not contain negative words. For
   example, "a lakeside without boats" should be rewritten such that the word "boat" does not appear at all.
9. Except for text content explicitly requested by the user, **adding any extra text content is prohibited**.
Here are examples of rewrites for different types of prompts: # Examples (Few-Shot Learning)
  1. User Input: An animal with nine lives.
    Rewrite Output: A cat bathed in soft sunlight, its fur soft and glossy. The background is a comfortable home
    environment with light from the window filtering through curtains, creating a warm light and shadow effect. The
    shot uses a medium distance perspective to highlight the cat's leisurely and stretched posture. Light cleverly hits
    the cat's face, emphasizing its spirited eyes and delicate whiskers, adding depth and affinity to the image.
  2. User Input: Create an anime-style tourism flyer with a grassland theme.
    Rewrite Output: In the lower right of the center, a short-haired girl sits sideways on a gray, irregularly shaped
    rock. She wears a white short-sleeved dress and brown flat shoes, holding a bunch of small white flowers in her
    left hand, smiling with her legs hanging naturally. The girl has dark brown shoulder-length hair with bangs
    covering her forehead, brown eyes, and a slightly open mouth. The rock surface has textures of varying depths. To
    the girl's left and front is lush grass, with long, yellow-green blades, some glowing golden in the sunlight. The
    grass extends into the distance, forming rolling green hills that fade in color as they recede. The sky occupies
    the upper half of the picture, pale blue dotted with a few fluffy white clouds. In the upper left corner, there is
    a line of text in italic, dark green font reading "Explore Nature's Peace". Colors are dominated by green, blue,
    and yellow, fluid lines, and distinct light and shadow contrast, creating a quiet and comfortable atmosphere.
  3. User Input: A Christmas sale poster with a red background, promoting a Buy 1 Get 1 Free milk tea offer.
    Rewrite Output: The poster features an overall red tone, embellished with white snowflake patterns on the top and
    left side. The upper right features a bunch of holly leaves with red berries and a pine cone. In the upper center,
    golden 3D text reads "Christmas Heartwarming Feedback" centered, along with red bold text "Buy 1 Get 1". Below, two
    transparent cups filled with bubble tea are placed side by side; the tea is light brown with dark brown pearls
    scattered at the bottom and middle. Below the cups, white snow piles up, decorated with pine branches, red berries,
    and pine cones. A blurry Christmas tree is faintly visible in the lower right corner. The image has high clarity,
    accurate text content, a unified design style, a prominent Christmas theme, and a reasonable layout, providing
    strong visual appeal.
  4. User Input: A woman indoors shot in natural light, smiling with arms crossed, showing a relaxed and confident
     posture.
    Rewrite Output: The image features a young Asian woman with long dark brown hair naturally falling over her
    shoulders, with some strands illuminated by light, showing a soft sheen. Her features are delicate, with long
    eyebrows, bright and spirited dark brown eyes looking directly at the camera, revealing peace and confidence. She
    has a high nose bridge, full lips with nude lipstick, and corners of the mouth slightly raised in a faint smile.
    Her skin is fair, with cheeks and collarbones illuminated by warm light, showing a healthy ruddiness. She wears a
    black spaghetti strap tank top revealing graceful collarbone lines, and a thin gold necklace with small beads and
    metal bars glinting in the light. Her outer layer is a beige knitted cardigan, soft in texture with visible
    knitting patterns on the sleeves. Her arms are crossed over her chest, hands covered by the cardigan sleeves, in a
    relaxed posture. The background is a pure dark brown without extra decoration, making the figure the absolute
    focus. The figure is located in the center of the frame. Light enters from the upper right, creating bright spots
    on her left cheek, neck, and collarbone, while the right side is slightly shadowed, creating a three-dimensional
    and soft tone. Image details are clear, showcasing skin texture, hair, and clothing materials well. Colors are
    dominated by warm tones, with the combination of beige and dark brown creating a warm and comfortable atmosphere.
    The overall style is natural, elegant, and artistic.
  5. User Input: Create a series of images showing the growth process of an apple from seed to fruit. The series should
     include four stages: 1. Sowing, 2. Seedling growth, 3. Plant maturity, 4. Fruit harvesting.
    Rewrite Output: A 4-panel exquisite illustration depicting the growth process of an apple, capturing each stage
    precisely and clearly. 1. "Sowing": A close-up shot of a hand gently placing a small apple seed into fertile dark
    soil, with visible soil texture and the seed's smooth surface. The background is a soft-focus garden dotted with
    green leaves and sunlight filtering through. 2. "Seedling Growth": A young apple sapling breaks through the soil,
    stretching tender green leaves toward the sky. The scene is set in a vibrant garden illuminated by warm golden
    light, highlighting the seedling's delicate structure. 3. "Plant Maturity": A mature apple tree, lush with branches
    and leaves, covered in tender green foliage and developing small apples. The background is a vibrant orchard under
    a clear blue sky, with dappled sunlight creating a peaceful atmosphere. 4. "Fruit Harvesting": A hand reaches into
    the tree to pick a ripe red apple, its smooth skin glistening in the sun. The scene shows the abundance of the
    orchard, with baskets of apples in the background, giving a sense of fulfillment. Each illustration uses a
    realistic style, focusing on details and harmonious colors to showcase the natural beauty and development of the
    apple's life cycle.
  6. User Input: If 1 represents red, 2 represents green, 3 represents purple, and 4 represents yellow, please generate
     a four-color rainbow based on this rule. The color order from top to bottom is 3142.
    Rewrite Output: The image consists of four horizontally arranged colored stripes, ordered from top to bottom as
    purple, red, yellow, and green. A white number is centered on each stripe. The top purple stripe features the
    number "3", the red stripe below it has the number "1", the yellow stripe further down has the number "4", and the
    bottom green stripe has the number "2". All numbers use a sans-serif font in pure white, forming a sharp contrast
    with the background colors to ensure good readability. The stripes have high color saturation and a slight texture.
    The overall layout is simple and clear, with distinct visual effects and no extra decorative elements, emphasizing
    the numerical information. The image is high definition, with accurate colors and a consistent style, offering
    strong visual appeal.
  7. User Input: A stone tablet carved with "Guan Guan Ju Jiu, On the River Isle", natural light, background is a
     Chinese garden.
    Rewrite Output: An ancient stone tablet carved with "Guan Guan Ju Jiu, On the River Isle", the surface covered with
    traces of time, the writing clear and deep. Natural light falls from above, softly illuminating every detail of the
    stone tablet and enhancing its sense of history. The background is an elegant Chinese garden featuring lush bamboo
    forests, winding paths, and quiet pools, creating a serene and distant atmosphere. The overall picture uses a
    realistic style with rich details and natural light and shadow effects, highlighting the cultural heritage of the
    stone tablet and the classical beauty of the garden.
# Output Format Please directly output the rewritten and optimized Prompt content. Do not include any explanatory
language or JSON formatting, and do not add opening or closing quotes yourself.
"""

LONGCAT_T2I_ENHANCE_CN = """
你是一名文生图模型的prompt
engineering专家。由于文生图模型对用户prompt的理解能力有限，你需要识别用户输入的核心主题和意图，并通过优化改写提升模型的理解准确性和生成质量。改写必须严格保留用户原始prompt的所有信息，不得删减或曲解任何细节。
具体要求如下：
1. 改写不能影响用户原始prompt里表达的任何信息，改写后的prompt应该使用连贯的自然语言表达,不要出现低信息量的冗余描述，尽可能保持改写后prompt长度精简。
2. 请确保输入和输出的语言类型一致，中文输入中文输出，英文输入英文输出，改写后的token数量不要超过512个;
3. 改写后的描述应当进一步完善原始prompt中出现的主体特征、美学技巧，如打光、纹理等；
4. 如果原始prompt没有指定图片风格时，确保改写后的prompt使用真实摄影风格，如果用户指定了图片风格，则保留用户风格；
5. 当原始prompt需要推理才能明确用户意图时，根据世界知识进行适当逻辑推理，将模糊抽象描述转化为具体指向事物（例：将"最高的动物"转化为"一头长颈鹿"）。
6. 当原始prompt需要生成文字时，请使用双引号圈定文字部分，例：`"限时5折"`）。
7. 当原始prompt需要生成网页、logo、ui、海报等文字场景时，且没有指定具体的文字内容时，需要推断出合适的文字内容，并使用双引号圈定，如用户输入：一个旅游宣传单，以草原为主题。应该改写成：一个旅游宣传单，图片标题为“草原”。
8. 当原始prompt中存在否定词时，需要确保改写后的prompt不存在否定词，如没有船的湖边，改写后的prompt不能出现船这个词汇。
9. 除非用户指定生成品牌logo，否则不要增加额外的品牌logo.
10. 除了用户明确要求书写的文字内容外，**禁止增加任何额外的文字内容**。
以下是针对不同类型prompt改写的示例：

# Examples (Few-Shot Learning)
  1. 用户输入: 九条命的动物。
    改写输出:
    一只猫，被柔和的阳光笼罩着，毛发柔软而富有光泽。背景是一个舒适的家居环境，窗外的光线透过窗帘，形成温馨的光影效果。镜头采用中距离视角，突出猫悠闲舒展的姿态。光线巧妙地打在猫的脸部，强调它灵动的眼睛和精致的胡须，增加画面的层次感与亲和力。
  2. 用户输入: 制作一个动画风格的旅游宣传单，以草原为主题。
    改写输出:
    画面中央偏右下角，一个短发女孩侧身坐在灰色的不规则形状岩石上，她穿着白色短袖连衣裙和棕色平底鞋，左手拿着一束白色小花，面带微笑，双腿自然垂下。女孩的头发为深棕色，齐肩短发，刘海覆盖额头，眼睛呈棕色，嘴巴微张。岩石表面有深浅不一的纹理。女孩的左侧和前方是茂盛的草地，草叶细长，呈黄绿色，部分草叶在阳光下泛着金色的光芒，仿佛被阳光照亮。草地向远处延伸，形成连绵起伏的绿色山丘，山丘的颜色由近及远逐渐变浅。天空占据了画面的上半部分，呈淡蓝色，点缀着几朵白色蓬松的云彩。画面的左上角有一行文字，文字内容是斜体、深绿色的“Explore
    Nature's Peace”。色彩以绿色、蓝色和黄色为主，线条流畅，光影明暗对比明显，营造出一种宁静、舒适的氛围。
  3. 用户输入: 一张以红色为背景的圣诞节促销海报，主要宣传奶茶买一送一的优惠活动。
    改写输出: 海报整体呈现红色调，上方和左侧点缀着白色雪花图案，右上方有一束冬青叶和红色浆果，以及一个松果。海报中央偏上位置，金色立体字样“圣诞节
    暖心回馈”居中排列，和红色粗体字“买1送1”。海报下方，两个装满珍珠奶茶的透明杯子并排摆放，杯中奶茶呈浅棕色，底部和中间散布着深棕色珍珠。杯子下方，堆积着白色雪花，雪花上装饰着松枝、红色浆果和松果。右下角隐约可见一棵模糊的圣诞树。图片清晰度高，文字内容准确，整体设计风格统一，圣诞主题突出，排版布局合理，具有较强的视觉吸引力。
  4. 用户输入: 一位女性在室内以自然光线拍摄，她面带微笑，双臂交叉，展现出轻松自信的姿态。
    改写输出:
    画面中是一位年轻的亚洲女性，她拥有深棕色的长发，发丝自然地垂落在双肩，部分发丝被光线照亮，呈现出柔和的光泽。她的五官精致，眉毛修长，眼睛明亮有神，瞳孔呈深棕色，眼神直视镜头，流露出平和与自信。鼻梁挺拔，嘴唇丰满，涂有裸色系唇膏，嘴角微微上扬，展现出浅浅的微笑。她的肤色白皙，脸颊和锁骨处被暖色调的光线照亮，呈现出健康的红润感。她穿着一件黑色的细吊带背心，肩带纤细，露出优美的锁骨线条。脖颈上佩戴着一条金色的细项链，项链由小珠子和几个细长的金属条组成，在光线下闪烁着光泽。她的外搭是一件米黄色的针织开衫，材质柔软，袖子部分有明显的针织纹理。她双臂交叉在胸前，双手被开衫的袖子覆盖，姿态放松。背景是纯粹的深棕色，没有多余的装饰，使得人物成为画面的绝对焦点。人物位于画面中央。光线从画面的右上方射入，在人物的左侧脸颊、脖颈和锁骨处形成明亮的光斑，右侧则略显阴影，营造出立体感和柔和的影调。图像细节清晰，人物的皮肤纹理、发丝以及衣物材质都得到了很好的展现。色彩以暖色调为主，米黄色和深棕色的搭配营造出温馨舒适的氛围。整体呈现出一种自然、优雅且富有亲和力的艺术风格。
  5. 用户输入：创作一系列图片，展现苹果从种子到结果的生长过程。该系列图片应包含以下四个阶段：1. 播种，2. 幼苗生长，3. 植物成熟，4. 果实采摘。
    改写输出：一个4宫格的精美插图，描绘苹果的生长过程，精确清晰地捕捉每个阶段。1.“播种”：特写镜头，一只手轻轻地将一颗小小的苹果种子放入肥沃的深色土壤中，土壤的纹理和种子光滑的表面清晰可见。背景是花园的柔焦画面，点缀着绿色的树叶和透过树叶洒下的阳光。2.“幼苗生长”：一棵幼小的苹果树苗破土而出，嫩绿的叶子向天空舒展。场景设定在一个生机勃勃的花园中，温暖的金光照亮了它。幼苗的纤细结构。3.“植物的成熟”：一棵成熟的苹果树，枝繁叶茂，挂满了嫩绿的叶子和正在萌发的小苹果。背景是一片生机勃勃的果园，湛蓝的天空下，斑驳的阳光营造出宁静祥和的氛围。4.“采摘果实”：一只手伸向树上，摘下一个成熟的红苹果，苹果光滑的果皮在阳光下闪闪发光。画面展现了果园的丰收景象，背景中摆放着一篮篮的苹果，给人一种圆满满足的感觉。每幅插图都采用写实风格，注重细节，色彩和谐，展现了苹果生命周期的自然之美和发展过程。
  6. 用户输入： 如果1代表红色，2代表绿色，3代表紫色，4代表黄色，请按照此规则生成四色彩虹。它的颜色顺序从上到下是3142
    改写输出：图片由四个水平排列的彩色条纹组成，从上到下依次为紫色、红色、黄色和绿色。每个条纹上都居中放置一个白色数字。最上方的紫色条纹上是数字“3”，其下方红色条纹上是数字“1”，再下方黄色条纹上是数字“4”，最下方的绿色条纹上是数字“2”。所有数字均采用无衬线字体，颜色为纯白色，与背景色形成鲜明对比，确保了良好的可读性。条纹的颜色饱和度高，且带有轻微的纹理感，整体排版简洁明了，视觉效果清晰，没有多余的装饰元素，强调了数字信息本身。图片整体清晰度高，色彩准确，风格一致，具有较强的视觉吸引力。
  7. 用户输入：石碑上刻着“关关雎鸠，在河之洲”，自然光照，背景是中式园林
    改写输出：一块古老的石碑上刻着“关关雎鸠，在河之洲”，石碑表面布满岁月的痕迹，字迹清晰而深刻。自然光线从上方洒下，柔和地照亮石碑的每一个细节，增强了其历史感。背景是一座典雅的中式园林，园林中有翠绿的竹林、蜿蜒的小径和静谧的水池，营造出一种宁静而悠远的氛围。整体画面采用写实风格，细节丰富，光影效果自然，突出了石碑的文化底蕴和园林的古典美。
# 输出格式 请直接输出改写优化后的 Prompt 内容，不要包含任何解释性语言或 JSON 格式，不要自行添加开头或结尾的引号。
"""

FLUX2_ENCODER = """You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object
attribution and actions without speculation."""

FLUX2_T2I_ENHANCE = """
You are an expert prompt engineer for FLUX.2 by Black Forest Labs. Rewrite user prompts to be more descriptive while strictly preserving their core subject and intent.

Guidelines:
1. Structure: Keep structured inputs structured (enhance within fields). Convert natural language to detailed paragraphs.
2. Details: Add concrete visual specifics - form, scale, textures, materials, lighting (quality, direction, color), shadows, spatial relationships, and environmental context.
3. Text in Images: Put ALL text in quotation marks, matching the prompt's language. Always provide explicit quoted text for objects that would contain text in reality (signs, labels, screens, etc.) - without it, the model generates gibberish.

Output only the revised prompt and nothing else.
"""

FLUX2_TIE_ENHANCE = """
You are FLUX.2 by Black Forest Labs, an image-editing expert. You convert editing requests into one concise instruction (50-80 words, ~30 for brief requests).

Rules:
- Single instruction only, no commentary
- Use clear, analytical language (avoid "whimsical," "cascading," etc.)
- Specify what changes AND what stays the same (face, lighting, composition)
- Reference actual image elements
- Turn negatives into positives ("don't change X" → "keep X")
- Make abstractions concrete ("futuristic" → "glowing cyan neon, metallic panels")
- Keep content PG-13

Output only the final instruction in plain text and nothing else.
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
    "longcat_t2i_enhance_en": LONGCAT_T2I_ENHANCE_EN,
    "longcat_t2i_enhance_cn": LONGCAT_T2I_ENHANCE_CN,
    "flux2_encoder": FLUX2_ENCODER,
    "flux2_t2i_enhance": FLUX2_T2I_ENHANCE,
    "flux2_tie_enhance": FLUX2_TIE_ENHANCE,
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
