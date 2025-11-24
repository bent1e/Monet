from PIL import Image
from qwen_vl_utils import process_vision_info
from PIL import Image

def resize_image_if_needed(img: Image.Image, max_resolution: int = 768) -> Image.Image:
    """
    Make sure the resolution <= max_resolution, if not, then resize。
    """
    if max(img.size) > max_resolution:
        img = img.copy()
        img.thumbnail((max_resolution, max_resolution), Image.Resampling.LANCZOS)
    return img

def get_qwen2_vl_processed_input_batch(processor, sys_prompt, questions, partial_solutions, img_paths):
    # 初始化 messages 列表

    messages = []
    texts = []
    
    for question, img_path, partial_solution in zip(questions, img_paths, partial_solutions):
        #img = Image.open(img_path).convert("RGB")
        #img = resize_image_if_needed(img, max_resolution=768)
        message = [
            {
                'role': "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": question + partial_solution},
                ],
            },
        ]

        texts.append(processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
        messages.append(message)
            
    image_inputs, video_inputs = process_vision_info(messages)
    # 批量处理
    batch_inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        use_fast=True
    )

    return batch_inputs




def process_multimodal_input(model_name, processor, img_paths, sys_prompt, questions, partial_solutions):
    if 'qwen' in model_name.lower() or "ThinkLite-VL" in model_name:
        assert type(img_paths) == list
        return get_qwen2_vl_processed_input_batch(processor, sys_prompt, questions, partial_solutions, img_paths)
       