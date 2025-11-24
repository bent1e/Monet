import argparse
import json
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from multimodal.device_map import set_device_map
import copy
from accelerate.hooks import remove_hook_from_submodules, attach_align_device_hook
# Image normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        tr = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tr)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    tr = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    tw, th = image_size * tr[0], image_size * tr[1]
    blocks = tr[0] * tr[1]
    resized = image.resize((tw, th))
    imgs = []
    cols = tw // image_size
    for idx in range(blocks):
        x0 = (idx % cols) * image_size
        y0 = (idx // cols) * image_size
        imgs.append(resized.crop((x0, y0, x0 + image_size, y0 + image_size)))
    assert len(imgs) == blocks
    if use_thumbnail and blocks != 1:
        imgs.append(image.resize((image_size, image_size)))
    return imgs


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_pixel_values(image_path, input_size=448, max_num=12):
    img = Image.open(image_path).convert('RGB')
    imgs = dynamic_preprocess(img, image_size=input_size, max_num=max_num, use_thumbnail=True)
    transform = build_transform(input_size)
    return torch.stack([transform(im) for im in imgs])


def get_module(model, layer_id):
    return model.language_model.model.layers[layer_id]



def set_module(model, layer_id, new_module):
    device = next(iter(model.language_model.model.layers[layer_id].parameters())).device
    replaced_layer = copy.deepcopy(new_module).to(device)
    model.language_model.model.layers[layer_id] = replaced_layer
    src_first_weight = next(iter(replaced_layer.parameters())).view(-1)[:5]
    tgt_first_weight = next(iter(model.language_model.model.layers[layer_id].parameters())).view(-1)[:5]
    assert torch.allclose(src_first_weight, tgt_first_weight), "failed to replace the weights"



def main():
    parser = argparse.ArgumentParser(
        description="Replace VisualPRM layers with InternVL layers, distribute across GPUs, and score a response"
    )
    parser.add_argument('--visualprm_path', type=str, required=True,
                        help='Path to VisualPRM-8B')
    parser.add_argument('--internvl_path', type=str, required=True,
                        help='Path to InternVL2.5-8B')
    parser.add_argument('--layers', type=int, required=True,nargs='+',
                        help='Comma-separated module paths to replace')
    parser.add_argument('--visualprm_devices', type=int, default=None, nargs='+',
                        help='JSON string or file for VisualPRM device_map')
    parser.add_argument('--internvl_devices', type=int, default=None, nargs='+',
                        help='JSON string or file for InternVL device_map')
    parser.add_argument('--image', type=str, required=True, help='Image file path')
    parser.add_argument('--question', type=str, help='Question to ask',
                        default="As shown in the diagram, EF is the axis of symmetry for quadrilateral ABCD. Given that CD = 5 cm and AD = 3 cm, what is the perimeter of parallelogram ABFD in cm?\nA. 12\nB. 10\nC. 14\nD. 16\nE. No correct answer")
    parser.add_argument('--response', type=str, help='Single response to score',
                        default="To find the perimeter of parallelogram ABFD, we need to use the given information about the quadrilateral ABCD and its symmetry with respect to EF. \n1. EF is the axis of symmetry for the quadrilateral ABCD. This means that EF divides ABCD into two congruent halves. \n2. Since EF is the axis of symmetry and AD = 3 cm, AD = DF.")
    parser.add_argument('--input_size', type=int, default=448, help='Image crop size')

    args = parser.parse_args()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.visualprm_path, trust_remote_code=True, use_fast=False
    )

    # load models with optional device_map
    v_map = set_device_map(args.visualprm_devices, args.visualprm_path.split('/')[-1])
    visualprm = AutoModel.from_pretrained(
        args.visualprm_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map=v_map
    ).eval()

    i_map = set_device_map(args.internvl_devices, args.internvl_path.split('/')[-1])
    internvl = AutoModel.from_pretrained(
        args.internvl_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map=i_map
    ).eval()

    # replace specified layers
    for layer_id in args.layers:
        new_mod = get_module(internvl, layer_id)
        set_module(visualprm, layer_id, new_mod)
        print(f"Replaced {layer_id}")

    remove_hook_from_submodules(visualprm)
    attach_align_device_hook(visualprm, execution_device=None) 

    for layer_id in range(32):
        layer = visualprm.language_model.model.layers[layer_id] 
        print("hook exists:", hasattr(layer, "_hf_hook"))
        print("hook target:", visualprm._hf_hook.module_to_device.get(layer, "??"))
        print("param device:", next(layer.parameters()).device)


    # preprocess image
    pixel_values = load_pixel_values(args.image, input_size=args.input_size)
    pixel_values = pixel_values.to(torch.bfloat16).to(args.visualprm_devices[0])
    num_patches = pixel_values.shape[0]

    # score
    for idx, layer in enumerate(visualprm.language_model.model.layers):
        print(idx, next(layer.parameters()).device)
    
    scored = visualprm.select_best_response(
        tokenizer=tokenizer,
        question=args.question,
        response_list=[args.response],
        pixel_values=pixel_values,
        num_patches_list=[num_patches],
        return_scores=True
    )
    resp, score = scored[0]
    print(f"Response: {resp}\nScore: {score}")

if __name__ == '__main__':
    main()