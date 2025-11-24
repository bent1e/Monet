import os
import torch
import math
from accelerate import infer_auto_device_map
def traverse_device_map(modules, allowed_devices):
    device_map = {}
    total_devices = len(allowed_devices)
    device_id = 0
    for module in modules:
        device_map[module] = allowed_devices[device_id]
        device_id = (device_id + 1)%total_devices
    return device_map

def set_device_map(allowed_devices, model_name, model, max_memory=None):
    assert isinstance(allowed_devices, list) and all(isinstance(x, int) for x in allowed_devices)
    if "qwen" in model_name.lower() or "ThinkLite" in model_name:
        num_language_layers = 28
        visual_modules = ["visual.patch_embed", "visual.rotary_pos_emb", "visual.merger"] + [f"visual.blocks.{l}" for l in range(32)]
        language_modules = ["model.embed_tokens", "model.norm", "model.rotary_emb", "lm_head"] + [f"model.layers.{l}" for l in range(num_language_layers)]
        device_map = {**traverse_device_map(visual_modules, allowed_devices), **traverse_device_map(language_modules, allowed_devices)}
        device_map["model.embed_tokens"] = allowed_devices[0]
        device_map["model.norm"] = allowed_devices[0]
        device_map["model.rotary_emb"] = allowed_devices[0]
        device_map["lm_head"] = allowed_devices[0]
        device_map[f'model.layers.{num_language_layers - 1}'] = allowed_devices[0]
        
    elif "internvl2" in model_name.lower() or "visualprm" in model_name.lower():
        device_map = {}
        world_size = len(allowed_devices)
        num_layers = {
            'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32, 
            'VisualPRM-8B': 32,
            'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = allowed_devices[i]
                layer_cnt += 1
        device_map['vision_model'] = allowed_devices[0]
        device_map['mlp1'] = allowed_devices[0]
        device_map['language_model.model.tok_embeddings'] = allowed_devices[0]
        device_map['language_model.model.embed_tokens'] = allowed_devices[0]
        device_map['language_model.output'] = allowed_devices[0]
        device_map['language_model.model.norm'] = allowed_devices[0]
        device_map['language_model.model.rotary_emb'] = allowed_devices[0]
        device_map['language_model.lm_head'] = allowed_devices[0]
        device_map[f'language_model.model.layers.{num_layers - 1}'] = allowed_devices[0]

    #device_map[""] = allowed_devices[0]
    return device_map