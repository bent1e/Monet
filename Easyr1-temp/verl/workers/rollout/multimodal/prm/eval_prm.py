import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoProcessor
from accelerate import dispatch_model, init_empty_weights, load_checkpoint_and_dispatch
from functools import partial
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
from rstar_deepthink.config import BaseConfig
from omegaconf import OmegaConf
from multimodal.process_input import process_multimodal_input
from multimodal.prm.prm_dataloader import ReasoningDatasetWhole, dataset_name_mapping, split_train_test_by_image, BalancedDataset
from multimodal.device_map import set_device_map
import copy
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from sklearn.metrics import f1_score
def load_qwen_model(config):

    allowed_devices = [int(device) for device in config.allowed_devices.split(",")]
    # 这里直接给出各设备的内存分配
    device = allowed_devices[0] # main device
    max_memory = {}
    for device in allowed_devices:
        max_memory[device] = '20GB'
    '''max_memory = {
        2: '22GB',
        5: '9GB',
        7: '3GB',
        8: '4GB',
        9: '5GB'
    }'''

    config_obj = AutoConfig.from_pretrained(config.mllm_dir)
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration._from_config(config_obj)
    visual_modules = ["visual.patch_embed", "visual.rotary_pos_emb", "visual.merger"] + \
                        [f"visual.blocks.{l}" for l in range(32)]
    language_modules = ["model.embed_tokens", "model.norm", "model.rotary_emb", "lm_head"] + \
                        [f"model.layers.{l}" for l in range(28)]
    device_map = {**set_device_map(visual_modules, allowed_devices),
                    **set_device_map(language_modules, allowed_devices)}
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=config.mllm_dir,
        device_map=device_map,
        max_memory=max_memory, 
        dtype=torch.bfloat16,  # 如果环境支持 bf16
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 600 * 28 * 28 # 1280*28*28
    processor = AutoProcessor.from_pretrained(config.mllm_dir, min_pixels=min_pixels, max_pixels=max_pixels)
    

    verifier_backbone = Qwen2_5_VLDecoderLayer(config=model.config, layer_idx=config.layer+1).to(device)
    
    return model, processor, verifier_backbone


class Qwen25VLWrapper(nn.Module):
    def __init__(self, model, processor, config):
        super().__init__()
        self.model = model
        self.processor = processor
        self.hook_layer:int = config.layer
        self.hook_features = None
        self.config = config
        self.register_hook()
        self.verifier_layer_forward_args = None

    def register_hook(self):
        '''
        resigister hook, get the input parameter of the forward function of the next layer
        '''
        hook_module = self.model.model.layers[self.hook_layer]
        verifier_module = self.model.model.layers[self.hook_layer+1]
        original_forward = verifier_module.forward 
        
        def wrapped_forward(*args, **kwargs): # capturing the necessary parameters (positional_embeddings, attn_mask, ...) for the forward of the verifier backbone
            self.verifier_layer_forward_args = (args, kwargs)
            return original_forward(*args, **kwargs)
        verifier_module.forward = wrapped_forward        
        
        def hook(module, input, output):
            self.hook_features = output[0] # hidden features
        hook_module.register_forward_hook(hook)

    def forward(self, img_paths, questions, steps):
        multimodal_processed_inputs = process_multimodal_input(self.config.mllm_dir, self.processor, img_paths=img_paths, sys_prompt="", questions=questions, partial_solutions=steps)
        with torch.inference_mode():
            generated_ids = self.model(**multimodal_processed_inputs, use_cache=False)
            '''generated_ids = self.model.generate(**multimodal_processed_inputs, 
                                                            temperature=self.config.temperature, # 0.7
                                                            top_k=100,
                                                            do_sample=True,
                                                            #top_p=config.top_p,
                                                            #num_return_sequences=1, # if>1, OOM
                                                            max_new_tokens=self.config.max_tokens, 
                                                            #stop=config.stop,
                                                            #skip_special_tokens=False,
                                                            seed=self.config.seed if self.config.temperature == 0 else None, # vllm0.6.6.post1 
                                                            #output_attentions=True,
                                                            #output_hidden_states=True
                                                            #output_scores=True
                                                            )'''
            del generated_ids, multimodal_processed_inputs
        return self.hook_features
    
    
class Verifier(nn.Module):
    def __init__(self, backbone, mlp_hidden_size=128, dropout_rate: float = 0.1):
        super().__init__()
        self.backbone = backbone  # Qwen2.5-VL LLM 的第 l+1 层
        # 假设三个特征每个维度为768，拼接后为 768*3
        self.mlp = nn.Sequential(
            nn.Linear(3584, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, 1)
        )
        self.kwargs = None

    def move_device(self, verifier_layer_forward_args, device=0, layer_id=-1):
        args, kwargs = verifier_layer_forward_args
        kwargs['use_cache'] = False
        kwargs['past_key_value'] = None
        #kwargs['past_key_value'].key_cache[layer_id] = kwargs['past_key_value'].float().key_cache[layer_id].to(device)
        #kwargs['past_key_value'].value_cache[layer_id] = kwargs['past_key_value'].float().value_cache[layer_id].to(device)
        if kwargs['attention_mask']!=None: # torch.Size([2, 1, 480, 480])
            kwargs['attention_mask'] = kwargs['attention_mask'].float().to(device)
        if kwargs['position_ids']!=None:
            kwargs['position_ids'] = kwargs['position_ids'].to(device)
        if kwargs['cache_position']!=None:
            kwargs['cache_position'] = kwargs['cache_position'].to(device)
        if kwargs['position_embeddings']!=None:
            new_pos_embs = []
            for e in kwargs['position_embeddings']:
                new_pos_embs.append(e.to(device))
            kwargs['position_embeddings'] = tuple(new_pos_embs)
        self.kwargs = kwargs
        
    def forward(self, x):
        x = self.backbone(x.float(), **self.kwargs) #x torch.Size([2, 480, 3584])
        logits = self.mlp(x[0][:,-1,:])
        logits = torch.sigmoid(logits)
        return logits.squeeze(-1)


def bradley_terry_loss(scores, labels, group_ids):
    loss = 0.0
    count = 0
    groups = {}
    for i, gid in enumerate(group_ids):
        groups.setdefault(gid, []).append(i)
    for group in groups.values():
        pos_indices = [i for i in group if labels[i] == 1]
        neg_indices = [i for i in group if labels[i] == 0]
        for i in pos_indices:
            for j in neg_indices:
                s_pos = scores[i]
                s_neg = scores[j]
                p = torch.sigmoid(s_pos - s_neg)
                loss = loss - torch.log(p + 1e-8)
                count += 1
    if count > 0:
        loss = loss / count
    else:
        loss = torch.tensor(0.0, requires_grad=True)
    return loss



def main(args, config):
    
    # set seeds
    seed = 42
    random.seed(seed)
    
    # save name
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt_dir))[0]
    os.makedirs(args.test_result_save_dir, exist_ok=True)
    test_files = ".".join([dataset_name_mapping(os.path.splitext(os.path.basename(f))[0]) for f in args.test_dataset_dirs])
    saved_file_name =   f"Tst{test_files}-{ckpt_name}"
    use_question = ("-use_question" in ckpt_name)
    use_step = ("-use_step" in ckpt_name)
    if use_question:
        saved_file_name += "-use_question"
    if use_step:
        saved_file_name += "-use_step"
    final_save_path = os.path.join(args.test_result_save_dir, saved_file_name + '.txt')

    device = int(config.allowed_devices.split(',')[0])
    if "Qwen2.5" in config.mllm_dir:
        model, processor, verifier_backbone = load_qwen_model(config)
        wrapper = Qwen25VLWrapper(model, processor, config)
    else:
        raise NotImplementedError

    # build verifier
    verifier_layer_id = config.layer + 1
    verifier = Verifier(backbone=verifier_backbone).to(device)
    verifier.load_state_dict(torch.load(args.ckpt_dir))
    
    # load data, train:val = 0.95:0.05
    val_dataset = ReasoningDatasetWhole(args.test_dataset_dirs, label_type=args.label_type).all_data
    total_test_num = len(val_dataset)
    val_dataset = val_dataset[:int(total_test_num)]#
        
    
    print("Validation samples (before balance):", len(val_dataset))
    if args.balance_testset:
        val_dataset = BalancedDataset(val_dataset)
        print("Validation samples (after balance):", len(val_dataset))

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    verifier.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []

        for batch_id, batch in enumerate(tqdm(val_loader, desc=f"Evaluation...")):
            img_paths = batch['image']
            questions = batch['question']
            steps = batch['step']
            labels = batch['label']

            if not use_question:
                questions = [""] * len(questions)
            if not use_step:
                steps = [""] * len(steps)

            hidden_features = wrapper(img_paths, questions, steps)
            verifier.move_device(wrapper.verifier_layer_forward_args, device, verifier_layer_id)
            logits = verifier(hidden_features.to(device))

            preds = (logits > 0.5).long().view(-1).cpu()
            labels = labels.view(-1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"========== Test macro F1: {macro_f1:.4f} ===========")

        with open(final_save_path, 'w') as f:
            f.write(f"Test macro F1: {macro_f1:.4f}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_cfg', type=str, default="config/sample_mcts.yaml")
    parser.add_argument('--test_dataset_dirs', type=str, required=True, nargs="+",
                        help="separated by commas", default="")
    parser.add_argument('--mllm_dir', type=str, required=True,
                        help="directory of the MLLM")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--allowed_devices', type=str, default="0,1,2,3,4,5,6,7",
                        help="devices to load the huggingface MLLM")
    parser.add_argument('--ckpt_dir', type=str, default="./multimodal/verifier/checkpoints/TrGeometry3K.Geo170K.Geometry3K-TstGeomVerse-Qwen2.5-VL-7B-Instruct-L10-lr0.0001-bsz2-use_step-ep1.pt")
    parser.add_argument('--test_result_save_dir', type=str, default="./multimodal/prm/test_results")
    parser.add_argument("--label_type", type=str, default="step_correctness", choices=["step_correctness", "lead_to_correctness"])
    parser.add_argument("--balance_testset", action="store_true", default=False, help="whether to balance the test set")
    args = parser.parse_args()
    
    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))

    config.allowed_devices = args.allowed_devices
    config.mllm_dir = args.mllm_dir
    config.layer = int(args.ckpt_dir[args.ckpt_dir.find("-L")+2:args.ckpt_dir.find("-lr")])
    main(args, config)
