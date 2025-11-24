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
        max_memory[device] = '10GB'
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
    

    prm_backbone = Qwen2_5_VLDecoderLayer(config=model.config, layer_idx=config.layer+1).to(device)
    
    return model, processor, prm_backbone


class Qwen25VLWrapper(nn.Module):
    def __init__(self, model, processor, config):
        super().__init__()
        self.model      = model
        self.processor  = processor
        self.hook_layers: list[int] = config.layers          # ← 多层
        self.features: dict[int, torch.Tensor] = {}          # {l: hiddens}
        self.prm_args: dict[int, tuple] = {}                 # {l+1: (args,kwargs)}
        self.register_hooks()

    def register_hooks(self):
        """为每个 (l, l+1) 成对注册 hook"""
        for l in self.hook_layers:
            # 1) 抓取 layer l 输出
            self.model.model.layers[l].register_forward_hook(
                lambda m, inp, out, l=l: self.features.__setitem__(l, out[0].detach())
            )
            # 2) 包装 (l+1).forward 以保存参数
            next_layer = self.model.model.layers[l+1]
            orig_fwd   = next_layer.forward
            def wrapped_forward(*args, **kwargs):
                self.prm_args[l+1] = (args, kwargs)
                return orig_fwd(*args, **kwargs)
            next_layer.forward = wrapped_forward

    def forward(self, img_paths, questions, steps):
        inputs = process_multimodal_input(
            self.config.mllm_dir, self.processor,
            img_paths=img_paths, sys_prompt="",
            questions=questions, partial_solutions=steps
        )
        with torch.inference_mode():          # 只做一次前传
            _ = self.model(**inputs, use_cache=False)
        return self.features, self.prm_args   # dict 形式返回

    
    
class PRM(nn.Module):
    def __init__(self, backbone, mlp_hidden_size=128, dropout_rate: float = 0.1):
        super().__init__()
        self.backbone = backbone  # Qwen2.5-VL LLM 的第 l+1 层
        self.mlp = nn.Sequential(
            nn.Linear(3584, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, 1)
        )
        self.kwargs = None

    def move_device(self, prm_layer_forward_args, device=0, layer_id=-1):
        args, kwargs = prm_layer_forward_args
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



# ---------------- 评估函数 ----------------
@torch.no_grad()
def evaluate_multi_layer(prms: dict[int, PRM],
                         wrapper: Qwen25VLWrapper,
                         val_loader: DataLoader,
                         device: int,
                         layer_ids: list[int],
                         use_question: bool,
                         use_step: bool):
    """返回每个 layer 的 macro‑F1，形如 {20:0.83, 24:0.81,…}"""

    from sklearn.metrics import f1_score

    # 先给每层准备预测列表
    preds_dict = {l: [] for l in layer_ids}
    all_labels = []

    for batch in tqdm(val_loader, desc="Validation"):
        img_paths  = batch['image']
        questions  = batch['question']
        steps      = batch['step']
        labels     = batch['label']            # (B,)

        if not use_question:
            questions = [""] * len(questions)
        if not use_step:
            steps = [""] * len(steps)

        # —— 1️⃣ 一次前传拿到所有层的隐藏 + 下一层参数 ——
        feats_dict, prm_args_dict = wrapper(img_paths, questions, steps)

        # —— 2️⃣ 按层推断，推完立刻把 PRM 放回 CPU 释放显存 ——
        for l in layer_ids:
            prm = prms[l].to(device).eval()
            prm.move_device(prm_args_dict[l + 1], device, l + 1)

            logits = prm(feats_dict[l].to(device))          # (B,)
            pred   = (logits > 0.5).long().cpu().tolist()   # 二分类阈值 0.5
            preds_dict[l].extend(pred)

            prm.to('cpu'); torch.cuda.empty_cache()

        # 同一个 batch 的 label 仅需保存一次
        all_labels.extend(labels.view(-1).cpu().tolist())

        # 释放当前 batch 的隐藏特征占用
        for k in list(feats_dict.keys()):
            del feats_dict[k]
        torch.cuda.empty_cache()

    # —— 3️⃣ 计算每层 macro‑F1 ——
    f1_dict = {l: f1_score(all_labels, preds_dict[l], average='macro')
               for l in layer_ids}
    return f1_dict




def main(args, config):
    
    # set seeds
    seed = 42
    random.seed(seed)
    validate_every = 2000  
    # save name
    mllm_name = os.path.basename(args.mllm_dir)
    training_files = ".".join([dataset_name_mapping(os.path.splitext(os.path.basename(f))[0]) for f in args.dataset_dirs])
    os.makedirs(args.save_dir, exist_ok=True)

    
    
    
    # for tensorboard visualization
    log_dir = os.path.join(args.save_dir, 'logs', saved_file_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    device = int(config.allowed_devices.split(',')[0])
    if "Qwen2.5" in config.mllm_dir:
        model, processor, prm_backbone = load_qwen_model(config)
        wrapper = Qwen25VLWrapper(model, processor, config)
    else:
        raise NotImplementedError

    # build prm
    # ---------- 构建每层 PRM ----------
    prms, optimizers = {}, {}
    for l in config.layers:
        prm_backbone = Qwen2_5_VLDecoderLayer(model.config, layer_idx=l+1)
        # 用主干模型的预训练权重初始化
        prm_backbone.load_state_dict(model.model.layers[l+1].state_dict())
        prms[l] = PRM(backbone=prm_backbone).to('cpu')   # 先放 CPU
        optimizers[l] = torch.optim.AdamW(prms[l].parameters(), lr=args.lr)


    # load data, train:val = 0.95:0.05
    full_dataset = ReasoningDatasetWhole(args.dataset_dirs).all_data
    if args.test_dataset_dirs != "":
        train_dataset = full_dataset
        val_dataset = ReasoningDatasetWhole(args.test_dataset_dirs, args.mcts_img_dir, label_type=args.label_type).all_data
        total_test_num = len(val_dataset)
        val_dataset = val_dataset[:int(total_test_num*0.05)]
      
    print("Train samples (before balance):", len(train_dataset))
    print("Validation samples (before balance):", len(val_dataset))  
    if args.balance_dataset:    
        train_dataset = BalancedDataset(train_dataset)
        val_dataset = BalancedDataset(val_dataset)
        print("Train samples (after balance):", len(train_dataset))
        print("Validation samples (after balance):", len(val_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    
    bce_loss_fn = nn.BCELoss()

    for epoch in range(args.epochs):
        prm_global_step = 0
        for batch in train_loader:
            img_paths, q, stp, labels = batch['image'], batch['question'], batch['step'], batch['label']
            if not args.use_question: q   = [""]*len(q)
            if not args.use_step:     stp = [""]*len(stp)

            # 前传一次获得全部 layer 特征
            feats_dict, prm_args_dict = wrapper(img_paths, q, stp)

            # 逐层把 PRM 挂到 GPU → 计算 → 立即转回 CPU
            for l in config.layers:
                prm = prms[l].to(device)
                prm.move_device(prm_args_dict[l+1], device, l+1)      # 处理 kwargs
                logits = prm(feats_dict[l].to(device))
                loss   = bce_loss_fn(logits, labels.float().to(device))

                optimizers[l].zero_grad()
                loss.backward()
                optimizers[l].step()

                # ⬇️释放显存
                prm.to('cpu')
                del feats_dict[l]; torch.cuda.empty_cache()

            if prm_global_step%10 == 0:
                print(f"Epoch {epoch}: Train loss: {loss.item():.4f}")
                writer.add_scalar("Training loss", loss.item(), prm_global_step)

            prm_global_step += 1

            if prm_global_step % validate_every == 0 and args.test_dataset_dirs != "":
                # 调用验证
                f1_results = evaluate_multi_layer(
                    prms=prms,
                    wrapper=wrapper,
                    val_loader=val_loader,
                    device=device,
                    layer_ids=config.layers,
                    use_question=args.use_question,
                    use_step=args.use_step
                )

                # 输出并写 TensorBoard
                for l, f1 in f1_results.items():
                    print(f"[Val] Layer {l:02d}  Macro‑F1 = {f1:.4f}")
                    writer.add_scalar(f"MacroF1/Layer{l}", f1, prm_global_step)


        for l in config.layers:
            saved_file_name =   f"{training_files}-{mllm_name}-L{l}-lr{args.lr}-bsz{args.batch_size}"
            if args.use_question:
                saved_file_name += "-use_question"
            if args.use_step:
                saved_file_name += "-use_step"
            saved_file_name_epoch = saved_file_name + f"-ep{epoch}"
            final_save_name = os.path.join(args.save_dir,saved_file_name_epoch)
            torch.save(prm.state_dict(),final_save_name+".pt")

    writer.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_cfg', type=str, default="config/sample_mcts.yaml")
    parser.add_argument('--dataset_dirs', type=str, required=True, nargs="+",
                        help="separated by commas")
    parser.add_argument('--mcts_img_dir', type=str, default="", help="Path to the image directory for mcts results.")
    parser.add_argument('--test_dataset_dirs', type=str, required=True, nargs="+",
                        help="separated by commas", default="")
    parser.add_argument('--layers', type=int, default=20, nargs="+",
                        help="use the features from layer l, and use the parameter from l+1 as the initialization as the prm")
    parser.add_argument('--mllm_dir', type=str, required=True,
                        help="directory of the MLLM")
    parser.add_argument('--allowed_devices', type=str, default="0,1,2,3,4,5,6,7",
                        help="devices to load the MLLM")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--save_dir', type=str, default="multimodal/prm/checkpoints")
    parser.add_argument('--use_question', action='store_true', help='whether to use question as the input of the prm')
    parser.add_argument('--use_step', action='store_true', help='whether to use step as the input of the prm')
    parser.add_argument('--label_type', type=str, default="step_correctness", choices=["step_correctness", "ans_contribution"], help='the label type of the dataset')
    parser.add_argument('--balance_dataset', action='store_true', help='whether to balance the dataset')
    
    args = parser.parse_args()
    
    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))

    config.dataset_dir = args.dataset_dirs
    config.allowed_devices = args.allowed_devices
    config.mllm_dir = args.mllm_dir
    config.layers = args.layers
    main(args, config)
