import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
from torch import Tensor
from typing import Union
import numpy as np
from tqdm import tqdm
import json
from tools.hash_dict import StepHashDict
model_fpath = "/data1/qxwang/checkpoints/simcse-large-gsm8k"
tokenizer = AutoTokenizer.from_pretrained(model_fpath)
model = AutoModel.from_pretrained(model_fpath).cuda()
max_seq_length = 256

def compute_emb(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_seq_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu().numpy()


data_path = "/data1/qxwang/codes/rStar-math/results/MCTS_rollouts/MathVista/rollout00MathVista.mcts.Qwen2.5-VL-7B-Instruct.datasize1024.20250411203825.jsonl"
full_tree_dicts = []
with open(data_path, "r") as f:
    for line in tqdm(f):
        full_tree_dict = json.loads(line)
        full_tree_dicts.append(full_tree_dict)
       
step_hash = StepHashDict(threshold=0.85, rep_mode="medoid")

import time
start_time = time.time()
for i, full_tree_dict in tqdm(enumerate(full_tree_dicts[:100])):
    tree_dict = full_tree_dict["rstar"]
    texts_of_a_problem = [] 
    for node_tag, node_dict in tree_dict.items():
        if node_tag == "0":
            continue
        text = node_dict["text"]
        texts_of_a_problem.append(text)
    
    embeds_of_a_problem = compute_emb(texts_of_a_problem)


    clusters = step_hash.update_sample_step_hash_dict(
        sample_id=i,
        embeddings= embeds_of_a_problem,
        texts=texts_of_a_problem,
    )

    for cid, info in clusters.items():
        print(f"\n\n==============Cluster {cid}  |  representative: “{info['rep_text'][:400]}”")
        for t in info["members_texts"]:
            print("   ·", t[:400])
            
print("Time taken:", time.time() - start_time)