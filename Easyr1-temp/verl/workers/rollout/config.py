# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

@dataclass
class MCTSConfig:
    use_vllm_for_multimodal: bool = True
    stop: list = field(default_factory=lambda: ["###"])
    rollouts: int = 10
    max_depth: int = 10
    c_puct: float = 2.0
    save_rollout_json: bool = False
    is_sampling: bool = True
    positive_reward: float = 1.0
    negative_reward: float = -1.0
    step_beam_width: int = 1
    update_leaf_value: bool = False
    terminal_sample: bool = True
    correctness_threshold_for_training: float = 0.4
    min_depth_for_complete: int = 5
    stop_when_find_correct_answer: bool = False
    """auto keys"""
    allow_mcts_samples_more_than_rollout_batch_size: bool = True
    
@dataclass
class MCConfig:
    embed_model_path: str = ""
    select_acc_threshold: float = 0.6
    step_hash_threshold: float = 0.7
    correct_cluster_threshold: float = 0.5  # whether a cluster is correct or not
    delim: str = "### Step"
    hash_server_name: str = "step_hash_server"
    embed_server_name: str = "embedding_server"
    drop_last_steps: int = 2
    remove_all_correct_groups: bool = False
    remove_high_acc_group: bool = True
    group_high_acc_threshold: float = 0.8
    shuffle_groups_across_samples: bool = True
    normalize_step_wise_adv: bool = True
    ablation_process_reward: bool = False
    #embedding_model_device: str = "cuda:0"
    embedding_model_path: str = "Qwen3-Embedding-4B"

@dataclass
class AVTConfig:
    select_acc_threshold: float = 0.6
    delim: str = "### Step"
    hash_server_name: str = "sample_hash_server"


    

@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 8
    temperature: float = 1.0
    max_num_seqs: int = 50
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 4
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = True #False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 16384
    disable_log_stats: bool = True
    val_override_config: Dict[str, Any] = field(default_factory=dict)
    sampling_strategy: str = field(default="greedy")
    repetition_penalty: float = 1.1
    online_difficulty_sampling: bool = False
    offline_difficulty_sampling: bool = False
    """auto keys"""
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)
    n_gpus_per_node: int = 4
    pr_batch_size: int = 512
    
    # mcts config
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    mc: MCConfig = field(default_factory=MCConfig)
    avt: AVTConfig = field(default_factory=AVTConfig)

    def to_dict(self):
        return asdict(self)
    
