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

import os
from contextlib import contextmanager
import pdb
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.tokenizer import get_processor
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig

# mcts
from .rstar_deepthink.solver import Solver
from .rstar_deepthink.agents import BS, MCTS
from tqdm import tqdm
from verl.models.transformers.qwen2_vl import get_rope_index
from tools.actors import StepHashServer, SampleHashServer
from verl.workers.reward.function import FunctionRuleBasedJudgeManager

# mc
import ray
from collections import defaultdict, OrderedDict
import random
import torch.nn.functional as F
from verl.workers.rollout.rstar_deepthink.agents.utils import extract_and_check
import re
import time

# avt
from avt.vllm.latent_recorder import LatentRecorder
import os, json, shutil, tempfile, pathlib

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(model_path: str, trust_remote_code: bool) -> Optional[Dict[int, float]]:
    processor = get_processor(model_path, trust_remote_code=trust_remote_code)
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None

def _make_vllm_shadow_model_dir(orig_dir: str) -> str:
    """Create a shadow model dir for vLLM: copy config.json and remove 'text_config',
    symlink all other files. Return the shadow dir path."""
    orig_dir = os.path.abspath(orig_dir)
    shadow = tempfile.mkdtemp(prefix="vllm_shadow_")
    # 1) copy & patch config.json
    with open(os.path.join(orig_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    if "text_config" in cfg:
        # Remove it so vLLM falls back to top-level hf_config
        del cfg["text_config"]
    with open(os.path.join(shadow, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # 2) symlink every other file/dir
    for entry in os.listdir(orig_dir):
        if entry == "config.json":
            continue
        src = os.path.join(orig_dir, entry)
        dst = os.path.join(shadow, entry)
        try:
            # For dirs, create a symlink to the whole dir; for files, link file.
            pathlib.Path(dst).symlink_to(src, target_is_directory=os.path.isdir(src))
        except Exception:
            # Fallback: copy if symlink is not allowed
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    return shadow

def remove_text_config_inplace(path: str) -> bool:
    """Remove the entire 'text_config' field from config.json on disk.
    - `path` can be a directory containing config.json or a direct path to config.json.
    - Only the 'text_config' key is removed; nothing else is changed.
    - Returns True if a write happened, False if 'text_config' did not exist.
    """
    # Resolve config.json path
    cfg_path = path
    if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found at: {cfg_path}")

    # Load current JSON
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # If no text_config, nothing to do
    if "text_config" not in cfg:
        return False

    # Remove the entire text_config section
    del cfg["text_config"]

    # Atomic write-back to avoid partial writes
    dir_name = os.path.dirname(cfg_path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=dir_name) as tmp:
        json.dump(cfg, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name

    os.replace(tmp_name, cfg_path)
    return True


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer, processor:  Optional[ProcessorMixin], 
                 hash_server: Optional[Union[StepHashServer, SampleHashServer]] = None,
                 rule_based_judge_server: Optional[FunctionRuleBasedJudgeManager] = None,
                 embed_model = None,
                 embed_tokenizer = None):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.processor = processor
        self.tokenizer = tokenizer
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        #print("config.prompt_length: ", config.prompt_length)
        #print("config.response_length: ", config.response_length)
        #print("config.gpu_memory_utilization", config.gpu_memory_utilization)
        #model_for_vllm = _make_vllm_shadow_model_dir(model_path)

        remove_text_config_inplace(model_path)

        self.inference_engine = LLM(
            model=model_path,
            #tokenizer=model_for_vllm,
            #tokenizer_mode="mmap",
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="auto",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            limit_mm_per_prompt={"image": config.limit_images},
            #disable_mm_preprocessor_cache=True,
            #mm_processor_cache_gb=config.mm_processor_cache_gb,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(model_path, trust_remote_code=config.trust_remote_code),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

        self.mcts_solver = Solver(
            config=self.config.mcts, 
            mllm=self.inference_engine)
        
        self.mcts_agent = MCTS
        self.hash_server = hash_server
        self.rule_based_judge_server = rule_based_judge_server

        self.embed_model = embed_model
        self.embed_tokenizer = embed_tokenizer

        self.latent_size = int(os.getenv("ABS_VIS_LATENT_SIZE", '0'))

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if self.config.sampling_strategy in ["mc2", "avt"]:
            batch_sample_idx = list(non_tensor_batch.pop("global_index"))
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # users can customize different sampling_params at different run
        batch_min_mean_correct_resp_lens = []
        
        with self.update_sampling_params(**prompts.meta_info):

            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=(self.rank == 0)
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            #breakpoint()

            
            if self.config.sampling_strategy in ["mc2", "avt"]:
                response_ids = []
                for completion, global_id in zip(completions, batch_sample_idx):
                    for output in completion.outputs:
                        response_ids.append(output.token_ids)
                    min_len, mean_len = ray.get(self.hash_server.look_up_min_mean_correct_resp_len.remote(global_id))
                    batch_min_mean_correct_resp_lens.extend([min_len] * self.sampling_params.n if min_len<float("inf") else [mean_len] * self.sampling_params.n)

            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                # value = torch.tensor([1, 2, 3])
                # value = value.repeat_interleave(2, dim=0)
                # output: tensor([1, 1, 2, 2, 3, 3])
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)
        if self.config.sampling_strategy in ["mc", "mc2", "avt"]:
            non_tensor_batch["ref_resp_lengths"] = np.array(batch_min_mean_correct_resp_lens)
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    @torch.no_grad()
    def generate_sequences_avt(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_sample_idx = list(non_tensor_batch.pop("global_index"))
        gts = list(non_tensor_batch["ground_truth"])
        questions = list(non_tensor_batch["problem"])
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # users can customize different sampling_params at different run
        batch_min_mean_correct_resp_lens = []
        
        with self.update_sampling_params(**prompts.meta_info):
            with LatentRecorder(set_env=True, prefer_tcp=True, filter_rank=self.rank) as rec:
                completions: List[RequestOutput] = self.inference_engine.generate(
                    prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=(self.rank == 0)
                )
                response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            
            #breakpoint() non_tensor_batch['latents'][1].shape
            
            
            min_req_id = 99999
            for completion in completions:
                min_req_id = min(min_req_id, int(completion.request_id))
            
            non_tensor_batch['latents'] = rec.to_object_array_auto(bsz=batch_size, rollout_n=self.sampling_params.n, min_req_id=min_req_id)

            #if self.rank == 0:
            #    pdb.set_trace()
            #breakpoint()
            response_ids = []
            for completion, global_id in zip(completions, batch_sample_idx):
                for output in completion.outputs:
                    response_ids.append(output.token_ids)
                min_len, mean_len = ray.get(self.hash_server.look_up_min_mean_correct_resp_len.remote(global_id))
                batch_min_mean_correct_resp_lens.extend([min_len] * self.sampling_params.n if min_len<float("inf") else [mean_len] * self.sampling_params.n)

            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            '''if self.rank == 0:
                latent_sum = 0
                latent_cnt = 0
                for i  in range(batch_size*self.sampling_params.n):
                    if non_tensor_batch["latents"][i] is not None:
                        latent_cnt+=1
                        latent_sum+=non_tensor_batch["latents"][i].shape[0]
                
                latent_start_cnt = 0
                latent_end_cnt = 0

                for i  in range(batch_size*self.sampling_params.n):
                    latent_start_cnt += (response_ids[i]==151666).nonzero().numel()
                    latent_end_cnt += (response_ids[i]==151667).nonzero().numel()
                
                if latent_sum != latent_start_cnt*self.latent_size:
                    print(f"[ERROR] latent_sum {latent_sum} != latent_start_cnt*self.latent_size {latent_start_cnt*self.latent_size}")
                    pdb.set_trace()

                for i in range(batch_size*self.sampling_params.n):
                    if non_tensor_batch["latents"][i] is not None:
                        if non_tensor_batch["latents"][i].shape[0] != (response_ids[i]==151666).nonzero().numel()*self.latent_size:
                            latent_num = non_tensor_batch["latents"][i].shape[0]
                            latent_poss_num = (response_ids[i]==151666).nonzero().numel()*self.latent_size
                            print(f"[ERROR] sample latent_cnt {latent_num} != sample latent_start_cnt * latent_size {latent_poss_num}")
                            pdb.set_trace()'''


            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                # value = torch.tensor([1, 2, 3])
                # value = value.repeat_interleave(2, dim=0)
                # output: tensor([1, 1, 2, 2, 3, 3])
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)
        non_tensor_batch["ref_resp_lengths"] = np.array(batch_min_mean_correct_resp_lens) 
        non_tensor_batch["ground_truth"] = _repeat_interleave(np.array(gts), self.sampling_params.n)
        non_tensor_batch["problem"] = _repeat_interleave(np.array(questions), self.sampling_params.n)
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    

    @torch.no_grad()
    def generate_sequences_mcts(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        if self.config.pr_batch_size!=-1:
            assert self.config.pr_batch_size % self.config.n_gpus_per_node == 0, "pr_batch_size should be divisible by n_gpus_per_node"
            original_batch_size = self.config.pr_batch_size // self.config.n_gpus_per_node
        else:
            original_batch_size = input_ids.size(0)
        device = input_ids.device

        non_tensor_batch = prompts.non_tensor_batch
        if self.config.pr_batch_size == -1 and original_batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        #question_key = non_tensor_batch["prompt_key"][0]
        gts = list(non_tensor_batch["ground_truth"])
        questions = list(non_tensor_batch["problem"])
        prompts_before_processor = list(non_tensor_batch.pop("prompt_before_processor")) # formated prompt text, for processor

        agents=[]
        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for batch_id, (raw_prompt_ids, multi_modal_data, question, gt, prompt_before_processor) in enumerate(zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), questions, gts, prompts_before_processor
            )):
                vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})
                agents.append(self.mcts_agent(config=self.config.mcts, batch_id=batch_id, question=question, \
                    ground_truth=gt, multi_modal_data = multi_modal_data, mllm=self.mcts_solver.mllm, prompt_before_processor=prompt_before_processor))
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]
        
        with self.update_sampling_params(**prompts.meta_info):
            for rollout in tqdm(range(self.config.mcts.rollouts), desc="MCTS Rollout Processing"):
                agents = self.mcts_solver.solve(agents, vllm_inputs=vllm_inputs, rollout=rollout, sampling_params=self.sampling_params)
                if rollout == self.config.mcts.rollouts - 1:
                    setattr(self.sampling_params, "stop", [])
                    self.mcts_solver.collect_complete_update(agents, sampling_params=self.sampling_params)
                    rollout_results: List[tuple[List[int], List[str], Any, List[List[int]], List[float], List[str], List[str], List[int]]] = self.mcts_solver.collect_rollout_results(agents)
                    '''
                    rollout_results: 
                    a list of tuples, each tuple contains:
                    1. prompt ids of step_1, step_2, ..., step_{t-1}
                    2. prompt text of step_1, step_2, ..., step_{t-1}
                    3. image for processor
                    4. response ids of the rollout steps at step_t: step^1_t, step^2_t, ..., step^k_t
                    5. step rewards of the rollout steps at step_t: step^1_t, step^2_t, ..., step^k_t
                    6. question of the rollout steps at step_t: step^1_t, step^2_t, ..., step^k_t
                    7. ground truth of the rollout steps at step_t: step^1_t, step^2_t, ..., step^k_t
                    8. batch id of the rollout steps at step_t: step^1_t, step^2_t, ..., step^k_t
                    '''

            # users can customize different sampling_params at different run
            #breakpoint()
            batch_size_after_mcts = len(rollout_results)
            if batch_size_after_mcts == 0:
                print("No valid samples from MCTS (which means no correct answer is reached or all rollouts lead to correct answers), just randomly select some zero-advantage samples") 
                
            input_ids_after_mcts = []
            input_text = [] # selected prompt text, including the question and partial solutions, for processor
            images = [] # selected images
            response_ids_after_mcts = []
            step_rewards_all = []
            questions_after_mcts = []
            gts_after_mcts = []
            batch_ids_after_mcts = []

            for (prompt_ids, prompt_text, image, reponse_ids, step_rewards, question, gt, batch_ids) in rollout_results:
                input_ids_after_mcts.append(prompt_ids)
                input_text.append(prompt_text)
                images.append(image)
                response_ids_after_mcts.extend(reponse_ids)
                step_rewards_all.extend(step_rewards)
                questions_after_mcts.extend(question)
                gts_after_mcts.extend(gt)
                batch_ids_after_mcts.extend(batch_ids)
            
            no_correct_answer = False
            
            # if the totoal number of selected prompts is less than the batch size, then randomly select some of them to fill the batch size
            if batch_size_after_mcts < original_batch_size or batch_size_after_mcts > original_batch_size:
                if batch_size_after_mcts < original_batch_size:
                    print(f"Original bsz: {original_batch_size}, selected samples from MCTS: {batch_size_after_mcts}, randomly select {original_batch_size-batch_size_after_mcts} samples to fill the batch size")
                    selected_indices = np.random.choice(
                        list(range(batch_size_after_mcts)), original_batch_size-batch_size_after_mcts, replace=True
                    )
                elif batch_size_after_mcts > original_batch_size:
                    print(f"Original bsz: {original_batch_size}, selected samples from MCTS: {batch_size_after_mcts}, randomly select {original_batch_size} samples from the selected samples")
                    selected_indices = np.random.choice(
                        list(range(batch_size_after_mcts)), original_batch_size, replace=False
                    )
                    input_ids_after_mcts = []
                    input_text = [] # selected prompt text, including the question and partial solutions, for processor
                    images = [] # selected images
                    response_ids_after_mcts = []
                    step_rewards_all = []
                    questions_after_mcts = []
                    gts_after_mcts = []
                    batch_ids_after_mcts = []
                for id in selected_indices:
                    (prompt_ids, prompt_text, image, reponse_ids, step_rewards, question, gt, batch_ids) = rollout_results[id]
                    input_ids_after_mcts.append(prompt_ids)
                    input_text.append(prompt_text)
                    images.append(image)
                    response_ids_after_mcts.extend(reponse_ids)
                    step_rewards_all.extend(step_rewards)
                    questions_after_mcts.extend(question)
                    gts_after_mcts.extend(gt)
                    batch_ids_after_mcts.extend(batch_ids)
            #print(f"response_ids_after_mcts2={response_ids_after_mcts}")
            # left pad the collected input_ids (list of int) from the MCTS rollout results and convert to tensor
            '''input_ids = VF.pad_2d_list_to_length(
                input_ids_after_mcts, self.pad_token_id, max_length=self.config.prompt_length, left_or_right="left"
            ).to(input_ids.device)'''
            #breakpoint()
                
                
            response_ids = VF.pad_2d_list_to_length(
                    response_ids_after_mcts, self.pad_token_id, max_length=self.config.response_length, left_or_right="right"
                ).to(input_ids.device)
            
            
            
            # recompute the input_ids, attention mask, position ids and multimodal_inputs for the selected prompts from MCTS
            input_ids, attention_mask, position_ids, reprocessed_multimodal_inputs, reprocessed_raw_input_ids = self.recompute_attn_mask_and_pos_ids(
                images, input_text, device
            )
                

            #breakpoint()
            if self.sampling_params.n > 1:
                batch_size = original_batch_size * self.sampling_params.n
                #print(f"Total samples selected from MCTS: {batch_size}")
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        
        
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)


        non_tensor_batch["ground_truth"] = np.array(gts_after_mcts)
        non_tensor_batch["problem"] = np.array(questions_after_mcts)
        non_tensor_batch["step_rewards"] = np.array(step_rewards_all)
        non_tensor_batch["batch_ids"] = np.array(batch_ids_after_mcts)
        non_tensor_batch["multi_modal_inputs"] = np.array(reprocessed_multimodal_inputs)


        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)  
    
    
    @torch.no_grad()
    def generate_sequences_mc(self, prompts: DataProto) -> DataProto:
        print("Actor RSS (GB):", ray.get(self.hash_server.get_rss.remote()))
        #print(ray.memory())         
        #print(ray.memory_stats()) 
        
        # left-padded attention_mask
        delim = self.config.mc.delim
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]

        assert self.config.pr_batch_size % self.config.n_gpus_per_node == 0, "pr_batch_size should be divisible by n_gpus_per_node"
        original_batch_size = self.config.pr_batch_size // self.config.n_gpus_per_node

        device = input_ids.device

        non_tensor_batch = prompts.non_tensor_batch
        if self.config.pr_batch_size == -1 and original_batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")


        correct_solutions_text = list(non_tensor_batch.pop("correct_solutions_text"))
        batch_sample_idx = list(non_tensor_batch.pop("global_index"))
        batch_gts = list(non_tensor_batch["ground_truth"])
        batch_problems = list(non_tensor_batch["problem"])
        non_tensor_batch.pop("raw_prompt_ids")

        assert "multi_modal_data" in non_tensor_batch
        
        with self.update_sampling_params(**prompts.meta_info):
            batch_images = []
            batch_input_text = []

            batch_accum_steps, batch_indiv_steps, _ = self.mcts_solver.split_solutions_into_accum_steps(solutions=correct_solutions_text, delim=delim, add_empty_step_in_front=True)
            batch_prompts_before_processor = list(non_tensor_batch.pop("prompt_before_processor"))
            batch_multi_modal_data = list(non_tensor_batch.pop("multi_modal_data")) 
            batch_next_correct_step = [] # partial solutions with the next step for building the GRPO group later, no question str
            new_batch_global_ids = [] # id in the whole training dataset
            new_batch_gts = []
            new_batch_problems = []
            new_batch_multimodal_data = []
            batch_ids = [] # id in the batch
            group_ids = [] # group id for the GRPO group, used to group the same question together
            group_id = 0
            
            # traverse each question in the batch
            for batch_id, (questions_str, multi_modal_data, accum_steps_strs, indiv_steps_strs, id, gt, problem) in \
                enumerate(zip(batch_prompts_before_processor, batch_multi_modal_data, batch_accum_steps, batch_indiv_steps, batch_sample_idx, batch_gts, batch_problems)):
                num_steps = len(accum_steps_strs)
                # traverse each step in the solution

                remove_last_steps = self.config.mc.drop_last_steps
                while num_steps <= remove_last_steps:
                    remove_last_steps -= 1
                if remove_last_steps > 1:
                    accum_steps_strs = accum_steps_strs[:-remove_last_steps]
                    indiv_steps_strs = indiv_steps_strs[:-remove_last_steps+1]
                elif remove_last_steps == 1:
                    accum_steps_strs = accum_steps_strs[:-remove_last_steps]
                    

                for i, (accum_steps_str, indiv_steps_str) in enumerate(zip(accum_steps_strs, indiv_steps_strs)): 
                    batch_images.append(multi_modal_data["image"])
                    batch_input_text.append(questions_str + accum_steps_str)
                    batch_next_correct_step.append(indiv_steps_str) # collect the partial solution with the next step for building the GRPO group later
                    new_batch_global_ids.append(id)
                    new_batch_gts.append(gt)
                    new_batch_problems.append(problem)
                    new_batch_multimodal_data.append(multi_modal_data)
                    batch_ids.append(batch_id)
                    group_ids.append(group_id)
                    group_id += 1
            

            '''
            input_text becomes:
            [ question_1+step_1, question_1+step_2, ..., question_1+step_{t-1},
              question_2+step_1, question_2+step_2, ..., question_2+step_{t-1},
              ...
              question_{bsz}+step_1, question_{bsz}+step_2, ..., question_{bsz}+step_{t-1} ]
            '''    
            
            # resize (resample or discard) the data to fit the pr_batch_size, ensure the samples of the same question are grouped together
            if not self.config.mc.remove_all_correct_groups and not self.config.mc.remove_high_acc_group:
                [batch_images, batch_input_text, batch_next_correct_step, new_batch_gts, new_batch_problems, new_batch_multimodal_data, batch_ids, group_ids], new_batch_global_ids = \
                    resize_lists_by_sampling([batch_images, batch_input_text, batch_next_correct_step, new_batch_gts, new_batch_problems, new_batch_multimodal_data, batch_ids, group_ids], new_batch_global_ids,\
                                                    self.config.pr_batch_size//self.config.n_gpus_per_node, seed=42)
                
            # construct vllm inputs
            input_ids, attention_mask, position_ids, reprocessed_multimodal_inputs, reprocessed_raw_input_ids = self.recompute_attn_mask_and_pos_ids(
                batch_images, batch_input_text, device
            )
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(reprocessed_raw_input_ids, new_batch_multimodal_data):
                vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})

            #breakpoint()
            # RL rollout
            original_sampling_params_n = self.sampling_params.n
            setattr(self.sampling_params, "n", self.sampling_params.n-1)
            setattr(self.sampling_params, "detokenize", True)
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=(self.rank == 0)
            )
            setattr(self.sampling_params, "n", original_sampling_params_n)
            setattr(self.sampling_params, "detokenize", False)

            # traverse all partial solutions of all samples in the batch. Here, all steps of all samples are listed in completions
            # each completion contains the self.sampling_params.n-1 rollouts for a partial step of a sample
            # judge the correctness, get the `lead_to_correct` for each step
            batch_first_steps = defaultdict(list) # sample_id -> list of first step str collected from the RL rollouts
            batch_correctness_refs, batch_embeds, batch_step_infos = [], [], []
            for completion, global_id, gt in tqdm(zip(completions, new_batch_global_ids, new_batch_gts),
                desc="Using rule-based judge to check the correctness of the RL rollouts... (msg from rank 0)",
                total=len(completions), disable=(self.rank != 0)):
                # traverse the rollouts starting from a partial solution
                for output in completion.outputs: 
                    response_str = output.text
                    batch_correctness_refs.append(self.rule_based_judge_server.compute_rule_based_judge_with_string.remote(response_str, gt))
                    steps = self.mcts_solver.split_solution_into_individual_steps(response_str, delim=delim) # List[str]
                    if steps == []:
                        steps = [response_str] # to avoid the error of self.embed_tokenizer([])
                    batch_embeds.append(self.compute_embeds(steps))
                    batch_step_infos.append((steps, global_id))
                                
             
            #update the global step hash dict       
            batch_correctness = ray.get(batch_correctness_refs)
            for step_info, embeds, correctness in tqdm(zip(batch_step_infos, batch_embeds, batch_correctness),
                desc="Updating the step hash dict using the RL rollout results... (msg from rank 0)",
                total=len(batch_step_infos), disable=(self.rank != 0)):
                steps, global_id = step_info
                batch_first_steps[global_id].append({"str": steps[0], "emb": embeds[0]})
                upd_ref = self.hash_server.update_sample_step_hash_dict.remote(
                            global_id, steps, embeds, [correctness] * len(steps)
                        )
            upd_refs = ray.get(upd_ref) # wait for the update to finish
            # Only when the preceeding codes rely on the previous execution results do we need to use ray.get to wait for the execution to complete.
            #breakpoint()
            
            # lookup the step_hash_dict to get the correctness of the first steps, and get the encoded ids of the first steps
            tokenizer = self.inference_engine.get_tokenizer()
            batch_first_step_correctness_ref = []
            batch_first_step_ids = []
            for global_id, first_steps in tqdm(batch_first_steps.items(), \
                desc="Looking up the step hash dict to get the correctness of the first steps... (msg from rank 0)",
                total=len(batch_first_steps), disable=(self.rank != 0)):
                first_steps_strs = [first_step["str"] for first_step in first_steps]
                #first_steps_embeds = np.array([first_step["emb"] for first_step in first_steps])
                batch_first_step_correctness_ref.append(self.hash_server.look_up_step_correctness.remote(global_id, first_steps_strs)) # list[bool]
                batch_first_step_ids.append(
                    tokenizer.batch_encode_plus(
                        first_steps_strs, # list[str]
                        add_special_tokens=False,
                        padding=False,
                        return_attention_mask=False,
                    )["input_ids"]  # List[List[int]], each list has different length
                )
            # flatten the list (flatten the sublists of different sample_ids, stack them together)
            batch_first_step_correctness = ray.get(batch_first_step_correctness_ref) # wait for the lookup to finish
            batch_first_step_correctness = [correctness for correctness_of_a_sample_id in batch_first_step_correctness for correctness in correctness_of_a_sample_id]    
            batch_first_step_ids = [first_step_id for first_step_ids_of_a_sample_id in batch_first_step_ids for first_step_id in first_step_ids_of_a_sample_id] # flatten the list of lists
                
            # collecting the first steps, and look up their correctness in the step_hash_dict
            batch_response_ids = []
            batch_step_correctness = []
            first_ids_iter = iter(batch_first_step_ids)
            first_correctness_iter = iter(batch_first_step_correctness)
            avg_group_acc = []
            group_all_correct_record = []
            high_acc_group_record = []
            for completion, correct_step_str in\
                zip(completions, batch_next_correct_step):
                group_acc = []
                for output in completion.outputs:
                    first_step_ids = next(first_ids_iter)               
                    correctness    = next(first_correctness_iter) 
                    batch_response_ids.append(first_step_ids)
                    batch_step_correctness.append(1.0 if correctness else 0.0)
                    group_acc.append(1 if correctness else 0)
                batch_response_ids.append(tokenizer.encode(correct_step_str, add_special_tokens=False))
                batch_step_correctness.append(1.0) # this step is on the correct path always correct
                group_acc.append(1)
                avg_group_acc.append(sum(group_acc) / len(group_acc)) # average accuracy of the group
                group_all_correct_record.append(0 if sum(group_acc) == len(group_acc) else 1) # whether all steps in the group are correct
                high_acc_group_record.append(0 if sum(group_acc) >= len(group_acc) * self.config.mc.group_high_acc_threshold and sum(group_acc) < len(group_acc) else 1) # whether the group is high accuracy

            batch_rl_rollout_acc = batch_correctness.count(True) / len(batch_correctness)
            batch_step_look_up_acc = sum(avg_group_acc) / len(avg_group_acc)
            batch_all_correct_ratio = 1 - sum(group_all_correct_record) / len(group_all_correct_record)
            batch_high_acc_ratio = 1 - sum(high_acc_group_record) / len(high_acc_group_record)
            #step_dict_info = self.hash_server.get_step_dict_info.remote()
            
            
            if self.rank == 0:
                print(f"RL rollout batch acc: {batch_rl_rollout_acc }; Avg group acc of a batch: {batch_step_look_up_acc}; All correct ratio: {batch_all_correct_ratio}; Group acc>={self.config.mc.group_high_acc_threshold} ratio: {batch_high_acc_ratio}")
            assert len(batch_response_ids) % self.sampling_params.n == 0 and len(batch_step_correctness) % self.sampling_params.n == 0, \
                f"len(batch_response_ids) % self.sampling_params.n = {len(batch_response_ids) % self.sampling_params.n}, len(batch_step_correctness) % self.sampling_params.n = {len(batch_step_correctness) % self.sampling_params.n}"


            # remove zero-advantage samples (all correct)
            # if 1, then the group is not all correct, select it
            #breakpoint()
            if self.config.mc.remove_all_correct_groups or self.config.mc.remove_high_acc_group:
                if self.config.mc.remove_all_correct_groups:
                    mask = group_all_correct_record
                elif self.config.mc.remove_high_acc_group:
                    mask = high_acc_group_record
                    
                new_batch_global_ids, batch_response_ids, new_batch_gts, new_batch_problems, batch_step_correctness, batch_ids, group_ids, reprocessed_multimodal_inputs, input_ids, attention_mask, position_ids = select_by_mask_for_list_of_sequences(
                    [new_batch_global_ids, batch_response_ids, new_batch_gts, new_batch_problems, batch_step_correctness, batch_ids, group_ids, reprocessed_multimodal_inputs, input_ids, attention_mask, position_ids],
                    mask
                )
                
                #breakpoint()
                [batch_response_ids, new_batch_gts, new_batch_problems, batch_step_correctness, batch_ids, group_ids, reprocessed_multimodal_inputs, input_ids, attention_mask, position_ids], new_batch_global_ids = resize_lists_by_sampling(
                    [batch_response_ids, new_batch_gts, new_batch_problems, batch_step_correctness, batch_ids, group_ids, reprocessed_multimodal_inputs, input_ids, attention_mask, position_ids],
                    new_batch_global_ids,
                    self.config.pr_batch_size//self.config.n_gpus_per_node, seed=42)

            if self.config.mc.shuffle_groups_across_samples:
                [batch_response_ids, new_batch_gts, new_batch_problems, batch_step_correctness, batch_ids, group_ids, reprocessed_multimodal_inputs, input_ids, attention_mask, position_ids], new_batch_global_ids = shuffle_lists(
                    [batch_response_ids, new_batch_gts, new_batch_problems, batch_step_correctness, batch_ids, group_ids, reprocessed_multimodal_inputs, input_ids, attention_mask, position_ids],
                    new_batch_global_ids,
                    seed=42)


            response_ids = VF.pad_2d_list_to_length(
                batch_response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)


            if self.sampling_params.n > 1:
                batch_size = original_batch_size * self.sampling_params.n
                #print(f"Total samples selected from MCTS: {batch_size}")
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        #breakpoint()
        
        
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)


        non_tensor_batch["ground_truth"] = np.array(repeat_elements_interleaved(new_batch_gts, self.sampling_params.n))
        non_tensor_batch["problem"] = np.array(repeat_elements_interleaved(new_batch_problems, self.sampling_params.n))
        non_tensor_batch["step_rewards"] = np.array(batch_step_correctness)
        non_tensor_batch["batch_ids"] = np.array(repeat_elements_interleaved(batch_ids, self.sampling_params.n))
        #non_tensor_batch["group_ids"] = np.array(self.repeat_elements_interleaved(group_ids, self.sampling_params.n))
        non_tensor_batch["multi_modal_inputs"] = np.array(reprocessed_multimodal_inputs)
        
        meta_info = {
            "rl_rollout_acc": batch_rl_rollout_acc,
            "step_look_up_acc": batch_step_look_up_acc,
            "all_correct_group_ratio": batch_all_correct_ratio,
            "high_acc_group_ratio": batch_high_acc_ratio
            #"step_dict_info": step_dict_info
        }


        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        #breakpoint()
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    @torch.no_grad()
    def generate_sequences_mc2(self, prompts: DataProto) -> DataProto:
        print("Actor RSS (GB):", ray.get(self.hash_server.get_rss.remote()))
        #print(ray.memory())         
        #print(ray.memory_stats()) 
        #breakpoint()
        # left-padded attention_mask
        
        delim = self.config.mc.delim # "### Step "
        tokenizer = self.inference_engine.get_tokenizer()
        delim_ids_list = [tokenizer.encode(prefix+delim+suffix, add_special_tokens=False) for prefix in [""," "] for suffix in ["", "s"]]
        im_end_token = tokenizer.encode("<|im_end|>")
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]

        assert self.config.pr_batch_size % self.config.n_gpus_per_node == 0, "pr_batch_size should be divisible by n_gpus_per_node"
        original_batch_size = self.config.pr_batch_size // self.config.n_gpus_per_node

        device = input_ids.device

        non_tensor_batch = prompts.non_tensor_batch
        if self.config.pr_batch_size == -1 and original_batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")


        correct_solutions_text_ = list(non_tensor_batch.pop("correct_solutions_text"))
        correct_solutions_text = []
        for text in correct_solutions_text_: # remove >= 4 consecutive '#' 
            correct_solutions_text.append(clean_header(text))
        batch_sample_idx = list(non_tensor_batch.pop("global_index"))
        batch_gts = list(non_tensor_batch["ground_truth"])
        batch_problems = list(non_tensor_batch["problem"])
        non_tensor_batch.pop("raw_prompt_ids")

        assert "multi_modal_data" in non_tensor_batch
        #breakpoint()
        with self.update_sampling_params(**prompts.meta_info):
            batch_images = []
            batch_input_text = []

            batch_accum_steps, _, batch_remain_steps = self.mcts_solver.split_solutions_into_accum_steps(solutions=correct_solutions_text, delim=delim, add_empty_step_in_front=True)
            batch_prompts_before_processor = list(non_tensor_batch.pop("prompt_before_processor"))
            batch_multi_modal_data = list(non_tensor_batch.pop("multi_modal_data")) 
            batch_remain_correct_step = [] # partial solutions with the next step for building the GRPO group later, no question str
            new_batch_global_ids = [] # id in the whole training dataset
            new_batch_gts = []
            new_batch_problems = []
            new_batch_multimodal_data = []
            batch_ids = [] # id in the batch
            group_ids = [] # group id for the GRPO group, used to group the same question together
            group_id = 0
            
            # traverse each question in the batch
            for batch_id, (questions_str, multi_modal_data, accum_steps_strs, remain_steps_strs, id, gt, problem) in \
                enumerate(zip(batch_prompts_before_processor, batch_multi_modal_data, batch_accum_steps, batch_remain_steps, batch_sample_idx, batch_gts, batch_problems)):
                num_steps = len(accum_steps_strs)
                

                remove_last_steps = self.config.mc.drop_last_steps
                while num_steps <= remove_last_steps:
                    remove_last_steps -= 1
                if remove_last_steps > 1:
                    accum_steps_strs = accum_steps_strs[:-remove_last_steps]
                    remain_steps_strs = remain_steps_strs[:-remove_last_steps+1]
                elif remove_last_steps == 1:
                    accum_steps_strs = accum_steps_strs[:-remove_last_steps]
                    
                # traverse each step in the solution
                for i, (accum_steps_str, remain_steps_str) in enumerate(zip(accum_steps_strs, remain_steps_strs)):
                    batch_images.append(multi_modal_data["image"])
                    batch_input_text.append(questions_str + accum_steps_str)
                    batch_remain_correct_step.append(remain_steps_str) # collect the partial solution with the next step for building the GRPO group later
                    new_batch_global_ids.append(id)
                    new_batch_gts.append(gt)
                    new_batch_problems.append(problem)
                    new_batch_multimodal_data.append(multi_modal_data)
                    batch_ids.append(batch_id)
                    group_ids.append(group_id)
                    group_id += 1
            

            '''
            input_text becomes:
            [ question_1+step_1, question_1+step_2, ..., question_1+step_{t-1},
              question_2+step_1, question_2+step_2, ..., question_2+step_{t-1},
              ...
              question_{bsz}+step_1, question_{bsz}+step_2, ..., question_{bsz}+step_{t-1} ]
            '''    
            
            # resize (resample or discard) the data to fit the pr_batch_size, ensure the samples of the same question are grouped together
            if not self.config.mc.remove_all_correct_groups and not self.config.mc.remove_high_acc_group:
                [batch_images, batch_input_text, batch_remain_correct_step, new_batch_gts, new_batch_problems, new_batch_multimodal_data, batch_ids, group_ids], new_batch_global_ids = \
                    resize_lists_by_sampling([batch_images, batch_input_text, batch_remain_correct_step, new_batch_gts, new_batch_problems, new_batch_multimodal_data, batch_ids, group_ids], new_batch_global_ids,\
                                                    self.config.pr_batch_size//self.config.n_gpus_per_node, seed=42)
                
            # construct vllm inputs
            input_ids, attention_mask, position_ids, reprocessed_multimodal_inputs, reprocessed_raw_input_ids = self.recompute_attn_mask_and_pos_ids(
                batch_images, batch_input_text, device
            )
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(reprocessed_raw_input_ids, new_batch_multimodal_data):
                vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})

            #breakpoint()
            # RL rollout
            original_sampling_params_n = self.sampling_params.n
            setattr(self.sampling_params, "n", self.sampling_params.n-1)
            setattr(self.sampling_params, "detokenize", True)
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=(self.rank == 0)
            )
            setattr(self.sampling_params, "n", original_sampling_params_n)
            setattr(self.sampling_params, "detokenize", False)

            # traverse all partial solutions of all samples in the batch. Here, all steps of all samples are listed in completions
            # each completion contains the self.sampling_params.n-1 rollouts for a partial step of a sample
            # judge the correctness, get the `lead_to_correct` for each step
            #breakpoint( )
            batch_steps = defaultdict(list) # sample_id -> list of first step str collected from the RL rollouts
            batch_correctness_refs, batch_embeds, batch_step_infos = [], [], []
            for completion, global_id, gt in tqdm(zip(completions, new_batch_global_ids, new_batch_gts),
                desc="Using rule-based judge to check the correctness of the RL rollouts... (msg from rank 0)",
                total=len(completions), disable=(self.rank != 0)):
                # traverse the rollouts starting from a partial solution
                for output in completion.outputs: 
                    response_str_ = output.text
                    response_ids = output.token_ids
                    response_str = clean_header(response_str_) # remove >= 4 consecutive '#'
                    #if self.rank == 0:
                    #    print("########################################")
                    #    print(f"re response_str={response_str}")
                    
                    response_ids = tokenizer.encode(response_str, add_special_tokens=False)[:self.config.response_length]
                    if len(response_ids) < self.config.response_length:
                        response_ids+=im_end_token
                    batch_correctness_refs.append(self.rule_based_judge_server.compute_rule_based_judge_with_string.remote(response_str, gt))
                    steps = self.mcts_solver.split_solution_into_individual_steps(response_str, delim=delim) # List[str]
                    if steps == []:
                        steps = [response_str] # to avoid the error of self.embed_tokenizer([])
                    batch_embeds.append(self.compute_embeds(steps))
                    batch_step_infos.append((response_str, response_ids, steps, global_id))
                                
            #breakpoint()
            #update the global step hash dict       
            batch_correctness = ray.get(batch_correctness_refs)
            upd_refs=[]
            for step_info, embeds, correctness in tqdm(zip(batch_step_infos, batch_embeds, batch_correctness),
                desc="Updating the step hash dict using the RL rollout results... (msg from rank 0)",
                total=len(batch_step_infos), disable=(self.rank != 0)):
                response_str, response_ids, steps, global_id = step_info
                batch_steps[global_id].append({"full_resp_str": response_str, "full_resp_ids": response_ids, "steps": steps, "emb": embeds})
                upd_refs.append(self.hash_server.update_sample_step_hash_dict.remote(
                            global_id, steps, embeds, [correctness] * len(steps)
                        ))
            start_time =time.time()
            batch_resp_step_correctness = ray.get(upd_refs) # wait for the update to finish

            print(f"[Rank {self.rank}] Time taken to update the step hash dict and get the correctness: {time.time()-start_time:.2f} seconds")
            # Only when the preceeding codes rely on the previous execution results do we need to use ray.get to wait for the execution to complete.
            #breakpoint()
            
            
            # lookup the step_hash_dict to get the correctness of the first steps, and get the encoded ids of the first steps
            #batch_resp_step_correctness_ref = []
            
            batch_resp_step_correctness_poss = []
            batch_resp_delim_poss_rl_rollout = []
            batch_resp_step_ids = []
            for global_id, resp_steps_and_embeds in tqdm(batch_steps.items(), \
                desc="Looking up the step hash dict to get the correctness of the first steps... (msg from rank 0)",
                total=len(batch_steps), disable=(self.rank != 0)):
                for resp_steps_and_embeds_of_a_intermediate_step in resp_steps_and_embeds:
                    full_resp_str = resp_steps_and_embeds_of_a_intermediate_step["full_resp_str"]
                    full_resp_ids = resp_steps_and_embeds_of_a_intermediate_step["full_resp_ids"]
                    steps = resp_steps_and_embeds_of_a_intermediate_step["steps"]
                    delim_poss, poss = find_delim_positions(full_resp_ids, delim_ids_list)
                    #if len(poss) == 1:
                    #    print(steps)
                    try:
                        assert len(poss) == len(steps), \
                            f"len(poss)={len(poss)}, len(steps)={len(steps)}, full_resp_str={full_resp_str}, full_resp_ids={full_resp_ids}"
                    except AssertionError as e:
                        print(f"[Warning] {e}")
                        min_len = min(len(poss), len(steps))
                        if len(poss) > len(steps):
                            poss[:] = poss[:min_len]
                            print(f"[Fix] Truncated poss to length {min_len}. This may cause inaccurate reward for this response.")
                        elif len(steps) > len(poss):
                            steps[:] = steps[:min_len]
                            print(f"[Fix] Truncated steps to length {min_len}. This may cause inaccurate reward for this response.")
                    
                    batch_resp_step_ids.append(full_resp_ids)
                    #steps_correctness = self.hash_server.look_up_step_correctness.remote(global_id, steps)
                    #batch_resp_step_correctness_ref.append(steps_correctness) # list[bool]                        
                    batch_resp_step_correctness_poss.append(poss) # list[List[int]], each list has different length
                    batch_resp_delim_poss_rl_rollout.append(delim_poss) # list[List[int]], each list has different length
            # flatten the list (flatten the sublists of different sample_ids, stack them together)

            #batch_resp_step_correctness = ray.get(batch_resp_step_correctness_ref)
            start_time =time.time()
            

            batch_resp_ids = []
            batch_resp_step_rewards = []
            batch_resp_step_end_poss = []
            batch_min_mean_correct_resp_lens = []
            batch_resp_delim_poss = []
            batch_resp_ids_iter = iter(batch_resp_step_ids)
            batch_resp_steps_correctness_iter = iter(batch_resp_step_correctness)
            batch_resp_step_correctness_poss_iter = iter(batch_resp_step_correctness_poss)
            batch_resp_delim_poss_rl_rollout_iter = iter(batch_resp_delim_poss_rl_rollout)
            avg_group_acc = []
            group_all_correct_record = []
            high_acc_group_record = []
            for completion, correct_remain_step_str, global_id in\
                zip(completions, batch_remain_correct_step, new_batch_global_ids):
                group_acc = []
                for output in completion.outputs:
                    step_ids = next(batch_resp_ids_iter)               
                    steps_correctness = next(batch_resp_steps_correctness_iter) 
                    resp_step_correctness_poss = next(batch_resp_step_correctness_poss_iter)
                    resp_delim_poss = next(batch_resp_delim_poss_rl_rollout_iter)
                    #########################################################################################
                    # crucial values
                    batch_resp_ids.append(step_ids)
                    batch_resp_step_rewards.append([1.0 if correct else 0.0 for correct in steps_correctness]) # all steps in the response are correct
                    batch_resp_step_end_poss.append(resp_step_correctness_poss)
                    batch_resp_delim_poss.append(resp_delim_poss)
                    #########################################################################################
                    for correctness in steps_correctness:
                        group_acc.append(1 if correctness else 0)
                correct_ids = tokenizer.encode(correct_remain_step_str, add_special_tokens=False)[:self.config.response_length]
                if len(correct_ids) < self.config.response_length:
                    correct_ids+=im_end_token
                batch_resp_ids.append(correct_ids)
                correct_delim_poss, correct_step_poss = find_delim_positions(correct_ids, delim_ids_list)
                batch_resp_step_end_poss.append(correct_step_poss)
                batch_resp_delim_poss.append(correct_delim_poss)
                batch_resp_step_rewards.append([1.0]*len(correct_step_poss))
                min_len, mean_len = ray.get(self.hash_server.look_up_min_mean_correct_resp_len.remote(global_id))
                batch_min_mean_correct_resp_lens.extend([min_len] * original_sampling_params_n if min_len<float("inf") else [mean_len] * original_sampling_params_n) # following SOL https://arxiv.org/pdf/2504.21370#page=4.56, use the mean length if min_len is inf
                for _ in correct_step_poss:
                    group_acc.append(1)
                avg_group_acc.append(sum(group_acc) / len(group_acc)) # average accuracy of the group
                group_all_correct_record.append(0 if sum(group_acc) == len(group_acc) else 1) # whether all steps in the group are correct
                high_acc_group_record.append(0 if sum(group_acc) >= len(group_acc) * self.config.mc.group_high_acc_threshold and sum(group_acc) < len(group_acc) else 1) # whether the group is high accuracy

            batch_rl_rollout_acc = batch_correctness.count(True) / len(batch_correctness)
            batch_step_look_up_acc = sum(avg_group_acc) / len(avg_group_acc)
            batch_all_correct_ratio = 1 - sum(group_all_correct_record) / len(group_all_correct_record)
            batch_high_acc_ratio = 1 - sum(high_acc_group_record) / len(high_acc_group_record)
            #step_dict_info = self.hash_server.get_step_dict_info.remote()
            if self.rank==0:
                print(f"Time taken to build the return values: {time.time()-start_time:.2f} seconds")
            
            if self.rank == 0:
                print(f"RL rollout batch acc: {batch_rl_rollout_acc }; Avg group of a batch after looking up: {batch_step_look_up_acc}; All correct ratio: {batch_all_correct_ratio}; Group acc>={self.config.mc.group_high_acc_threshold} ratio: {batch_high_acc_ratio}")
            assert len(batch_resp_ids) % self.sampling_params.n == 0 and  len(batch_resp_step_end_poss) % self.sampling_params.n == 0 and len(batch_resp_step_rewards) % self.sampling_params.n == 0, \
                f"len(batch_resp_ids) == {len(batch_resp_ids)}, len(batch_resp_step_end_poss) == {len(batch_resp_step_end_poss)}, len(batch_resp_step_rewards) == {len(batch_resp_step_rewards)}"


            # remove zero-advantage samples (all correct)
            # if 1, then the group is not all correct, select it
            #breakpoint()
            start_time =time.time()
            if self.config.mc.remove_all_correct_groups or self.config.mc.remove_high_acc_group:
                if self.config.mc.remove_all_correct_groups:
                    mask = group_all_correct_record
                elif self.config.mc.remove_high_acc_group:
                    mask = high_acc_group_record
                    
                new_batch_global_ids, batch_resp_ids, new_batch_gts, new_batch_problems, batch_resp_step_rewards, batch_min_mean_correct_resp_lens, batch_resp_step_end_poss, batch_resp_delim_poss, batch_ids, group_ids, reprocessed_multimodal_inputs, input_ids, attention_mask, position_ids = select_by_mask_for_list_of_sequences(
                    [new_batch_global_ids, batch_resp_ids, new_batch_gts, new_batch_problems, batch_resp_step_rewards, batch_min_mean_correct_resp_lens, batch_resp_step_end_poss, batch_resp_delim_poss, batch_ids, group_ids, reprocessed_multimodal_inputs, input_ids, attention_mask, position_ids],
                    mask
                )
                
                #breakpoint()
                [batch_resp_ids, new_batch_gts, new_batch_problems, batch_resp_step_rewards, batch_min_mean_correct_resp_lens, batch_resp_step_end_poss, batch_resp_delim_poss, batch_ids, group_ids, reprocessed_multimodal_inputs, input_ids, attention_mask, position_ids], new_batch_global_ids = resize_lists_by_sampling(
                    [batch_resp_ids, new_batch_gts, new_batch_problems, batch_resp_step_rewards, batch_min_mean_correct_resp_lens, batch_resp_step_end_poss, batch_resp_delim_poss, batch_ids, group_ids, reprocessed_multimodal_inputs, input_ids, attention_mask, position_ids],
                    new_batch_global_ids,
                    self.config.pr_batch_size//self.config.n_gpus_per_node, seed=42)

            if self.config.mc.shuffle_groups_across_samples:
                [batch_resp_ids, new_batch_gts, new_batch_problems, batch_resp_step_rewards, batch_min_mean_correct_resp_lens, batch_resp_step_end_poss, batch_resp_delim_poss, batch_ids, group_ids, reprocessed_multimodal_inputs, input_ids, attention_mask, position_ids], new_batch_global_ids = shuffle_lists(
                    [batch_resp_ids, new_batch_gts, new_batch_problems, batch_resp_step_rewards, batch_min_mean_correct_resp_lens, batch_resp_step_end_poss, batch_resp_delim_poss, batch_ids, group_ids, reprocessed_multimodal_inputs, input_ids, attention_mask, position_ids],
                    new_batch_global_ids,
                    seed=42)


            response_ids = VF.pad_and_clip_2d_list_to_length(
                batch_resp_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)


            if self.sampling_params.n > 1:
                batch_size = original_batch_size * self.sampling_params.n
                #print(f"Total samples selected from MCTS: {batch_size}")
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
            if self.rank==0:
                print(f"Time taken of remaining part 1: {time.time()-start_time:.2f} seconds")
            
            
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        #breakpoint()
        
        
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)


        non_tensor_batch["ground_truth"] = np.array(repeat_elements_interleaved(new_batch_gts, self.sampling_params.n))
        non_tensor_batch["problem"] = np.array(repeat_elements_interleaved(new_batch_problems, self.sampling_params.n))
        non_tensor_batch["full_step_rewards"] = np.array(batch_resp_step_rewards,  dtype=object)  # list of lists, each list contains the rewards of the steps in the response
        non_tensor_batch["ref_resp_lengths"] = np.array(batch_min_mean_correct_resp_lens)  # list of int, each element is the min or mean correct response length
        non_tensor_batch["step_end_positions"] = np.array(batch_resp_step_end_poss,  dtype=object)
        non_tensor_batch["delim_positions"] = np.array(batch_resp_delim_poss,  dtype=object)  # list of lists, each list contains the positions of the delimiters in the response
        non_tensor_batch["batch_ids"] = np.array(repeat_elements_interleaved(batch_ids, self.sampling_params.n))
        #non_tensor_batch["group_ids"] = np.array(self.repeat_elements_interleaved(group_ids, self.sampling_params.n))
        non_tensor_batch["multi_modal_inputs"] = np.array(reprocessed_multimodal_inputs)
        #breakpoint()
        meta_info = {
            "rl_rollout_acc": batch_rl_rollout_acc,
            "step_look_up_acc": batch_step_look_up_acc,
            "all_correct_group_ratio": batch_all_correct_ratio,
            "high_acc_group_ratio": batch_high_acc_ratio
            #"step_dict_info": step_dict_info
        }

        #for name, value in zip(['input_ids', 'response_ids', 'sequence_ids', 'attention_mask', 'response_mask', 'position_ids'],
        #                      [input_ids, response_ids, sequence_ids, attention_mask, response_mask, position_ids]):
        #    if value.shape[1]>self.config.response_length:
        #        print(f"[Warning] {name} shape {value.shape} exceeds response_length {self.config.response_length}")
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        #breakpoint()
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    
    @torch.no_grad()
    def compute_embeds(self, texts: List[str]) -> np.ndarray:
        outputs = self.embed_model.embed(texts, use_tqdm=False)
        #outputs = ray.get(self.embed_model.encode.remote(texts, use_tqdm=False)) # old, worked
        return torch.tensor([o.outputs.embedding for o in outputs]).detach().half().cpu().numpy().copy()
        '''def compute_embeds(self, texts):
            inputs = self.embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.embed_model.device) for k, v in inputs.items()}
            # Get the embeddings
            with torch.no_grad():
                embeddings = self.embed_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings.detach().cpu().numpy().copy()'''
    
    '''@torch.no_grad()
    def compute_rule_based_judge_with_string(self, data: DataProto):
        assert self._is_rollout
        correctness = []
        response_strs = []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
            response_strs.append(response_str)
            ground_truth = data.non_tensor_batch["ground_truth"][i]
            correctness.append(extract_and_check(response_str, ground_truth))
        return DataProto(meta_info={"correctness": correctness, "response_strs": response_strs})'''
    
    @torch.no_grad()
    def recompute_attn_mask_and_pos_ids(self, images: List[Any], input_text: List[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Any]]:
        '''
        Reprocess the input images and text to recompute the attention mask and position ids.
        Args:
            images: list of images
            input_text: list of input text
        '''
        reprocessed_input_ids = []
        reprocessed_attention_mask = []
        reprocessed_position_ids = []
        reprocessed_multimodal_inputs = [] # including pixel_values and image_grid_thw
        reprocessed_raw_input_ids = []
        for image, prompt in zip(images, input_text):
            model_inputs = self.processor(image, [prompt], add_special_tokens=False, return_tensors="pt")

            reprocessed_multimodal_inputs.extend([{"pixel_values":model_inputs["pixel_values"], "image_grid_thw":model_inputs["image_grid_thw"]}] * self.sampling_params.n)
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
                # qwen2vl mrope
                position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    attention_mask=attention_mask,
                )
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
            input_ids, attention_mask, position_ids = VF.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=self.config.prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="right",
            )
            
            reprocessed_input_ids.append(input_ids)
            reprocessed_attention_mask.append(attention_mask)
            reprocessed_position_ids.append(position_ids)
            
            raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(raw_prompt_ids) > self.config.prompt_length:
                raw_prompt_ids = raw_prompt_ids[: self.config.prompt_length]
            reprocessed_raw_input_ids.append(raw_prompt_ids)

            
        input_ids = torch.stack(reprocessed_input_ids, dim=0)
        attention_mask = torch.stack(reprocessed_attention_mask, dim=0)
        position_ids = torch.stack(reprocessed_position_ids, dim=0)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)
        
        return input_ids, attention_mask, position_ids, reprocessed_multimodal_inputs, reprocessed_raw_input_ids
    
    


def resize_lists_by_sampling(
    lists: List[Any],
    sample_ids: List[Any],
    target_len: int,
    seed: int | None = None
) -> Tuple[List[Any], List[Any]]:
    """
    将多组对齐列表/张量扩充/裁剪到 target_len，并保持同一 ID 的样本在补齐后仍连续排列。

    参数
    ----
    lists : List[list 或 torch.Tensor]
        若干对齐的列表或张量。
    sample_ids : List[Any]
        与 lists 同长的 ID 列表；相同 ID 已经连续排列。
    target_len : int
        目标长度。
    seed : int | None
        随机种子，便于复现。

    返回
    ----
    resized_lists : List[list 或 torch.Tensor]
        补齐/裁剪后的各列表/张量。
    resized_ids : List[Any]
        补齐/裁剪后的 ID 列表。
    """
    if seed is not None:
        random.seed(seed)

    if len(sample_ids) == 0:
        raise ValueError("sample_ids cannot be empty.")

    original_len = len(sample_ids)
    indices = list(range(original_len))

    # 情况一：裁剪
    if original_len > target_len:
        selected_indices = random.sample(indices, k=target_len)
        selected_indices.sort()

    # 情况二：补齐
    elif original_len < target_len:
        extra_indices = random.choices(indices, k=target_len - original_len)

        id2idx: "OrderedDict[Any, List[int]]" = OrderedDict()
        for idx, sid in enumerate(sample_ids):
            id2idx.setdefault(sid, []).append(idx)

        for idx in extra_indices:
            sid = sample_ids[idx]
            id2idx[sid].append(idx)

        selected_indices = [i for idx_list in id2idx.values() for i in idx_list]

    # 情况三：长度一致
    else:
        selected_indices = indices

    # 生成结果
    resized_lists = []
    for lst in lists:
        if isinstance(lst, torch.Tensor):
            lst_len = lst.shape[0]
        else:
            lst_len = len(lst)


        if lst_len > original_len:
            assert lst_len % original_len == 0, "When expanding, the length must be divisible by the original length."
            m = lst_len // original_len
            # 扩展版索引
            if isinstance(lst, torch.Tensor):
                resized = torch.cat([lst[i * m : (i + 1) * m] for i in selected_indices], dim=0)
            elif isinstance(lst, list):
                resized = sum([lst[i * m : (i + 1) * m] for i in selected_indices], [])
            else:
                raise TypeError("`lst` must be a list or a torch.Tensor.")
        else:
            # 普通索引
            if isinstance(lst, torch.Tensor):
                resized = lst[selected_indices]
            elif isinstance(lst, list):
                resized = [lst[i] for i in selected_indices]
            else:
                raise TypeError("`lst` must be a list or a torch.Tensor.")

        resized_lists.append(resized)

    resized_ids = [sample_ids[i] for i in selected_indices]

    print(f"Resized data of length {original_len} to target length {target_len}.")
    return resized_lists, resized_ids
    

def shuffle_lists(lists, sample_ids, seed=42):
    if seed is not None:
        random.seed(seed)
    original_len = len(sample_ids)
    shuffled_indices = list(range(original_len))
    random.shuffle(shuffled_indices)
    shuffled_lists = []
    for lst in lists:
        if isinstance(lst, torch.Tensor):
            lst_len = lst.shape[0]
        else:
            lst_len = len(lst)


        if lst_len > original_len:
            assert lst_len % original_len == 0, "When shuffling, the length must be divisible by the original length."
            m = lst_len // original_len
            if isinstance(lst, torch.Tensor):
                resized = torch.cat([lst[i * m : (i + 1) * m] for i in shuffled_indices], dim=0)
            elif isinstance(lst, list):
                resized = sum([lst[i * m : (i + 1) * m] for i in shuffled_indices], [])
            else:
                raise TypeError("`lst` must be a list or a torch.Tensor.")
        else:
            if isinstance(lst, torch.Tensor):
                resized = lst[shuffled_indices]
            elif isinstance(lst, list):
                resized = [lst[i] for i in shuffled_indices]
            else:
                raise TypeError("`lst` must be a list or a torch.Tensor.")

        shuffled_lists.append(resized)
    shuffled_ids = [sample_ids[i] for i in shuffled_indices]
    return shuffled_lists, shuffled_ids
    

def repeat_elements_interleaved(lst, n):
    return [item for item in lst for _ in range(n)]




MaskLike = Union[List[int], Sequence[int], torch.Tensor]
DataLike = Union[List, torch.Tensor]


def _to_bool_tensor(mask: MaskLike) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        return mask.bool()
    return torch.tensor(list(mask), dtype=torch.bool)


def select_by_mask(mask: MaskLike, data: DataLike, m: int | None = None) -> DataLike:
    """
    Returns a new list or Tensor by selecting elements from `data` according to the binary `mask`.

    args:
    ----
    mask : list or torch.Tensor filled with 0/1 with length n
    data : list or tensor with length n or n * m
           - L = n         → case 1：pick lines by mask
           - L = n * m     → case 2：pick segments by mask
    m    : length of each segment in case 2, if not provided, will be inferred from `data`.

    return: a new list or Tensor with the same type as `data`
    """
    mask_b = _to_bool_tensor(mask)
    n = mask_b.numel()

    # -------- list -------- #
    if isinstance(data, list):
        L = len(data)
        # case 1
        if L == n:
            return [x for x, keep in zip(data, mask_b) if keep]
        # case 2
        if m is None:
            if L % n != 0:
                raise ValueError("Cannot infer m, please provide it explicitly.")
            m = L // n
        elif L != n * m:
            raise ValueError(f"`len(data)` ({L}) must be equal to n*m ({n*m}).")

        out: list = []
        for idx, keep in enumerate(mask_b):
            if keep:
                out.extend(data[idx * m : (idx + 1) * m])
        return out

    # -------- Tensor -------- #
    if not torch.is_tensor(data):
        raise TypeError("`data` must be list or torch.Tensor")

    if data.dim() < 2:
        raise ValueError("Only 2D or higher dimensional Tensors are supported, using the first dimension as row index")

    L = data.size(0)
    # case 1
    if L == n:
        return data[mask_b]

    # case 2
    if m is None:
        if L % n != 0:
            raise ValueError("Cannot infer m, please provide it explicitly.")
        m = L // n
    elif L != n * m:
        raise ValueError(f"`data.size(0)` ({L}) must be equal to n*m ({n*m}).")

    segs = [data[idx * m : (idx + 1) * m] for idx, keep in enumerate(mask_b) if keep]
    return torch.cat(segs, dim=0) if segs else data.new_empty((0, *data.shape[1:]))

def select_by_mask_for_list_of_sequences(lists, mask, m: int = None):
    return [select_by_mask(mask, seq, m) for seq in lists]

def find_delim_positions(
    response_ids: List[int],
    delim_ids_list: List[str]
) -> List[int]:
    positions: List[int] = []
    i = 0
    n = len(response_ids)

    while i < n:
        matched = False
        for delim_ids in delim_ids_list:
            m = len(delim_ids)
            if m == 0 or i + m > n:
                continue
            if response_ids[i : i + m] == delim_ids:
                positions.append(i)  
                i += m               
                matched = True
                break                
        if not matched:
            i += 1   
    delim_poss = positions.copy()  # copy the positions to avoid modifying the original list
    positions = positions + [n-1]            
    if positions[0] == 0 and len(positions) > 1:
        positions = positions[1:]    
    return delim_poss, positions

def clean_header(text: str) -> str:
    # 1) 四个及以上 # → ### 
    text = re.sub(r'#{4,}', '###', text)

    # 2) “### Step<内容>”，但 <内容> 不是 “ 空格+数字” 时，删掉 Step<内容>
    #    (保留前导 “###” 与换行)
    text = re.sub(
        r'### Step(?! \d+)\S*',   # 不是 “ Step <数字>” 的整行/剩余部分
        '',
        text)

    # don't fix single '###' here, leave it to the format reward function to penalize it because it won't affect 
    # the locating of '### Step x:' here
    '''# 3) 只要出现 “###” 且其后面不是 “ Step”，就删掉 “###”
    text = re.sub(
        r'###(?! Step)',     # 负向前瞻：后面不是 " Step"
        '',
        text)'''

    return text