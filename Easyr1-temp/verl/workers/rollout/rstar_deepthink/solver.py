# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
import os
import json
import copy
import os.path as osp
from tqdm import tqdm
from termcolor import colored
from functools import partial
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from pebble import ProcessPool
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Any, Dict, List, Callable, Type, Tuple, Union
from pydantic import BaseModel, ConfigDict, field_validator
from .agents.tree import BaseTree
from .agents.mcts import MCTS
from .llms.llms import llm_generate, rm_generate
from .llms.llm_engine import llm_engine, rm_engine
from .constants import TIMEOUT_SECONDS, ERROR_COLOR

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, \
    AutoConfig, MllamaForConditionalGeneration,\
    Qwen2_5_VLForConditionalGeneration, AutoTokenizer
import torch
from verl.workers.rollout.multimodal.process_input import process_multimodal_input
from verl.workers.rollout.multimodal.device_map import set_device_map
from verl.workers.rollout.multimodal.generate import generate_multimodal_responses
from PIL import Image
from accelerate import dispatch_model, init_empty_weights, load_checkpoint_and_dispatch
import numpy as np
from transformers import AutoTokenizer
from vllm.outputs import CompletionOutput

from verl.workers.rollout.rstar_deepthink.agents.utils import extract_math_answer, rstar_equiv
from verl.workers.rollout.multimodal.registered_models import qwen_series
from verl.workers.rollout.config import MCTSConfig
import re

class Solver(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: MCTSConfig
    stop: List[str] = None
    llm: Optional[Callable[[...], List[str]]] = None
    llm_engine: Optional[LLM] = None
    generate_sampling_params: Optional[SamplingParams] = None
    need_value_func: bool = False
    max_agent_steps: int = 1
    reward_model: Optional[Any] = None
    mllm_dir: str = None
    mllm: Optional[LLM] = None
    mllm_processor: Optional[Callable[[...], Any]] = None
    tokenizer: Optional[Callable[[...], Any]] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        '''self.need_value_func = self.config.need_value_func
        if self.need_value_func:
            self.reward_model = self.create_rm()
        if self.config.mllm_dir != "":
            if self.config.use_vllm_for_multimodal:
                self.mllm = self.create_model_from_vllm()
                self.mllm_processor = None
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.mllm_dir)
            else:
                self.mllm, self.mllm_processor = self.create_mllm_from_hf()
                self.tokenizer = self.mllm_processor.tokenizer
            self.llm = None
        if self.config.llm_dir != "":
            self.mllm = None
            self.mllm_processor = None
            self.llm = self.create_model_from_vllm()
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_dir)'''
        
        
        '''if self.config.mode == "sbs":
            self.max_agent_steps = 1
        elif self.config.mode == "mcts":
            self.max_agent_steps = self.config.iterations
            self.config.step_beam_width = 1'''
            


    def create_rm(self):
        rm, v_head, tokenizer = rm_engine(self.config)
        return partial(
            rm_generate,
            model=rm,
            v_head=v_head,
            tokenizer=tokenizer,
            max_model_len=self.config.max_model_len,
        )


    def create_model_from_vllm(self):
        engine, sampling_params = llm_engine(self.config)
        self.llm_engine = engine
        self.generate_sampling_params = sampling_params
        return partial(
            llm_generate,
            engine=self.llm_engine,
        )
        
        
    @staticmethod
    def processor(agent, output) -> BaseTree:
        agent.generate_next_step(output)
        return agent


    @staticmethod
    def selector(agent, output) -> BaseTree:
        agent.select_next_step(output) # will select the first node from agent.current_nodes, then clear agent.current_nodes and select the chid with the highest UCT to add into agent.current_nodes
        return agent

    def resize_img_if_needed(self, img):
        w, h = img.size
        min_side = min(w, h)
        min_side_length = 32
        scale = min_side_length / min_side
        if min_side < 32:
            img =  img.resize((int(round(w*scale)), int(round(h*scale))), Image.Resampling.LANCZOS)
        return img
    
    def generate_preprocess(self, agents):
        prompts = []
        rewards = []
        prompts_span = [0]
        valid_agents = []
        invalid_agents = []
        expanded_agents = []
        multimodal_processed_inputs = []

        for agent in agents:
            if agent.should_generate_next():# if all nodes in self.current_nodes is_terminal, or depth>max_depth, or self.current_nodes==[] then False, and this agent will be added to invalid_agents
                if agent.has_expanded(): # true when agent.current_nodes[0] has children
                    expanded_agents.append(agent)
                else:
                    multimodal_processed_inputs.extend(agent.create_vllm_input())

                    rewards.extend(agent.get_rewards())
                    prompts_span.append(prompts_span[-1] + len(multimodal_processed_inputs))
                    valid_agents.append(agent)
            else: 
                invalid_agents.append(agent)


        return prompts, prompts_span, valid_agents, invalid_agents, expanded_agents, rewards, multimodal_processed_inputs


    def generate_postprocess_multi_process(
        self, 
        outputs: List[List[RequestOutput]], 
        valid_agents: List[BaseTree],
    ) -> List[BaseTree]:
        post_agents = []
        #with ProcessPool(max_workers=min(len(valid_agents), os.cpu_count())) as pool:
        with ProcessPool(max_workers=12) as pool:
            future = pool.map(self.__class__.processor, valid_agents, outputs, timeout=TIMEOUT_SECONDS)
            iterator = future.result()
        
        progress_bar = tqdm(total=len(valid_agents), desc="generate_postprocess")  
        while True:
            try:
                result = next(iterator)
                post_agents.append(result)
            except StopIteration:
                break
            except Exception as error:
                print(colored(f"{error}\n", ERROR_COLOR))
                post_agents.append(None)
            progress_bar.update(1) 
        progress_bar.close() 
            
        # update agents
        updated_agents = [
            post_agent if post_agent is not None else valid_agent
            for post_agent, valid_agent in zip(post_agents, valid_agents)
        ]
        return updated_agents
    
    def generate_postprocess_single_process(
        self, 
        outputs: List[List['RequestOutput']], 
        valid_agents: List['BaseTree'],
    ) -> List['BaseTree']:
        post_agents = []
        progress_bar = tqdm(total=len(valid_agents), desc="generate_postprocess")  

        for agent, output in zip(valid_agents, outputs):
            try:
                result = self.__class__.processor(agent, output)
                post_agents.append(result)
            except Exception as error:
                print(colored(f"{error}\n", ERROR_COLOR))
                post_agents.append(None)
            progress_bar.update(1) 

        progress_bar.close() 

        updated_agents = [
            post_agent if post_agent is not None else valid_agent
            for post_agent, valid_agent in zip(post_agents, valid_agents)
        ]
        return updated_agents


    def value_preprocess(self, agents: List[BaseTree]) -> Tuple[List[str], List[int]]:
        prompts = []
        prompts_span = [0]
        for agent in agents:
            agent_prompts, _, _, _ = agent.create_prompt(is_value_only=True)
            prompts.extend(agent_prompts)
            prompts_span.append(prompts_span[-1] + len(agent_prompts))
        return prompts, prompts_span
    
    
    def value_postprocess( 
        self, 
        outputs, 
        valid_agents,
    ) -> List[BaseTree]:
        for agent, output in zip(valid_agents, outputs):
            if agent is not None:
                self.selector(agent, output) # 
        return valid_agents
    

    def save_intermediate_metric(self, path: str, agents: List[MCTS], rollout) -> None:
        if self.config.is_sampling: return
        states = [s.intermediate_metric for s in agents]
        statics = []
        for i in range(rollout + 1):
            pass1, passn = 0, 0
            for idx, state in enumerate(states):
                max_value = -100
                max_value_result = False
                pass1_ans = False
                for idx, rollout_index in enumerate(state["rollout_indexs"]):
                    if rollout_index <= i:
                        if state["value_estimate"][idx] > max_value:
                            max_value = state["value_estimate"][idx]
                            max_value_result = state["judgements"][idx]
                        if state["judgements"][idx]:
                            pass1_ans = True
                if max_value_result:
                    pass1 += 1
                if pass1_ans:
                    passn += 1
            statics.append({
                "rollout": i,
                "pass1": pass1,
                "passn": passn,
                "len": len(states),
            })
        with open(path, "w", encoding='utf-8') as f:
            json.dump([statics,states], f, ensure_ascii=False, indent=4)

    
    def collect_rollout_results(self, agents):
        rollout_results = self.output(agents)
        return rollout_results
    
    def output(self, agents: List[BaseTree], do_reward_analysis=False):
        results = []
        for i, agent in enumerate(agents):   
            results.extend(agent.return_filtered_prompt_and_response_ids())
            #if len(results)>0:
            #    print(agent.batch_id, results[-1][5])
            #if do_reward_analysis:
            #    reward_stat_all_agents.append(reward_stat)
        return results
    
    def solve(self, agents: List[BaseTree], vllm_inputs: List[Dict], rollout: int, sampling_params= None):

        # Initialize the initial search starting point of agents, and the initial point of each rollout is root
        assert len(agents) == len(vllm_inputs)
        for agent, vllm_input in zip(agents, vllm_inputs):
            if agent.root.state["ids"] is None: # Initialize the prompt token ids (the question) of the root node
                agent.root.state["ids"] = vllm_input["prompt_token_ids"]
            agent.select_next_step(from_root=True)
            agent.rollout_idx = rollout

        setattr(sampling_params, "stop", self.config.stop)
        setattr(sampling_params, "detokenize", True)

        for step in range(self.config.max_depth):
            print("-----------------Current Rollout: ", rollout, "-----------------")
            print("-----------------Current Step: ", step, "-----------------")
            prompts, prompts_span, valid_agents, invalid_agents, expanded_agents, valid_rewards, multimodal_processed_inputs = self.generate_preprocess(agents)
            
            if len(valid_agents + expanded_agents) < 1:
                break
            
            # step expansion
            outputs = self.mllm.generate(multimodal_processed_inputs, sampling_params)
            
            #breakpoint()  
            for output, reward in zip(outputs, valid_rewards): # outputs[0].outputs变为None# attach reward to prevent repeat rewarding
                output.value_estimate = reward
            reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
            
            # process output and run python code, use the parser_result to determine the state of next nodes (mcts.py, `create_child``)
            # if final answer comes up in a child node, then its `.is_terminal` will be set to True, and its value (+1 or -1) will bp along its parents til the root
            valid_agents = self.generate_postprocess_single_process(reconstructed_outputs, valid_agents) 

            # step evaluation
            prompts, prompts_span = self.value_preprocess(valid_agents)
            if self.need_value_func:
                outputs = self.reward_model(prompts=prompts)
                reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
            else:
                reconstructed_outputs = [None] * (len(prompts_span) - 1)
            
            # selection
            # select the first node from agent.current_nodes, then clear agent.current_nodes and select the chid with the highest UCT to add into agent.current_nodes
            valid_agents = self.value_postprocess(reconstructed_outputs, valid_agents)
            expanded_agents = self.value_postprocess([None] * len(expanded_agents), expanded_agents) # for expanded agents, just do selection step
            
            # keep all agents
            agents = valid_agents + invalid_agents + expanded_agents

        return agents
    
    def collect_complete_update(self, agents: List[BaseTree],  sampling_params= None):
        """
        1. Complete the uncomplete nodes with sufficient large depth in the MCTS tree using VLLM, or underexplored nodes with correct siblings
        2. Check the correctness of the final answer, return the `lead_to_correct` field of the nodes
        3. Traverse the tree again, write back the `lead_to_correct` field to the uncomplete nodes.
        """
        vllm_inputs = []
        cnt_for_each_agent = []
        new_sampling_params = copy.deepcopy(sampling_params)
        nodes_to_update_for_each_agent = []
        for agent in agents:
            if self.config.stop_when_find_correct_answer: # sampling_params.n == rollout.n, > 1
                vllm_input, nodes_to_update = agent.collect_underexplored_nodes_with_correct_sibling(rollout_threshold=sampling_params.n)
            else:
                setattr(new_sampling_params, "n", 1)
                vllm_input, nodes_to_update = agent.collect_uncomplete_nodes()
            vllm_inputs.extend(vllm_input)
            cnt_for_each_agent.append(len(vllm_input))
            nodes_to_update_for_each_agent.append(nodes_to_update)

        print("Completing uncomplete nodes or underexplored nodes with correct siblings...")
        setattr(new_sampling_params, "stop", [])
        outputs = self.mllm.generate(vllm_inputs, new_sampling_params)
        
        outputs_for_each_agent = [] # for all agents (questions)
        for cnt in cnt_for_each_agent:
            outputs_for_each_agent.append(outputs[:cnt]) 
            outputs = outputs[cnt:]

        lead_to_correct_lists = [] # for all agents (questions), each element is a list of lead_to_correct for each node in the agent
        for single_agent_outputs, agent in zip(outputs_for_each_agent, agents): # all agents (questions)
            lead_to_correct_list = []
            for output in single_agent_outputs: # all nodes in the agent (question)
                if self.config.stop_when_find_correct_answer:
                    correctness = False
                    for resp in output.outputs:
                        correctness |= self.extract_check_answer(resp.text, agent.ground_truth)
                else:
                    correctness = self.extract_check_answer(output.outputs[0].text, agent.ground_truth)
                lead_to_correct_list.append(correctness)
                #print("Recompleted response_text: ", response_text, "Recompleted extracted_answer: ", extracted_answer, "ground_truth: ", agent.ground_truth, "correctness: ", correctness)
            lead_to_correct_lists.append(lead_to_correct_list)

        for lead_to_correct_list, agent, nodes_to_update in zip(lead_to_correct_lists, agents, nodes_to_update_for_each_agent):
            agent.update_node_set_lead_to_correct(nodes_to_update, lead_to_correct_list)
                
    def extract_check_answer(self, text:str, gt:str) -> bool:
        if "boxed" in text:
            extracted_answer = extract_math_answer(text, answer_format="boxed")
        elif "answer" in text:
            extracted_answer = extract_math_answer(text, answer_format="no_boxed")
        else:
            extracted_answer = ""
        correctness = rstar_equiv(extracted_answer, gt)
        return correctness
    
    def split_solutions_into_accum_steps(self, solutions: Union[List[str], str], delim: str, add_empty_step_in_front=False) -> Union[List[List[str]], List[str]]:
        '''
        solutions should be a list of strings, each string is a solution with no question prefix
        This function will only handle "xxx. ### Step 2: yyy. ### Step 3: zz.", i.e., no "### Step" at the beginning.
        '''
        batch_indiv_steps = []
        batch_remain_steps = []
        if isinstance(solutions, List):
            batch_accum_steps = []
            for solution in solutions:
                partial_solutions = []
                indiv_steps = []
                remain_steps = []
                steps = solution.split(delim)
                if add_empty_step_in_front:
                    steps = [""] + steps # add a empty string at the beginning
                partial_solution = ""
                for i, step in enumerate(steps):
                    if i>1:
                        step = delim + step
                    if i>0:
                        indiv_steps.append(step)
                        if i < len(steps) - 1:
                            remain_steps.append("".join([step] + [delim+step for step in steps[i+1:]]))
                        else:
                            remain_steps.append(step)
                    partial_solution = partial_solution + step
                    partial_solutions.append(partial_solution)
                batch_accum_steps.append(partial_solutions)
                batch_indiv_steps.append(indiv_steps)
                batch_remain_steps.append(remain_steps)
            
            return batch_accum_steps, batch_indiv_steps, batch_remain_steps
        elif isinstance(solutions, str):
            partial_solutions = []
            indiv_steps = []
            steps = solutions.split(delim)
            if add_empty_step_in_front:
                steps = [""] + steps # add a empty string at the beginning
            partial_solution = ""
            for i, step in enumerate(steps):
                if i>1:
                    step = delim + step
                if i>0:
                    indiv_steps.append(step)
                    if i < len(steps) - 1:
                        remain_steps.append("".join([step] + [delim+step for step in steps[i+1:]]))
                    else:
                        remain_steps.append(step)
                partial_solution = partial_solution + step
                partial_solutions.append(partial_solution)
                batch_indiv_steps.append(indiv_steps)
                batch_remain_steps.append(remain_steps)
            return partial_solutions, batch_indiv_steps

    def split_solution_into_individual_steps(self, solutions: Union[List[str],str], delim: str) -> Union[List[List[str]], List[str]] :
        '''
        Split each solution into individual steps.
        solutions should be a list of strings, each string is a solution with no question prefix
        e.g., "xxx. ### Step 2: yyy. ### Step 3: zz."
        '''
        if isinstance(solutions, List):
            batch_individual_steps = []
            for solution in solutions:
                steps = solution.split(delim)
                if steps[0] in ["", "#", " "]:
                    steps = steps[1:]
                steps = [self.remove_step_numbers(step).strip() for step in steps]
                batch_individual_steps.append(steps)
            return batch_individual_steps
        elif isinstance(solutions, str):
            steps = solutions.split(delim)
            if steps[0] in ["", "#", " "]:
                steps = steps[1:]
            steps = [self.remove_step_numbers(step).strip() for step in steps]
            return steps

    def remove_step_numbers(self, text: str):
        return re.sub(r"^ \d+(\.\d+)?:", "", text)

    def get_first_step(self, solution_str: str, delim="### Step") -> str:
        """
        Get the first step from the solution string.
        """

        return solution_str.split(delim)[0].strip()
        
        