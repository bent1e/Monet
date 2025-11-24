# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
import os
from abc import abstractmethod
from typing import Optional, Any, Dict, List, Callable, Type, Tuple, Union
from pydantic import BaseModel, PrivateAttr, conlist, ConfigDict, field_validator
from omegaconf import DictConfig, OmegaConf
from timeout_decorator import timeout
from verl.workers.rollout.rstar_deepthink.config import BaseConfig
from verl.workers.rollout.rstar_deepthink.nodes.base_node import BaseNode
from verl.workers.rollout.rstar_deepthink.tools.python_tool import PythonInterpreter
from verl.workers.rollout.rstar_deepthink.constants import TIMEOUT_SECONDS, TIMEOUT_MESSAGE, CODE_END, OUTPUT_END, CODE, ANSWER
from verl.workers.rollout.config import MCTSConfig
import numpy as np

def _python_ast_init():
    python = PythonInterpreter(globals=globals(), locals=None)
    return python


def tool_wrapper(tool):
    def _tool(query):
        return tool.run(query)
    return _tool


def no_action_wrapper(tool):
    def _tool(query):
        return "No action, no observation. Please continue to solve."
    return _tool


tools = {
    "python_interpreter": tool_wrapper(_python_ast_init()),
    "None": no_action_wrapper(_python_ast_init()),
}


class BaseTree(BaseModel):

    config: MCTSConfig
    question: str
    ground_truth: Optional[Union[str, List[str]]] = None
    llm: Any = None
    mllm: Any = None
    root: Optional[Type[BaseNode]] = None
    current_node: Optional[Type[BaseNode]] = None 
    stop: Optional[List[str]] = None
    node_max_retry: int = 5
    img_path: str = None # for mllm
    annotated_steps: Optional[List[str]] = None # for reward analysis
    step_rewards: Optional[List[int]] = None # for reward analysis
    multi_modal_data: Any = None
    batch_id: int = None
    prompt_before_processor: str = None
    found_correct_path: bool = False
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stop = self.config.stop
        self.root = self.create_root()
        self.current_node = self.root
    
    
    def create_root(self) -> Type[BaseNode]:
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root

    @abstractmethod
    def create_node(self, parent: Optional[Type[BaseNode]] = None) -> Type[BaseNode]:
        """
        subclass must implement
        """
    
    def collect_partial_solution(self, node: Type[BaseNode]) -> str:
        # from leaf to root, and reverse
        # not including the question text (root node)
        trajectory = []
        while node:
            if node.state['text']:
                trajectory.append(node.state['text'])
            node = node.parent
        return "".join(reversed(trajectory))

    def collect_question_concat_partial_solution(self, node: Type[BaseNode]) -> str:
        # from leaf to root, and reverse
        # including the question text (root node)
        trajectory = []
        while node:
            if node.state['text']:
                trajectory.append(node.state['text'])
            else: # root
                trajectory.append(self.question)
            node = node.parent
        return "".join(reversed(trajectory))
    
    def collect_partial_solution_ids(self, node: Type[BaseNode]) -> str:
        # from leaf to root, and reverse
        # including the ids of the question (root node)
        trajectory = []
        while node:
            if node.state['ids']:
                trajectory.append(node.state['ids'])
            node = node.parent
        return [id for ids in reversed(trajectory) for id in ids]
    
    def return_states(self) -> Dict[str, Dict[str, str]]:
        candidates = [self.root]
        states = {}
        reward_stat = {}

        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            if node.has_children():
                candidates.extend(node.children)

        return states
    
    def return_filtered_prompt_and_response_ids(self) -> List[Tuple[List[int], List[List[int]]]]:
        """
        Traverse the completed MCTS tree. If the correct child ratio < self.config.correctness_threshold_for_training, 
        then we will add the prompt token ids and response ids of this step.
        
        Return:
            a list of tuples, each tuple contains two lists:
                1. prompt ids of step_1, step_2, verl.workers.rollout., step_{t-1}
                2. response ids of the rollout steps at step_t: step^1_t, step^2_t, verl.workers.rollout., step^k_t
        """
        candidates = [self.root]
        results = []

        while candidates:
            node = candidates.pop(0)
            if node.has_children():
                candidates.extend(node.children)
                prompt_ids = self.collect_partial_solution_ids(node)
                prompt_text = self.prompt_before_processor + self.collect_partial_solution(node)
                correct_cnt = 0
                valid_cnt = 0
                for child in node.children:
                    if child.visit_count() > 0: # once lead to a final answer
                        valid_cnt += 1
                    if child.lead_to_correct == True:
                        correct_cnt += 1
                if valid_cnt < len(node.children): # only when all children lead to a final answer, can we move on to compute the correctness ratio and check the threshold
                    continue
                correct_ratio = correct_cnt/len(node.children)
                if correct_ratio > 0 and correct_ratio < self.config.correctness_threshold_for_training:
                    print(f"Adding a group from problem {self.batch_id} to the training set, correct ratio: {correct_ratio}")
                    #print(f"Prompt: {prompt_text}")
                    response_ids = [child.state['ids'] for child in node.children]
                    results.append((prompt_ids, prompt_text, self.multi_modal_data["image"], response_ids, \
                                    [1.0 if child.lead_to_correct else 0.0 for child in node.children], \
                                    [self.question]*len(node.children), \
                                    [self.ground_truth]*len(node.children),\
                                    [self.batch_id]*len(node.children)))
                
        return results

    def collect_uncomplete_nodes(self):
        """
        Traverse the completed MCTS tree. If a node is deep enough and hasn't lead to a final answer, collect it.

        Return:
            a list of dicts, each dict is an input for vllm
        """
        candidates = [self.root]
        vllm_input = []
        nodes_to_update = []
        deep_node_cnt = 0
        uncomplete_node_cnt = 0
        while candidates:
            node = candidates.pop(0)
            if node.depth >= self.config.min_depth_for_complete:
                deep_node_cnt += 1
                if node.visit_count() == 0:
                    uncomplete_node_cnt += 1
                    partial_solution = self.collect_partial_solution_ids(node)
                    vllm_input.append({"prompt_token_ids":  partial_solution, "multi_modal_data": self.multi_modal_data})
                    nodes_to_update.append(node)
            if node.has_children():
                candidates.extend(node.children)
        print(f"Completed in collecting the uncomplete nodes of a tree. Deep node count: {deep_node_cnt}, uncomplete node count: {uncomplete_node_cnt}")
        return vllm_input, nodes_to_update
    
    def collect_underexplored_nodes_with_correct_sibling(self, rollout_threshold: int = 5):
        """
        Traverse the completed MCTS tree. If a node has a sibling that lead to a correct answer, and its visit count is less than the threshold (rollout.n), collect it.

        Return:
            a list of dicts, each dict is an input for vllm
        """
        candidates = [self.root]
        vllm_input = []
        nodes_to_update = []
        while candidates:
            node = candidates.pop(0)
            if node.has_children():
                candidates.extend(node.children)
                lead_to_incorrect_children = []
                correct_cnt = 0
                for child in node.children:
                    if child.visit_count() < rollout_threshold and child.lead_to_correct == False:
                        lead_to_incorrect_children.append(child)
                    if child.lead_to_correct == True:
                        correct_cnt += 1
                correct_ratio = correct_cnt/len(node.children)
                if correct_ratio > 0 and correct_ratio < self.config.correctness_threshold_for_training:
                    for child in lead_to_incorrect_children:
                        partial_solution = self.collect_partial_solution_ids(child)
                        vllm_input.append({"prompt_token_ids":  partial_solution, "multi_modal_data": self.multi_modal_data})
                        nodes_to_update.append(child)
        print(f"Collected {len(vllm_input)} underexplored nodes with lead_to_correct=True siblings from a tree.")
        return vllm_input, nodes_to_update
    
    def update_node_set_lead_to_correct(self, nodes_to_update: List[Type[BaseNode]], lead_to_correct_list: List[bool]):
        """
        Update the lead_to_correct of the nodes in the set.
        """
        cnt = 0
        for node, lead_to_correct in zip(nodes_to_update, lead_to_correct_list):
            node.lead_to_correct = lead_to_correct
            if lead_to_correct:
                node.update(self.config.positive_reward)
            else:
                node.update(self.config.negative_reward)
            cnt += 1
        print(f"Completed in updating the `lead_to_correct` and `visit_count` of uncomplete nodes of a tree. Updated {cnt} nodes.")
    
    
    def write_back_lead_to_correct_of_uncomplete_nodes(self, lead_to_correct_list: List[bool]):
        """
        Traverse the completed MCTS tree. If a node is deep enough and hasn't lead to a final answer, update it according to the lead_to_correct_list.
        """
        candidates = [self.root]
        vllm_input = []
        p = 0

        while candidates:
            node = candidates.pop(0)
            if node.depth >= self.config.min_depth_for_complete and node.visit_count() == 0:
                if lead_to_correct_list[p]:
                    node.update(self.config.positive_reward)
                else:
                    node.update(self.config.negative_reward)
                p+=1
            if node.has_children():
                candidates.extend(node.children)
        #print(f"Completed in updating the `lead_to_correct` and `visit_count` of uncomplete nodes of a tree.")
        return vllm_input

    def write_back_lead_to_correct_of_nodes_with_correct_sibling(self, rollout_threshold: int, lead_to_correct_list: List[bool]):
        """
        Traverse the completed MCTS tree. If a node has a sibling that lead to a correct answer, update it according to the lead_to_correct_list.

        Return:
            a list of dicts, each dict is an input for vllm
        """
        candidates = [self.root]
        vllm_input = []
        while candidates:
            node = candidates.pop(0)
            if node.has_children():
                candidates.extend(node.children)
                lead_to_incorrect_children = []
                correct_cnt = 0
                for child in node.children:
                    if child.visit_count() < rollout_threshold and child.lead_to_correct == False:
                        lead_to_incorrect_children.append(child)
                    if child.lead_to_correct == True:
                        correct_cnt += 1
                correct_ratio = correct_cnt/len(node.children)
                if correct_ratio > 0 and correct_ratio < self.config.correctness_threshold_for_training:
                    for child in lead_to_incorrect_children:
                        partial_solution = self.collect_partial_solution_ids(child)
                        vllm_input.append({"prompt_token_ids":  partial_solution, "multi_modal_data": self.multi_modal_data})
            
        print(f"Collected {len(vllm_input)} underexplored nodes with lead_to_correct=True siblings from a tree.")
        return vllm_input


def extract_program(result: str, last_only=False):
    program = ""
    start = False
    result = result.replace("<end_of_step>", "")
    for line in result.split("\n"):
        if line.find("<code>") != -1:
            if last_only:
                program = "" # only extract the last program
            else:
                program += "\n# ========\n"
            start = True
        elif line.find("<end_of_code>") != -1:
            start = False
        elif start:
            program += line + "\n"
    # maybe all output is a program
    if not program:
        program = result
    return program.strip()

def code_execution(
    node: Type[BaseNode], 
    parser_result: Dict[str, str],
) -> str:


    @timeout(TIMEOUT_SECONDS, use_signals=True, exception_message=TIMEOUT_MESSAGE)
    def _code_execution(node: Type[BaseNode], parser_result: Dict[str, str]) -> str:
        # Define tool
        action = parser_result["action"]
        tool_func = tools[action]

        history_action_inputs = collect_action_inputs(node, action)

        # then, we execute current code snippets
        action_input = parser_result["action_input"]
        action_input = extract_program(''.join(history_action_inputs) + action_input)
        observation = str(tool_func(action_input)).strip()
        del tool_func
        return observation
    try:
        observation = _code_execution(node, parser_result)
    except Exception as e:
        observation = "{}: {}".format(type(e).__name__, str(e))
    
    return observation


def collect_action_inputs(
    node: Type[BaseNode], 
    action: str,
) -> List[str]:
    action_inputs = []
    while node: 
        if OUTPUT_END in node.state['text'] or CODE_END in node.state['text']:
            break
        if node.state["action"] == action:
            action_input = node.state["action_input"]
            if action_input and "TimeoutError" not in node.state["text"].split(action_input)[-1]:
                action_inputs.append(action_input)
        node = node.parent
    return action_inputs[::-1]


def code_run(solution):
    if CODE not in solution or CODE_END not in solution or OUTPUT_END not in solution or ANSWER not in solution:
        return solution
    
    @timeout(TIMEOUT_SECONDS, use_signals=True, exception_message=TIMEOUT_MESSAGE)
    def _code_execution(solution: str) -> str:
        tool_func = tools['python_interpreter']
        action_input = extract_program(solution)
        observation = str(tool_func(action_input)).strip()
        del tool_func
        return observation
    
    try:
        observation = _code_execution(solution)
    except Exception as e:
        observation = "{}: {}".format(type(e).__name__, str(e))
    
    return observation