# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
from typing import Dict, Any, Optional, Type, List, Tuple, Callable, Union
from pydantic import field_validator
from vllm.outputs import CompletionOutput, RequestOutput
from verl.workers.rollout.rstar_deepthink.agents.utils import math_equiv as is_equiv
from verl.workers.rollout.rstar_deepthink.nodes.base_node import BaseNode
from verl.workers.rollout.rstar_deepthink.nodes import MCTSNode
from verl.workers.rollout.rstar_deepthink.constants import (
    TOO_MANY_CODE_ERRORS, 
    TOO_MANY_STEPS, 
    NO_VALID_CHILD, 
    CODE_END,
)
from .tree import BaseTree, code_execution
from .beam_search import BS

#from multimodal.mutual_check_prompt import mutual_check_func
import numpy as np
from verl.workers.rollout.rstar_deepthink.agents.utils import on_annotated_path, multimodal_step_result_unwrap

class MCTS(BS):
    search_node: Type[BaseNode] = None

    intermediate_metric: Dict = {
        "question": "",
        "gt": "", 
        "answers": [],
        "judgements": [],
        "value_estimate": [],
        "rollout_indexs": [],
    }


    def create_node(self, parent: Optional[Type[MCTSNode]] = None) -> Type[MCTSNode]:
        return MCTSNode(
            parent=parent, 
            additional_state_keys=self.NODE_KEYS,
            c_puct=self.config.c_puct,
        )


    def selection(self, from_root=False) -> Optional[Type[MCTSNode]]:
        if from_root:
            start_node = self.root
        else:
            start_node = self.search_node
        # select a child node
        node = start_node
        if node is None: return None
        if node.has_children() or node.is_terminal:
            next_node = self.select_child(node)     # To encourage exploration, select from non-terminal children
            if next_node is None:                   # if None, it means all children are terminal
                node.is_terminal = True
            node = next_node
        return None if (node is None or node.is_terminal) else node

    def select_child(self, node: Type[MCTSNode]) -> Optional[Type[MCTSNode]]:
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:
                continue
            puct_value = child.puct()
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        #return random.choice(best_childs) if best_childs else None
        return best_childs[0] if best_childs else None

    def expand_node(self, outputs: List[CompletionOutput], node: Type[MCTSNode]) -> None:
        for idx, output in enumerate(outputs):
            if not output.stop_reason: output.stop_reason = ""
            step_text, parser_result = self.step_unwrap(output.text + output.stop_reason)
            step_ids = output.token_ids
            self.create_child(step_text, step_ids, parser_result, node, idx) # also perform: evaluate each created child and the child is the final answer, then do backpropagation

    def create_child(
        self, 
        step_text: str, 
        step_ids: List[int],
        parser_result: Dict[str, str], 
        node: Type[MCTSNode],
        idx: int,
    ) -> None:
        new_node = self.create_node(parent=node)
        parent_child_count = len(node.children)
        new_node.tag = f"{node.tag}.{parent_child_count + 1}"
        new_node.depth = node.depth + 1
        #print(f"#########child id {idx}######", step_text, "##############################################################################")
        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_tprintext
            new_node.state["ids"] = step_ids
            new_node.state["final_answer"] = NO_VALID_CHILD
            self.eval_final_answer(new_node) # backpropagation
        elif parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_text
            new_node.state["ids"] = step_ids
            new_node.state["final_answer"] = parser_result["final_answer"]
            self.eval_final_answer(new_node)
        elif parser_result["action"] == "python_interpreter":
            observation = code_execution(node, parser_result)
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]
            new_node.state["observation"] = observation
            if CODE_END in parser_result["action_input"]:
                observation = self.obs_wrap(observation)
                new_node.state["text"] = f"{step_text}{self.config.step_delim}{observation}"
            else:
                new_node.state["text"] = step_text
                
            if "error" in observation.lower():
                new_node.consecutive_errors = node.consecutive_errors + 1
                if new_node.consecutive_errors >= self.config.errors_threshold:
                    observation = self.obs_wrap(observation)
                    step_text = step_text + CODE_END if CODE_END not in step_text else step_text
                    new_node.state["text"] = f"{step_text}{self.config.step_delim}{observation}"
                    new_node.is_terminal = True
                    new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
                    self.eval_final_answer(new_node)
        elif parser_result["action"] == "perception": # TO-DO
            #observation = mutual_check_func(self.mllm, self.img_path, self.question, step_text)
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]
            new_node.state["observation"] = observation
            new_node.state["text"] = step_text
            new_node.state["ids"] = step_ids
        elif parser_result["action"] == "reasoning": # TO-DO
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]
            new_node.state["text"] = step_text
            new_node.state["ids"] = step_ids
            if step_text == "": # empty step
                new_node.is_terminal = True
                new_node.state["final_answer"] = NO_VALID_CHILD
                self.eval_final_answer(new_node) # backpropagation
        else:
            new_node.state["text"] = step_text
            new_node.state["ids"] = step_ids

        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS
            self.eval_final_answer(new_node)

        node.children.append(new_node)

    def eval_final_answer(self, node: Type[MCTSNode]) -> None:
        if node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            # if the final answer is not valid, update the node with negative reward
            node.update(self.config.negative_reward)
            return 
        
        if self.config.is_sampling:
            final_answer = node.state["final_answer"]
            correct = is_equiv(self.ground_truth, final_answer)
            node.update_recursive(self.config.positive_reward if correct else self.config.negative_reward, self.root)
            if correct:
                self.found_correct_path = True
        else:
            # just append the node to candidate_nodes, will update the value in select_next_step()
            self.candidate_nodes.append(node)


    def record_intermediate_metric(self, answer, value_estimate):
        self.intermediate_metric["question"] = self.question
        self.intermediate_metric["gt"] = self.ground_truth
        # each rollout retains the answer with the highest value_estimate
        # Check if the rollout's answer is already in the list
        if self.intermediate_metric["rollout_indexs"] and self.rollout_idx in self.intermediate_metric["rollout_indexs"]:
            # Find the index of the existing rollout
            index = self.intermediate_metric["rollout_indexs"].index(self.rollout_idx)
            if value_estimate > self.intermediate_metric["value_estimate"][index]:
                self.intermediate_metric["answers"][index] = answer
                self.intermediate_metric["judgements"][index] = is_equiv(self.ground_truth, answer)
                self.intermediate_metric["value_estimate"][index] = value_estimate
        else:
            # If the rollout's answer is not in the list, add it
            self.intermediate_metric["answers"].append(answer)
            self.intermediate_metric["judgements"].append(is_equiv(self.ground_truth, answer))
            self.intermediate_metric["value_estimate"].append(value_estimate)
            self.intermediate_metric["rollout_indexs"].append(self.rollout_idx)


    def select_next_step(self, outputs=None, from_root=False) -> None:
        self.search_node = self.current_nodes[0] if self.current_nodes else None
        self.current_nodes = []
        if outputs:
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                if candidate_node.is_terminal and self.config.is_sampling:
                    continue
                value_estimate = output.value_estimate if output.value_estimate is not None else self.config.negative_reward
                if output.value_estimate is None:
                    candidate_node.is_terminal = True

                # backup
                if candidate_node.is_terminal and candidate_node.state["final_answer"]:
                    # for terminal node: update_recursive
                    if candidate_node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
                        candidate_node.update(self.config.negative_reward)
                    else:
                        # save intermediate metric
                        self.record_intermediate_metric(answer=candidate_node.state["final_answer"], value_estimate=value_estimate)

                        candidate_node.update_recursive(value_estimate, self.root)
                else:
                    # for intermediate node: just update the value
                    if self.config.terminal_sample:
                        pass # according to the default setting, self.config.terminal_sample==True, will go from here, so if not from_root, just select the best child of the candidate_node
                    else:
                        candidate_node.update(value_estimate)

                if self.__class__.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)
        selection_node = self.selection(from_root=from_root)
        if selection_node is not None:
            self.current_nodes.append(selection_node)
    
    
    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs): # output contains all responses for a sample in the batch
            value_estimate = output.value_estimate
            if value_estimate is not None:  
                self.expand_node(output.outputs, current_node) # after this, current_node has `best_of` children
            else:
                value_estimate = self.config.negative_reward
                current_node.is_terminal = True

            if self.config.update_leaf_value: # False
                # if need update leaf node value, just append the node to candidate_nodes, will update the value in select_next_step()
                for value_node in current_node.children:
                    if value_node not in self.candidate_nodes and value_node.visit_count() < 1:
                        self.candidate_nodes.append(value_node) 
                    
                    
    def return_states(self, analyze_reward = False) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        reward_stat = {}
        if analyze_reward:
            correct_reward = np.zeros(self.config.reward_analysis_max_path_len)
            correct_cnt = np.zeros(self.config.reward_analysis_max_path_len)
            neutral_reward = np.zeros(self.config.reward_analysis_max_path_len)
            neutral_cnt = np.zeros(self.config.reward_analysis_max_path_len)
            incorrect_reward = np.zeros(self.config.reward_analysis_max_path_len)
            incorrect_cnt = np.zeros(self.config.reward_analysis_max_path_len)
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            states[node.tag]["q_value"] = node.q_value()
            states[node.tag]["visit_count"] = node.visit_count()
            states[node.tag]["lead_to_correct"] = node.lead_to_correct
            states[node.tag]["step_correctness"] = node.step_correctness
            
            if analyze_reward and on_annotated_path(node.tag) and reward_stat!=None and node.depth<len(self.step_rewards): # annotated path
                dist_to_answer = len(self.step_rewards)-1-node.depth # distance to the final answer
                gt_reward = self.step_rewards[node.depth]
                if gt_reward == 1:
                    correct_cnt[dist_to_answer] += 1
                    correct_reward[dist_to_answer] += node.q_value()    
                elif gt_reward == 0:
                    neutral_cnt[dist_to_answer] += 1
                    neutral_reward[dist_to_answer] += node.q_value()
                elif gt_reward == -1:
                    incorrect_cnt[dist_to_answer] += 1
                    incorrect_reward[dist_to_answer] += node.q_value()
            
            if node.has_children():
                candidates.extend(node.children)
        if analyze_reward:        
            reward_stat["correct_reward"] = correct_reward
            reward_stat["correct_cnt"] = correct_cnt
            reward_stat["neutral_reward"] = neutral_reward  
            reward_stat["neutral_cnt"] = neutral_cnt
            reward_stat["incorrect_reward"] = incorrect_reward
            reward_stat["incorrect_cnt"] = incorrect_cnt         
        
        return states, reward_stat

