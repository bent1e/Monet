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

import re
from typing import Dict, List, Union
import numpy as np
from mathruler.grader import extract_boxed_content, grade_answer
#from math_evaluation import is_equiv
from examples.reward_function.answer_transformation import answer_transformation_fn
from verl.workers.rollout.rstar_deepthink.agents.utils import remove_text_box, extract_no_boxed_answer
import re
import torch

BOXED_RE     = re.compile(r"The final answer is: \\boxed\{.*?\}", re.DOTALL)          # must have \\boxed
INVALID_HASH = re.compile(r"#{4,}")                              # 4 or more #
BAD_STEP12   = re.compile(r"(?<!#)#{1,2}\s*Step\s+\d+(?:\.\d+)*:")  # 1~2 #
BAD_TRIPLE   = re.compile(r"\#{3}(?!\s*Step\s+\d+(?:\.\d+)*:)")     # 3 # but illegal
AFTER_BOXED_RE = re.compile(r"The final answer is: \\boxed\{.*?\}\s*\S", re.DOTALL)

def format_reward(predict: str):
    if not BOXED_RE.search(predict):
        return 0.0 
    if AFTER_BOXED_RE.search(predict):
        return 0.0         
    if INVALID_HASH.search(predict):
        return 0.0          
    if BAD_STEP12.search(predict):
        return 0.0          
    if BAD_TRIPLE.search(predict):
        return 0.0      
    return 1.0              



def accuracy_reward(predict: str, ground_truth: str) -> float:
    return 1.0 if extract_and_check(predict, ground_truth) else 0.0


# use for the rule-based judge in the RL rollouts, V2
'''def extract_and_check(predict: str, ground_truth: str) -> float:
    answer = remove_text_box(extract_boxed_content(predict))
    answer, ground_truth = answer_transformation_fn(answer), answer_transformation_fn(ground_truth)
    if grade_answer(answer, ground_truth):
        return True
    else:
        return is_equiv(ground_truth, answer)'''

# use for the rule-based judge in the RL rollouts, V1  
def extract_and_check(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    if answer == 'None':
        answer = extract_no_boxed_answer(predict)
    return grade_answer(answer, ground_truth)

def compute_score(predicts: List[str], step_reward_or_gts: Union[List[float], List[str], List[List[float]]], format_weight: float = 0.1, length_penalty_weight = 0.001, resp_lengths = None, ref_resp_lengths = None) -> List[Dict[str, float]]:
    # test: step_reward_or_gts is the ground truth str
    # train: step_reward_or_gts is the step reward
    scores = []
    if isinstance(step_reward_or_gts[0], float) or isinstance(step_reward_or_gts[0], str):
        for predict, step_reward_or_gt in zip(predicts, step_reward_or_gts):
            predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
            format_score = format_reward(predict)
            if isinstance(step_reward_or_gt, str):
                accuracy_score = accuracy_reward(predict, step_reward_or_gt)
            else:
                accuracy_score = float(step_reward_or_gt) #accuracy_reward(predict, step_rewards)
            scores.append(
                {
                    "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,#accuracy_score, #(1 - format_weight) * accuracy_score + format_weight * format_score,
                    "format": format_score,
                    "accuracy": accuracy_score,
                }
            )
    elif isinstance(step_reward_or_gts[0], list):
        #breakpoint()
        ref_resp_lengths = torch.tensor(ref_resp_lengths) # (bs, )
        length_penalty = torch.where(resp_lengths > ref_resp_lengths, resp_lengths - ref_resp_lengths, torch.zeros_like(resp_lengths)).numpy()
        for i, (predict, steps_rewards) in enumerate(zip(predicts, step_reward_or_gts)):
            predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
            format_score = format_reward(predict)
            accuracy_score = np.array(steps_rewards)

    
            scores.append(
                {
                    "overall_mean": np.mean((1 - format_weight) * accuracy_score + format_weight * format_score),#accuracy_score, #(1 - format_weight) * accuracy_score + format_weight * format_score,
                    "format": format_score,
                    "accuracy_mean": np.mean(accuracy_score),
                    "overall_step_wise": (1 - format_weight) * np.array(steps_rewards) + format_weight * format_score - length_penalty_weight * length_penalty[i],  # step-wise reward
                }
            )
            
    return scores
