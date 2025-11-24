# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
from typing import List, Dict, Any, Optional, Type, Tuple, Union
#from math_evaluation import is_equiv
from verl.workers.rollout.rstar_deepthink.prompts.prompt_rstar import PROMPT_RSTAR
from verl.workers.rollout.rstar_deepthink.tools.python_tool import PythonInterpreter
from verl.workers.rollout.rstar_deepthink.constants import *
from verl.workers.rollout.utils.math_equal import math_equal
from verl.workers.rollout.utils.checker import check_one_answer
from verl.workers.rollout.utils.util import equiv, strip_string, choice_answer_clean
from verl.workers.rollout.multimodal.registered_models import qwen_series
import re

def remove_text_box(text: str | None) -> str | None:
    if text is None:
        return None

    start = text.find(r"\text{")
    if start == -1:
        return text.strip()

    # ---------- 初始化哨兵 ----------
    start_text = end_text = None
    stack = []
    answer = text[start:]

    for i, c in enumerate(answer):
        if c == "{":
            stack.append(i)
        elif c == "}":
            if not stack:          # 括号不匹配，直接返回原串
                return text.strip()
            start_text = stack.pop()
            if not stack:          # 最外层 '}' 配对完成
                end_text = i
                break

    # ---------- 若没匹配成功，直接返回原串 ----------
    if start_text is None or end_text is None:
        return text.strip()

    in_text_string = text[start + start_text + 1 : start + end_text]
    if in_text_string.strip() == "and":
        ex_text = text[:start] + text[start + end_text + 1 :]
    else:
        ex_text = (
            text[:start]
            + in_text_string.strip()
            + text[start + end_text + 1 :]
        )
    return ex_text.strip()



def extract_boxed_answer(text, debug=False):
    if text is None:
        return None
    start = text.rfind(r"\boxed{")
    if start == -1:
        return text
    end = None
    stack = []
    answer = text[start:]
    for i, c in enumerate(answer):
        if c == "{":
            stack.append(i)
        elif c == "}":
            start = stack.pop()  # \boxed start{
            if len(stack) == 0:
                end = i  # \boxed end}
                break
    if end is None and debug:
        print("brack not closing", answer)
        return None
    return answer[start + 1 : end]

def extract_no_boxed_answer(text, debug=False):
    if text is None:
        return None
    start = -1
    answer_indicators = ["is"]
    for answer_indicator in answer_indicators:
        if answer_indicator in text.lower():
            start = text.lower().rfind(answer_indicator)
            if start == -1:
                continue
            else:
                start = start + len(answer_indicator)
                break
    
    end = text.find("</answer>")
    if start == -1:
        return 'None'
    if end !=1:
        return text[start:end]
    return text[start:]

INVALID_ANS = "[invalid]"

def extract_math_answer(answer, answer_format = "boxed"):
    try:
        extract_ans_temp = answer.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if answer_format == "boxed":
            extract_ans = remove_text_box(extract_boxed_answer(extract_ans))
        elif answer_format == "no_boxed":
            extract_ans = remove_text_box(extract_no_boxed_answer(extract_ans))
    except:
        extract_ans = INVALID_ANS
    return extract_ans


python_tool_string = f"{PythonInterpreter().name}: {PythonInterpreter().description}"
python_tool_name = PythonInterpreter().name
    

def rstar_prompt_wrap(
    question: str, 
    partial_solution: str,
    config,
) -> str:
    step_delim = config.step_delim
    prompt_pot = PROMPT_RSTAR(config)
    inputs = f"{question}{step_delim}"  

    rstar_examples = prompt_pot.random_examples()
    
    if len(rstar_examples) > 1:
        example_prefix = "The following are %d demonstration examples." % len(rstar_examples)
    elif len(rstar_examples) == 1:
        example_prefix = "The following is a demonstration example."

    format_instructions = prompt_pot.pot_format_instructions
    
    if len(rstar_examples) > 0:
        sys_prompt = step_delim.join([format_instructions, example_prefix, *rstar_examples, ""])
    else:
        sys_prompt = step_delim.join([format_instructions, ""])
    if sys_prompt.strip() == "":
        prompt = step_delim.join([prompt_pot.pot_suffix.format(input=inputs)])
    else:
        prompt = step_delim.join([sys_prompt, prompt_pot.pot_suffix.format(input=inputs)])
    if partial_solution:
        prompt = "".join([prompt, partial_solution])
    return prompt + "", sys_prompt 


def multimodal_prompt_wrap(
    question: str, 
    partial_solution: str,
    config,
    first_annotated_step: str = None
) -> str:
    step_delim = config.step_delim
    prompt_pot = PROMPT_RSTAR(config)

    rstar_examples = prompt_pot.random_examples()
    
    if len(rstar_examples) > 1:
        example_prefix = "The following are %d demonstration examples." % len(rstar_examples)
    elif len(rstar_examples) == 1:
        example_prefix = "The following is a demonstration example."

    format_instructions = prompt_pot.pot_format_instructions
    
    
    if len(rstar_examples) > 0:
        sys_prompt = step_delim.join([format_instructions, example_prefix, *rstar_examples, ""])
    else:
        sys_prompt = step_delim.join([format_instructions, ""])
    
    if any([x in config.mllm_dir for x in qwen_series]):
        prompt = f"<|im_start|>system\n{sys_prompt}\n<|im_end|>"
        user_prompt = ("<|im_start|>user\nNow, it's your turn!<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n###Step 1: "
        )
    else:
        raise NotImplementedError
    
    if config.task == "reward_analysis":
        user_prompt += first_annotated_step # when analyzing the reward, enforce the first step to be the first annotated step
        
    if prompt.strip() == "":
        prompt = step_delim.join([user_prompt])
    else:
        prompt = step_delim.join([prompt, user_prompt])
    if partial_solution:
        prompt = "".join([prompt, partial_solution])
    return prompt + "", sys_prompt


def rstar_obs_wrap(observation: str) -> str:
    return f"{OUTPUT}{observation}{OUTPUT_END}"


def rstar_step_result_unwrap(
    text: str,
) -> Tuple[str, Dict[str, str]]:
    parser_result = {
        "action": "",
        "action_input": "",
        "final_answer": "",
    }
    #if ANSWER_END in text or "boxed" in text:
    if "boxed" in text:
        parser_result["final_answer"] = extract_math_answer(text)
        return text, parser_result
    else:
        parser_result["action"] = "python_interpreter"
        parser_result["action_input"] = text
        return text, parser_result
    
def multimodal_step_result_unwrap(
    text: str,
) -> Tuple[str, Dict[str, str]]:
    parser_result = {
        "action": "",
        "action_input": "",
        "final_answer": "",
    }
    #if ANSWER_END in text or "boxed" in text:
    if "boxed" in text:
        parser_result["final_answer"] = extract_math_answer(text, answer_format="boxed")
        return text, parser_result
    if "answer" in text:
        parser_result["final_answer"] = extract_math_answer(text, answer_format="no_boxed")
        return text, parser_result
    if "<visual_clues>" in text: # to-do: change into judged by which macro action is chosen, rather than determined by the information in the output
        parser_result["action"] = "perception"
        parser_result["action_input"] = text
        return text, parser_result
    else:
        parser_result["action"] = "reasoning"
        parser_result["action_input"] = text
        return text, parser_result


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def remove_single_dollar(s):
    if not s:
        return s
    if isinstance(s, list):
        s = s[0]
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return s


def any_condition(conditions):
    return any(conditions)


def rstar_equiv(gt, pred, grt_choices = None): # grt_choices is a list that lists the ground truth choices if it's a multi-choice question
    # In this function, I integrated multiple open-source evaluation tools
    # each with its own judgment logic and strengths in handling special cases such as LaTeX, units, etc.
    gt = str(gt)
    pred = str(pred)
    try:
        if gt.strip().lower() == pred.strip().lower():
            return True
        
        pred_choice = None
        if pred.lower() in ["a", "b", "c", "d", "e", "f"]:
            pred_choice = pred[0]
        
        if pred[:2].lower() in ['a:', 'b:', 'c:', 'd:', 'e:', 'f:', 'a.', 'b.', 'c.', 'd.', 'e.', 'f.']: # for preds like "A: 1.414"
            pred_choice = pred[0]

        if pred[:3].lower() in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']: # for preds like "(a) 1.414"
            pred_choice = pred[1]

            

        # If gt is not in ['A', 'B', 'C', 'D', 'E'] but pred is in ['A', 'B', 'C', 'D', 'E']
        if gt.lower() not in ['a', 'b', 'c', 'd', 'e', 'f'] and pred_choice is not None and grt_choices is not None:
            choices = ['a', 'b', 'c', 'd', 'e', 'f']
            ground_truth_choice = choices[grt_choices.index(gt)]
            if ground_truth_choice.lower() == pred_choice.lower():
                return True
            
        # If gt is in ['A', 'B', 'C', 'D', 'E'] but pred is not in ['A', 'B', 'C', 'D', 'E']
        if gt.lower() in ['a', 'b', 'c', 'd', 'e', 'f'] and pred_choice is None and grt_choices is not None:
            choices = ['a', 'b', 'c', 'd', 'e', 'f']
            pred_choice = choices[grt_choices.index(pred)]
            #print("pred_choice", pred_choice)
            if gt.lower() == pred_choice.lower():
                return True

        # Check if both gt and pred are words (no numbers) and pred is a substring of gt
        if not any(char.isdigit() for char in gt) and not any(char.isdigit() for char in pred):
            if pred.strip().lower() in gt.strip().lower():
                return True

        # Check if gt or pred contains "√{*}" and convert to "\sqrt{*}"
        sqrt_pattern = r"√\{(.*?)\}"
        gt = re.sub(sqrt_pattern, r"\\sqrt{\1}", gt)
        pred = re.sub(sqrt_pattern, r"\\sqrt{\1}", pred)
            

        # For college-math and omni-math, the pred and gt positions need to be changed.
        # Because we found that the quality of ground truth in a small subset of problems within benchmarks like college-math is relatively low.
        if any(
            func(x, y) for func in [math_equal, check_one_answer] for x, y in [(gt, pred), (pred, gt)]
        ):
            return True
        # special for college-math, etc.
        gt_strip, pred_strip = strip_string(gt), strip_string(pred)
        if any(
            func(x, y) for func in [math_equal, check_one_answer] for x, y in [(gt_strip, pred_strip), (pred_strip, gt_strip)]
        ):
            return True

        # for choice question
        if gt in ["A", "B", "C", "D", "E"] and pred not in ["A", "B", "C", "D", "E"]:
            pred = choice_answer_clean(pred)
            if math_equal(gt, pred):
                return True
        elif is_multi_choice(gt) and not is_multi_choice(pred):
            pred = "".join(
                [c for c in pred if c in ["A", "B", "C", "D", "E"]]
            )
            if math_equal(gt, pred):
                return True
    except Exception as e:
        #print("maroi_equiv error")
        #print(e)
        pass
    return False
        

def math_equiv(grt: Union[str, list[str]], prd: str, grt_choice = None):
    prd = (prd)
    if isinstance(grt, list):
        for g in grt:
            if rstar_equiv(g, prd, grt_choice):
                return True
        return False
    else:
        return rstar_equiv(grt, prd, grt_choice)


def truncate_prompt(tokenizer, prompts, max_input_len):
    encoded_batch = tokenizer(
        prompts,
        truncation=True,
        max_length=max_input_len,
        padding=False, # whether to padding to the same_size
        return_tensors=None
    )
    input_ids_list = encoded_batch["input_ids"]
    return [tokenizer.decode(ids) for ids in input_ids_list]


def on_annotated_path(node_tag):
    if node_tag == '0':
        return True
    ids = node_tag.split('.')[1:]
    for i in ids:
        if i!='1':
            return False
    return True
    
def extract_and_check(response_str: str, gt: str) -> bool:
    # Extract the answer from the response string
    answer = remove_text_box(extract_boxed_answer(response_str))
    if answer is None:
        answer = remove_text_box(extract_no_boxed_answer(response_str))
    if answer is None:
        return False
    
    return rstar_equiv(gt, answer)