'''
Traverse the MCTS trees stored in JSONL files, and perform three tasks:
1. Mask reasoning steps using LLM.
2. Fill in masked reasoning steps using MLLM.
3. Check the consistency of original and filled reasoning steps using LLM.
4. Save the results back to the JSONL file.
Other effects:
- Ensure each step ('text') starts with 'Step i:', except for the root node.
'''
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from rstar_deepthink.config import BaseConfig
from rstar_deepthink.llms.llm_engine import llm_init, mllm_init
from rstar_deepthink.llms.llms import llm_generate
from rstar_deepthink.llms.llm_engine import llm_engine
from rstar_deepthink.solver import Solver
from functools import partial
from pydantic import BaseModel
from transformers import AutoTokenizer
from typing import Optional, Any
from multimodal.step_correctness.mutual_check_prompt import sys_prompt_mask_accum_step, sys_prompt_mask_single_step, sys_prompt_mask_consec_step,\
    sys_prompt_fill_single_step, sys_prompt_fill_consec_step, sys_prompt_fill_accum_step, \
    sys_prompt_check_single_step, sys_prompt_check_consec_step
from PIL import Image
from pydantic import Field
import os
from rstar_deepthink.agents.utils import multimodal_prompt_wrap, rstar_equiv, multimodal_step_result_unwrap


NO_VALID_STEP_CORRECTNESS = -999
NO_VALID_RECOMPLETION_CORRECTNESS = -999

class Traverser(Solver):
    llm_sampling_params: Any = Field(default=None)
    mllm_sampling_params: Any = Field(default=None)

    def __init__(self, config, load_llm=False, load_mllm=False, llm_dir=None, mllm_dir=None, **kwargs):
        super().__init__(config=config, **kwargs)

        if load_llm and llm_dir:
            self.config.llm_dir = llm_dir
            self.llm, self.llm_sampling_params = llm_engine(self.config)
            self.llm = partial(llm_generate, engine=self.llm)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_dir)

        if load_mllm and mllm_dir:
            self.config.mllm_dir = mllm_dir
            if self.config.use_vllm_for_multimodal:
                self.mllm, self.mllm_sampling_params = llm_engine(self.config)
                self.mllm = partial(llm_generate, engine=self.mllm)
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.mllm_dir)
            else:
                self.mllm, self.mllm_processor = self.create_mllm_from_hf()
                self.tokenizer = self.mllm_processor.tokenizer

    def mask_steps(self, input_file: str, output_file: str, mode = "consec_step", process_num = -1, q_value_threshold_for_check = 0.0):
        '''
        Mask crucial information in reasoning steps using LLM.
        '''
        assert mode in ["single_step", "consec_step", "accum_step"]
        updated_records = []

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines[:process_num], desc="Masking with LLM"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "rstar" in data and isinstance(data["rstar"], dict):
                rstar_dict = data["rstar"]
                question = data["question"]
                node_tags = list(rstar_dict.keys())

                batch_size = self.config.batch_size
                for i in range(0, len(node_tags), batch_size):
                    batch_tags = node_tags[i:i + batch_size]
                    batch_input_for_mask_llm = []
                    valid_tags = []
                    for tag in batch_tags:
                        curr_step = rstar_dict[tag]["text"]
                        #if "input_for_mask_llm" in rstar_dict[tag].keys():
                        #    continue
                        
                        if "final_answer" in curr_step or "final answer" in curr_step or curr_step == "" or tag == "0":
                            rstar_dict[tag]["masked_step"] = None
                            rstar_dict[tag]["step_correctness"] = NO_VALID_STEP_CORRECTNESS
                        else:
                            parent_tag = tag[:-2]
                            
                            # Ensure each step starts with 'Step i:', except for the root node and the final answer nodes.
                            curr_step = self.add_step_prefix(rstar_dict, curr_step, tag)
                            rstar_dict[tag]["text"] = curr_step

                            
                            # Construct the input for the LLM to mask
                            if rstar_dict[tag]['lead_to_correct'] == 1 and rstar_dict[tag]['q_value']>q_value_threshold_for_check: # on the correct path, directly set `step_correctness` to 1
                                rstar_dict[tag]["masked_step"] = None
                                rstar_dict[tag]["step_correctness"] = 1
                            else:
                                valid_tags.append(tag)
                                if mode == "single_step":
                                    input_for_mask_llm = curr_step
                                elif mode == "consec_step":
                                    input_for_mask_llm = (rstar_dict[parent_tag]["text"], curr_step)
                                elif mode == "accum_step":
                                    input_for_mask_llm = rstar_dict[parent_tag].get("input_for_mask_llm","") + "\n" + curr_step
                                    rstar_dict[tag]["input_for_mask_llm"] = input_for_mask_llm
                                batch_input_for_mask_llm.append(input_for_mask_llm)

                    prompts_mask = []
                    for input_for_mask_llm in batch_input_for_mask_llm:
                        if mode == "single_step":
                            prompts_mask.append(f"{sys_prompt_mask_single_step}\n\nQuestion $Q$: {question}\nReasoning step $s$: {input_for_mask_llm}\nYour output: ")
                        elif mode == "consec_step":
                            if input_for_mask_llm[0] == "": # input_for_mask_llm[1] is the first step
                                prompts_mask.append(f"{sys_prompt_mask_consec_step}\n\nQuestion $Q$: {question}\n\nReasoning step $i$: {input_for_mask_llm[1]}\n\nYour output: ")
                            else:
                                prompts_mask.append(f"{sys_prompt_mask_single_step}\n\nQuestion $Q$: {question}\n\nReasoning step $i-1$: {input_for_mask_llm[0]}\nReasoning step $i$: {input_for_mask_llm[1]}\n\nYour output: ")
                        elif mode == "accum_step":
                            prompts_mask.append(f"{sys_prompt_mask_accum_step}\n\nQuestion $Q$: {question}\n\nReasoning trajectory $S$: {input_for_mask_llm}\n\nYour output: ")

                    mask_outputs = self.llm(prompts_mask, self.llm_sampling_params)
                    for tag, moutput in zip(valid_tags, mask_outputs):
                        output_mask_step = self.add_step_prefix(rstar_dict, moutput.outputs[0].text, tag)
                        if output_mask_step != "no" and "<blank>" not in output_mask_step:
                            output_mask_step += " <blank>"
                        rstar_dict[tag]["masked_step"] = output_mask_step

                data["rstar"] = rstar_dict
            updated_records.append(data)

        with open(output_file, 'w', encoding='utf-8') as f:
            for record in updated_records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write('\n')

    def fill_steps(self,input_file: str, output_file: str, mode = "accum_step"):
        """
        Fill in masked reasoning steps using MLLM.
        """
        updated_records = []
        assert mode in ["single_step", "consec_step", "accum_step"]

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Filling with MLLM"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "rstar" in data and isinstance(data["rstar"], dict):
                rstar_dict = data["rstar"]
                question = data["question"]
                img_path = os.path.join(self.config.dataset_dir, data["image"]) 
                node_tags = list(rstar_dict.keys())

                batch_size = self.config.batch_size
                
                for i in range(0, len(node_tags), batch_size):
                    batch_tags = node_tags[i:i + batch_size]
                    valid_tags = []
                    multimodal_processed_inputs = []
                    for tag in batch_tags:
                        if rstar_dict[tag].get("masked_step", None) is None:
                            rstar_dict[tag]["step_correctness"] = NO_VALID_STEP_CORRECTNESS # invalid
                            rstar_dict[tag]["filled_text"] = None
                            continue
                        masked_step = rstar_dict[tag]["masked_step"]
                        if '<blank>' not in masked_step:
                            rstar_dict[tag]["step_correctness"] = NO_VALID_STEP_CORRECTNESS # invalid
                            rstar_dict[tag]["filled_text"] = None
                            continue

                        previous_steps = self.get_previous_steps(rstar_dict, tag)

                        if "Qwen2.5-VL" in self.config.mllm_dir or "Qwen2-VL" in self.config.mllm_dir:
                            if mode == "accum_step":
                                sys_prompt = f"<|im_start|>\n{sys_prompt_fill_accum_step}<|im_end|>"
                                user_prompt = (
                                    "<|im_start|>IMAGE $I$: <|vision_start|><|image_pad|><|vision_end|>"
                                    f"QUESTION $Q$: {question}\nREASONING TRAJECTORY $S$: {previous_steps}\n{masked_step}<|im_end|>\n"
                                    "<|im_start|>YOUR OUTPUT:\n"
                                )
                            elif mode == "consec_step" and len(tag) == 3 or mode == "single_step":
                                sys_prompt = f"<|im_start|>\n{sys_prompt_fill_single_step}<|im_end|>"
                                user_prompt = (
                                    "<|im_start|>IMAGE $I$: <|vision_start|><|image_pad|><|vision_end|>"
                                    f"QUESTION $Q$: {question}\nREASONING STEP $i$: {masked_step}\n" 
                                    "<|im_start|>YOUR OUTPUT:\n"
                                )
                            else: # "consec_step", step>1 (len(tag) > 3)
                                parent_tag = tag[:-2]
                                last_step = rstar_dict[parent_tag]["text"]
                                sys_prompt = f"<|im_start|>\n{sys_prompt_fill_consec_step}<|im_end|>"
                                user_prompt = (
                                    "<|im_start|>IMAGE $I$: <|vision_start|><|image_pad|><|vision_end|>"
                                    f"QUESTION $Q$: {question}\nREASONING STEP $i-1$: {last_step}\nREASONING STEP $i$: {masked_step}\n" 
                                    "<|im_start|>YOUR OUTPUT:\n"
                                    )

                            img = Image.open(img_path).convert("RGB") if img_path else None
                            img = self.resize_img_if_needed(img)
                            multimodal_processed_inputs.append({
                                "prompt": f"{sys_prompt}\n{user_prompt}",
                                "multi_modal_data": {"image": img},
                            })
                            valid_tags.append(tag)
                        else:
                            raise NotImplementedError()

                    if not multimodal_processed_inputs:
                        continue

                    fill_outputs = self.mllm(
                        multimodal_processed_inputs,
                        self.mllm_sampling_params
                    )

                    for out, tag in zip(fill_outputs, valid_tags):
                        filled_text = out.outputs[0].text
                        rstar_dict[tag]["filled_text"] = self.add_step_prefix(rstar_dict, filled_text, tag)

                data["rstar"] = rstar_dict
            updated_records.append(data)


        with open(output_file, 'w', encoding='utf-8') as f:
            for record in updated_records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write('\n')

    def check_steps(self, input_file: str, output_file: str, mode = "single_step"):
        updated_records = []
        assert mode in ["single_step"]
        sys_prompt_check = globals()[f"sys_prompt_check_{mode}"]
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Checking consistency with LLM"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "rstar" in data and isinstance(data["rstar"], dict):
                rstar_dict = data["rstar"]
                node_tags = list(rstar_dict.keys())

                batch_size = self.config.batch_size
                for i in range(0, len(node_tags), batch_size):
                    valid_tags = []
                    batch_tags = node_tags[i:i + batch_size]
                    prompts_check = []
                    for tag in batch_tags:
                        if rstar_dict[tag].get("filled_text", None) is None:
                            continue
                        original_text = rstar_dict[tag]["text"]
                        filled_text = rstar_dict[tag]["filled_text"]
                        if mode == "single_step":
                            prompt_check = f"{sys_prompt_check}\n\nReasoning step $s_1$: {original_text}\nReasoning step $s_2$: {filled_text}\nYour output: "
                        prompts_check.append(prompt_check)
                        valid_tags.append(tag)  
                    if not prompts_check:
                        continue
                    check_outputs = self.llm(prompts_check, self.llm_sampling_params)

                    for tag, chk_out in zip(batch_tags, check_outputs):
                        out_str = chk_out.outputs[0].text
                        if "1" in out_str:
                            rstar_dict[tag]["step_correctness"] = 1
                        elif "0" in out_str:
                            rstar_dict[tag]["step_correctness"] = 0
                        else:
                            # 无法判断时自行定义处理
                            rstar_dict[tag]["step_correctness"] = NO_VALID_STEP_CORRECTNESS

                data["rstar"] = rstar_dict
            updated_records.append(data)

        # 覆盖写回原文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in updated_records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write('\n')

    def annotate_ans_contribution(self, input_file: str, output_file: str):
        updated_records = []

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Checking consistency with LLM"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "rstar" in data and isinstance(data["rstar"], dict):
                rstar_dict = data["rstar"]
                node_tags = list(rstar_dict.keys())
                batch_size = self.config.batch_size
                for i in range(0, len(node_tags), batch_size):
                    batch_tags = node_tags[i:i + batch_size]
                    for tag in batch_tags:
                        if tag == "0" or rstar_dict[tag].get("step_correctness", 0) !=1: # Only consider correct steps
                            continue
                        if rstar_dict[tag]["lead_to_correct"]:
                            rstar_dict[tag]["ans_contribution"] = 1
                        else:
                            rstar_dict[tag]["ans_contribution"] = 0
                   
                data["rstar"] = rstar_dict
            updated_records.append(data)

        # 覆盖写回原文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in updated_records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write('\n')

    def get_previous_steps(self, rstar_dict, tag):
        """
        Get the steps before the current step.
        """
        previous_steps = ""
        parent_tag = tag[:-2]
        while parent_tag != "0":
            previous_steps = rstar_dict[parent_tag]["text"] + "\n" + previous_steps
            parent_tag = parent_tag[:-2]
        #if "Step 1" not in previous_steps:
        #    previous_steps = "Step 1\n" + previous_steps
        return previous_steps.strip()
    
    def add_step_prefix(self, rstar_dict, step, tag):
        """
        Add the prefix 'Step i:' to the step text.
        """
        step = step.strip()
        if step == "no":
            return step
        if step[:5]!="Step ": # step doesn't start with "Step "
            if len(tag) == 3: # step 1. According to the format of the MCTS rollout reults, all steps 1s have no "Step 1:" prefix.
                step = "Step 1: " + step
            else:
                parent_tag = tag[:-2]
                step = "Step " + str(int(rstar_dict[parent_tag]["text"][5])+1) + ": " + step
        return step
    
    def complete_rollout(self, input_file: str, output_file: str, mode = "consec_step", process_num = -1, q_value_threshold_for_check = 0.0):
        '''
        Complete the rollout from the given reasoning intermediate steps.
        '''
        updated_records = []

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines[:process_num], desc="Completing Rollout"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "rstar" in data and isinstance(data["rstar"], dict):
                rstar_dict = data["rstar"]
                question = data["question"]
                answer = data["answer"]
                node_tags = list(rstar_dict.keys())

                batch_size = self.config.batch_size
                for i in range(0, len(node_tags), batch_size):
                    batch_tags = node_tags[i:i + batch_size]
                    batch_input = []
                    valid_tags = []
                    for tag in batch_tags:
                        curr_step = rstar_dict[tag]["text"]
                        #if "input_for_mask_llm" in rstar_dict[tag].keys():
                        #    continue
                        
                        if "final_answer" in curr_step or "final answer" in curr_step or curr_step == "" or len(tag) <=3:
                            rstar_dict[tag]["recompleted_rollout"] = None 
                            rstar_dict[tag]["recompletion_correctness"] = NO_VALID_RECOMPLETION_CORRECTNESS
                        else:
                            # Ensure each step starts with 'Step i:', except for the root node and the final answer nodes.
                            curr_step = self.add_step_prefix(rstar_dict, curr_step, tag)
                            rstar_dict[tag]["text"] = curr_step

                            # Construct the input for the LLM to mask
                            if rstar_dict[tag]['lead_to_correct'] == 1 and rstar_dict[tag]['q_value']>q_value_threshold_for_check: # on the correct path, directly set `step_correctness` to 1
                                rstar_dict[tag]["recompleted_rollout"] = None 
                                rstar_dict[tag]["recompletion_correctness"] = NO_VALID_RECOMPLETION_CORRECTNESS
                            else:
                                valid_tags.append(tag)
                                previous_steps = self.get_previous_steps(rstar_dict, tag) 
                                if previous_steps.startswith("Step 1:"):
                                    previous_steps = previous_steps[len("Step 1:"):]
                                input_prompt = multimodal_prompt_wrap(question, previous_steps, self.config)[0]
                                batch_input.append(input_prompt)

                    recompletion_outputs = self.mllm(batch_input, self.mllm_sampling_params)
                    for tag, output in zip(valid_tags, recompletion_outputs):
                        recompleted_rollout = self.add_step_prefix(rstar_dict, output.outputs[0].text, tag)
                        rstar_dict[tag]["recompleted_rollout"] = recompleted_rollout
                        extracted_final_answer = multimodal_step_result_unwrap(recompleted_rollout)[1]['final_answer']
                        rstar_dict[tag]["recompletion_correctness"] = rstar_equiv(extracted_final_answer, answer) # 1 for correct, 0 for incorrect

                data["rstar"] = rstar_dict
            updated_records.append(data)

        with open(output_file, 'w', encoding='utf-8') as f:
            for record in updated_records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write('\n')
    
def main():
    parser = argparse.ArgumentParser(description="Process JSONL file with three separate steps (mask, fill, check).")
    parser.add_argument('--custom_cfg', type=str, default="config/traverse.yaml")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the MCTS input JSONL file.")
    parser.add_argument("--llm_dir", type=str, default="", help="Path to the LLM directory.")
    parser.add_argument("--mllm_dir", type=str, default="", help="Path to the MLLM directory.")
    parser.add_argument("--task", type=str, required=True, choices=["mask", "fill", "check", "annotate_ans_contribution", "complete_rollout"], help="Task to perform. Get step correctness: mask, fill, or check; get answer contribution: annot_ans_contribution.")
    parser.add_argument("--step_correctness_task_mode", type=str, default="consec_step", choices=["single_step", "consec_step", "accum_step"], help="Masking/filling/checking mode for the reasoning steps.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for processing.")
    parser.add_argument('--process_num', type=int, default=-1, help="Number of records to process. Default is -1 for all records.")
    parser.add_argument('--dataset_dir', type=str, default="", help="Path to the image directory.")
    parser.add_argument('--q_value_threshold_for_check', type=float, default=0.2, help="If below threshold, the step is likely to be incorrect. For checking the correctness of reasoning steps or rollout recompletion by the RL model.")
    args = parser.parse_args()

    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))

    load_llm = (args.task in ["mask", "check"])
    load_mllm = (args.task in ["fill", "complete_rollout"])

    config.use_vllm_for_multimodal = (args.task in ["fill", "complete_rollout"])
    config.batch_size = args.batch_size
    config.dataset_dir = args.dataset_dir
    traverser = Traverser(
        config=config,
        load_llm=load_llm,
        load_mllm=load_mllm,
        llm_dir=args.llm_dir,
        mllm_dir=args.mllm_dir
    )

    if args.task in ["mask", "fill", "check"]:
        step_correctness_task_output_file = os.path.splitext(args.input_file)[0] + f"-{args.task}ed-{args.step_correctness_task_mode}.jsonl"

    if args.task == "mask":
        assert not any(task in args.input_file for task in ["-masked-", "-filled-", "-checked-"]) 
        traverser.mask_steps(args.input_file, step_correctness_task_output_file, mode=args.step_correctness_task_mode, q_value_threshold_for_check=args.q_value_threshold_for_check, process_num=args.process_num)
    elif args.task == "fill":
        assert "-masked" in args.input_file and "-filled-" not in args.input_file and "-checked-" not in args.input_file
        traverser.fill_steps(args.input_file, step_correctness_task_output_file, mode=args.step_correctness_task_mode)
    elif args.task == "check":
        assert "-masked" in args.input_file and "-filled-" in args.input_file and "-checked-" not in args.input_file
        traverser.check_steps(args.input_file, step_correctness_task_output_file, mode=args.step_correctness_task_mode)
    elif args.task == "annotate_ans_contribution":
        assert "-masked" in args.input_file and "-filled-" in args.input_file and "-checked-" in args.input_file
        step_correctness_task_output_file = os.path.splitext(args.input_file)[0] + f"-w-ans_contri_annot.jsonl"
        traverser.annotate_ans_contribution(args.input_file, step_correctness_task_output_file)
    elif args.task == "complete_rollout":
        complete_rollout_output_file = os.path.splitext(args.input_file)[0] + f"-recompleted.jsonl"
        traverser.complete_rollout(args.input_file, complete_rollout_output_file, q_value_threshold_for_check=args.q_value_threshold_for_check, process_num=args.process_num)
if __name__ == "__main__":
    main()