'''
Evaluate the accuracy of the mutual check module
Currently only support single-step mutual checking.
'''
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from rstar_deepthink.config import BaseConfig
from rstar_deepthink.llms.llm_engine import llm_init, mllm_init
from rstar_deepthink.llms.llms import llm_generate
from rstar_deepthink.llms.llm_engine import llm_engine
from multimodal.step_correctness.traverse_tree import Traverser
from functools import partial
from pydantic import BaseModel
from transformers import AutoTokenizer
from typing import Optional, Any
from multimodal.step_correctness.mutual_check_prompt import sys_prompt_mask_accum_step, sys_prompt_mask_single_step, sys_prompt_mask_consec_step, sys_prompt_mask_single_step_no_question,\
    sys_prompt_fill_single_step, sys_prompt_fill_consec_step, sys_prompt_fill_accum_step, \
    sys_prompt_check_single_step, sys_prompt_check_consec_step
from PIL import Image
from pydantic import Field
from multimodal.prm.prm_dataloader import ReasoningDatasetWhole, dataset_name_mapping
import os
import torch
import random
from torch.utils.data import DataLoader

NO_VALID_STEP_CORRECTNESS = -999

class MutualChecker(Traverser):
    '''
    Given a batch of reasoning steps, check their step-correctness using 
    '''
    def __init__(self, config, load_llm=False, load_mllm=False, llm_dir=None, mllm_dir=None, **kwargs):
        super().__init__(config=config, load_llm=load_llm, load_mllm=load_mllm, llm_dir=llm_dir, mllm_dir=mllm_dir, **kwargs)
        
    def mask_batch(self, batch, mode = "single_step"):
        '''
        Mask crucial information in reasoning steps using LLM.
        '''
        assert mode in ["single_step", "single_step_no_question", "single_step_mask_second_half",  "single_step_mask_random"]

        returned_data = []
        prompts_mask = []
        img_paths = batch['image']
        questions = batch['question']
        steps = batch['step']
        labels = batch['label']
        previous_stepss = batch["previous_steps"]
        mask_outputs = []
        for question, step in zip(questions, steps):
            if mode == "single_step":
                prompts_mask.append(f"{sys_prompt_mask_single_step}\n\n\nQuestion $Q$: {question}\n\nReasoning step $s$: {step}\n\nYour output: ")
            elif mode == "single_step_no_question":
                prompts_mask.append(f"{sys_prompt_mask_single_step_no_question}\n\nREASONING STEP $s$: {step}\nYOUR OUTPUT: ")
            elif mode == "single_step_mask_second_half":
                pos = int(len(step)*random.uniform(0.3, 0.6))
                mask_outputs.append(step[:pos] + "<blank>")
            elif mode == "single_step_mask_random":
                step_words = step.split(" ")
                candidate_mask_ids = []
                for i,word in enumerate(step_words):
                    if word.isdigit():
                        candidate_mask_ids.append(i)
                mask_ids = random.sample(range(len(step_words)), int(len(step_words)*0.3))
                raise NotImplementedError(f"Mode {mode} is not implemented for mask_batch.")
            else:
                raise NotImplementedError(f"Mode {mode} is not implemented for mask_batch.")

        if mode in ["single_step", "single_step_no_question"]:
            mask_outputs = self.llm(prompts_mask, self.llm_sampling_params)
        
        for img_path, question, step, label, moutput, previous_steps in zip(img_paths, questions, steps, labels, mask_outputs, previous_stepss):
            if mode in ["single_step", "single_step_no_question"]:
                output_mask_step = moutput.outputs[0].text.strip()
            elif mode in ["single_step_mask_second_half"]:
                output_mask_step = moutput
            #if output_mask_step != "no" and "<blank>" not in output_mask_step:
            #    output_mask_step += " <blank>"
            returned_data.append(
                {
                    "image": img_path,
                    "question": question,
                    "step": step,
                    "label": label.item(),
                    "masked_step": output_mask_step,
                    "previous_steps": previous_steps
                }
            )
        return returned_data

    def fill_batch(self, batch, mode = "single_step"):
        '''
        Fill the masked crucial information in reasoning steps using LLM.
        '''
        assert mode in ["single_step", "accum_step", "accum_step_mask_random"]

        returned_data = []
        multimodal_processed_inputs = []
        img_paths = batch['image']
        questions = batch['question']
        steps = batch['step']
        labels = batch['label']
        masked_steps = batch['masked_step']
        previous_stepss = batch["previous_steps"]
        
        valid_img_paths, valid_questions, valid_steps, valid_labels, valid_masked_steps = [], [], [], [], []
        for img_path, question, step, label, masked_step, previous_steps in zip(img_paths, questions, steps, labels, masked_steps, previous_stepss):
            masked_step = masked_step.strip()
            if masked_step == "no" or "<blank>" not in masked_step or label == 0:
                continue
            if mode == "single_step":
                sys_prompt = f"<|im_start|>\n{sys_prompt_fill_single_step}<|im_end|>"
                user_prompt = (
                    "<|im_start|>Image $I$: <|vision_start|><|image_pad|><|vision_end|>"
                    f"Question $Q$: {question}\nReasoning step $s_i$: {masked_step}\n" 
                    "<|im_start|>Your output:\n"
                )
            elif mode in ["accum_step", "accum_step_mask_random"]:
                sys_prompt = f"<|im_start|>\n{sys_prompt_fill_accum_step}<|im_end|>"
                user_prompt = (
                    "<|im_start|>Image $I$: <|vision_start|><|image_pad|><|vision_end|>"
                    f"Question $Q$: {question}\nReasoning trajectory $S$: {previous_steps}\nLast step $s_T$: {masked_step}<|im_end|>\n"
                    "<|im_start|>Your output: "
                )

            else:
                raise NotImplementedError(f"Mode {mode} is not implemented for fill_batch.")
            
            img = Image.open(img_path).convert("RGB") if img_path else None
            img = self.resize_img_if_needed(img)
            multimodal_processed_inputs.append({
                "prompt": f"{sys_prompt}\n{user_prompt}",
                "multi_modal_data": {"image": img},
            })
            valid_img_paths.append(img_path)
            valid_questions.append(question)
            valid_steps.append(step)
            valid_labels.append(label)
            valid_masked_steps.append(masked_step)

        fill_outputs = self.mllm(multimodal_processed_inputs, self.mllm_sampling_params)
                                 
        for img_path, question, step, label, fill_output, masked_step in zip(valid_img_paths, valid_questions, valid_steps, valid_labels, fill_outputs, valid_masked_steps):
            output_fill_step = fill_output.outputs[0].text.strip()
            returned_data.append(
                {
                    "image": img_path,
                    "question": question,
                    "step": step,
                    "label": label.item(),
                    "filled_step": output_fill_step,
                    "masked_step": masked_step
                }
            )
        return returned_data

    def check_batch(self, batch, mode = "single_step"):
        '''
        Mask crucial information in reasoning steps using LLM.
        '''
        assert mode in ["single_step"]
        mutual_check_correct_cnt = 0
        mutual_check_valid_cnt = 0
        returned_data = []
        prompts_check = []
        img_paths = batch['image']
        questions = batch['question']
        steps = batch['step']
        labels = batch['label']
        filled_steps = batch['filled_step']
        masked_steps = batch['masked_step']
        valid_img_paths, valid_questions, valid_steps, valid_labels, valid_fill_steps, valid_masked_steps = [], [], [], [], [], []
        for img_path, question, step, label, filled_step, masked_step in zip(img_paths, questions, steps, labels, filled_steps, masked_steps):
            if "<blank>" in filled_step:
                continue
            if mode == "single_step":
                prompts_check.append(f"{sys_prompt_check_single_step}\n\nReasoning step $s_1$: {step}\nReasoning step $s_2$: {filled_step}\nYour output: ")
                valid_img_paths.append(img_path)
                valid_questions.append(question)
                valid_steps.append(step)
                valid_labels.append(label)
                valid_fill_steps.append(filled_step)
                valid_masked_steps.append(masked_step)
            else:
                raise NotImplementedError(f"Mode {mode} is not implemented for check_batch.")
            
        check_outputs = self.llm(prompts_check, self.llm_sampling_params)
                                 
        for img_path, question, step, masked_step, filled_step, label, check_output in zip(valid_img_paths, valid_questions, valid_steps, valid_masked_steps, valid_fill_steps, valid_labels, check_outputs):
            label = label.item()
            output_check_step = check_output.outputs[0].text
            if "1" in output_check_step:
                mutual_check_step_correctness = 1
            elif "0" in output_check_step:
                mutual_check_step_correctness = -1
            else:
                # 无法判断时自行定义处理
                mutual_check_step_correctness = NO_VALID_STEP_CORRECTNESS
            
            if mutual_check_step_correctness in [-1,1] and label in [-1,1]: # for VisualPRMbench, step correcntess could be 0, we won't evaluate such labels for mutual check 
                mutual_check_valid_cnt += 1
            
                if mutual_check_step_correctness == label:
                    mutual_check_correct_cnt += 1
            
            returned_data.append(
                {
                    "image": img_path,
                    "question": question,
                    "step": step,
                    "masked_step": masked_step,
                    "filled_step": filled_step,
                    "label": label,
                    "mutual_check_step_correctness": mutual_check_step_correctness
                }
            )
        return returned_data, mutual_check_correct_cnt, mutual_check_valid_cnt

    
    
def main(args, config):
    
    # set seeds
    seed = 42
    random.seed(seed)
    
    # save name
    llm_name = os.path.basename(args.llm_dir)
    mllm_name = os.path.basename(args.mllm_dir)
    os.makedirs(args.test_result_dir, exist_ok=True)
    test_files = ".".join([dataset_name_mapping(os.path.splitext(os.path.basename(f))[0]) for f in args.test_dataset_dirs])
    if "mutual_check_intermediate" in test_files:
        saved_file_name =   f"{test_files}-{args.task}-{args.step_correctness_task_mode}"
    else:
        saved_file_name =   f"Tst{test_files}-mutual_check_intermediate-{args.task}-{args.step_correctness_task_mode}"

    load_llm =(args.task in ["mask", "check"])
    load_mllm = (args.task == "fill")
    mutualchecker=MutualChecker(config, load_llm=load_llm, load_mllm=load_mllm, llm_dir=args.llm_dir, mllm_dir=args.mllm_dir)
    
    # load data, train:val = 0.95:0.05
    val_data_before_balance = ReasoningDatasetWhole(args.test_dataset_dirs).all_data
    total_test_num = len(val_data_before_balance)
    val_dataset = val_data_before_balance[:int(total_test_num)]
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    results = []
    with torch.no_grad():
        total = 0
        correct_cnt = 0
        for batch_id, batch in enumerate(tqdm(val_loader, desc=f"Evluation...")):
            if args.task == "mask":
                output_batch = mutualchecker.mask_batch(batch, mode=args.step_correctness_task_mode)
            elif args.task == "fill":
                output_batch = mutualchecker.fill_batch(batch, mode=args.step_correctness_task_mode)
            elif args.task == "check":
                output_batch, mutual_check_correct_cnt_batch, mutual_check_valid_cnt_batch = mutualchecker.check_batch(batch, mode=args.step_correctness_task_mode)
                total += mutual_check_valid_cnt_batch
                correct_cnt += mutual_check_correct_cnt_batch
            else:
                raise NotImplementedError(f"Task {args.task} is not implemented.")

            results.extend(output_batch)
            
        with open(os.path.join(args.intermediate_result_dir, saved_file_name + '.jsonl'), 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')    
        
        if args.task == "check":
            print(f"========== Test acc: {correct_cnt/total:.4f} ===========")
            with open(os.path.join(args.test_result_dir, saved_file_name + '.jsonl'), 'w') as f:
                f.write(f"Test acc: {correct_cnt/total:.4f}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_cfg', type=str, default="config/mutual_check.yaml")
    parser.add_argument('--test_dataset_dirs', type=str, required=True, nargs="+",
                        help="separated by commas", default="")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_result_dir', type=str, default="./multimodal/step_correctness/results/test_results")
    parser.add_argument('--task', type=str, default="check", choices=["mask", "fill", "check"])
    parser.add_argument("--step_correctness_task_mode", type=str, default="consec_step", choices=["single_step", "single_step_no_question", "accum_step", "single_step_mask_second_half", "accum_step_mask_random",  "single_step_mask_random"], help="Masking/filling/checking mode for the reasoning steps.")
    parser.add_argument("--llm_dir", type=str, default="", help="Path to the LLM directory.")
    parser.add_argument("--mllm_dir", type=str, default="", help="Path to the MLLM directory.")
    parser.add_argument('--process_num', type=int, default=-1, help="Number of records to process. Default is -1 for all records.")
    parser.add_argument('--intermediate_result_dir', type=str, default="./multimodal/step_correctness/results/intermediate_results", help="Path to save intermediate results.")
    args = parser.parse_args()
    
    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))

    load_llm = (args.task in ["mask", "check"])
    load_mllm = (args.task == "fill")
    

    config.use_vllm_for_multimodal = (args.task == "fill")
    config.batch_size = args.batch_size
    main(args, config)