import argparse
import PIL.Image
#import new.avt_qwen_model.vllm.apply_qwen2_5_avt_gpu_model_runner
from new.AAA_vllm_toolkit.load_and_gen_vllm import *
from new.AAA_vllm_toolkit.load_and_gen_hf import *
import os
import PIL
from src.task import *


def main():
    parser = argparse.ArgumentParser(description="Quick inference script for AVT model")
    parser.add_argument('--model_path', default='/home/dids/shiyang/checkpoints/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--data_path', nargs='+', required=True, help='Path to the dataset files')
    parser.add_argument('--dataset_root', default='./new')
    parser.add_argument('--num_samples', default=10, type=int)
    args = parser.parse_args()

    mllm, sampling_params = vllm_mllm_init(args.model_path, tp=1, gpu_memory_utilization=0.8)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    # inputs = vllm_mllm_process_single_data("Describe this image in detail.", image_path='/data1/qxwang/codes/Mirage/new/debug_1.jpg', mllm_dir=model_path)
    
    # WARNING: the os.environ['ABS_VIS_START_ID'] must be set by `export ABS_VIS_START_ID=...`, otherwise
    #os.environ['ABS_VIS_START_ID'] = str(processor.tokenizer.encode('<abs_vis_token>')[0])
    #os.environ['ABS_VIS_END_ID'] = str(processor.tokenizer.encode('</abs_vis_token>')[0])

    preprocess_function = avt_single_input_images_preprocess_function_question_only
    all_train_dataset = []
    for data_path in args.data_path:
        if data_path.endswith('.jsonl'):
            train_dataset = load_jsonl_dataset(data_path)
        elif data_path.endswith('.json'):
            train_dataset = load_json_dataset(data_path)
        all_train_dataset.extend(train_dataset[:])
    all_train_dataset = all_train_dataset[:args.num_samples]
    train_dataset = []
    cur_max = -1
    for i, sample in tqdm(enumerate(all_train_dataset), desc="Collecting training data and length check...", total=len(all_train_dataset)):
        processed, cur_max = preprocess_function(sample, dataset_root=args.dataset_root, processor=processor, max_seq_len=6000, cur_max=cur_max, id=i, rank=0)
        if processed is not None:
            train_dataset.append(processed)


    inputs = vllm_mllm_process_batch_from_messages(train_dataset, processor)
    output = vllm_generate(inputs, sampling_params, mllm)
    #print(output[0].outputs[0].text)

    cnt = 0
    for o in output:
        if "</abs_vis_token>" in o.outputs[0].text:
            cnt+=1
            print("###############################################")
    print(cnt)


    conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Question: What's the price of the egg sandwhich? \nPut your final answer within \\boxed{}. If you cannot see relevant visual information to infer the answer from the image, just output \\boxed{None} and don't guess the answer based on your knowledge."},
                    {"type": "image", "image": PIL.Image.open('/home/dids/shiyang/codes/abstract-visual-token/new/vts_1.jpg').convert("RGB")}
                ]
            }
        ]

    inputs = vllm_mllm_process_batch_from_messages([conversation], processor)
    output = vllm_generate(inputs, sampling_params, mllm)
    print(output[0].outputs[0].text)

if __name__ == '__main__':
    # 必须放在主入口，避免 multiprocessing spawn 子进程导入本模块时再次执行推理代码
    main()

'''
In Stage 2, the model further refines its understanding by incorporating more detailed information. It now takes [Question], [Image], [Thoughts], [Latent], and [Answer] as inputs. The model generates embeddings for thoughts and answers, similar to Stage 1. However, in addition to CE and L2 losses, there is a "generate" step which likely involves generating new content or improving existing predictions. The overall goal is to improve the accuracy of generating answers by leveraging both textual and visual information.
'''