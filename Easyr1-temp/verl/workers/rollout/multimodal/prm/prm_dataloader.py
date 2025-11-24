import json
import os
import random  # 新增
from torch.utils.data import Dataset
from collections import OrderedDict
from tqdm import tqdm

def dataset_name_mapping(dataset_name):
    if dataset_name in DATASET_NAME_MAPPING:
        return DATASET_NAME_MAPPING[dataset_name]
    else:
        if "mcts" in dataset_name:
            dataset_name = dataset_name.split("mcts_")[0][len("rollout00"):]
        return dataset_name
    
DATASET_NAME_MAPPING = {
    "geo170k_extracted_full_prm": "Geo170K",
    "geometry3k_en_20240402_extracted_prm": "Geometry3K",
    "geometry3k_en_20240402_extracted_open_ended_only_prm": "Geometry3K",
    "geomverse_extracted_prm":"GeomVerse",
    "MathV360K_prompts_prm":"MathV360K"
    
}
def split_train_test_by_image(all_data, train_ratio=0.8):
    """
    给定完整数据，先取出所有 image 去重，然后随机切分为 train/test 两部分。
    之后返回 (train_data, test_data) 两部分。
    注意：这里没有进行类别平衡
    """

    # 1) 收集去重后的所有图片
    unique_images = list(OrderedDict.fromkeys(sample["image"] for sample in all_data))

    # 2) 随机打乱
    random.shuffle(unique_images)

    # 3) 划分
    train_size = int(len(unique_images) * train_ratio)
    train_image_set = set(unique_images[:train_size])
    test_image_set = set(unique_images[train_size:])

    # 4) 筛选对应数据
    train_data = [d for d in all_data if d["image"] in train_image_set]
    test_data = [d for d in all_data if d["image"] in test_image_set]

    return train_data, test_data




class ReasoningDatasetWhole(Dataset):
    """
    用来把全部数据都先读进来保存到 self.all_data 中，
    不做任何划分/平衡的操作。
    """
    def __init__(self, dataset_dirs, label_type="step_correctness"):
        """
        dataset_dirs: list[str], 每个元素是一个 jsonl 文件路径。
        """
        self.all_data = []
        files = [f.strip() for f in dataset_dirs]
        for file_path in files:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist, skipping.")
                continue 
            if "VisualPRM400K" in file_path:
                self.VisualPRM400K_load_data(file_path, label_type)
            elif "VisualPRM400K-v1.1" in file_path: # soft score, label is the frequency of leading to correct final answer
                self.VisualPRM400K_v1_1_load_data(file_path, label_type)
            elif "VisualProcessBench" in file_path:
                self.VisualProcessBench_load_data(file_path, label_type)
            elif "mcts" in file_path:
                self.MCTS_load_data(file_path, label_type)
            elif "mutual_check_intermediate" in file_path:
                self.eval_mutual_checker_intermediate_load_data(file_path)
            else:
                raise NotImplementedError(f"Dataset {file_path} not supported.")
                
                
    def VisualPRM400K_load_data(self, file_path, label_type="step_correctness"):
        json_name = os.path.splitext(os.path.basename(file_path))[0]
        img_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'images')
        image_folder_name = dataset_name_mapping(json_name)
        img_dataset_dir = os.path.join(img_dataset_dir, image_folder_name)     
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                

                image_path = os.path.join(img_dataset_dir, sample.get("image", ""))
                convs = sample.get("conversations", [])
                # 去除 system prompt
                filtered = [c for c in convs if c.get("from") in {"human", "gpt"}]
                human_msgs = [c for c in filtered if c.get("from") == "human"]
                gpt_msgs = [c for c in filtered if c.get("from") == "gpt"]

                # a question + at least one mid step + a final answer >=3
                if len(human_msgs) < 3 or len(gpt_msgs) < 3:
                    continue
                question_str = human_msgs[0].get("value", "").strip()
                question_start = question_str.find("\n\nQuestion:")
                question_end = question_str.find("### Solution Process:")
                question = question_str[question_start:question_end]
                # 去掉第一条问题和最后一条答案，只保留中间的 step
                steps = [msg.get("value", "").strip() for msg in human_msgs[1:-1]]
                annotations = [
                    1 if msg.get("value", "").strip() == '+'
                    else 0
                    for msg in gpt_msgs[1:-1]
                ]

                if len(steps) != len(annotations):
                    continue

                for step, annotation in zip(steps, annotations):
                    self.all_data.append({
                        "image": image_path,
                        "question": question.strip(),
                        "step": step.strip(),
                        "label": annotation
                    })  
                    
                    
    def VisualPRM400K_v1_1_load_data(self, file_path, label_type="step_correctness"):
        img_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'images')
 
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                
                img_file_name = '/'.join(sample.get("image", "").split('/')[1:])
                image_path = os.path.join(img_dataset_dir, img_file_name)
                steps_with_score = sample["steps_with_score"]

                steps = [s["step"] for s in steps_with_score[:-1]]
                scores = [s["score"] for s in steps_with_score][:-1]
                question = sample["question_orig"].strip()

                for step, score in zip(steps, scores):
                    self.all_data.append({
                        "image": image_path,
                        "question": question.strip(),
                        "step": step.strip(),
                        "label": score
                    })                  
                                

    def VisualProcessBench_load_data(self, file_path, label_type="step_correctness"):
        img_dataset_dir = os.path.dirname(file_path)     
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue


                image_path = os.path.join(img_dataset_dir, sample.get("image", ""))
                steps = [step for step in sample["response"]["steps"]]
                annotations = [annotation for annotation in sample["response"]["process_correctness"]]
                question = sample["question"]

                if len(steps) != len(annotations):
                    continue

                accum_steps = ""
                for i, (step, annotation) in enumerate(zip(steps, annotations)):
                    if label_type == "step_correctness":
                        if annotation == 0:
                           continue
                    if label_type == "lead_to_correctness":
                        if annotation == -1:
                            continue
                    step = step.strip()
                    if not step.endswith(".") and not step.endswith("?") and not step.endswith("!") and not step.endswith(":"):
                        step += "."
                        
                    self.all_data.append({
                        "image": image_path,
                        "question": question.strip(),
                        "step": step,
                        "previous_steps": accum_steps,
                        "label": annotation
                    })    
                    delim = " "
                    if step.startswith("Step"):
                        delim = "\n"
                    if step[0].isdigit():
                        delim = "\n"
                    accum_steps += delim + step

    def MCTS_load_data(self, file_path, label_type="step_correctness"):
        '''
        Load data from MCTS structured JSONL file.
        file_path: str, path to the JSONL file.
        '''
        img_dataset_dir = os.path.dirname(os.path.dirname(file_path))     
        # 1. Read the JSONL file.
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc="Processing each JSON object"):
            line = line.strip()
            if not line:
                continue  # Skip empty lines if necessary

            # Parse the JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Handle parsing errors as needed; here we skip the line
                continue

            # If 'rstar' exists and is a dictionary, proceed with traversal
            if "rstar" in data and isinstance(data["rstar"], dict):
                rstar_dict = data["rstar"]
                question = data["question"]
                image_path = os.path.join(img_dataset_dir, data["image"])
                
                # 3. Traverse each sub-node within 'rstar'
                for node_tag, node_data in rstar_dict.items():
                    # Extract the step and correctness from the node data
                    step = node_data.get["text"]
                    if node_tag != "0":
                        parent_tag = node_tag[:-2]
                        previous_step = rstar_dict[parent_tag]["text"]
                    else:
                        previous_step = ""
                    
                    label = node_data[label_type]
                    
                    # Append the data to all_data
                    self.all_data.append({
                        "image": image_path,
                        "question": question.strip(),
                        "step": step.strip(),
                        "previous_step": previous_step,
                        "label": label
                    })

    def eval_mutual_checker_intermediate_load_data(self, file_path):
         with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                self.all_data.append(sample)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]


class BalancedDataset(Dataset):
    """
    Given a data_list, balance the positive and negative samples (optional).
    """
    def __init__(self, data_list, label_type="step_correctness"):

        neg_label = -1 if label_type == "lead_to_correctness" else 0 
        data_0 = [item for item in data_list if item["label"] == 0]
        data_1 = [item for item in data_list if item["label"] == 1]

        len0, len1 = len(data_0), len(data_1)
        # 如果存在某一类为0，就没法平衡，看自己需求怎么处理
        if len0 == 0 or len1 == 0:
            print("Warning: One of the classes is zero, can't balance.")
            self.data = data_list
        else:
            # 下采样多数类
            if len0 > len1:
                data_0 = random.sample(data_0, len1)
            elif len1 > len0:
                data_1 = random.sample(data_1, len0)
            # 合并并随机打乱
            self.data = data_0 + data_1
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]