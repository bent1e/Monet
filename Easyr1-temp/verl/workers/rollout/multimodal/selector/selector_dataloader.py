import json
import os
import random  # 新增
from torch.utils.data import Dataset
from collections import OrderedDict
from tqdm import tqdm


VISUALPRM_IMG_DIR_MAPPING = {
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
    def __init__(self, dataset_dirs):
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
                self.VisualPRM400K_load_data(file_path)
            elif "VisualProcessBench" in file_path:
                self.VisualProcessBench_load_data(file_path)
            elif "rollout" in file_path:
                self.MCTS_load_data(file_path)
            

    def VisualProcessBench_load_data(self, file_path):
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

                for step, annotation in zip(steps, annotations):
                    self.all_data.append({
                        "image": image_path,
                        "question": question,
                        "step": step,
                        "label": annotation
                    })    

    def MCTS_load_data(self, file_path):
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
                    if node_data["step_correctness"] == 0:
                        continue
                    step = node_data.get["text"]
                    correctness = node_data["step_correctness"]
                    
                    # Append the data to all_data
                    self.all_data.append({
                        "image": image_path,
                        "question": question,
                        "step": step,
                        "label": correctness
                    })


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]


class ReasoningDatasetSplit(Dataset):
    """
    给定一份 data_list（对应某些 image），并执行类别平衡（可选）。
    """
    def __init__(self, data_list, balance=True):
        """
        data_list: 已经切好只包含某些 image 的原始数据(字典列表).
        balance: 是否对 label 进行平衡.
        """
        if not balance:
            self.data = data_list
        else:
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