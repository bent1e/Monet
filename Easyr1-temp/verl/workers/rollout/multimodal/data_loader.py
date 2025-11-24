from datasets import load_dataset
from datasets import get_dataset_split_names
import os
def multimodal_data_loader(dataset_path, cache_dir=None):
    #print(get_dataset_split_names(dataset_path, cache_dir=cache_dir))
    if "mathvista" in dataset_path.lower():
        '''dataset = load_dataset(
            "parquet",
            data_files={"testmini": os.path.join(dataset_path, "data/testmini-00000-of-00001-725687bf7a18d64b.parquet")},
            cache_dir=cache_dir
        )'''
        dataset = load_dataset(dataset_path, cache_dir=cache_dir)
        dataset = dataset["testmini"].to_pandas().to_dict(orient="records")# 1000 samples for "testmini"
    if "mathverse" in dataset_path.lower():
        dataset = load_dataset(
            "parquet",
            data_files={"testmini": os.path.join(dataset_path, "testmini.parquet")},
            cache_dir=cache_dir
        )
        dataset = dataset["testmini"].to_pandas().to_dict(orient="records")# 1000 samples for "testmini"
        for data in dataset:
            img_path = data["image"]["path"]
            data["image"] = f"images/{img_path}"
    return dataset