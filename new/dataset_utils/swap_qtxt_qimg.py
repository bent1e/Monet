import json
import os
import pdb
DEFAULT_FILES = [
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_arc_agi/raw_train_w_obs_w_metadata.json",
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_checkers/raw_train_w_obs_w_metadata.json",
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_connect_four/raw_train_w_obs_w_metadata.json",
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_rpm/raw_train_w_obs_w_metadata.json",
    "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual/Zebra_CoT_tetris/raw_train_w_obs_w_metadata.json"

]

for path in DEFAULT_FILES:
    cnt = 0
    with open(path, 'r') as f:
        data = json.load(f)

    new_data = []
    for item in data[:]:
        #pdb.set_trace()
        user = item['data'][1]
        user_content = user['content']
        if user_content[0]['type'] != 'image':
            cnt += 1
            new_content = user_content[1:] + user_content[:1]
            user['content'] = new_content
        item['data'][1] = user
        new_data.append(item)
    print(f"{path} swapped {cnt}")

    with open(path.replace(".json", "") + "_swap.json",'w') as f:
        json.dump(data, f, indent=2)