import json
with open("./Tsttest_neat_labels_for_rstar_len10-mutual_check_intermediate-mask-single_step-fill-accum_step-check-single_step.jsonl", "r") as f:
            
    lines = f.readlines()
    for line in lines[:500]:
        data = json.loads(line)
        label = data["label"]
        pred = data["mutual_check_step_correctness"]
        if label == -1 and pred == 1:
            
            print(data["step"])
            print("****")
            print(data["masked_step"])
            print("****")
            print(data["filled_step"])
            print("****")
            
            print(f"gt: {label}, pred: {pred}")
            print("########################################")