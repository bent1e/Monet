from vllm import LLM
import time
model=LLM(model="/home/dids/shiyang/checkpoints/Qwen3-Embedding-8B", task="embed", gpu_memory_utilization=0.2)
texts = ["Then, draw a line segment from the leftmost side of the bush-troop.", "Output: <observation>It can be known that the length of the \"bush-troop\" is 1."]
res = model.embed(texts, use_tqdm=False)
print(res)
time.sleep(20)