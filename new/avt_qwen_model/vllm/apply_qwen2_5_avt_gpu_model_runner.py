# apply_qwen_patch.py  —— 在 import transformers 之前运行
import importlib.util, sys, pathlib, os

patch_path = pathlib.Path(__file__).with_name("avt_gpu_model_runner.py")

# 1) 动态加载你的完整文件
spec  = importlib.util.spec_from_file_location(
    "vllm.v1.worker.gpu_model_runner",
    patch_path,
)
patched_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patched_mod)

# 2) 替换 sys.modules 中的同名条目
sys.modules["vllm.v1.worker.gpu_model_runner"] = patched_mod

print("Replaced the original Qwen2.5-VL model with the AVT version.")
