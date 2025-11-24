# sitecustomize.py
# Runs in every Python process (parent + spawned workers)
import os, sys, importlib
import traceback
from functools import wraps
try:
    from msgspec.structs import replace as struct_replace  # type: ignore[import-not-found]
except Exception as e:
    struct_replace = None
    # Send diagnostic to stderr only; avoid polluting stdout of subprocess callers (e.g., wandb)
    print(f"[AVT sitecustomize] Optional dependency 'msgspec' not available: {e}", file=sys.stderr, flush=True)
import os, sys, importlib, functools, typing as T

# ---- env you requested ----

os.environ["VLLM_USE_V1"] = "1"  # force V1 engine if desired
os.environ["VLLM_NO_USAGE_STATS"] = "1"  # disable usage stats
workspace = os.path.abspath(".")
old_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
os.environ["ABS_VIS_START_ID"] = "151666"
os.environ["ABS_VIS_END_ID"] = "151667"

os.environ["AVT_LATENT_HOOK_BIN"] = "1"
# ---------------------------
AVT_DEBUG = int(os.environ.get("AVT_DEBUG", "0"))

def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

def _dprint(msg: str) -> None:
    if AVT_DEBUG:
        _eprint(msg)

# Optional entry log (stderr only) when debugging is enabled
if AVT_DEBUG:
    _eprint("Entering sitecustomize.py")


def patch_model_runner_output_init() -> None:
    mod = importlib.import_module("vllm.v1.outputs")
    MRO = mod.ModelRunnerOutput

    if getattr(MRO, "_avt_patched_latents", False):
        _dprint("[AVT] ModelRunnerOutput already patched for latents")
        return

    # Add typing annotation for clarity (best-effort).
    try:
        anns = getattr(MRO, "__annotations__", None)
        if isinstance(anns, dict) and "latents" not in anns:
            anns["latents"] = T.Optional[list[list[float]]]
    except Exception:
        pass

    if not hasattr(MRO, "latents"):
        setattr(MRO, "latents", None)

    orig_init = getattr(MRO, "__init__", None)
    if not callable(orig_init):
        raise TypeError("ModelRunnerOutput.__init__ is not callable")

    @functools.wraps(orig_init)
    def _avt_init(self, *args, **kwargs):
        latents = kwargs.pop("latents", None)
        orig_init(self, *args, **kwargs)
        setattr(self, "latents", latents)

    setattr(MRO, "__init__", _avt_init)
    setattr(MRO, "_avt_patched_latents", True)
    _dprint("[AVT] patched ModelRunnerOutput.__init__ to accept latents=...")



#patch_model_runner_output_init()
try:
    sys.modules['vllm.v1.worker.gpu_model_runner'] = importlib.import_module("avt.vllm.avt_gpu_model_runner")
    sys.modules['transformers.models.qwen2_5_vl.modeling_qwen2_5_vl'] = importlib.import_module("avt.transformers.avt_modeling_qwen2_5_vl")
    _eprint("[Easyr1 AVT vllm init] Replaced original vllm.v1.worker.gpu_model_runner with avt.vllm.avt_gpu_model_runner")
except Exception as e:
    _eprint(f"[AVT vllm init] Override failed: {e}")
    traceback.print_exc()