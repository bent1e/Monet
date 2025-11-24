# sitecustomize.py
# All comments in English. Target: vLLM 0.8.5 (old-style OutputProcessor).
# We modify the IPC schema by redefining EngineCoreOutput to include `latents`.

import os, sys, importlib, functools, typing as T
from collections import OrderedDict

# --- Required environment per your request ---
os.environ["VLLM_USE_V1"] = "1"  # force V1 engine if desired
os.environ["VLLM_NO_USAGE_STATS"] = "1"  # disable usage stats
workspace = os.path.abspath(".")
old_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
os.environ["ABS_VIS_START_ID"] = "151666"
os.environ["ABS_VIS_END_ID"] = "151667"
os.environ["ABS_VIS_LATENT_SIZE"] = "10"

AVT_DEBUG = os.environ.get("AVT_DEBUG", "0") == "1"
def _dprint(msg: str) -> None:
    if AVT_DEBUG:
        print(msg, flush=True)

# ---------------------------------------------------------------------
# 0) Redefine vllm.v1.engine.EngineCoreOutput schema with an extra field:
#    latents: Optional[List[float]] = None
#    We derive from the original to preserve all existing fields.
# ---------------------------------------------------------------------
def patch_engine_core_output_add_latents() -> None:
    """Rebuild vllm.v1.engine.EngineCoreOutput using msgspec.defstruct
    so that the constructor accepts the same keyword args as before,
    plus a new field: latents: Optional[List[float]] = None.
    """
    eng_mod = importlib.import_module("vllm.v1.engine")
    msgspec = importlib.import_module("msgspec")

    Old = eng_mod.EngineCoreOutput
    # Idempotent: if already has 'latents', skip
    if "latents" in getattr(Old, "__annotations__", {}):
        _dprint("[AVT] EngineCoreOutput already has 'latents'")
        return

    old_ann = getattr(Old, "__annotations__", None)
    if not isinstance(old_ann, dict):
        raise TypeError("EngineCoreOutput has no __annotations__; cannot patch safely")

    # Build fields list in the original order, with defaults preserved
    fields = []
    for name in old_ann.keys():
        typ = old_ann[name]
        if hasattr(Old, name):
            fields.append((name, typ, getattr(Old, name)))
        else:
            fields.append((name, typ))

    # Append our new field at the end
    fields.append(("latents", T.Optional[list[float]], None))

    # Recreate the struct with the same flags; this registers fields at class creation.
    New = msgspec.defstruct(
        "EngineCoreOutput",
        fields=fields,
        array_like=True,
        omit_defaults=True,
        gc=False,
        module=eng_mod.__name__,   # keep import path stable
    )

    # Swap the symbol in vllm.v1.engine to the new class
    setattr(eng_mod, "EngineCoreOutput", New)
    _dprint("[AVT] EngineCoreOutput patched via msgspec.defstruct (with 'latents')")


# ---------------------------------------------------------------------
# 1) Patch vllm.v1.outputs.ModelRunnerOutput to accept/hold `latents`.
#    This class is not a msgspec.Struct in your build; wrap __init__.
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 2) Wrap Scheduler.update_from_output:
#    - Call original to get EngineCoreOutputs.
#    - Replace each EngineCoreOutput with the same object but latents=vec.
#    We use msgspec.structs.replace to set the new field immutably.
# ---------------------------------------------------------------------
def wrap_scheduler_update_from_output() -> None:
    sched_mod  = importlib.import_module("vllm.v1.core.sched.scheduler")
    Scheduler  = sched_mod.Scheduler
    orig       = Scheduler.update_from_output
    if getattr(orig, "_avt_wrapped", False):
        _dprint("[AVT] Scheduler.update_from_output already wrapped")
        return

    from msgspec.structs import replace as _replace
    eng_mod = importlib.import_module("vllm.v1.engine")

    @functools.wraps(orig)
    def wrapped(self, scheduler_output, model_runner_output):
        # CRITICAL: rebind the scheduler module's cached reference so it uses the patched class
        setattr(sched_mod, "EngineCoreOutput", getattr(eng_mod, "EngineCoreOutput"))

        res = orig(self, scheduler_output, model_runner_output)

        outs            = res.outputs                      # list[EngineCoreOutput]
        if not outs:
            _dprint("[AVT] The vllm.v1.core.sched.scheduler.Scheduler.update_from_output has no new_token_ids")
            return res
        req_id_to_index = model_runner_output.req_id_to_index  # dict[str,int]
        latents_step    = getattr(model_runner_output, "latents")  # fail-fast if missing

        if not isinstance(outs, list):
            raise TypeError("EngineCoreOutputs.outputs is not a list")
        if not isinstance(req_id_to_index, dict):
            raise TypeError("model_runner_output.req_id_to_index is not a dict")
        if not isinstance(latents_step, list):
            raise TypeError("model_runner_output.latents must be a list")

        replaced = 0

        for i, out in enumerate(outs):
            rid = out.request_id
            idx = req_id_to_index[rid]     # ensure mapping exists
            raw = latents_step[idx]        # List[float] or []
            if raw is None or (isinstance(raw, list) and len(raw) == 0):
                continue
            #if not isinstance(raw, list):
            #    raise TypeError("Each per-req latent must be List[float] (or empty list)")

            # At this point 'out' should be of the patched class with 'latents' field.
            outs[i] = _replace(out, latents=raw)
            replaced += 1

        _dprint(f"[AVT sched] set latents on {replaced}/{len(outs)} EngineCoreOutput(s)")
        
        return res

    wrapped._avt_wrapped = True
    Scheduler.update_from_output = wrapped
    _dprint("[AVT] wrapped Scheduler.update_from_output (rebound EngineCoreOutput)")


# ---------------------------------------------------------------------
# 3) Patch vllm.outputs.CompletionOutput to add: latents: Optional[List[List[float]]]
#    This is driver-side only; does not cross processes.
# ---------------------------------------------------------------------
def patch_completion_output_class() -> None:
    out_mod = importlib.import_module("vllm.outputs")
    msgspec = importlib.import_module("msgspec")

    CO = out_mod.CompletionOutput
    if "latents" in getattr(CO, "__annotations__", {}):
        _dprint("[AVT] CompletionOutput already has 'latents'")
        return

    if not (isinstance(CO, type) and issubclass(CO, msgspec.Struct)):
        # Fallback for non-Struct variants (unlikely on v1)
        setattr(CO, "latents", None)
        _dprint("[AVT] CompletionOutput setattr(latents) (non-Struct fallback)")
        return

    anns     = dict(CO.__annotations__)
    defaults = {k: getattr(CO, k) for k in anns.keys() if hasattr(CO, k)}

    fields: list[tuple[str, T.Any, T.Any]] = []
    for n, t in anns.items():
        if n in defaults:
            fields.append((n, t, defaults[n]))
        else:
            fields.append((n, t))
    fields.append(("latents", T.Optional[list[list[float]]], None))

    class _CompletionOutputNew(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):  # type: ignore[misc]
        pass

    __ann: dict[str, T.Any] = {}
    for f in fields:
        if len(f) == 3:
            n, t, d = f
            __ann[n] = t
            setattr(_CompletionOutputNew, n, d)
        else:
            n, t = f
            __ann[n] = t

    _CompletionOutputNew.__annotations__ = __ann  # type: ignore[attr-defined]
    _CompletionOutputNew.__module__      = out_mod.__name__
    setattr(out_mod, "CompletionOutput", _CompletionOutputNew)
    _dprint("[AVT] patched vllm.outputs.CompletionOutput.latents")


# ---------------------------------------------------------------------
# 4) Wrap OutputProcessor.process_outputs (old-style):
#    - PRE: collect per-step latents from EngineCoreOutput.latents.
#    - POST: when a request finishes, attach full trajectory to each
#            CompletionOutput.latents in that RequestOutput.
# ---------------------------------------------------------------------
def wrap_output_processor_process_outputs() -> None:
    op_mod = importlib.import_module("vllm.v1.engine.output_processor")
    OutputProcessor = op_mod.OutputProcessor
    orig = OutputProcessor.process_outputs
    if getattr(orig, "_avt_wrapped", False):
        _dprint("[AVT] OutputProcessor.process_outputs already wrapped")
        return

    from vllm.outputs import RequestOutput  # type check

    eng_mod = importlib.import_module("vllm.v1.engine")
    @functools.wraps(orig)
    def wrapped(self, engine_core_outputs, engine_core_timestamp=None, iteration_stats=None):
        # PRE: accumulate per-step latents
        setattr(op_mod, "EngineCoreOutput", getattr(eng_mod, "EngineCoreOutput"))

        if not hasattr(self, "_avt_latents_accum"):
            self._avt_latents_accum: dict[str, list[list[float]]] = {}

        if not isinstance(engine_core_outputs, list):
            raise TypeError("engine_core_outputs must be a list[EngineCoreOutput]")

        pre_cnt = 0
        for eco in engine_core_outputs:
            rid = eco.request_id  # fail fast
            vec = getattr(eco, "latents")
            if vec is None:
                continue
            if not isinstance(vec, list):
                raise TypeError("EngineCoreOutput.latents must be list[float]")
            self._avt_latents_accum.setdefault(rid, []).append(vec)
            pre_cnt += 1

        _dprint(f"[AVT op] pre-accumulated {pre_cnt} step-latents")

        # RUN official implementation
        out = orig(self, engine_core_outputs, engine_core_timestamp, iteration_stats)

        # POST: attach full trajectory to finished RequestOutputs
        req_outputs = out.request_outputs  # list[RequestOutput]
        post_cnt = 0
        for ro in req_outputs:
            if not isinstance(ro, RequestOutput):
                raise TypeError("Unknown request output type (expected RequestOutput)")
            rid = ro.request_id
            if ro.finished:
                traj = self._avt_latents_accum.pop(rid, None)  # List[List[float]] or None
                if traj is None:
                    continue
                for co in ro.outputs:  # CompletionOutput entries
                    co.latents = traj
                post_cnt += 1

        _dprint(f"[AVT op] post-attached latents to {post_cnt} finished RequestOutput(s)")
        if post_cnt > 0:
            pass
        return out

    wrapped._avt_wrapped = True
    OutputProcessor.process_outputs = wrapped
    _dprint("[AVT] wrapped OutputProcessor.process_outputs")


# ---------------------------------------------------------------------
# 5) Ensure your custom gpu_model_runner is used (so you can set MRO.latents).
# ---------------------------------------------------------------------
def map_gpu_model_runner() -> None:
    sys.modules['vllm.v1.worker.gpu_model_runner'] = importlib.import_module(
        "avt.vllm.avt_gpu_model_runner"
    )
    _dprint("[AVT] mapped gpu_model_runner -> avt.vllm.avt_gpu_model_runner")


# ---------------------------------------------------------------------
# Apply patches in correct order at import time.
# IMPORTANT: sitecustomize runs before vLLM imports in each process.
# ---------------------------------------------------------------------
patch_engine_core_output_add_latents()
patch_model_runner_output_init()
map_gpu_model_runner()
patch_completion_output_class()
wrap_scheduler_update_from_output()
wrap_output_processor_process_outputs()


print("[AVT] vLLM patched via sitecustomize (EngineCoreOutput+latents):", __file__)
