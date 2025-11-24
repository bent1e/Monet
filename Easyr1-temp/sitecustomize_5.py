# sitecustomize.py
# Target: vLLM 0.8.5 (old-style OutputProcessor).
# All comments in English.

import os, sys, importlib, functools, typing as T

# ----------------- Required envs -----------------
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

# --------------- Helpers -----------------
# --- helper: clone msgspec.Struct 并追加/改型字段，开启 kw_only ---
def _defstruct_clone_with_extra_field(struct_cls, extra_field: tuple[str, T.Any, T.Any], module_name: str):
    msgspec = importlib.import_module("msgspec")
    anns = getattr(struct_cls, "__annotations__", None)
    if not isinstance(anns, dict):
        raise TypeError(f"{struct_cls} has no __annotations__")

    # 收集现有字段默认值，保持原顺序
    defaults = {name: getattr(struct_cls, name) for name in anns.keys() if hasattr(struct_cls, name)}
    fields = []
    for name, typ in anns.items():
        if name in defaults:
            fields.append((name, typ, defaults[name]))
        else:
            fields.append((name, typ))

    # 总是以“带默认值”的三元组形式追加新字段，避免把它当作必填字段
    fname, ftype, fdefault = extra_field
    fields.append((fname, ftype, fdefault))

    New = msgspec.defstruct(
        struct_cls.__name__,
        fields=fields,
        array_like=True,
        omit_defaults=True,
        gc=False,
        kw_only=True,          # 关键：放宽必填/可选字段顺序约束
        module=module_name,
    )
    return New


def _defstruct_clone_retype_field(struct_cls, field_name: str, new_type: T.Any, module_name: str):
    msgspec = importlib.import_module("msgspec")
    anns = getattr(struct_cls, "__annotations__", None)
    if not isinstance(anns, dict):
        raise TypeError(f"{struct_cls} has no __annotations__")

    defaults = {name: getattr(struct_cls, name) for name in anns.keys() if hasattr(struct_cls, name)}
    fields = []
    for name, typ in anns.items():
        typ2 = new_type if name == field_name else typ
        if name in defaults:
            fields.append((name, typ2, defaults[name]))
        else:
            fields.append((name, typ2))

    New = msgspec.defstruct(
        struct_cls.__name__,
        fields=fields,
        array_like=True,
        omit_defaults=True,
        gc=False,
        kw_only=True,
        module=module_name,
    )
    return New


# --- 0) 仍然按你的方式给 EngineCoreOutput 增加 latents，但使用上面的 helper ---
def patch_engine_core_output_add_latents():
    eng_mod = importlib.import_module("vllm.v1.engine")
    Old = eng_mod.EngineCoreOutput
    if "latents" in getattr(Old, "__annotations__", {}):
        _dprint("[AVT] EngineCoreOutput already has 'latents'")
        return
    New = _defstruct_clone_with_extra_field(
        Old,
        ("latents", T.Optional[list[float]], None),   # 新字段显式默认 None
        eng_mod.__name__,
    )
    setattr(eng_mod, "EngineCoreOutput", New)
    _dprint("[AVT] EngineCoreOutput patched via msgspec.defstruct (with 'latents')")


# --- 0.5) 重新绑定 serial_utils 的编解码器，并兜底 enc_hook ---
def _refresh_serial_utils_codecs():
    su = importlib.import_module("vllm.v1.serial_utils")
    eng = importlib.import_module("vllm.v1.engine")
    msgspec = importlib.import_module("msgspec")

    # 包一层 enc_hook：遇到 EngineCoreOutputs/EngineCoreOutput 时，转换为内置结构，避免走 pickle
    if not hasattr(su, "_avt_orig_enc_hook"):
        su._avt_orig_enc_hook = su.enc_hook
        def _avt_enc_hook(obj):
            try:
                if isinstance(obj, eng.EngineCoreOutputs) or isinstance(obj, eng.EngineCoreOutput):
                    # 将 msgspec.Struct 转成纯内置类型，保证可编码
                    return msgspec.to_builtins(obj)
            except Exception:
                pass
            return su._avt_orig_enc_hook(obj)
        su.enc_hook = _avt_enc_hook
        _dprint("[AVT] serial_utils.enc_hook wrapped")

    # 重新构造模块里所有“持有 msgspec.Encoder/Decoder”的对象
    enc_cnt = dec_cnt = 0
    for name, obj in su.__dict__.items():
        try:
            enc = getattr(obj, "encoder", None)
            if isinstance(enc, msgspec.Encoder):
                setattr(obj, "encoder", msgspec.Encoder(type=eng.EngineCoreOutputs, enc_hook=su.enc_hook))
                enc_cnt += 1
            dec = getattr(obj, "decoder", None)
            if isinstance(dec, msgspec.Decoder):
                setattr(obj, "decoder", msgspec.Decoder(type=eng.EngineCoreOutputs))
                dec_cnt += 1
        except Exception:
            pass
    _dprint(f"[AVT] serial_utils re-bound encoders={enc_cnt}, decoders={dec_cnt}")
# --------------- (1) Patch ModelRunnerOutput to accept latents= ---------------
def patch_model_runner_output_init() -> None:
    mod = importlib.import_module("vllm.v1.outputs")
    MRO = mod.ModelRunnerOutput

    if getattr(MRO, "_avt_patched_latents", False):
        return

    try:
        anns = getattr(MRO, "__annotations__", None)
        if isinstance(anns, dict) and "latents" not in anns:
            anns["latents"] = T.Optional[list[T.Optional[list[float]]]]
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

# --------------- (2) Wrap Scheduler.update_from_output -----------------------
def wrap_scheduler_update_from_output() -> None:
    sched  = importlib.import_module("vllm.v1.core.sched.scheduler")
    if getattr(sched.Scheduler.update_from_output, "_avt_wrapped", False):
        return

    orig = sched.Scheduler.update_from_output
    from msgspec.structs import replace as _replace
    eng = importlib.import_module("vllm.v1.engine")

    @functools.wraps(orig)
    def wrapped(self, scheduler_output, model_runner_output):
        # Ensure scheduler module references patched classes
        setattr(sched, "EngineCoreOutput",  getattr(eng, "EngineCoreOutput"))
        setattr(sched, "EngineCoreOutputs", getattr(eng, "EngineCoreOutputs"))

        res = orig(self, scheduler_output, model_runner_output)

        outs         = getattr(res, "outputs", None)
        req2idx      = getattr(model_runner_output, "req_id_to_index", None)
        latents_step = getattr(model_runner_output, "latents", None)

        if not isinstance(outs, list) or not isinstance(req2idx, dict):
            return res
        if latents_step is None:
            return res
        if not isinstance(latents_step, list):
            _dprint("[AVT sched] model_runner_output.latents is not list -> skip")
            return res

        n_set = 0
        for i, eco in enumerate(outs):
            rid = getattr(eco, "request_id", None)
            if rid is None or rid not in req2idx:
                continue
            idx = req2idx[rid]
            if idx >= len(latents_step):
                continue
            vec = latents_step[idx]
            if vec is None:
                continue
            try:
                # If mutable, direct set works
                eco.latents = vec
            except Exception:
                outs[i] = _replace(eco, latents=vec)
            n_set += 1
        _dprint(f"[AVT sched] attached latents to {n_set}/{len(outs)} EngineCoreOutput(s)")
        return res

    wrapped._avt_wrapped = True
    sched.Scheduler.update_from_output = wrapped
    _dprint("[AVT] wrapped Scheduler.update_from_output")

# --------------- (3) Patch CompletionOutput to add .latents ------------------
def patch_completion_output_class() -> None:
    out_mod = importlib.import_module("vllm.outputs")
    msgspec = importlib.import_module("msgspec")

    CO = out_mod.CompletionOutput
    if "latents" in getattr(CO, "__annotations__", {}):
        return

    # For msgspec.Struct variant, rebuild; otherwise, just setattr.
    if isinstance(CO, type) and issubclass(CO, msgspec.Struct):
        anns     = dict(CO.__annotations__)
        defaults = {k: getattr(CO, k) for k in anns if hasattr(CO, k)}

        fields: list[tuple] = []
        for n, t in anns.items():
            if n in defaults:
                fields.append((n, t, defaults[n]))
            else:
                fields.append((n, t))
        fields.append(("latents", T.Optional[list[list[float]]], None))

        NewCO = msgspec.defstruct(
            "CompletionOutput",
            fields=fields,
            array_like=True,
            omit_defaults=True,
            gc=False,
            module=out_mod.__name__,
        )
        setattr(out_mod, "CompletionOutput", NewCO)
        _dprint("[AVT] patched vllm.outputs.CompletionOutput.latents (Struct)")
    else:
        setattr(CO, "latents", None)
        _dprint("[AVT] CompletionOutput setattr(latents) (fallback)")

# --------------- (4) Wrap OutputProcessor.process_outputs --------------------
def wrap_output_processor_process_outputs() -> None:
    op_mod = importlib.import_module("vllm.v1.engine.output_processor")
    if getattr(op_mod.OutputProcessor.process_outputs, "_avt_wrapped", False):
        return

    orig = op_mod.OutputProcessor.process_outputs

    @functools.wraps(orig)
    def wrapped(self, engine_core_outputs, engine_core_timestamp=None, iteration_stats=None):
        # PRE: accumulate per-step latents
        if not hasattr(self, "_avt_latents_accum"):
            self._avt_latents_accum: dict[str, list[list[float]]] = {}

        pre = 0
        if isinstance(engine_core_outputs, list):
            for eco in engine_core_outputs:
                rid = getattr(eco, "request_id")
                vec = getattr(eco, "latents")
                if rid is None or not vec:
                    continue
                self._avt_latents_accum.setdefault(rid, []).append(vec)
                pre += 1
        _dprint(f"[AVT op] pre-accumulated {pre} step-latents")

        out = orig(self, engine_core_outputs, engine_core_timestamp, iteration_stats)

        # POST: attach to finished RequestOutput(s)
        post = 0
        try:
            req_outputs = getattr(out, "request_outputs", None)
            if isinstance(req_outputs, list):
                for ro in req_outputs:
                    rid = getattr(ro, "request_id", None)
                    finished = getattr(ro, "finished", False)
                    if not finished or rid is None:
                        continue
                    traj = self._avt_latents_accum.pop(rid, None)
                    if not traj:
                        continue
                    outs = getattr(ro, "outputs", None)
                    if isinstance(outs, list) and outs:
                        try:
                            outs[-1].latents = traj
                        except Exception:
                            pass
                        post += 1
        except Exception:
            pass

        _dprint(f"[AVT op] post-attached latents to {post} finished RequestOutput(s)")
        return out

    wrapped._avt_wrapped = True
    op_mod.OutputProcessor.process_outputs = wrapped
    _dprint("[AVT] wrapped OutputProcessor.process_outputs")

# --------------- (5) Map custom gpu_model_runner -----------------------------
def map_gpu_model_runner() -> None:
    sys.modules['vllm.v1.worker.gpu_model_runner'] = importlib.import_module(
        "avt.vllm.avt_gpu_model_runner"
    )
    _dprint("[AVT] mapped gpu_model_runner -> avt.vllm.avt_gpu_model_runner")

# --------------- Apply patches (ORDER MATTERS) -------------------------------
patch_engine_core_output_add_latents()          # 先改 schema
patch_model_runner_output_init()                # 让 MRO 支持 latents
map_gpu_model_runner()                          # 你的自定义 runner
patch_completion_output_class()                 # 给 CompletionOutput 加落地字段
wrap_scheduler_update_from_output()             # 从 MRO.latents -> EngineCoreOutput.latents
_refresh_serial_utils_codecs()                  # 关键：重绑 serial_utils 编解码器 + hook 兜底
wrap_output_processor_process_outputs()         # 汇总并挂到最终 outputs

print("[AVT] vLLM patched via sitecustomize (EngineCoreOutput+EngineCoreOutputs+latents):", __file__)
