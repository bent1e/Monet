# sitecustomize.py
import os, sys, functools, importlib

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
try:
    _wk = os.path.abspath(".")
    _old = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{_wk}:{_old}" if _old else _wk
except Exception:
    pass

_AVT_DEBUG = os.environ.get("AVT_DEBUG", "0") == "1"
_AVT_GLOBAL_LATENTS_ACCUM: dict[str, list] = {}

# 1. Patch CompletionOutput to add latents
def _patch_completion_output():
    for mod_name in ("vllm.outputs", "vllm.v1.outputs"):
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, "CompletionOutput")
            if cls and not hasattr(cls, "latents"):
                # Patch __init__ to accept latents
                orig_init = cls.__init__
                @functools.wraps(orig_init)
                def new_init(self, *args, latents=None, **kwargs):
                    orig_init(self, *args, **kwargs)
                    self.latents = latents
                cls.__init__ = new_init
        except Exception as e:
            if _AVT_DEBUG: print(f"[AVT] Patch CompletionOutput failed: {e}")

_patch_completion_output()

# 2. Patch EngineCore.step to attach latents to EngineCoreOutputs
def _patch_enginecore_step():
    # Disabled: engine core patch is fragile across processes and not needed.
    if _AVT_DEBUG:
        print("[AVT] Skipping EngineCore.step patch (using LLMEngine.step instead)")

_patch_enginecore_step()

# 3. Patch OutputProcessor to propagate latents to CompletionOutput
def _patch_output_processor():
    # Disabled: engine_core_outputs is a list in V1; we inject latents in LLMEngine.step instead.
    if _AVT_DEBUG:
        print("[AVT] Skipping OutputProcessor.process_outputs patch (using LLMEngine.step instead)")

_patch_output_processor()

# 4. Do NOT wrap LLM.generate to avoid recursion and API breakage.
def _note_no_generate_wrap():
    if _AVT_DEBUG:
        print("[AVT] Skipping LLM.generate wrapping; latents will be attached to CompletionOutput objects when available.")

_note_no_generate_wrap()

# 5. Provide a worker-side function to pull last-step latents safely
def _avt_worker_get_latents(worker):  # executed on worker via collective_rpc
    sys.modules['vllm.v1.worker.gpu_model_runner'] = importlib.import_module("avt.vllm.avt_gpu_model_runner")
    try:
        # Try common attribute names first
        candidates = []
        for name in ("gpu_model_runner", "model_runner"):
            if hasattr(worker, name):
                candidates.append(getattr(worker, name))
        # Fallback: scan attributes
        if not candidates:
            for v in worker.__dict__.values():
                try:
                    if hasattr(v, "_avt_last_latents_step"):
                        candidates.append(v)
                except Exception:
                    continue
        for runner in candidates:
            lat = getattr(runner, "_avt_last_latents_step")
            if lat is not None:
                return lat
        return None
    except Exception:
        return None


# 6. Wrap LLM._run_engine to attach latents to finished CompletionOutput
def _patch_llm_run_engine():
    try:
        llm_mod = importlib.import_module("vllm.entrypoints.llm")
        LLM = getattr(llm_mod, "LLM")
        if LLM is None:
            if _AVT_DEBUG:
                print("[AVT] LLM class not found; skip _run_engine patch")
            return

        orig_run_engine = getattr(LLM, "_run_engine")
        if orig_run_engine is None or getattr(orig_run_engine, "_avt_wrapped", False):
            return

        @functools.wraps(orig_run_engine)
        def _run_engine_with_latents(self, *, use_tqdm: bool):
            # This mirrors vllm.entrypoints.llm.LLM._run_engine with latents injection
            from tqdm.auto import tqdm  # local import to match original
            from vllm.outputs import RequestOutput, PoolingRequestOutput

            if use_tqdm:
                num_requests = self.llm_engine.get_num_unfinished_requests()
                pbar = tqdm(
                    total=num_requests,
                    desc="Processed prompts",
                    dynamic_ncols=True,
                    postfix=(f"est. speed input: {0:.2f} toks/s, "
                             f"output: {0:.2f} toks/s"),
                )

            outputs: list[RequestOutput | PoolingRequestOutput] = []
            total_in_toks = 0
            total_out_toks = 0

            while self.llm_engine.has_unfinished_requests():
                step_outputs = self.llm_engine.step()

                # Fetch last-step latents from workers (best-effort)
                latents = None
                try:
                    results = self.collective_rpc(_avt_worker_get_latents)
                    # Pick the first non-None result
                    for r in results:
                        if r is not None:
                            latents = r
                            break
                except Exception as e:
                    if _AVT_DEBUG:
                        print(f"[AVT] collective_rpc get_latents failed: {e}")

                for output in step_outputs:
                    # If this is a RequestOutput, attach latents to each CompletionOutput
                    try:
                        from vllm.outputs import RequestOutput as _RO
                        if isinstance(output, _RO) and getattr(output, "outputs"):
                            for comp in output.outputs:
                                # Attach without changing ctor usage
                                try:
                                    setattr(comp, "latents", latents)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    if output.finished:
                        outputs.append(output)
                        if use_tqdm:
                            # Maintain the same progress semantics
                            try:
                                from vllm.outputs import RequestOutput as _RO2
                                if isinstance(output, _RO2):
                                    n = len(output.outputs)
                                    assert output.prompt_token_ids is not None
                                    total_in_toks += len(output.prompt_token_ids) * n
                                    in_spd = (total_in_toks /
                                              max(1e-6, pbar.format_dict["elapsed"]))
                                    total_out_toks += sum(len(stp.token_ids) for stp in output.outputs)
                                    out_spd = (total_out_toks /
                                               max(1e-6, pbar.format_dict["elapsed"]))
                                    pbar.postfix = (
                                        f"est. speed input: {in_spd:.2f} toks/s, "
                                        f"output: {out_spd:.2f} toks/s")
                                    pbar.update(n)
                                else:
                                    pbar.update(1)
                            except Exception:
                                # If tqdm internals change, don't fail generation
                                pbar.update(1)

            if use_tqdm:
                pbar.close()

            return sorted(outputs, key=lambda x: int(x.request_id))

        _run_engine_with_latents._avt_wrapped = True
        setattr(LLM, "_run_engine", _run_engine_with_latents)
        if _AVT_DEBUG:
            print("[AVT] Patched LLM._run_engine to attach latents")
    except Exception as e:
        if _AVT_DEBUG:
            print(f"[AVT] Failed to patch LLM._run_engine: {e}")


_patch_llm_run_engine()

print("[AVT] vLLM patched via sitecustomize:", __file__)

# 7. Patch v1 LLMEngine.step to attach latents directly on RequestOutputs
def _patch_v1_llmengine_step():
    try:
        mod = importlib.import_module("vllm.v1.engine.llm_engine")
        LLMEngineV1 = getattr(mod, "LLMEngine")
        if LLMEngineV1 is None:
            if _AVT_DEBUG:
                print("[AVT] v1 LLMEngine not found; skip step patch")
            return

        orig_step = getattr(LLMEngineV1, "step")
        if orig_step is None or getattr(orig_step, "_avt_wrapped", False):
            return

        def _avt_find_latents(obj, max_depth=2, seen=None):
            if obj is None or max_depth < 0:
                return None
            if seen is None:
                seen = set()
            try:
                oid = id(obj)
                if oid in seen:
                    return None
                seen.add(oid)
            except Exception:
                pass
            try:
                lat = getattr(obj, "_avt_last_latents_step")
                if lat is not None:
                    return lat
            except Exception:
                pass
            # look into common child attributes only (shallow)
            child_names = (
                "gpu_model_runner", "model_runner", "runner",
                "worker", "workers", "engine_core", "model_executor",
            )
            for name in child_names:
                try:
                    child = getattr(obj, name)
                except Exception:
                    continue
                # handle list/tuple/dict containers
                if isinstance(child, (list, tuple)):
                    for ch in child:
                        lat = _avt_find_latents(ch, max_depth - 1, seen)
                        if lat is not None:
                            return lat
                elif isinstance(child, dict):
                    for ch in child.values():
                        lat = _avt_find_latents(ch, max_depth - 1, seen)
                        if lat is not None:
                            return lat
                else:
                    lat = _avt_find_latents(child, max_depth - 1, seen)
                    if lat is not None:
                        return lat
            return None

        @functools.wraps(orig_step)
        def _step_with_latents(self):
            outputs = orig_step(self)

            # Fetch latents from workers via EngineCore client
            latents = None
            try:
                results = self.collective_rpc(_avt_worker_get_latents)
                for r in results:
                    if r is not None:
                        latents = r
                        break
            except Exception as e:
                if _AVT_DEBUG:
                    print(f"[AVT] v1 LLMEngine.step: collective_rpc failed: {e}")

            # Fallback: try to find latents locally (non-multiprocess)
            if latents is None:
                try:
                    for candidate in (getattr(self, "model_executor"),
                                      getattr(self, "engine_core")):
                        latents = _avt_find_latents(candidate)
                        if latents is not None:
                            break
                except Exception:
                    pass

            # Attach latents to each CompletionOutput in each RequestOutput
            try:
                from vllm.outputs import RequestOutput as _RO
                if isinstance(outputs, list):
                    for ro in outputs:
                        if isinstance(ro, _RO) and getattr(ro, "outputs"):
                            for comp in ro.outputs:
                                try:
                                    setattr(comp, "latents", latents)
                                except Exception:
                                    pass
            except Exception:
                pass

            # Strict diagnostics to help debug missing latents when AVT_DEBUG=1
            if _AVT_DEBUG and latents is None:
                def _has_attr_path(root, path: list[str]):
                    cur = root
                    chain = [type(root).__name__]
                    for p in path:
                        if not hasattr(cur, p):
                            return False, "->".join(chain + [p])
                        cur = getattr(cur, p)
                        chain.append(p + f"({type(cur).__name__})")
                    return True, "->".join(chain)

                checks = []
                for label, obj in ("self", self), ("model_executor", getattr(self, "model_executor")), ("engine_core", getattr(self, "engine_core")):
                    try:
                        if obj is None:
                            checks.append(f"{label}: None")
                            continue
                        ok, chain = _has_attr_path(obj, ["gpu_model_runner", "_avt_last_latents_step"])
                        checks.append(f"{label}.gpu_model_runner path: {'OK' if ok else 'MISS'} | {chain}")
                        ok2, chain2 = _has_attr_path(obj, ["model_runner", "_avt_last_latents_step"])
                        checks.append(f"{label}.model_runner path: {'OK' if ok2 else 'MISS'} | {chain2}")
                    except Exception as e:
                        checks.append(f"{label}: error {e}")

                raise RuntimeError(
                    "[AVT] latents not found. Enable AVT_DEBUG=0 to suppress.\n"
                    + "\n".join(checks)
                )

            return outputs

        _step_with_latents._avt_wrapped = True
        setattr(LLMEngineV1, "step", _step_with_latents)
        if _AVT_DEBUG:
            print("[AVT] Patched v1 LLMEngine.step to attach latents")
    except Exception as e:
        if _AVT_DEBUG:
            print(f"[AVT] Failed to patch v1 LLMEngine.step: {e}")


_patch_v1_llmengine_step()