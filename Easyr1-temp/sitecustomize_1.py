# sitecustomize.py (top-level)
# Runs in every Python process (parent + spawned workers)

import os, sys, importlib
import sys
import importlib
import typing as T
from msgspec.structs import replace as struct_replace
import functools

os.environ["VLLM_USE_V1"] = "1"  # force V1 engine if desired
os.environ["VLLM_NO_USAGE_STATS"] = "1"  # disable usage stats
workspace = os.path.abspath(".")
old_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
os.environ["ABS_VIS_START_ID"] = "151666"
os.environ["ABS_VIS_END_ID"] = "151667"
os.environ["ABS_VIS_LATENT_SIZE"] = "10"

def _patch_engine_core_output():
    mod = importlib.import_module("vllm.v1.engine")
    msgspec = importlib.import_module("msgspec")
    torch = importlib.import_module("torch")

    # Pull referenced types from the original module
    LogprobsLists = getattr(mod, "LogprobsLists")
    LogprobsTensors = getattr(mod, "LogprobsTensors")
    FinishReason = getattr(mod, "FinishReason")
    EngineCoreEvent = getattr(mod, "EngineCoreEvent")

    Old = getattr(mod, "EngineCoreOutput")

    # If already patched with "latents", do nothing
    if "latents" in getattr(Old, "__annotations__", {}):
        return

    class EngineCoreOutput(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):  # type: ignore[misc]
        # Required fields
        request_id: str
        new_token_ids: list[int]

        # Optional fields (preserve defaults)
        new_logprobs: T.Optional[LogprobsLists] = None
        new_prompt_logprobs_tensors: T.Optional[LogprobsTensors] = None

        pooling_output: T.Optional[torch.Tensor] = None

        finish_reason: T.Optional[FinishReason] = None
        stop_reason: T.Union[int, str, None] = None
        events: T.Optional[list[EngineCoreEvent]] = None
        kv_transfer_params: T.Optional[dict[str, T.Any]] = None

        # The number of tokens with prefix cache hits.
        num_cached_tokens: int = 0

        # NEW: make this serializable; encode tensors yourself
        latents: T.Optional[list[T.Optional[list[float]]]] = None

        @property
        def finished(self) -> bool:
            return self.finish_reason is not None

    # Keep import/pickle path stable
    EngineCoreOutput.__module__ = mod.__name__
    setattr(mod, "EngineCoreOutput", EngineCoreOutput)

# --- after your EngineCoreOutput patch is applied ---

def _patch_scheduler_update_from_output():
    # Import the original scheduler module
    sched_mod = importlib.import_module("vllm.v1.core.sched.scheduler")

    # Fetch utility functions/classes the body relies on
    check_stop = getattr(sched_mod, "check_stop")
    EngineCoreOutputs = getattr(sched_mod, "EngineCoreOutputs")
    SpecDecodingStats = getattr(sched_mod, "SpecDecodingStats", None)  # may be None on some versions

    # Define the replacement method. All comments are in English per your preference.
    def _update_from_output(self, scheduler_output, model_runner_output):
        # Late-bind EngineCoreOutput to ensure our patched class (with `latents`) is used.
        EngineCoreOutput = importlib.import_module("vllm.v1.engine").EngineCoreOutput

        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        latents = getattr(model_runner_output, "latents", None)
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        new_running = []
        outputs = []
        spec_decoding_stats = None

        # Main loop over running requests
        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # Not scheduled in this step
                new_running.append(request)
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids:
                # Adjust num_computed_tokens based on accepted/rejected spec tokens
                num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 - len(generated_token_ids))
                request.num_computed_tokens -= num_tokens_rejected
                if SpecDecodingStats is not None:
                    spec_decoding_stats = self.make_spec_decoding_stats(
                        spec_decoding_stats,
                        num_draft_tokens=len(scheduled_spec_token_ids),
                        num_accepted_tokens=len(generated_token_ids) - 1,
                    )

            # Free encoder cache entries that are fully consumed
            cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(request)
            if cached_encoder_input_ids:
                for input_id in list(cached_encoder_input_ids):
                    mm_positions = request.mm_positions[input_id]
                    start_pos = mm_positions.offset
                    num_tokens = mm_positions.length
                    if start_pos + num_tokens <= request.num_computed_tokens:
                        self.encoder_cache_manager.free_encoder_input(request, input_id)

            # Update scheduled spec token ids for the request
            if spec_token_ids is not None:
                request.spec_token_ids = spec_token_ids[req_index]

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids

            # Append new tokens (if any), check stop, and trim tail after stop
            for num_new, output_token_id in enumerate(new_token_ids, 1):
                request.append_output_token_ids(output_token_id)
                stopped = check_stop(request, self.max_model_len)
                if stopped:
                    self._free_request(request)
                    del new_token_ids[num_new:]  # Trim tokens after stop
                    break

            # Extract sample logprobs if requested
            if request.sampling_params.logprobs is not None and logprobs:
                # On spec decode this may carry >1 items per step; we slice per request
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            # Apply structured output grammar state if needed
            if new_token_ids and request.use_structured_output:
                # structured_output_request is guaranteed not None in this branch
                request.structured_output_request.grammar.accept_tokens(req_id, new_token_ids)  # type: ignore[union-attr]

            # Prompt logprobs for this request (prefill-only; invariant: none on partial prefill)
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)

            if new_token_ids:
                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        # NEW FIELD propagated from ModelRunnerOutput
                        latents=latents,
                    )
                )
            else:
                # Invariant: EngineCore should not return partial prefill outputs.
                assert not prompt_logprobs_tensors

            if not stopped:
                new_running.append(request)

        # Return cached request data to the queue so they can be reused
        for req_data in scheduler_output.scheduled_cached_reqs:
            self._cached_reqs_data[req_data.req_id].append(req_data)

        self.running = new_running
        engine_core_outputs = EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(spec_decoding_stats),
        )
        if self.include_finished_set:
            # NOTE: currently sending duplicates here; can be optimized later
            engine_core_outputs.finished_requests = (
                scheduler_output.finished_req_ids | self.finished_req_ids
            )

        return engine_core_outputs

    # Monkey-patch the class method
    sched_mod.Scheduler.update_from_output = _update_from_output

def _wrap_enginecore_step():
    core_mod = importlib.import_module("vllm.v1.engine.core")

    # Ensure the method exists on this version
    if not hasattr(core_mod.EngineCore, "step"):
        # Fallback: nothing to do on unknown versions
        return

    current = core_mod.EngineCore.step

    # Idempotence: avoid double-wrapping (which can cause recursion/lockups)
    if getattr(current, "_avt_wrapped", False):
        print("[AVT] return 1")
        return

    @functools.wraps(current)
    def wrapped(self):
        # 1) Call the official implementation to keep scheduling semantics intact.
        res = current(self)  # -> EngineCoreOutputs

        # 2) Read per-step latents cached by the model runner in execute_model().
        mr = getattr(self, "model_runner", None)
        req_ids = getattr(mr, "_avt_last_req_ids", None)
        latents_step = getattr(mr, "_avt_last_latents_step", None)

        outputs = getattr(res, "outputs", None)
        if not (isinstance(outputs, list) and req_ids and latents_step):
            return res  # Nothing to attach

        # Build request_id -> latent map for this step
        rmap = {}
        for i, rid in enumerate(req_ids):
            if i < len(latents_step):
                rmap[rid] = latents_step[i]  # Optional[List[float]]

        # 3) Attach to each EngineCoreOutput by request_id
        for i, out in enumerate(outputs):
            rid = getattr(out, "request_id", None)
            if rid is None:
                continue
            val = rmap.get(rid)
            if val is None:
                continue
            try:
                # Works if msgspec.Struct is mutable in this version
                out.latents = val
            except Exception:
                # Immutable Struct fallback
                from msgspec.structs import replace as _replace
                outputs[i] = _replace(out, latents=val)

        return res

    wrapped._avt_wrapped = True  # idempotence flag
    core_mod.EngineCore.step = wrapped


_patch_engine_core_output()
_wrap_enginecore_step()
for key in ("vllm.v1.worker.gpu_model_runner","vllm.worker.gpu_model_runner","vllm.worker.model_runner",):
    sys.modules[key] = importlib.import_module("avt.vllm.avt_gpu_model_runner")
#sys.modules['vllm.v1.core.sched.scheduler'] = importlib.import_module("avt.vllm.avt_scheduler")
print("[AVT] vLLM runner patched via sitecustomize:", __file__)
