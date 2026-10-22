"""Single-stage deployment client and low-level engine import."""

import copy
import time
from collections import deque
from dataclasses import replace
from uuid import uuid4

import torch

from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.engine.worker import start_stage_worker
from vllm_rlt.models import prepare_model
from vllm_rlt.worker.logprobs import check_logprobs_mode

LLMEngine = StageEngine


class StageClient:
    """Common control and lifetime owner for one prefill, decode or prefill_decode worker.

    Request methods serve a single-stage deployment. Coordinators use send/poll
    for connector messages and retain responsibility for cross-stage ordering.
    """

    def __init__(
        self,
        model,
        *,
        device=None,
        dtype=None,
        revision=None,
        model_config=None,
        seed=0,
        startup_timeout=120,
        shutdown_timeout=30,
        stage_config=None,
        connector_factory=None,
        max_control_messages=32,
        wait_for_ready=True,
        **engine_options,
    ):
        if "logprobs_mode" in engine_options:  # Fail before starting a worker.
            check_logprobs_mode(engine_options["logprobs_mode"])
        model, revision, self.model_config = prepare_model(
            model, revision=revision, model_config=model_config
        )
        # An explicitly supplied model is a weight source, never a second runner.
        # Checkpoint paths load directly inside the worker without a host copy.
        if isinstance(model, torch.nn.Module):
            parameter = next(model.parameters())
            device = parameter.device if device is None else device
            dtype = parameter.dtype if dtype is None else dtype
            model = copy.deepcopy(model).cpu()
        device = device if device is not None else "cpu"
        dtype = str(dtype).removeprefix("torch.") if dtype is not None else "bfloat16"
        self.stage_config = stage_config or StageConfig()
        self.connector_factory = connector_factory
        self.max_control_messages = max_control_messages
        self.info = {}
        self.ready = False
        self.connected = False
        self.trace_mode = getattr(engine_options.get("exit_config"), "mode", None) == "trace"
        self.closed = False
        self.failure = None
        self.timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.requests = {}  # External ID -> unique execution generation, no Request copies.
        self.generations = {}
        self.outputs = deque()
        self.replies = {}
        self.pending_controls = set()
        self.stopped = False
        self.process, self.channel = start_stage_worker(
            self.stage_config,
            device,
            model,
            dict(
                seed=seed,
                dtype=dtype,
                revision=revision,
                engine=engine_options,
                model_config=self.model_config,
            ),
            connector_factory=connector_factory,
            max_control_messages=max_control_messages,
        )
        if wait_for_ready:
            try:
                self.wait_ready()
            except BaseException:
                self.terminate()
                self.join(shutdown_timeout, force=True)
                raise

    def wait_ready(self):
        deadline = time.monotonic() + self.timeout
        while not self.ready:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("stage startup timed out")
            self._receive(timeout=remaining)
        return self.info

    def send(self, kind, **fields):
        if self.closed or self.failure:
            raise RuntimeError("stage client is unavailable")
        try:
            self.channel.send(dict(kind=kind, **fields))
        except OSError as exc:
            self.failure = str(exc)
            raise RuntimeError("stage control channel disconnected") from exc

    def poll(self, *, max_messages=None):
        """Handle common worker events and return messages for routing/coordination."""
        messages = []
        for _ in range(self.max_control_messages if max_messages is None else max_messages):
            if self.stopped:
                break
            message = self._receive()
            if message is None:
                break
            messages.append(message)
        if not self.stopped and not self.process.is_alive():
            raise RuntimeError(f"stage {self.stage_config.stage_id} exited {self.process.exitcode}")
        return messages

    def _receive(self, *, timeout=0):
        if not self.channel.poll(timeout):
            if not self.process.is_alive():
                raise RuntimeError(f"stage worker exited with code {self.process.exitcode}")
            if timeout:
                raise TimeoutError("stage worker response timed out")
            return None
        try:
            message = self.channel.recv()
        except EOFError as exc:
            raise RuntimeError("stage worker closed its control channel") from exc
        kind = message["kind"]
        if kind == "fatal":
            self.failure = message["error"]
            error = ValueError if message.get("error_type") == "ValueError" else RuntimeError
            raise error(f"stage worker failed:\n{self.failure}")
        if kind == "ready":
            self.info, self.ready = message, True
        elif kind == "connected":
            self.connected = True
        elif kind == "reply":
            if message["control_id"] in self.pending_controls:
                self.replies[message["control_id"]] = message
        elif kind == "output":
            generation = message["request_id"]
            request_id = self.generations.get(generation)
            if request_id is not None:
                output = replace(message["output"], request_id=request_id)
                self.outputs.append(output)
                if output.finished:
                    self.requests.pop(request_id, None)
                    self.generations.pop(generation, None)
        elif kind == "stopped":
            self.stopped = True
            self.final_stats = dict(requests=0, used_blocks=message["used_blocks"])
        return message

    def submit_control(self, method, *args, session_id=None, **kwargs):
        """Send a control call without waiting, so coordinators can broadcast."""
        control_id = uuid4().hex
        self.pending_controls.add(control_id)
        try:
            self.send(
                "control",
                control_id=control_id,
                method=method,
                args=args,
                kwargs=kwargs,
                session_id=session_id,
            )
        except BaseException:
            self.discard_control(control_id)
            raise
        return control_id

    def take_control_reply(self, control_id):
        """Consume a reply after poll(); errors remain available for aggregation."""
        reply = self.replies.pop(control_id, None)
        if reply is not None:
            self.pending_controls.remove(control_id)
        return reply

    def discard_control(self, control_id):
        """Stop retaining a completed/abandoned call, including any late reply."""
        self.pending_controls.discard(control_id)
        self.replies.pop(control_id, None)

    def _call(self, method, *args, **kwargs):
        control_id = self.submit_control(method, *args, **kwargs)
        try:
            deadline = time.monotonic() + self.timeout
            while control_id not in self.replies:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"stage control timed out: {method}")
                self._receive(timeout=remaining)
            reply = self.take_control_reply(control_id)
        except (OSError, RuntimeError, TimeoutError) as exc:
            self.failure = str(exc)
            raise
        finally:
            self.discard_control(control_id)
        if "error" in reply:
            error = RuntimeError if reply.get("error_type") == "RuntimeError" else ValueError
            raise error(reply["error"])
        return reply.get("result")

    def add_request(self, request_id, prompt_token_ids, sampling_params=None, *, trace_id=None):
        if (
            not isinstance(request_id, str)
            or not request_id
            or request_id in self.requests
            or any(output.request_id == request_id for output in self.outputs)
        ):
            raise ValueError("request ID must be nonempty and unique among active requests")
        generation = uuid4().hex
        self.requests[request_id] = generation
        self.generations[generation] = request_id
        # Preserve external trace identity while wire IDs prevent stale-result reuse.
        if trace_id is None:
            # Explicit traces are resolved by the engine; ordinary requests omit it.
            trace_id = request_id if self.trace_mode else None
        try:
            self._call(
                "add_request", generation, prompt_token_ids, sampling_params, trace_id=trace_id
            )
        except BaseException:
            self.requests.pop(request_id, None)
            self.generations.pop(generation, None)
            raise

    def abort_request(self, request_id):
        generation = self.requests.get(request_id)
        result = self._call("abort_request", generation) if generation is not None else None
        self.requests.pop(request_id, None)
        self.generations.pop(generation, None)
        self.outputs = type(self.outputs)(o for o in self.outputs if o.request_id != request_id)
        return replace(result, request_id=request_id) if result is not None else None

    def request_ids(self):
        return tuple(self.requests)

    def has_request(self, request_id):
        return request_id in self.requests

    def has_unfinished_requests(self):
        return bool(self.requests or self.outputs)

    def step(self):
        if self.requests and not self.outputs:
            if self.channel.poll(0.01):
                self._receive()
            elif not self.process.is_alive():
                raise RuntimeError("stage worker exited before completing requests")
        self.poll()
        outputs = list(self.outputs)
        self.outputs.clear()
        return outputs

    def get_weight_version(self):
        return self._call("get_weight_version")

    def pause_generation(self):
        return self._call("pause_generation")

    def resume_generation(self):
        return self._call("resume_generation")

    def reset_prefix_cache(self):
        if self.has_unfinished_requests():
            raise RuntimeError("Finish active requests before resetting cached KV")
        return self._call("reset_prefix_cache")

    def start_weight_update(self, version=None):
        if self.has_unfinished_requests():
            raise RuntimeError("Weight updates require an idle engine")
        return self._call("start_weight_update", version=version)

    def update_weights(self, weights):
        weights = [(name, tensor.detach().cpu()) for name, tensor in weights]
        return self._call("update_weights", weights=weights)

    def finish_weight_update(self):
        return self._call("finish_weight_update")

    def stats(self):
        return self.final_stats if self.stopped else self._call("stats")

    def start_profile(self, config, *, scheduled=False):
        return self._call("start_profile", config, scheduled=scheduled)

    def stop_profile(self):
        return self._call("stop_profile")

    def profile_status(self):
        return self._call("profile_status")

    def wait_for_profile_artifacts(self, timeout=None):
        return self._call("wait_for_profile_artifacts", timeout=timeout)

    def request_stop(self):
        # A connected stage must already be drained by its coordinator. A local
        # stage can cancel its remaining requests inside the owner process.
        self.send("stop" if self.connector_factory is not None else "shutdown")

    def terminate(self):
        """Signal termination; coordinators order writer/receiver groups before join."""
        if not self.closed and self.process.is_alive():
            self.process.terminate()

    def join(self, timeout=None, *, force=False):
        if self.closed:
            return
        self.process.join(self.shutdown_timeout if timeout is None else timeout)
        if self.process.is_alive() and force:
            self.process.kill()
            self.process.join(self.shutdown_timeout)
        if self.process.is_alive():
            raise TimeoutError("stage worker did not exit")
        self.channel.close()
        self.requests.clear()
        self.generations.clear()
        self.outputs.clear()
        self.replies.clear()
        self.pending_controls.clear()
        self.closed = True

    def close(self):
        if self.closed:
            return
        if self.connector_factory is not None:
            if not self.stopped:
                raise RuntimeError("connected stage shutdown must be ordered by its coordinator")
            self.join()
            return
        try:
            if self.process.is_alive() and not self.failure:
                self.request_stop()
                deadline = time.monotonic() + self.shutdown_timeout
                while not self.stopped:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("stage shutdown timed out")
                    self._receive(timeout=remaining)
                self.join()
        finally:
            if not self.closed:
                self.terminate()
                self.join(force=True)

    def __del__(self):
        # Connected workers belong to a coordinator: only it knows when remote
        # writers are stopped. Never reclaim their pools through GC ordering.
        if hasattr(self, "process") and not self.closed and self.connector_factory is None:
            try:
                self.close()
            except Exception:
                pass
