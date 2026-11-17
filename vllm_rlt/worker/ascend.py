"""Optional host executor; every model operation remains ordinary eager ATen."""

import hashlib
import os
import sysconfig
import weakref
from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch.nn.modules import module as module_hooks

from vllm_rlt.layers.normalization import RMSNorm
from vllm_rlt.request import Operation
from vllm_rlt.worker.executor import ForwardResult, ModelRuntime
from vllm_rlt.worker.model_runner import ModelRunner
from vllm_rlt.worker.sampler import Sampler


@lru_cache(maxsize=1)
def load_native_plan():
    from torch.utils.cpp_extension import load

    source = Path(__file__).parent / "csrc" / "ascend_eager_plan.cpp"
    identity = (
        source.read_bytes()
        + str(torch.__version__).encode()
        + str(torch._C._GLIBCXX_USE_CXX11_ABI).encode()
        + str(sysconfig.get_config_var("SOABI")).encode()
    )
    digest = hashlib.sha256(identity).hexdigest()[:12]
    root = os.environ.get(
        "VLLM_RLT_NATIVE_BUILD_DIR", str(Path.home() / ".cache/vllm-rlt/native-eager")
    )
    build = Path(root) / digest
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="rlt_ascend_eager_" + digest,
        sources=[str(source)],
        build_directory=str(build),
        extra_cflags=["-O3", "-g0"],
        with_cuda=False,
        verbose=True,
    )


class NativeOuroExecutor:
    """Eager layer calls with current weights and conservative Python fallback.

    The extension stores no model tensors. Each call borrows the module's live
    parameters, and returns owned output/KV tensors. KV is published through its
    traversal owner only after a complete layer call has enqueued successfully.
    """

    def __init__(self):
        self.extension = load_native_plan()
        self.native_layers = 0
        self.fallbacks = Counter()

    def _fallback(self, reason):
        self.fallbacks[reason] += 1
        return None

    def layer(self, layer, hidden, position_embeddings, batch, traversal):
        from vllm_rlt.models.ouro import OuroAttention, OuroDecoderLayer, OuroMLP

        if (
            type(layer) is not OuroDecoderLayer
            or type(layer.self_attn) is not OuroAttention
            or type(layer.mlp) is not OuroMLP
        ):
            return self._fallback("module_schema")
        if type(hidden) is not torch.Tensor or torch._C._get_tracing_state():
            return self._fallback("tensor_or_tracing")
        if any(
            getattr(module_hooks, name, {})
            for name in (
                "_global_forward_hooks",
                "_global_forward_pre_hooks",
                "_global_backward_hooks",
                "_global_backward_pre_hooks",
            )
        ):
            return self._fallback("global_hooks")
        if (
            layer._forward_hooks
            or layer._forward_pre_hooks
            or layer._backward_hooks
            or layer._backward_pre_hooks
            or "forward" in layer.__dict__
            or getattr(layer, "_compiled_call_impl", None) is not None
        ):
            return self._fallback("layer_hooks_or_forward")
        # Hooked layers run through their ordinary nn.Module call. Child callbacks,
        # custom forwards and compiled children retain the Python implementation.
        for child in layer.modules():
            if child is layer:
                continue
            if (
                child._forward_hooks
                or child._forward_pre_hooks
                or child._backward_hooks
                or child._backward_pre_hooks
                or "forward" in child.__dict__
                or getattr(child, "_compiled_call_impl", None) is not None
            ):
                return self._fallback("child_hooks_or_forward")
        a, m = layer.self_attn, layer.mlp
        norms = (
            layer.input_layernorm,
            layer.input_layernorm_2,
            layer.post_attention_layernorm,
            layer.post_attention_layernorm_2,
        )
        linears = (a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj)
        if any(type(x) is not RMSNorm for x in norms) or any(
            type(x) is not torch.nn.Linear for x in linears
        ):
            return self._fallback("leaf_schema")
        if any(x.bias is not None for x in linears):
            return self._fallback("linear_bias")
        if any(x.variance_epsilon != norms[0].variance_epsilon for x in norms):
            return self._fallback("norm_epsilon")
        weights = [x.weight for x in (*norms, *linears)]
        if any(type(x) is not torch.nn.Parameter for x in weights):
            return self._fallback("parameter_schema")
        prepare = getattr(traversal, "prepare_native", None)
        if prepare is None:
            return self._fallback("paged_storage")
        inputs = prepare(a.layer_idx, hidden)
        if inputs is None:
            return self._fallback("append_geometry")
        if a.config.head_dim != inputs.head_dim:
            return self._fallback("head_geometry")
        output, key, value = self.extension.layer_forward(
            hidden,
            *position_embeddings,
            weights,
            [] if inputs.history is None else inputs.history,
            inputs.mask,
            inputs.batch_size,
            inputs.query_tokens,
            inputs.token_start,
            inputs.head_dim,
            inputs.kv_heads,
            norms[0].variance_epsilon,
        )
        traversal.commit_native(a.layer_idx, inputs, (key, value))
        self.native_layers += 1
        return output


class AscendSampler(Sampler):
    """Keep batched greedy dispatch out of the CUDA/ROCm sampling path."""

    def sample_batch(self, logits, params, generators):
        if getattr(self.sample, "__func__", None) is Sampler.sample and all(
            item.temperature == 0 for item in params
        ):
            return logits[: len(params)].argmax(dim=-1)
        return torch.stack(
            [
                self.sample(row, item, generator)
                for row, item, generator in zip(logits, params, generators)
            ]
        )


# The shared runtime and model keep their upstream implementation. Only an NPU
# runner creates this adapter; it owns preparation reused within one forward.
class AscendOuroRuntime(ModelRuntime):
    def __init__(self, model, cache, *, native=False):
        super().__init__(model, cache)
        self.layer_executor = NativeOuroExecutor() if native else None

    @property
    def owns_hidden(self):
        from vllm_rlt.models.ouro import OuroForCausalLM

        if type(self.model) is not OuroForCausalLM:
            return False
        norm = self.model.model.norm
        return (
            "recurrent_prepared" not in self.model.__dict__
            and type(norm) is RMSNorm
            and not norm._forward_hooks
            and not norm._forward_pre_hooks
            and "forward" not in norm.__dict__
        )

    def forward(self, batch):
        from vllm_rlt.models.ouro import OuroForCausalLM

        if (
            type(self.model) is not OuroForCausalLM
            or "recurrent_prepared" in self.model.__dict__
            or batch.graphs is not None
            or batch.operation not in (Operation.PRELUDE, Operation.RECURRENT)
        ):
            return super().forward(batch)
        hidden, gate = batch.hidden, None
        try:
            if batch.operation == Operation.PRELUDE:
                if batch.tokens is None:
                    raise ValueError("PRELUDE requires token IDs")
                hidden = self.model.prelude(batch.tokens)
            if hidden is None:
                raise ValueError("RECURRENT requires hidden state")
            position_key = position_embeddings = None
            for kv in batch.metadata:
                positions = kv.position_ids
                key = (
                    positions.data_ptr(),
                    positions.shape,
                    positions.stride(),
                    positions.dtype,
                    positions.device,
                    hidden.dtype,
                )
                if key != position_key:
                    position_embeddings = self.model.model.rotary_emb(hidden, positions)
                    position_key = key
                traversal = self.cache._begin_traversal(kv)
                for layer in self.model.model.layers:
                    output = (
                        None
                        if self.layer_executor is None
                        else self.layer_executor.layer(
                            layer, hidden, position_embeddings, kv, traversal
                        )
                    )
                    hidden = (
                        layer(hidden, position_embeddings, kv, traversal)
                        if output is None
                        else output
                    )
                hidden = self.model.model.norm(hidden)
                gate = (
                    self.model.model.early_exit_gate(hidden).squeeze(-1)
                    if batch.compute_gate
                    else None
                )
                if traversal is not self.cache:
                    traversal.finish()
            return ForwardResult(hidden, gate)
        finally:
            if isinstance(batch.metadata, Generator):
                batch.metadata.close()


@dataclass(frozen=True)
class HiddenBatch:
    """Immutable eager output; Request.hidden_state remains authoritative.

    Row views retain their owning allocation. Weak request identities and exact
    view identities prevent reuse after cancellation, replacement or regrouping.
    Only the latest batch is retained; this is not a second request-state store.
    """

    hidden: torch.Tensor
    requests: tuple[weakref.ReferenceType, ...]
    rows: tuple[torch.Tensor, ...]

    def matches(self, requests):
        return len(requests) == len(self.rows) and all(
            owner() is request and request.hidden_state is row
            for owner, request, row in zip(self.requests, requests, self.rows)
        )


@dataclass(frozen=True)
class TokenBatch:
    tokens: torch.Tensor
    requests: tuple
    rows: tuple

    def matches(self, requests, sources):
        return len(requests) == len(self.rows) and all(
            owner() is request and source is row
            for owner, request, source, row in zip(self.requests, requests, sources, self.rows)
        )


class AscendModelRunner(ModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.device.type != "npu":
            raise ValueError("AscendModelRunner requires an NPU")
        self.runtime = AscendOuroRuntime(
            self.model, self.cache_manager, native=self.execution_config.native_eager_plan
        )
        self.sampler = AscendSampler()
        self.hidden_batch = self.token_batch = None
        self.retain_hidden_batch = (
            self.execution_config.async_scheduling
            and self.cache_manager.backend == "ascend_eager"
            and not self.execution_config.static_buffers
            and not self.execution_config.cuda_graphs
        )

    def _save(self, request, state):
        self.hidden_batch = None
        return super()._save(request, state)

    def _save_batch(self, requests, hidden, routing):
        if (
            self.retain_hidden_batch
            and self.runtime.owns_hidden
            and routing is None
            and len(hidden) == len(requests)
        ):
            rows = tuple(hidden.unbind(0))
            for request, state in zip(requests, rows):
                request.hidden_state = state
            self.hidden_batch = HiddenBatch(
                hidden, tuple(weakref.ref(request) for request in requests), rows
            )
            return
        self.hidden_batch = None
        if routing is None:
            for request, state in zip(requests, hidden):
                self._save(request, state)
            return
        routing.scatter(hidden)
        for request in requests:
            request.hidden_state = self.states[self.state_slots[request.request_id]]

    def _gather(self, requests, workspace, size):
        saved = self.hidden_batch
        if (
            saved is not None
            and workspace is None
            and size == len(requests)
            and saved.matches(requests)
        ):
            if self.execution_config.multi_stream:
                # The next submission can replace all Request row views before
                # this stream has consumed the previous allocation.
                saved.hidden.record_stream(self.device_ops.current_stream(self.device))
            return saved.hidden
        states = [request.hidden_state for request in requests]
        if workspace is None:
            hidden = torch.stack(states)
        else:
            hidden = workspace.hidden[:size]
            hidden.zero_()
            for row, state in enumerate(states):
                hidden[row].copy_(state)
        if hidden.device.type == "npu" and self.execution_config.multi_stream:
            # Request states can be replaced before this stream consumes them.
            # NPU uses these tensors instead of CUDA's persistent state slots.
            stream = self.device_ops.current_stream(hidden.device)
            for state in states:
                state.record_stream(stream)
        return hidden

    def _prepare_tokens(self, prepared, size):
        """Materialize snapshotted token sources on the current compute stream."""
        sources = prepared.token_sources
        workspace = prepared.workspace
        borrowed = prepared.token_ids
        stream = (
            self.device_ops.current_stream(self.device) if self.device_ops is not None else None
        )
        if borrowed is not None and prepared.routing is None and size == len(borrowed):
            tokens = borrowed
            tokens.record_stream(stream)
            if self.token_batch is not None and self.token_batch.tokens is tokens:
                self.token_batch = None
        elif prepared.routing is not None:
            tokens = prepared.routing.gather(tokens=True)
        elif all(isinstance(source, torch.Tensor) for source in sources):
            tokens = torch.stack(sources)
        elif all(isinstance(source, tuple) and len(source) == 1 for source in sources):
            ids = [source[0] for source in sources]
            tokens = (
                workspace.tokens(ids, size)
                if workspace is not None
                else torch.tensor(ids, device=self.device, dtype=torch.long)
            )
        else:
            tokens = torch.cat(
                [
                    source.reshape(-1)
                    if isinstance(source, torch.Tensor)
                    else torch.tensor(source, device=self.device, dtype=torch.long)
                    for source in sources
                ]
            )
        if prepared.routing is None and size > len(tokens):
            tokens = torch.cat((tokens, tokens.new_zeros(size - len(tokens))))
        for span, source in zip(prepared.batch.spans, sources):
            if (
                tokens is not borrowed
                and isinstance(source, torch.Tensor)
                and source.device.type in ("cuda", "npu")
            ):
                source.record_stream(stream)
            if span.request.input_token_tensor is source:
                span.request.input_token_tensor = None
        return tokens

    def _wait_for_requests(self, batch, stream):
        # A previous batch often gives many requests the same completion event.
        dependencies = set()
        saved = self.hidden_batch
        retained = saved is not None and saved.matches([span.request for span in batch.spans])
        if retained:
            saved.hidden.record_stream(stream)
        for span in batch.spans:
            event = self.events.get(span.request.request_id)
            if event is not None and event not in dependencies:
                stream.wait_event(event)
                dependencies.add(event)
            if not retained and span.request.hidden_state is not None:
                span.request.hidden_state.record_stream(stream)

    def _retain_device_tokens(self, batch, values, rows):
        self.token_batch = None
        if (
            self.device.type != "npu"
            or self.cache_manager._contiguous is None
            or not self.retain_hidden_batch
            or len(rows) != len(batch.requests)
            or list(rows) != list(range(len(batch.requests)))
        ):
            return
        requests = [item.request for item in batch.requests]
        self.token_batch = TokenBatch(
            values,
            tuple(weakref.ref(r) for r in requests),
            tuple(r.input_token_tensor for r in requests),
        )

    def _finalize_npu_batch(self, requests):
        cache = self.cache_manager
        copies = []
        for request in requests:
            depth = request.loops_done - 1
            if cache.layout == "shared" or depth + 1 == cache.max_loops:
                # This still verifies every required layer. With no destination
                # there is no device work: the existing core event is sufficient.
                cache.finalize_token(request.request_id, request.position, depth)
            else:
                copies.append(request)
        if not copies:
            return
        stream = self.boundary_stream
        with self.device_ops.stream(stream):
            try:
                dependencies = {
                    self.events[request.request_id]
                    for request in copies
                    if request.request_id in self.events
                }
                for event in dependencies:
                    stream.wait_event(event)
                for request in copies:
                    cache.finalize_token(
                        request.request_id, request.position, request.loops_done - 1
                    )
                done = self.device_ops.Event()
                done.record(stream)
            except Exception:
                # A later request can fail after earlier copies were enqueued.
                # Finish those readers before cleanup may recycle their pages.
                stream.synchronize()
                raise
        for request in copies:
            self.events[request.request_id] = done

    def finalize_many(self, requests):
        if self.execution_config.async_scheduling and self.cache_manager.backend == "ascend_eager":
            self._finalize_npu_batch(requests)
        else:
            super().finalize_many(requests)

    def _free_runner_slot(self, request_id):
        self.hidden_batch = self.token_batch = None
        return super()._free_runner_slot(request_id)
