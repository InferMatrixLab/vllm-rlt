"""Ascend execution with device-resident state, batched sampling and KV finalization."""

import weakref
from dataclasses import dataclass

import torch

from vllm_rlt.worker.executor import ModelRuntime
from vllm_rlt.worker.model_runner import ModelRunner
from vllm_rlt.worker.sampler import Sampler


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


class AscendModelRunner(ModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.device.type != "npu":
            raise ValueError("AscendModelRunner requires an NPU")
        self.runtime = ModelRuntime(
            self.model,
            self.cache_manager,
            prepare_recurrent=getattr(self.model, "prepare_recurrent", None),
        )
        self.sampler = AscendSampler()
        self.hidden_batch = None
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
            and getattr(self.model, "recurrent_returns_owned_hidden", False)
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
        stream = (
            self.device_ops.current_stream(self.device) if self.device_ops is not None else None
        )
        if prepared.routing is not None:
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
            if isinstance(source, torch.Tensor) and source.device.type in ("cuda", "npu"):
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
        self.hidden_batch = None
        return super()._free_runner_slot(request_id)
