"""Ascend execution with device-resident state, batched sampling and KV finalization."""

import torch

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


class AscendModelRunner(ModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.device.type != "npu":
            raise ValueError("AscendModelRunner requires an NPU")
        self.sampler = AscendSampler()

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
