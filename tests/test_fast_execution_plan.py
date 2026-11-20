"""Ownership and failure boundaries of Ascend recurrent execution."""

import pytest
import torch

from tests.test_ascend import make_npu_model
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.device import resolve_device
from vllm_rlt.request import Operation
from vllm_rlt.worker.executor import ForwardBatch, ModelRuntime

pytestmark = pytest.mark.npu


def cache_for(model):
    c = model.config
    cache = KVCacheManager(
        c.num_hidden_layers,
        c.num_key_value_heads,
        c.head_dim,
        32,
        16,
        4,
        device="npu:0",
        dtype=torch.bfloat16,
        backend="ascend_eager",
    )
    assert cache.allocate("a", 8)
    return cache


def metadata(cache, start, count):
    return cache._prepare_batches(
        ["a"] * count,
        [[depth] * count for depth in range(4)],
        list(range(start, start + count)),
    )


@torch.inference_mode()
def test_failed_traversal_keeps_partial_layer_coverage_visible():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    cache = cache_for(model)
    runtime = ModelRuntime(model, cache)
    batch = metadata(cache, 0, 3)[0]

    def fail(*args):
        raise RuntimeError("injected layer failure")

    handle = model.model.layers[1].register_forward_pre_hook(fail)
    try:
        hidden = model.prelude(torch.tensor([3, 7, 11], device="npu:0"))
        with pytest.raises(RuntimeError, match="injected layer failure"):
            runtime.forward(ForwardBatch(Operation.RECURRENT, hidden=hidden, metadata=(batch,)))
    finally:
        handle.remove()
    with pytest.raises(RuntimeError, match="every layer"):
        cache.finalize_token("a", 1, 0)
    q = torch.zeros(
        3,
        model.config.num_attention_heads,
        model.config.head_dim,
        device="npu:0",
        dtype=torch.bfloat16,
    )
    assert torch.isfinite(cache._attend_prepared(0, batch, q)).all()
    with pytest.raises(RuntimeError, match="uninitialized KV history"):
        cache._attend_prepared(1, batch, q)
    # Rerun all layers for the same page slots after the interrupted write.
    actual = runtime.forward(
        ForwardBatch(Operation.RECURRENT, hidden=hidden, metadata=(batch,))
    ).hidden
    reference = cache_for(model)
    expected = model.recurrent_prepared(hidden, metadata(reference, 0, 3)[0], reference)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    cache.free("a")
    reference.free("a")


@torch.inference_mode()
def test_paged_coverage_after_truncation_and_restore():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    cache = cache_for(model)
    runtime = ModelRuntime(model, cache)
    hidden = model.prelude(torch.tensor([3, 7, 11], device="npu:0"))
    for batch in metadata(cache, 0, 3):
        runtime.forward(ForwardBatch(Operation.RECURRENT, hidden=hidden, metadata=(batch,)))
    cache.finalize_token("a", 2, 3)
    snapshot = cache.snapshot("a")
    cache.truncate_suffix("a", 1)
    with pytest.raises(RuntimeError, match="every layer"):
        cache.finalize_token("a", 2, 3)
    cache.truncate_suffix("a", 0)
    cache.restore("a", snapshot)
    cache.finalize_token("a", 2, 3)
    cache.truncate_suffix("a", 1)
    with pytest.raises(RuntimeError, match="every layer"):
        cache.finalize_token("a", 2, 3)
    cache.free("a")
