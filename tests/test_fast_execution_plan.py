"""Ownership and failure boundaries of the immutable Ascend decode plan."""

from dataclasses import replace

import pytest
import torch

from tests.test_ascend import make_npu_model
from tests.test_fixed_prelude import engine_for
from tests.test_npu_async import drain
from vllm_rlt import SamplingParams
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.device import resolve_device
from vllm_rlt.request import Operation
from vllm_rlt.worker.ascend import AscendOuroRuntime
from vllm_rlt.worker.executor import ForwardBatch, ModelRuntime
from vllm_rlt.worker.metadata import stage_range_metadata

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
        contiguous_kv=True,
    )
    assert cache.allocate("a", 8)
    return cache


def metadata(cache, start, count):
    layouts = cache.metadata_builder.build_ranges(["a"], [start], [count], [(d,) for d in range(4)])
    batches = stage_range_metadata(
        layouts, lambda values, dtype: torch.as_tensor(values, dtype=dtype, device="npu:0")
    )
    return tuple(replace(b, block_tables=b.table_plan.bind(torch.device("npu:0"))) for b in batches)


@torch.inference_mode()
def test_position_plan_reuses_depths_and_expires_between_tokens():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    expected = None
    for reuse in [False, True]:
        cache = cache_for(model)
        runtime = (AscendOuroRuntime if reuse else ModelRuntime)(model, cache)
        observed = []
        handle = model.model.rotary_emb.register_forward_hook(lambda *args: observed.append(1))
        actual = []
        try:
            for start, ids in [(0, [3, 7, 11]), (3, [13])]:
                hidden = model.prelude(torch.tensor(ids, device="npu:0"))
                result = runtime.forward(
                    ForwardBatch(
                        Operation.RECURRENT,
                        hidden=hidden,
                        metadata=metadata(cache, start, len(ids)),
                    )
                )
                actual.append(result.hidden.clone())
            assert len(observed) == (2 if reuse else 8)
            if expected is None:
                expected = actual
            else:
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        finally:
            handle.remove()
            cache.free("a")


@torch.inference_mode()
def test_failed_traversal_keeps_partial_layer_coverage_visible():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    cache = cache_for(model)
    batch = metadata(cache, 0, 3)[0]

    def fail(*args):
        raise RuntimeError("injected layer failure")

    handle = model.model.layers[1].register_forward_pre_hook(fail)
    try:
        hidden = model.prelude(torch.tensor([3, 7, 11], device="npu:0"))
        with pytest.raises(RuntimeError, match="injected layer failure"):
            AscendOuroRuntime(model, cache).forward(
                ForwardBatch(Operation.RECURRENT, hidden=hidden, metadata=(batch,))
            )
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
    # Retrying the write after an interrupted traversal must escape to validated
    # paged overwrite semantics, rather than retaining a stale fast lease.
    actual = (
        AscendOuroRuntime(model, cache)
        .forward(ForwardBatch(Operation.RECURRENT, hidden=hidden, metadata=(batch,)))
        .hidden
    )
    reference = cache_for(model)
    expected = model.recurrent_prepared(hidden, metadata(reference, 0, 3)[0], reference)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    cache.free("a")
    reference.free("a")


@torch.inference_mode()
def test_device_token_batch_is_borrowed_and_remains_owned():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    engine = engine_for(model, True)
    try:
        engine.add_request(
            "a", [3, 7, 11], SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
        )
        engine.step()
        engine.step()
        original = engine.model_runner.token_batch.tokens
        snapshot = original.clone()
        observed = []
        handle = model.model.embed_tokens.register_forward_pre_hook(
            lambda module, args: observed.append(args[0])
        )
        try:
            engine.step()
        finally:
            handle.remove()
        assert len(observed) == 1
        assert observed[0].data_ptr() == original.data_ptr()
        drain(engine)
        torch.testing.assert_close(original, snapshot, atol=0, rtol=0)
    finally:
        engine.close()


@torch.inference_mode()
def test_completion_proof_does_not_survive_truncation_or_restore():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    cache = cache_for(model)
    hidden = model.prelude(torch.tensor([3, 7, 11], device="npu:0"))
    for batch in metadata(cache, 0, 3):
        AscendOuroRuntime(model, cache).forward(
            ForwardBatch(Operation.RECURRENT, hidden=hidden, metadata=(batch,))
        )
    allocation = cache._get_allocation("a")
    proof = allocation.dense_submitted[cache._plane(3)]
    assert proof.all_layers_end == 3
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
