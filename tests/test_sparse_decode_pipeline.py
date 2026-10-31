"""Sparse decode attention with token pipelines: device-decided row modes and slots."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from tests.test_sparse_decode_attention import drain, make_store
from tests.test_sparse_decode_fused import _paged
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SparseDecodeAttentionConfig,
    SpeculativeConfig,
)
from vllm_rlt.attention.sparse import CAPTURE, CONSUME, DENSE

PROMPTS = [[2, 3, 4, 5, 6, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
LENGTHS = (23, 9, 17)


@pytest.fixture(autouse=True)
def _check_fast_launches(monkeypatch):
    """Every reused launch must be the kernel Triton's own dispatch would pick."""
    try:
        from vllm_rlt.kernels import sparse_decode
    except ImportError:  # CPU-only installs have no Triton and no fused launches
        return
    monkeypatch.setattr(sparse_decode, "_LAUNCH_CHECK", True)


@pytest.mark.gpu
def test_mixed_modes_match_uniform_fused_kernels():
    """One launch set with every mode equals the uniform launches row for row."""
    from vllm_rlt.kernels import sparse_decode
    from vllm_rlt.kernels.paged_attention import TritonPagedAttention

    rows, heads, kv_heads, dim, block_size, pages = 6, 4, 2, 64, 16, 9
    dense = TritonPagedAttention()
    q, keys, values, tables = _paged(
        rows, heads, kv_heads, dim, pages, block_size, torch.float32, 7
    )
    q2, _, _, _ = _paged(rows, heads, kv_heads, dim, pages, block_size, torch.float32, 8)
    lengths = torch.tensor([5, 33, 70, 129, 140, 1], device="cuda", dtype=torch.int32)
    modes = [DENSE, CAPTURE, CONSUME, CAPTURE, CONSUME, DENSE]
    stores, leases = [], []
    for _ in range(2):
        store = make_store(
            slots=rows,
            layers=2,
            heads=heads,
            head_dim=dim,
            key_bucket=pages * block_size,
            device="cuda",
            implementation="fused",
        )
        lease = store.reserve("r", 0, rows, generation=1)
        # The consume rows need a reference from an earlier loop.
        earlier = [row for row, mode in enumerate(modes) if mode == CONSUME]
        plan = store.build_plan(
            [(lease, row) for row in earlier], [CAPTURE] * len(earlier), key_retention=0.25
        )
        sparse_decode.fused_paged_attention(
            q[earlier],
            keys,
            values,
            tables[earlier],
            lengths[earlier],
            plan,
            layer=1,
            dense_attention=dense,
        )
        stores.append(store)
        leases.append(lease)
    mixed_store, uniform_store = stores
    plan = mixed_store.build_device_plan(
        [leases[0]],
        leases[0].slot_ids,
        torch.tensor(modes, device="cuda", dtype=torch.int8),
        key_retention=0.25,
    )
    mixed = sparse_decode.fused_paged_attention(
        q2, keys, values, tables, lengths, plan, layer=1, dense_attention=dense
    )
    expected = torch.empty_like(q2)
    for mode in (DENSE, CAPTURE, CONSUME):
        subset = [row for row, m in enumerate(modes) if m == mode]
        if mode == DENSE:
            expected[subset] = dense(q2[subset], keys, values, tables[subset], lengths[subset])
            continue
        host = uniform_store.build_plan(
            [(leases[1], row) for row in subset], [mode] * len(subset), key_retention=0.25
        )
        expected[subset] = sparse_decode.fused_paged_attention(
            q2[subset],
            keys,
            values,
            tables[subset],
            lengths[subset],
            host,
            layer=1,
            dense_attention=dense if mode == CAPTURE else None,
        )
    torch.cuda.synchronize()
    # Dense and capture rows are the dense kernel's output; consume rows the
    # uniform consume's; every store field matches slot for slot.
    assert torch.equal(mixed, expected)
    for name in ("selected", "counts", "mass", "output", "contribution", "valid", "uses"):
        assert torch.equal(getattr(mixed_store, name), getattr(uniform_store, name)), name


def _engine(mode, sparse, *, draft_loops=2, graphs=False, retention=0.5):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.models import OuroForCausalLM

    torch.manual_seed(123)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.float32)
    return StageEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(256, 16, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=16, prefill_chunk_size=8
        ),
        execution_config=ExecutionConfig(
            async_scheduling=True,
            static_buffers=graphs,
            pad_to_power_of_two=graphs,
            cuda_graphs=graphs,
        ),
        speculative_config=SpeculativeConfig(3, draft_loops=draft_loops, mode=mode),
        sparse_decode_attention_config=(
            SparseDecodeAttentionConfig(
                enabled=True, key_retention=retention, implementation="fused"
            )
            if sparse
            else None
        ),
    )


def _generate(e):
    for i, (prompt, length) in enumerate(zip(PROMPTS, LENGTHS)):
        e.add_request(str(i), prompt, SamplingParams(max_tokens=length, ignore_eos=True))
    outputs = drain(e)
    assert sorted(outputs) == ["0", "1", "2"]
    assert e.cache_manager.num_used_blocks == 0
    if e.reference_store is not None:
        assert e.reference_store.outstanding == 0
    return {rid: list(output.token_ids) for rid, output in outputs.items()}


@pytest.mark.gpu
@pytest.mark.parametrize("draft_loops", [1, 2])
@pytest.mark.parametrize("graphs", [False, True])
def test_pipeline_sparse_matches_block_sparse(draft_loops, graphs):
    """Both capture at loop 1 and consume at loops 2 and 3 of every token."""
    results = {}
    for mode in ("block", "pipeline"):
        for sparse in (False, True):
            e = _engine(mode, sparse, draft_loops=draft_loops, graphs=graphs)
            try:
                results[mode, sparse] = _generate(e)
                if sparse and mode == "pipeline" and graphs:
                    runner = e.speculative_runner
                    assert runner.graphs.sparse_replays > 0
            finally:
                e.close()
    # Precondition: the dense pipeline already reproduces block speculation.
    assert results["pipeline", False] == results["block", False]
    assert results["pipeline", True] == results["block", True]


@pytest.mark.gpu
def test_pipeline_sparse_slots_follow_requests_through_abort_and_reuse():
    e = _engine("pipeline", True)
    store = e.reference_store
    width = min(4, 4 // 2)
    try:
        e.add_request("a", PROMPTS[0], SamplingParams(max_tokens=40, ignore_eos=True))
        for _ in range(200):
            e.step()
            state = e.speculative_runner.states.get("a")
            if state is not None and state.reference is not None:
                break
        else:
            pytest.fail("pipeline state was not created")
        # One lease of `width` slots for the request's lifetime.
        assert store.outstanding == width
        assert sorted(state.reference.slots) == sorted(state.pipeline.slots.tolist())
        e.abort_request("a")
        drain(e)
        assert store.outstanding == 0
        # A reused id gets a new generation and fresh slots.
        assert _generate(e)
    finally:
        e.close()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "sparse",
    [
        SparseDecodeAttentionConfig(enabled=True, key_retention=0.5),
        SparseDecodeAttentionConfig(enabled=True, key_retention=1.0, implementation="fused"),
    ],
)
def test_pipeline_rejects_sparse_paths_without_device_plans(sparse):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.models import OuroForCausalLM

    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.float32)
    with pytest.raises(ValueError, match="token pipelines"):
        StageEngine(
            model,
            attention_backend="triton",
            execution_config=ExecutionConfig(async_scheduling=True),
            speculative_config=SpeculativeConfig(3, mode="pipeline"),
            sparse_decode_attention_config=sparse,
        )
