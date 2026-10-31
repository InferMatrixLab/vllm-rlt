"""Ordinary and synchronous block decode through the shared sparse contract."""

from dataclasses import replace

import pytest
import torch

from tests.helpers import tiny_ouro_model
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SparseDecodeAttentionConfig,
    SpeculativeConfig,
)
from vllm_rlt.attention import BackendCapabilities, create_backend
from vllm_rlt.attention.sparse import CAPTURE, CONSUME, DENSE
from vllm_rlt.engine.stage_engine import StageEngine


def engine(*, draft=None, retention=1.0, device="cpu", dtype=torch.float32, **options):
    values = dict(
        attention_backend="triton" if device == "cuda" else "torch",
        cache_config=CacheConfig(512, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=16, prefill_chunk_size=3
        ),
        execution_config=ExecutionConfig(static_buffers=True, pad_to_power_of_two=True),
        speculative_config=None if draft is None else SpeculativeConfig(2, draft_loops=draft),
        sparse_decode_attention_config=(
            None
            if retention is None
            else SparseDecodeAttentionConfig(enabled=True, key_retention=retention)
        ),
    )
    values.update(options)
    return StageEngine(tiny_ouro_model(device, dtype), **values)


def generate(e, count=3):
    for i in range(count):
        e.add_request(
            str(i), [2, 3, 4, 5][: 2 + i % 3], SamplingParams(max_tokens=8, ignore_eos=True)
        )
    outputs = {}
    for _ in range(1000):
        if not e.has_unfinished_requests():
            break
        for output in e.step():
            if output.finished:
                outputs[output.request_id] = list(output.token_ids)
    assert len(outputs) == count
    assert all(len(tokens) == 8 for tokens in outputs.values())
    assert e.cache_manager.num_used_blocks == 0
    if e.reference_store is not None:
        assert e.reference_store.outstanding == 0
    return outputs


@pytest.mark.parametrize("draft", [None, 1, 2])
def test_full_retention_matches_dense_and_reclaims_slots(draft):
    dense = engine(draft=draft, retention=None)
    sparse = engine(draft=draft)
    try:
        assert generate(sparse) == generate(dense)
    finally:
        sparse.close()
        dense.close()


@pytest.mark.parametrize("draft", [None, 1, 2])
def test_plans_follow_tokens_and_keep_prompt_dense(draft, monkeypatch):
    e = engine(draft=draft, retention=0.5)
    seen = []
    original = e.cache_manager._attend_prepared

    def attend(layer, kv, q):
        plan = getattr(kv, "reference_plan", None)
        if plan is not None:
            assert len(plan.modes) == len(q)
            for row, mode in enumerate(plan.modes):
                if mode == DENSE:
                    continue
                allocation, depth, position = kv.rows[row]
                request_id = next(rid for rid, owner in kv.allocations if owner is allocation)
                assert position >= len(e.scheduler.requests[request_id].prompt_token_ids)
                assert mode == (CAPTURE if depth == 1 else CONSUME)
                assert int(e.reference_store.position[plan.slots[row]]) == position
                seen.append((mode, position, plan.slots[row]))
        return original(layer, kv, q)

    monkeypatch.setattr(e.cache_manager, "_attend_prepared", attend)
    try:
        generate(e)
        assert {mode for mode, _, _ in seen} == {CAPTURE, CONSUME}
    finally:
        e.close()


@pytest.mark.parametrize("draft", [None, 1, 2])
def test_cancel_and_request_id_reuse_cannot_reuse_old_reference(draft):
    e = engine(draft=draft, retention=0.5)
    try:
        for _ in range(2):
            e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=32, ignore_eos=True))
            for _ in range(100):
                e.step()
                if e.reference_store.outstanding or (
                    draft is not None and e.speculative_runner.stats.rounds
                ):
                    break
            e.abort_request("r")
            assert e.reference_store.outstanding == 0
            assert e.cache_manager.num_used_blocks == 0
    finally:
        e.close()


def test_context_bound_is_enforced_before_request_admission():
    e = engine(
        sparse_decode_attention_config=SparseDecodeAttentionConfig(
            enabled=True, max_context_tokens=8
        )
    )
    try:
        with pytest.raises(ValueError, match="sparse.*context"):
            e.add_request("too-long", [2, 3, 4], SamplingParams(max_tokens=8))
        assert "too-long" not in e.scheduler.requests
    finally:
        e.close()


@pytest.mark.parametrize("setting", ["graphs", "pipeline", "preemption", "shared"])
def test_unqualified_execution_modes_fail_explicitly(setting):
    options = {
        "graphs": dict(execution_config=ExecutionConfig(cuda_graphs=True)),
        "pipeline": dict(speculative_config=SpeculativeConfig(2, mode="pipeline")),
        "preemption": dict(scheduler_config=SchedulerConfig(enable_preemption=True)),
        "shared": dict(cache_config=CacheConfig(512, 2, layout="shared")),
    }[setting]
    with pytest.raises(ValueError, match="sparse decode attention"):
        engine(**options)


def test_backend_explicitly_declares_sparse_reference_support():
    selected = create_backend(
        "auto",
        torch.device("cpu"),
        torch.float32,
        8,
        2,
        required_capabilities=BackendCapabilities(sparse_decode=True),
    )
    assert selected.capabilities.sparse_decode


@pytest.mark.gpu
@pytest.mark.parametrize("kv_heads", [2, 16])
@pytest.mark.parametrize("retention", [0.1, 1.0])
def test_cuda_sparse_preserves_dense_backend_outputs(kv_heads, retention):
    from vllm_rlt.core.kv_cache_manager import KVCacheManager
    from vllm_rlt.worker.attention_reference import AttentionReferenceStore

    # BF16 rounding of Torch softmax and Triton split-KV can differ even for
    # identical operands. Dense/capture rows and the full-retention control
    # must preserve the configured backend, including in a mixed plan.
    torch.manual_seed(157)
    rows, heads, dim, length = 32, 16, 128, 130
    cache = KVCacheManager(
        1,
        kv_heads,
        dim,
        rows * 9,
        16,
        1,
        device="cuda",
        dtype=torch.bfloat16,
        backend="triton",
    )
    store = AttentionReferenceStore(
        slots=rows, layers=1, heads=heads, head_dim=dim, key_bucket=length, device="cuda"
    )
    ids = [str(i) for i in range(rows)]
    owners = []
    for rid in ids:
        assert cache.allocate(rid, length)
        keys = torch.randn(length, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        cache.write(0, [rid] * length, [0] * length, list(range(length)), keys, values)
        lease = store.reserve(rid, 0, 1, generation=1)
        owners.append((lease, 0))
    batch = cache._prepare_batch(ids, [0] * rows, [length - 1] * rows)
    q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    expected = cache._attend_prepared(0, batch, q)
    modes = [DENSE] + [CAPTURE] * (rows - 1)
    plan = store.build_plan(owners, modes, key_retention=retention)
    actual = cache._attend_prepared(0, replace(batch, reference_plan=plan), q)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for row in range(1, rows):
        torch.testing.assert_close(
            store.output[plan.slots[row], 0], expected[row].float(), atol=0, rtol=0
        )
    if retention == 1.0:
        # Changed later-loop Q/K/V prevents an identity-only correction test.
        q = torch.randn_like(q)
        cache._write_prepared(
            0, batch, torch.randn_like(keys[:rows]), torch.randn_like(keys[:rows])
        )
        expected = cache._attend_prepared(0, batch, q)
        consume = store.build_plan(
            owners, [DENSE] + [CONSUME] * (rows - 1), key_retention=retention
        )
        actual = cache._attend_prepared(0, replace(batch, reference_plan=consume), q)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert int(store.uses.sum()) == rows - 1
    for lease, _ in owners:
        store.release(lease)
    for rid in ids:
        cache.free(rid)
    torch.cuda.synchronize()
    assert store.outstanding == cache.num_used_blocks == 0


@pytest.mark.parametrize("draft", [1, 2])
def test_sparse_verification_matches_serial_same_input_hidden_logits_and_kv(draft):
    from tests.helpers import make_cpu_cache
    from tests.test_sparse_decode_attention import make_reference_store
    from vllm_rlt.attention.sparse import reference_mode
    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput
    from vllm_rlt.request import Request, RequestPhase

    e = engine(draft=draft, retention=0.5)
    model, cache = e.model, e.cache_manager
    prefix, current = [3, 4, 5], 7
    assert cache.allocate("r", 16)
    for position, token in enumerate(prefix):
        hidden = model.prelude(torch.tensor([token]))
        for depth in range(4):
            hidden, _ = model.recurrent(hidden, ["r"], [depth], [position], cache)
    request = Request("r", prefix, SamplingParams(ignore_eos=True), generated_token_ids=[current])
    seen, heads = {}, []
    recurrent = model.recurrent_prepared

    def record(hidden, kv, cache, **kwargs):
        out, gate = recurrent(hidden, kv, cache, **kwargs)
        for row, (_, depth, position) in enumerate(kv.rows):
            seen[depth, position] = out[row].clone()
        return out, gate

    model.recurrent_prepared = record
    hook = model.lm_head.register_forward_hook(
        lambda module, inputs, output: heads.append(output.detach().clone())
    )
    try:
        e.speculative_runner.submit(
            SchedulerOutput(RequestPhase.SPECULATIVE, [ScheduledRequest(request, 3, 3)])
        ).collect()
    finally:
        hook.remove()
        model.recurrent_prepared = recurrent
    assert e.reference_store.outstanding == 0
    tokens = prefix + [current]
    tokens.extend(int(model.coda(seen[draft - 1, pos]).argmax()) for pos in (3, 4))
    oracle = make_cpu_cache(model.config)
    assert oracle.allocate("r", 16)
    store = make_reference_store(model)
    for position, token in enumerate(tokens):
        lease = store.reserve("r", 0, 1, generation=position)
        hidden = model.prelude(torch.tensor([token]))
        for depth in range(4):
            kv = oracle._prepare_batch(["r"], [depth], [position])
            mode = reference_mode(depth)
            if position >= len(prefix) and mode != DENSE:
                plan = store.build_plan([(lease, 0)], [mode], key_retention=0.5)
                kv = replace(kv, reference_plan=plan)
            hidden, _ = model.recurrent_prepared(hidden, kv, oracle, compute_gate=False)
            if position >= len(prefix):
                torch.testing.assert_close(seen[depth, position], hidden[0], atol=4e-5, rtol=4e-5)
        if position >= len(prefix):
            torch.testing.assert_close(
                heads[-1][position - len(prefix)], model.coda(hidden)[0], atol=4e-5, rtol=4e-5
            )
        store.release(lease)
    for depth in range(4):
        for layer in range(model.config.num_hidden_layers):
            for actual, expected in zip(
                cache.read(layer, "r", depth, len(tokens)),
                oracle.read(layer, "r", depth, len(tokens)),
            ):
                torch.testing.assert_close(actual, expected, atol=4e-5, rtol=4e-5)
    cache.free("r")
    oracle.free("r")
    e.close()


@pytest.mark.parametrize("draft", [1, 2])
@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_sync_rejection_retires_all_references_before_replacement(draft, accepted, monkeypatch):
    e = engine(draft=draft, retention=0.5)
    try:
        e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=16, ignore_eos=True))
        while not e.step():
            pass
        request = e.scheduler.requests["r"]
        start = request.position
        original = e.speculative_runner._coda
        calls = 0

        def controlled(hidden):
            nonlocal calls
            logits = torch.full((len(hidden), 64), -100.0)
            if calls < 2:
                logits[:, 10 + calls] = 100
            else:
                for row in range(3):
                    logits[row, 10 + row if row < accepted else 20 + row] = 100
            calls += 1
            return logits

        seen = []
        build = e.reference_store.build_plan

        def record(*args, **kwargs):
            plan = build(*args, **kwargs)
            seen.append(plan)
            return plan

        monkeypatch.setattr(e.reference_store, "build_plan", record)
        monkeypatch.setattr(e.speculative_runner, "_coda", controlled)
        output = e.step()[0]
        expected = [10 + row if row < accepted else 20 + row for row in range(accepted + 1)]
        assert output.token_ids[-(accepted + 1) :] == expected
        assert request.position == start + accepted + 1
        assert e.reference_store.outstanding == 0
        old = seen[0]
        with pytest.raises(RuntimeError, match="stale"):
            e.reference_store.read_reference(old, 0, 0)
        monkeypatch.setattr(e.speculative_runner, "_coda", original)
        e.step()
        assert seen[-1].incarnations[0] > old.incarnations[0]
        assert e.speculative_runner.stats.accepted_tokens >= accepted
    finally:
        e.close()


@pytest.mark.parametrize("draft", [None, 1, 2])
def test_failed_attention_releases_lease_and_allows_fresh_request(draft, monkeypatch):
    e = engine(draft=draft, retention=0.5)
    original = e.cache_manager._attend_prepared

    def fail(layer, kv, q):
        if getattr(kv, "reference_plan", None) is not None:
            raise RuntimeError("injected attention failure")
        return original(layer, kv, q)

    monkeypatch.setattr(e.cache_manager, "_attend_prepared", fail)
    try:
        e.add_request("r", [2, 3], SamplingParams(max_tokens=16, ignore_eos=True))
        with pytest.raises(RuntimeError, match="injected"):
            for _ in range(100):
                e.step()
        assert e.reference_store.outstanding == 0
        assert "r" not in e.scheduler.requests
        assert e.cache_manager.num_used_blocks == 0
        monkeypatch.setattr(e.cache_manager, "_attend_prepared", original)
        generate(e, count=1)
    finally:
        e.close()


@pytest.mark.gpu
@pytest.mark.parametrize("draft", [None, 1, 2])
def test_cuda_sparse_sync_executes_and_drains(draft):
    e = engine(draft=draft, retention=0.5, device="cuda", dtype=torch.bfloat16)
    try:
        generate(e)
        torch.cuda.synchronize()
        assert not bool(e.reference_store.counts.any())
    finally:
        e.close()


@pytest.mark.gpu
def test_cuda_ordinary_retirement_survives_boundary_allocator_reuse(monkeypatch):
    e = engine(device="cuda", dtype=torch.bfloat16)
    try:
        e.add_request("r", [2, 3], SamplingParams(max_tokens=16, ignore_eos=True))
        for _ in range(100):
            e.step()
            if e.reference_store.outstanding:
                break
        store = e.reference_store
        assert store.outstanding == 1
        slot = e.model_runner.attention_references["r"][2].slots[0]
        assert slot != 0
        torch.cuda.synchronize()
        store.counts[0] = 77
        torch.cuda.synchronize()
        clear = store._clear
        pointers = []

        def record(indices):
            assert torch.cuda.current_stream() == e.model_runner.core_stream
            pointers.append(indices.data_ptr())
            clear(indices)

        monkeypatch.setattr(store, "_clear", record)
        # Hold core while boundary allocations try to reuse any released index
        # storage. Correct stream ownership prevents that reuse before clearing.
        with torch.cuda.stream(e.model_runner.core_stream):
            torch.cuda._sleep(20_000_000)
        with torch.cuda.stream(e.model_runner.boundary_stream):
            e.model_runner._release_attention_reference("r")
            poison = [torch.zeros(1, device="cuda", dtype=torch.int64) for _ in range(32)]
        torch.cuda.synchronize()
        assert pointers and len(poison) == 32
        assert not bool(store.counts[slot].any())
        assert bool((store.counts[0] == 77).all())
    finally:
        e.close()


@pytest.mark.gpu
def test_cuda_close_drains_retirement_of_an_active_reference():
    e = engine(device="cuda", dtype=torch.bfloat16)
    e.add_request("r", [2, 3], SamplingParams(max_tokens=16, ignore_eos=True))
    for _ in range(100):
        e.step()
        if e.reference_store.outstanding:
            break
    assert e.reference_store.outstanding == 1
    e.close()
    assert e.reference_store.outstanding == 0
    assert e.cache_manager.num_used_blocks == 0
    assert e.model_runner.core_stream.query()
