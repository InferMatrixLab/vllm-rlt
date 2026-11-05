"""Ordinary and synchronous block decode through the shared sparse contract."""

from dataclasses import replace

import pytest
import torch

from tests.helpers import tiny_ouro_model
from tests.test_prefix_growth import finish
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SparseDecodeAttentionConfig,
    SpeculativeConfig,
)
from vllm_rlt.attention import BackendCapabilities, create_backend
from vllm_rlt.attention import sparse as sparse_attention
from vllm_rlt.attention.sparse import CAPTURE, CONSUME, DENSE
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.request import RequestPhase


def engine(
    *, draft=None, retention=1.0, prefix=False, device="cpu", dtype=torch.float32, **options
):
    values = dict(
        attention_backend="triton" if device == "cuda" else "torch",
        cache_config=CacheConfig(512, 2, enable_prefix_caching=prefix, incremental_allocation=True),
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
                assert plan.owners[row].request_id == request_id
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


def reference_payload(store, event):
    # read_reference returns only the retained selection, not unused capacity.
    return tuple(
        value.clone() for value in store.read_reference(event["plan"], event["layer"], event["row"])
    )


def trace_cpu_sparse(e, monkeypatch):
    """Observe real CPU correction calls and the request/slot that supplies them."""
    events, prefill_starts = [], {}
    active_rows = None
    attend = e.cache_manager._attend_prepared
    correct = sparse_attention.sparse_correction

    def correction(q, keys, values, mass, output, contribution):
        event = active_rows[q.data_ptr()]
        assert event["mode"] == CONSUME
        slot, layer = event["slot"], event["layer"]
        store = e.reference_store
        assert mass.data_ptr() == store.mass[slot, layer].data_ptr()
        assert output.data_ptr() == store.output[slot, layer].data_ptr()
        assert contribution.data_ptr() == store.contribution[slot, layer].data_ptr()
        selected = store.read_reference(event["plan"], layer, event["row"])[0]
        event["payload"] = tuple(value.clone() for value in (selected, mass, output, contribution))
        result = correct(q, keys, values, mass, output, contribution)
        assert keys.shape == values.shape
        event["correction_keys"] = keys.shape[1]
        return result

    def record(layer, kv, q):
        nonlocal active_rows
        if e.last_schedule.phase == RequestPhase.PREFILL:
            for item in e.last_schedule.requests:
                prefill_starts.setdefault(id(item.request), item.token_start)
        plan = getattr(kv, "reference_plan", None)
        if plan is None:
            return attend(layer, kv, q)
        rows = {}
        for row, mode in enumerate(plan.modes):
            if mode == DENSE:
                continue
            allocation, depth, position = kv.rows[row]
            rid = next(rid for rid, owner in kv.allocations if owner is allocation)
            request = e.scheduler.requests[rid]
            owner, token_position, lease = e.model_runner.attention_references[rid]
            assert owner is request and token_position == position
            assert position >= len(request.prompt_token_ids)
            assert mode == (CAPTURE if depth == 1 else CONSUME)
            assert lease.slots == (plan.slots[row],)
            assert plan.owners[row] is lease
            assert e.reference_store._bindings[plan.slots[row]] is lease
            rows[q[row].data_ptr()] = dict(
                request=request,
                position=position,
                depth=depth,
                layer=layer,
                mode=mode,
                slot=plan.slots[row],
                lease=lease,
                causal_length=int(kv.context_lengths[row]),
                plan=plan,
                row=row,
            )
        active_rows = rows
        try:
            result = attend(layer, kv, q)
        finally:
            active_rows = None
        for event in rows.values():
            slot = event["slot"]
            assert e.reference_store._bindings[slot] is event["lease"]
            count = e.reference_store.host_count(slot, layer)
            assert bool((e.reference_store.counts[slot, layer] == count).all())
            event["retained_count"] = count
            if "payload" not in event:
                event["payload"] = reference_payload(e.reference_store, event)
            events.append(event)
        return result

    monkeypatch.setattr(e.cache_manager, "_attend_prepared", record)
    monkeypatch.setattr(sparse_attention, "sparse_correction", correction)
    return events, prefill_starts


def assert_sparse_decode(events, request, retention, *, layers):
    params = request.sampling_params
    assert params.min_loops == params.max_loops == 4
    # Prefill emits the first output; only the next max_tokens - 1 inputs
    # traverse decode. Depth 0 is dense, depth 1 captures, depths 2/3 consume.
    start = len(request.prompt_token_ids)
    expected_steps = {
        (position, layer, depth)
        for position in range(start, start + params.max_tokens - 1)
        for layer in range(layers)
        for depth in range(1, params.max_loops)
    }
    seen = set()
    captured = {}
    for event in events:
        if event["request"] is not request:
            continue
        step = tuple(event[key] for key in ("position", "layer", "depth"))
        assert step not in seen, f"duplicate sparse step: {step}"
        seen.add(step)
        mode = event["mode"]
        assert mode == (CAPTURE if event["depth"] == 1 else CONSUME)
        assert event["plan"].owners[event["row"]] is event["lease"]
        assert event["lease"].request_id == request.request_id
        identity = (event["position"], event["layer"], event["slot"], id(event["lease"]))
        if mode == CAPTURE:
            captured[identity] = event["payload"]
        else:
            assert identity in captured
            for actual, expected in zip(event["payload"], captured[identity]):
                assert torch.equal(actual, expected)
        count, length = event["retained_count"], event["causal_length"]
        selected = event["payload"][0]
        assert selected.shape[-1] == count
        assert bool(((selected >= 0) & (selected < length)).all())
        assert bool((selected[:, 1:] > selected[:, :-1]).all())
        if retention == 1.0:
            assert count == length
        else:
            assert 0 < count < length
            if mode == CONSUME:
                assert event["correction_keys"] == count
    assert seen == expected_steps, (expected_steps - seen, seen - expected_steps)


def assert_prefix_requests_drained(e):
    assert not e.has_unfinished_requests()
    assert not e.scheduler.requests
    assert not e.cache_manager._allocations
    assert not e.model_runner.attention_references
    assert not e.model_runner.state_slots
    assert e.reference_store.outstanding == 0
    # Prefix-only pages remain cached and evictable; payload storage may persist.
    assert e.cache_manager.num_used_blocks == 0
    assert e.cache_manager.num_free_blocks == e.cache_manager.num_blocks


def cold_prefix_output(prompt, params, retention):
    # Every suffix gets a new engine, including empty prefix/reference state.
    cold = engine(retention=retention, prefix=False)
    try:
        cold.add_request("cold", prompt, params)
        cold.step()
        assert cold.last_schedule.requests[0].token_start == 0
        assert cold.cache_manager.prefix_hits == 0
        output = finish(cold)["cold"]
        assert_prefix_requests_drained(cold)
        return output
    finally:
        cold.close()


def cold_prefix_outputs(e, requests, *, cancel_and_replace=None, events=None):
    """Replay target arrivals/lifecycle on an independent cache-disabled engine.

    Matching arrivals does not imply equal execution batches: prefix hits and
    prefill chunking can still change the cold/warm schedules.
    """
    try:
        assert not e.cache_manager.enable_prefix_caching
        admitted = {
            rid: admit_prefix_request(e, rid, prompt, params, cached_tokens=0)[0]
            for rid, (prompt, params) in requests.items()
        }
        if cancel_and_replace is not None:
            victim_id, replacement_id, prompt, params = cancel_and_replace
            wait_for_unconsumed_capture(e, requests, events)
            assert {event["mode"] for event in events} == {CAPTURE}
            aborted = e.abort_request(victim_id)
            assert aborted.finished and aborted.finish_reason == "abort"
            del admitted[victim_id]
            assert e.reference_store.outstanding == len(admitted)
            admitted[replacement_id] = admit_prefix_request(
                e, replacement_id, prompt, params, cached_tokens=0
            )[0]
            wait_for_unconsumed_capture(e, [replacement_id], events)
            assert all(e.has_request(rid) for rid in admitted)
        outputs = finish(e)
        e.model_runner.synchronize()
        assert set(outputs) == set(admitted)
        assert e.cache_manager.prefix_hits == 0
        if events is not None:
            for request in admitted.values():
                assert_sparse_decode(events, request, 0.5, layers=e.model.config.num_hidden_layers)
        assert_prefix_requests_drained(e)
        return outputs
    finally:
        e.close()


def admit_prefix_request(e, rid, prompt, params, *, cached_tokens):
    cache = e.cache_manager
    prefix = cache.lookup_prefix(prompt)
    assert len(prefix) * cache.block_size == cached_tokens
    before = cache.prefix_hits
    e.add_request(rid, prompt, params)
    # Admit one arrival at a time so the counter delta belongs to this request,
    # even when another admitted request will share its prompt pages.
    e.scheduler._admit()
    assert cache.prefix_hits - before == cached_tokens
    request = e.scheduler.requests[rid]
    assert request.num_prefilled_tokens == cached_tokens
    for depth, table in enumerate(cache.plane_block_tables(rid)):
        assert table[: len(prefix)] == tuple(group[depth] for group in prefix)
    return request, prefix


def wait_for_unconsumed_capture(e, request_ids, events):
    for _ in range(100):
        e.step()
        states = [e.model_runner.attention_references.get(rid) for rid in request_ids]
        if all(state is not None for state in states):
            captured = [
                [event for event in events if event["lease"] is state[2]] for state in states
            ]
            if all(captured):
                for state, rows in zip(states, captured):
                    assert len(rows) == e.model.config.num_hidden_layers
                    assert {event["layer"] for event in rows} == set(range(len(rows)))
                    assert {event["mode"] for event in rows} == {CAPTURE}
                    assert e.reference_store._bindings[state[2].slots[0]] is state[2]
                return states
    pytest.fail("decode never yielded after CAPTURE and before CONSUME")


@pytest.mark.parametrize("retention", [0.5, 1.0])
@pytest.mark.parametrize("suffix,cached_tokens", [([6, 7, 8], 6), ([9, 10], 4)])
def test_prefix_hit_matches_its_own_cold_sparse_decode(
    retention, suffix, cached_tokens, monkeypatch
):
    prompt = [2, 3, 4, 5] + suffix
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    expected = cold_prefix_output(prompt, params, retention)
    e = engine(retention=retention, prefix=True)
    try:
        e.add_request("prime", [2, 3, 4, 5, 6, 7, 8], params)
        finish(e)
        assert_prefix_requests_drained(e)
        events, starts = trace_cpu_sparse(e, monkeypatch)
        request, prefix = admit_prefix_request(
            e, "target", prompt, params, cached_tokens=cached_tokens
        )
        pages = [block for group in prefix for block in group]
        keys = e.cache_manager.key_cache[pages].clone()
        values = e.cache_manager.value_cache[pages].clone()
        actual = finish(e)["target"]
        assert starts[id(request)] == cached_tokens
        assert actual.token_ids == expected.token_ids
        assert actual.exit_depths == expected.exit_depths == [4] * params.max_tokens
        assert actual.finish_reason == expected.finish_reason == "length"
        assert_sparse_decode(events, request, retention, layers=e.model.config.num_hidden_layers)
        assert torch.equal(e.cache_manager.key_cache[pages], keys)
        assert torch.equal(e.cache_manager.value_cache[pages], values)
        assert_prefix_requests_drained(e)
        assert e.cache_manager.lookup_prefix(prompt) == prefix
    finally:
        e.close()


def test_shared_prefix_keeps_live_sparse_references_request_local(monkeypatch):
    prompts = {"a": [2, 3, 4, 5, 6], "b": [2, 3, 4, 5, 9]}
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    expected = cold_prefix_outputs(
        engine(retention=0.5, prefix=False),
        {rid: (prompt, params) for rid, prompt in prompts.items()},
    )
    e = engine(retention=0.5, prefix=True)
    try:
        e.add_request("prime", prompts["a"], params)
        finish(e)
        assert_prefix_requests_drained(e)
        events, starts = trace_cpu_sparse(e, monkeypatch)
        requests = {
            rid: admit_prefix_request(e, rid, prompt, params, cached_tokens=4)[0]
            for rid, prompt in prompts.items()
        }
        states = wait_for_unconsumed_capture(e, prompts, events)
        assert e.reference_store.outstanding == 2
        assert states[0][2].slots != states[1][2].slots
        assert states[0][2] is not states[1][2]
        actual = finish(e)
        for rid, request in requests.items():
            assert starts[id(request)] == 4
            assert actual[rid].token_ids == expected[rid].token_ids
            assert actual[rid].exit_depths == expected[rid].exit_depths == [4] * params.max_tokens
            assert actual[rid].finish_reason == expected[rid].finish_reason == "length"
            assert_sparse_decode(events, request, 0.5, layers=e.model.config.num_hidden_layers)
        assert_prefix_requests_drained(e)
    finally:
        e.close()


@pytest.mark.parametrize("replacement_id", ["victim", "replacement"])
def test_prefix_hit_cancel_after_capture_reuses_slots_without_old_reference(
    replacement_id, monkeypatch
):
    prompt = [2, 3, 4, 5, 10]
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    cold = engine(retention=0.5, prefix=False)
    with monkeypatch.context() as patch:
        cold_events, _ = trace_cpu_sparse(cold, patch)
        expected = cold_prefix_outputs(
            cold,
            {"victim": ([2, 3, 4, 5, 9], params)},
            cancel_and_replace=("victim", replacement_id, prompt, params),
            events=cold_events,
        )[replacement_id]
    e = engine(retention=0.5, prefix=True)
    try:
        e.add_request("prime", [2, 3, 4, 5, 6], params)
        finish(e)
        assert_prefix_requests_drained(e)
        events, starts = trace_cpu_sparse(e, monkeypatch)
        victim, _ = admit_prefix_request(e, "victim", [2, 3, 4, 5, 9], params, cached_tokens=4)
        old_state = wait_for_unconsumed_capture(e, ["victim"], events)[0]
        assert starts[id(victim)] == 4
        assert {event["mode"] for event in events} == {CAPTURE}
        stale = events[0]
        old_slot = e.model_runner.state_slots["victim"]
        aborted = e.abort_request("victim")
        assert aborted.finished and aborted.finish_reason == "abort"
        assert_prefix_requests_drained(e)
        replacement, _ = admit_prefix_request(e, replacement_id, prompt, params, cached_tokens=4)
        new_state = wait_for_unconsumed_capture(e, [replacement_id], events)[0]
        assert replacement is not victim
        assert new_state[0] is replacement and old_state[0] is victim
        assert new_state[2].slots == old_state[2].slots
        assert new_state[2] is not old_state[2]
        assert e.model_runner.state_slots[replacement_id] == old_slot
        with pytest.raises(RuntimeError, match="stale"):
            e.reference_store.read_reference(stale["plan"], stale["layer"], stale["row"])
        with pytest.raises(RuntimeError, match="stale"):
            e.reference_store.consume(stale["plan"], stale["layer"])
        assert all(event["mode"] == CAPTURE for event in events if event["request"] is replacement)
        for event in events:
            if event["request"] is replacement:
                for actual, saved in zip(
                    reference_payload(e.reference_store, event), event["payload"]
                ):
                    assert torch.equal(actual, saved)
        actual = finish(e)[replacement_id]
        assert starts[id(replacement)] == 4
        assert actual.token_ids == expected.token_ids
        assert actual.exit_depths == expected.exit_depths == [4] * params.max_tokens
        assert actual.finish_reason == expected.finish_reason == "length"
        assert_sparse_decode(events, replacement, 0.5, layers=e.model.config.num_hidden_layers)
        assert_prefix_requests_drained(e)
        assert e.cache_manager.lookup_prefix(prompt)
    finally:
        e.close()


def test_sparse_trace_rejects_missing_duplicate_and_foreign_steps(monkeypatch):
    e = engine(retention=0.5, prefix=True)
    try:
        events, _ = trace_cpu_sparse(e, monkeypatch)
        params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
        e.add_request("target", [2, 3, 4, 5, 6], params)
        request = e.scheduler.requests["target"]
        finish(e)
        layers = e.model.config.num_hidden_layers
        assert_sparse_decode(events, request, 0.5, layers=layers)
        last_position = len(request.prompt_token_ids) + params.max_tokens - 2
        consume = next(event for event in events if event["mode"] == CONSUME)
        incomplete = (
            [event for event in events if event["position"] != last_position],
            [event for event in events if event["layer"] != layers - 1],
            [event for event in events if event is not consume],
            [*events, events[-1]],
            [dict(event, request=object()) if event is consume else event for event in events],
        )
        for corrupted in incomplete:
            with pytest.raises(AssertionError):
                assert_sparse_decode(corrupted, request, 0.5, layers=layers)
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
    from vllm_rlt.worker.attention_reference import SparseDecodeReferenceStore

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
    store = SparseDecodeReferenceStore(
        slots=rows, layers=1, heads=heads, head_dim=dim, key_bucket=length, device="cuda"
    )
    ids = [str(i) for i in range(rows)]
    owners = []
    for rid in ids:
        assert cache.allocate(rid, length)
        keys = torch.randn(length, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        cache.write(0, [rid] * length, [0] * length, list(range(length)), keys, values)
        lease = store.reserve(rid, 0, 1)
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
        # Consume rows read the reference: their slots stay bound until release.
        assert store.outstanding == len(owners)
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
        lease = store.reserve("r", 0, 1)
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
        assert seen[-1].owners[0] is not old.owners[0]
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
