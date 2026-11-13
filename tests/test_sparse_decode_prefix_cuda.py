"""CUDA prefix reuse, request isolation and sparse-reference retirement.

The small synchronous cases inspect real payloads on the owning stream. The
larger cases use dispatch-only observation without layer waits.
"""

import pytest
import torch

from tests.helpers import tiny_ouro_config, tiny_ouro_model
from tests.test_prefix_growth import finish
from tests.test_sparse_decode_sync import (
    admit_prefix_request,
    assert_prefix_requests_drained,
    assert_sparse_decode,
    cold_prefix_outputs,
    reference_payload,
    wait_for_unconsumed_capture,
)
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SparseDecodeAttentionConfig,
)
from vllm_rlt.attention import sparse as sparse_attention
from vllm_rlt.attention.sparse import CAPTURE, CONSUME, DENSE
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestPhase

pytestmark = pytest.mark.gpu


def cuda_prefix_engine(implementation, retention=0.5, *, larger=False, prefix=True):
    if larger:
        torch.manual_seed(123)
        model = OuroForCausalLM(
            tiny_ouro_config(
                vocab_size=256,
                hidden_size=256,
                intermediate_size=768,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=32,
                max_position_embeddings=2048,
            )
        ).to(device="cuda", dtype=torch.bfloat16)
    else:
        model = tiny_ouro_model("cuda", torch.bfloat16)
    return StageEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(
            1536 if larger else 256,
            16 if larger else 2,
            enable_prefix_caching=prefix,
            incremental_allocation=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=128 if larger else 16,
            prefill_chunk_size=64 if larger else 3,
        ),
        execution_config=ExecutionConfig(
            async_scheduling=False,
            multi_stream=True,
            static_buffers=True,
            cuda_graphs=False,
            use_uva=True,
        ),
        sparse_decode_attention_config=SparseDecodeAttentionConfig(
            enabled=True, key_retention=retention, implementation=implementation
        ),
    )


def assert_output(actual, expected, params):
    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths == [4] * params.max_tokens
    assert actual.finish_reason == expected.finish_reason == "length"


def cold_output(implementation, prompt, params, **options):
    # Every target gets its own fresh model, KV cache and reference pool.
    e = cuda_prefix_engine(implementation, prefix=False, **options)
    try:
        admit_prefix_request(e, "cold", prompt, params, cached_tokens=0)
        e.step()
        assert e.last_schedule.requests[0].token_start == 0
        output = finish(e)["cold"]
        e.model_runner.synchronize()
        assert_prefix_requests_drained(e)
        return output
    finally:
        e.close()


def snapshot_prefix(e, prefix):
    pages = [block for group in prefix for block in group]
    with torch.cuda.stream(e.model_runner.core_stream):
        return (
            pages,
            e.cache_manager.key_cache[pages].clone(),
            e.cache_manager.value_cache[pages].clone(),
        )


def assert_prefix_unchanged(e, snapshot):
    e.model_runner.synchronize()
    pages, keys, values = snapshot
    assert torch.equal(e.cache_manager.key_cache[pages], keys)
    assert torch.equal(e.cache_manager.value_cache[pages], values)


def trace_cuda_sparse(e, monkeypatch):
    """Check real synchronous CUDA arithmetic; this is not an async race observer."""
    from vllm_rlt.kernels import sparse_decode

    events, starts = [], {}
    active = []
    selected_events = []
    store = e.reference_store
    attend = e.cache_manager._attend_prepared
    scores = sparse_attention.fused_selected_scores
    values = sparse_attention.fused_selected_values
    fused = sparse_decode.fused_paged_attention

    def selected_scores(q, cache, selected, tables, *, layer, block_size):
        nonlocal selected_events
        assert cache.data_ptr() == e.cache_manager.key_cache[:, layer].data_ptr()
        selected_events = []
        for row, table in enumerate(tables):
            matches = [event for event in active if torch.equal(table, event["table"])]
            assert len(matches) == 1
            event = matches[0]
            assert event["mode"] == CONSUME and event["layer"] == layer
            assert torch.equal(q[row], event["q"])
            assert torch.equal(selected[row], event["payload"][0])
            assert 0 < selected.shape[-1] < event["causal_length"]
            event["correction_keys"] = selected.shape[-1]
            selected_events.append(event)
        result = scores(q, cache, selected, tables, layer=layer, block_size=block_size)
        for event, row_scores in zip(selected_events, result):
            event["scores"] = row_scores
        return result

    def selected_values(probabilities, cache, selected, tables, *, layer, block_size):
        assert cache.data_ptr() == e.cache_manager.value_cache[:, layer].data_ptr()
        assert len(selected_events) == len(probabilities)
        for row, event in enumerate(selected_events):
            assert torch.equal(tables[row], event["table"])
            assert torch.equal(selected[row], event["payload"][0])
            expected = torch.softmax(event["scores"] * event["q"].shape[-1] ** -0.5, dim=-1)
            torch.testing.assert_close(probabilities[row], expected, rtol=0, atol=0)
        result = values(probabilities, cache, selected, tables, layer=layer, block_size=block_size)
        for event, new in zip(selected_events, result):
            event["new_contribution"] = new
        return result

    def fused_attention(q, keys, values, tables, lengths, plan, *, layer, **kwargs):
        assert active and all(event["plan"] is plan for event in active)
        assert tuple(plan.slot_ids.tolist()) == plan.slots
        assert keys.data_ptr() == e.cache_manager.key_cache[:, layer].data_ptr()
        assert values.data_ptr() == e.cache_manager.value_cache[:, layer].data_ptr()
        result = fused(q, keys, values, tables, lengths, plan, layer=layer, **kwargs)
        for event in active:
            event["fused"] = True
            if event["mode"] != CONSUME:
                continue
            selected, mass, reference, contribution = event["payload"]
            assert torch.equal(q[event["row"]], event["q"])
            assert torch.equal(tables[event["row"]], event["table"])
            retained_keys, retained_values = sparse_attention.gather_retained(
                keys, values, event["table"], selected, keys.shape[1], q.shape[1] // keys.shape[2]
            )
            expected = sparse_attention.sparse_correction(
                event["q"], retained_keys, retained_values, mass, reference, contribution
            )
            # Fused FP32 reduction order can move a BF16 rounding step; this
            # matches the existing fused-versus-scalar kernel test's bound.
            torch.testing.assert_close(result[event["row"]].float(), expected, rtol=2e-2, atol=2e-2)
            event["correction_keys"] = selected.shape[-1]
        return result

    def record(layer, kv, q):
        nonlocal active
        if e.last_schedule.phase == RequestPhase.PREFILL:
            for item in e.last_schedule.requests:
                starts.setdefault(
                    id(item.request), (item.token_start, item.token_start + item.token_count)
                )
        plan = kv.reference_plan
        if plan is None or plan.dense():
            return attend(layer, kv, q)
        assert q.is_cuda and plan.store is store
        stream = torch.cuda.current_stream(q.device)
        assert stream == e.model_runner.core_stream == store.stream
        stream.synchronize()
        active = []
        batch = tuple(item.request.request_id for item in e.last_schedule.requests)
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
            slot = plan.slots[row]
            assert store._bindings[slot] is lease
            event = dict(
                request=request,
                position=position,
                depth=depth,
                layer=layer,
                mode=mode,
                slot=slot,
                lease=lease,
                causal_length=int(kv.context_lengths[row]),
                plan=plan,
                row=row,
                q=q[row],
                table=kv.block_tables[row],
                batch=batch,
                fused=False,
            )
            if mode == CONSUME:
                event["payload"] = reference_payload(store, event)
            active.append(event)
        try:
            result = attend(layer, kv, q)
            stream.synchronize()
            for event in active:
                slot, row = event["slot"], event["row"]
                assert store._bindings[slot] is event["lease"]
                count = int(store.counts[slot, layer, 0])
                assert bool((store.counts[slot, layer] == count).all())
                event["retained_count"] = count
                if event["mode"] == CAPTURE:
                    event["payload"] = reference_payload(store, event)
                elif plan.key_retention < 1.0 and not event["fused"]:
                    _, mass, reference, contribution = event["payload"]
                    expected = (
                        reference
                        - contribution
                        + mass.unsqueeze(-1) * event.pop("new_contribution")
                    )
                    torch.testing.assert_close(
                        result[row], expected.to(result.dtype), rtol=0, atol=0
                    )
                    del event["scores"]
                # Full retention and mixed dense/sparse submissions use batched;
                # fused kernels accept only uniform capture or consume rows.
                assert event["fused"] == (
                    store.implementation == "fused"
                    and plan.key_retention < 1.0
                    and set(plan.modes) in ({CAPTURE}, {CONSUME})
                )
                del event["q"], event["table"]
                events.append(event)
        finally:
            active = []
        return result

    monkeypatch.setattr(e.cache_manager, "_attend_prepared", record)
    monkeypatch.setattr(sparse_attention, "fused_selected_scores", selected_scores)
    monkeypatch.setattr(sparse_attention, "fused_selected_values", selected_values)
    monkeypatch.setattr(sparse_decode, "fused_paged_attention", fused_attention)
    return events, starts


@pytest.mark.parametrize("implementation", ["batched", "fused"])
@pytest.mark.parametrize("retention", [0.5, 1.0])
@pytest.mark.parametrize("suffix,cached_tokens", [([6, 7, 8], 6), ([9, 10], 4)])
def test_cuda_prefix_hit_matches_its_own_cold_decode(
    implementation, retention, suffix, cached_tokens, monkeypatch
):
    prompt = [2, 3, 4, 5] + suffix
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    expected = cold_output(implementation, prompt, params, retention=retention)
    e = cuda_prefix_engine(implementation, retention)
    try:
        e.add_request("prime", [2, 3, 4, 5, 6, 7, 8], params)
        finish(e)
        assert_prefix_requests_drained(e)
        request, prefix = admit_prefix_request(
            e, "target", prompt, params, cached_tokens=cached_tokens
        )
        snapshot = snapshot_prefix(e, prefix)
        events, starts = trace_cuda_sparse(e, monkeypatch)
        actual = finish(e)["target"]
        assert starts[id(request)] == (cached_tokens, min(cached_tokens + 3, len(prompt)))
        assert_output(actual, expected, params)
        assert_sparse_decode(events, request, retention, layers=e.model.config.num_hidden_layers)
        assert_prefix_unchanged(e, snapshot)
        assert_prefix_requests_drained(e)
        assert e.cache_manager.lookup_prefix(prompt) == prefix
    finally:
        e.close()


@pytest.mark.parametrize("implementation", ["batched", "fused"])
def test_cuda_shared_prefix_keeps_simultaneous_references_isolated(implementation, monkeypatch):
    prompts = {"a": [2, 3, 4, 5, 6], "b": [2, 3, 4, 5, 9]}
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    expected = cold_prefix_outputs(
        cuda_prefix_engine(implementation, prefix=False),
        {rid: (prompt, params) for rid, prompt in prompts.items()},
    )
    e = cuda_prefix_engine(implementation)
    try:
        e.add_request("prime", prompts["a"], params)
        finish(e)
        events, starts = trace_cuda_sparse(e, monkeypatch)
        admitted = {
            rid: admit_prefix_request(e, rid, prompt, params, cached_tokens=4)
            for rid, prompt in prompts.items()
        }
        assert admitted["a"][1] == admitted["b"][1]
        snapshot = snapshot_prefix(e, admitted["a"][1])
        assert all(e.cache_manager._refs[page] >= 3 for page in snapshot[0])
        states = wait_for_unconsumed_capture(e, prompts, events)
        assert e.reference_store.outstanding == 2
        assert states[0][2].slots != states[1][2].slots
        assert states[0][2] is not states[1][2]
        for layer in range(e.reference_store.layers):
            captured = [event for event in events if event["layer"] == layer]
            assert {event["request"].request_id for event in captured} == set(prompts)
            assert any(
                not torch.equal(a, b)
                for a, b in zip(captured[0]["payload"], captured[1]["payload"])
            )
        actual = finish(e)
        for mode in (CAPTURE, CONSUME):
            assert any(
                event["mode"] == mode and set(event["batch"]) == set(prompts) for event in events
            )
        for rid, (request, _) in admitted.items():
            assert starts[id(request)] == (4, 5)
            assert_output(actual[rid], expected[rid], params)
            assert_sparse_decode(events, request, 0.5, layers=e.model.config.num_hidden_layers)
        assert_prefix_unchanged(e, snapshot)
        assert_prefix_requests_drained(e)
    finally:
        e.close()


@pytest.mark.parametrize("implementation", ["batched", "fused"])
@pytest.mark.parametrize("replacement_id", ["victim", "replacement"])
def test_cuda_prefix_cancel_between_capture_and_consume_reuses_slot(
    implementation, replacement_id, monkeypatch
):
    prompt = [2, 3, 4, 5, 10]
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    survivor_prompt = [2, 3, 4, 5, 11]
    survivor_params = SamplingParams(max_tokens=12, min_loops=4, max_loops=4, ignore_eos=True)
    cold = cuda_prefix_engine(implementation, prefix=False)
    with monkeypatch.context() as patch:
        cold_events, _ = trace_cuda_sparse(cold, patch)
        expected = cold_prefix_outputs(
            cold,
            {
                "victim": ([2, 3, 4, 5, 9], params),
                "survivor": (survivor_prompt, survivor_params),
            },
            cancel_and_replace=("victim", replacement_id, prompt, params),
            events=cold_events,
        )
    e = cuda_prefix_engine(implementation)
    try:
        e.add_request("prime", [2, 3, 4, 5, 6], params)
        finish(e)
        events, starts = trace_cuda_sparse(e, monkeypatch)
        victim, prefix = admit_prefix_request(e, "victim", [2, 3, 4, 5, 9], params, cached_tokens=4)
        survivor, survivor_prefix = admit_prefix_request(
            e, "survivor", survivor_prompt, survivor_params, cached_tokens=4
        )
        assert survivor_prefix == prefix
        snapshot = snapshot_prefix(e, prefix)
        assert all(e.cache_manager._refs[page] >= 3 for page in snapshot[0])
        old, survivor_state = wait_for_unconsumed_capture(e, ["victim", "survivor"], events)
        survivor_slot = survivor_state[2].slots[0]
        assert survivor_state[2].slots != old[2].slots
        assert starts[id(victim)] == (4, 5)
        assert {event["mode"] for event in events} == {CAPTURE}
        stale = next(event for event in events if event["request"] is victim)
        state_slot = e.model_runner.state_slots["victim"]
        store = e.reference_store
        with torch.cuda.stream(e.model_runner.core_stream):
            survivor_metadata = {
                name: getattr(store, name)[survivor_slot].clone() for name in ("counts", "mass")
            }
        binding = store._bindings[survivor_slot]
        cleared = []
        clear = e.reference_store._clear

        def retirement(indices):
            assert torch.cuda.current_stream() == e.model_runner.core_stream
            clear(indices)
            done = torch.cuda.Event()
            done.record(e.model_runner.core_stream)
            cleared.append(done)

        monkeypatch.setattr(e.reference_store, "_clear", retirement)
        aborted = e.abort_request("victim")
        assert aborted.finished and aborted.finish_reason == "abort"
        assert len(cleared) == 1
        assert store.outstanding == 1 and e.has_request("survivor")
        assert e.model_runner.attention_references["survivor"] is survivor_state
        assert store._bindings[survivor_slot] is binding
        with torch.cuda.stream(e.model_runner.core_stream):
            # Inspect immediately after abort, before any further scheduler step
            # can legitimately update B's payload or advance its token.
            for name, saved in survivor_metadata.items():
                assert torch.equal(getattr(store, name)[survivor_slot], saved)
            for event in events:
                if event["request"] is survivor:
                    # Its old batch plan also contains retired A and must be
                    # rejected as a whole. Inspect B's still-bound slot instead.
                    layer = event["layer"]
                    selected = event["payload"][0]
                    payload = (
                        store.selected[survivor_slot, layer, :, : selected.shape[-1]].long(),
                        store.mass[survivor_slot, layer],
                        store.output[survivor_slot, layer],
                        store.contribution[survivor_slot, layer],
                    )
                    for actual, saved in zip(payload, event["payload"]):
                        assert torch.equal(actual, saved)
            assert old[2].slots[0] not in store._bindings
            assert not bool(store.counts[old[2].slots[0]].any())
        assert_prefix_unchanged(e, snapshot)
        replacement, _ = admit_prefix_request(e, replacement_id, prompt, params, cached_tokens=4)
        new = wait_for_unconsumed_capture(e, [replacement_id], events)[0]
        assert e.has_request("survivor")
        assert cleared[0].query()  # The new capture completed after retirement on core.
        assert replacement is not victim and old[0] is victim and new[0] is replacement
        assert new[2].slots == old[2].slots
        assert new[2] is not old[2]
        assert e.model_runner.state_slots[replacement_id] == state_slot
        with torch.cuda.stream(e.model_runner.core_stream):
            with pytest.raises(RuntimeError, match="stale"):
                e.reference_store.read_reference(stale["plan"], stale["layer"], stale["row"])
            with pytest.raises(RuntimeError, match="stale"):
                e.reference_store.consume(stale["plan"], stale["layer"])
            assert all(
                event["mode"] == CAPTURE for event in events if event["request"] is replacement
            )
            for event in events:
                if event["request"] is replacement:
                    for actual, saved in zip(
                        reference_payload(e.reference_store, event), event["payload"]
                    ):
                        assert torch.equal(actual, saved)
        actual = finish(e)
        assert starts[id(replacement)] == (4, 5)
        assert_output(actual[replacement_id], expected[replacement_id], params)
        assert_output(actual["survivor"], expected["survivor"], survivor_params)
        assert_sparse_decode(events, replacement, 0.5, layers=e.model.config.num_hidden_layers)
        assert_sparse_decode(events, survivor, 0.5, layers=e.model.config.num_hidden_layers)
        assert all(event["mode"] == CAPTURE for event in events if event["request"] is victim)
        assert_prefix_unchanged(e, snapshot)
        assert_prefix_requests_drained(e)
    finally:
        e.close()


def trace_dispatch(e, monkeypatch):
    """Record host plan ownership without device reads or extra synchronization."""
    from vllm_rlt.kernels import sparse_decode

    calls, starts = [], {}
    attend = e.cache_manager._attend_prepared

    def prefill(layer, kv, q):
        if e.last_schedule.phase == RequestPhase.PREFILL:
            for item in e.last_schedule.requests:
                starts.setdefault(id(item.request), item.token_start)
        return attend(layer, kv, q)

    def observe(function, implementation):
        def call(q, keys, values, tables, lengths, plan, *, layer, **kwargs):
            assert q.is_cuda and q.dtype == torch.bfloat16
            assert torch.cuda.current_stream(q.device) == e.model_runner.core_stream
            bindings = tuple(plan.store._bindings[slot] for slot in plan.slots if slot >= 0)
            assert all(
                plan.store._bindings[slot] is owner
                for slot, owner in zip(plan.slots, plan.owners)
                if slot >= 0
            )
            calls.append((implementation, plan.modes, plan.slots, bindings))
            return function(q, keys, values, tables, lengths, plan, layer=layer, **kwargs)

        return call

    monkeypatch.setattr(e.cache_manager, "_attend_prepared", prefill)
    monkeypatch.setattr(
        sparse_attention,
        "batched_paged_attention",
        observe(sparse_attention.batched_paged_attention, "batched"),
    )
    monkeypatch.setattr(
        sparse_decode,
        "fused_paged_attention",
        observe(sparse_decode.fused_paged_attention, "fused"),
    )
    return calls, starts


@pytest.mark.parametrize("implementation", ["batched", "fused"])
@pytest.mark.parametrize("length,concurrency", [(63, 2), (257, 3), (1025, 4)])
def test_cuda_larger_ragged_decode_matches_each_cold_prefix(
    implementation, length, concurrency, monkeypatch
):
    shared = [2 + (i * 17) % 240 for i in range(length - 1)]
    prompts = {str(i): shared + [11 + i] for i in range(concurrency)}
    params = {
        rid: SamplingParams(max_tokens=12 + int(rid), min_loops=4, max_loops=4, ignore_eos=True)
        for rid in prompts
    }
    expected = cold_prefix_outputs(
        cuda_prefix_engine(implementation, larger=True, prefix=False),
        {rid: (prompt, params[rid]) for rid, prompt in prompts.items()},
    )
    e = cuda_prefix_engine(implementation, larger=True)
    try:
        assert sum(p.numel() for p in e.model.parameters()) > 1_000_000
        e.add_request("prime", prompts["0"], params["0"])
        finish(e)
        calls, starts = trace_dispatch(e, monkeypatch)
        cached_tokens = (length - 1) // 16 * 16
        admitted = {
            rid: admit_prefix_request(e, rid, prompt, params[rid], cached_tokens=cached_tokens)
            for rid, prompt in prompts.items()
        }
        prefix = admitted["0"][1]
        assert all(item[1] == prefix for item in admitted.values())
        snapshot = snapshot_prefix(e, prefix)
        assert all(e.cache_manager._refs[page] >= concurrency + 1 for page in snapshot[0])
        actual = finish(e)
        for rid, (request, _) in admitted.items():
            assert starts[id(request)] == cached_tokens
            assert_output(actual[rid], expected[rid], params[rid])
        for mode in (CAPTURE, CONSUME):
            assert any(
                impl == implementation
                and set(modes) == {mode}
                and {binding.request_id for binding in bindings} == set(prompts)
                for impl, modes, _, bindings in calls
            )
        assert_prefix_unchanged(e, snapshot)
        assert_prefix_requests_drained(e)
    finally:
        e.close()
