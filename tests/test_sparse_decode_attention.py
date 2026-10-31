"""Sparse decode attention: reference lifecycle and plan contracts (RFC #156).

Covers the asynchronous self-speculation cell of the sparse decode attention
column: one reference slot per live token, plan modes for the draft depths,
GPU acceptance driving retirement of rejected candidates and their dependent
descendants, same-position replacement through incarnations, delayed
completion and reference-slot reuse.
"""

import math

import pytest
import torch

from tests.helpers import (
    bootstrap_speculation,
    make_cpu_cache,
    make_speculative_engine,
    tiny_ouro_model,
)
from vllm_rlt import SamplingParams, SparseDecodeAttentionConfig
from vllm_rlt.attention import sparse as sparse_module
from vllm_rlt.attention.sparse import (
    CAPTURE,
    CONSUME,
    DENSE,
    REFERENCE_DEPTH,
    SparseDecodePlan,
    reference_mode,
    reference_statistics,
    retained_count,
    sparse_correction,
    sparse_paged_attention,
)
from vllm_rlt.worker.attention_reference import AttentionReferenceStore
from vllm_rlt.worker.metadata import MetadataBuilder

tiny = tiny_ouro_model
bootstrap = bootstrap_speculation


def drain(engine):
    outputs = {}
    for _ in range(1000):
        if not engine.has_unfinished_requests():
            return outputs
        for output in engine.step():
            outputs[output.request_id] = output
    pytest.fail("engine failed to drain")


def make_store(**overrides):
    values = dict(slots=6, layers=2, heads=2, head_dim=4, key_bucket=8, device="cpu")
    values.update(overrides)
    return AttentionReferenceStore(**values)


def fill_payload(store, slots, value=1.0):
    for slot in slots:
        store.mass[slot] = value
        store.output[slot] = value
        store.contribution[slot] = value
        store.selected[slot] = 1
        store.counts[slot] = 1


# ---------------------------------------------------------------------------
# Plan and arithmetic contracts (RFC #156 sections 3.2 and 6.3)
# ---------------------------------------------------------------------------


def test_reference_mode_follows_reference_depth():
    # Loop 2 (depth index 1) captures; later loops consume; loop 1 stays dense.
    assert reference_mode(0) == DENSE
    assert reference_mode(REFERENCE_DEPTH) == CAPTURE
    assert reference_mode(2) == CONSUME
    assert reference_mode(3) == CONSUME


def test_plan_groups_rows_by_mode():
    plan = SparseDecodePlan(
        modes=(DENSE, CAPTURE, CAPTURE, CONSUME),
        slots=(-1, 3, 4, 5),
        incarnations=(0, 1, 1, 1),
        key_retention=0.5,
        key_bucket=None,
    )
    assert plan.groups() == {DENSE: [0], CAPTURE: [1, 2], CONSUME: [3]}
    assert not plan.dense()
    assert plan.reference_depth == REFERENCE_DEPTH


def test_retained_count_follows_flashloop_ceil_rule():
    # FlashLoop aeaadee select_decode_keys: k = min(N, max(1, ceil(N * r))).
    assert retained_count(13, 0.1) == 2
    assert retained_count(10, 0.1) == 1
    assert retained_count(10, 0.5) == 5
    assert retained_count(9, 0.5) == 5
    assert retained_count(1, 0.1) == 1
    assert retained_count(4, 1.0) == 4
    assert retained_count(0, 0.5) == 0
    with pytest.raises(ValueError, match="key_retention"):
        retained_count(4, 1.5)


def test_reference_statistics_fail_when_capacity_cannot_hold_retained_keys():
    # Capacity never truncates the algorithm: a count above it fails loudly.
    q = torch.ones(1, 1)
    keys = torch.log(torch.tensor([0.2, 0.3, 0.5])).reshape(3, 1, 1)
    values = torch.ones(3, 1, 1)
    with pytest.raises(ValueError, match="cannot hold"):
        reference_statistics(q, keys, values, key_retention=0.5, key_bucket=1)
    # The same query passes with sufficient capacity.
    output, _, _, positions = reference_statistics(q, keys, values, key_retention=0.5, key_bucket=2)
    assert positions.shape[1] == 2
    assert output.shape == (1, 1)


def test_reference_statistics_match_rfc_worked_example():
    # RFC #156 section 3.2: probs [0.1, 0.3, 0.6], values [10, 20, 30];
    # full output 25; retaining the last two keys gives m=0.9, C_ref=24.
    # A later loop with local softmax [0.75, 0.25] over values [20, 40]
    # corrects to 25 - 24 + 0.9 * (0.75*20 + 0.25*40) = 23.5.
    q = torch.ones(1, 1)
    keys = torch.log(torch.tensor([0.1, 0.3, 0.6])).reshape(3, 1, 1)
    values = torch.tensor([10.0, 20.0, 30.0]).reshape(3, 1, 1)
    output, contribution, mass, positions = reference_statistics(q, keys, values, key_retention=0.5)
    assert output.item() == pytest.approx(25.0)
    assert mass.item() == pytest.approx(0.9)
    assert contribution.item() == pytest.approx(24.0)
    assert positions.tolist() == [[1, 2]]

    later_keys = torch.log(torch.tensor([3.0, 1.0])).reshape(1, 2, 1)
    later_values = torch.tensor([20.0, 40.0]).reshape(1, 2, 1)
    corrected = sparse_correction(q, later_keys, later_values, mass, output, contribution)
    assert corrected.item() == pytest.approx(23.5)


def test_full_retention_matches_dense_attention():
    torch.manual_seed(7)
    length, kv_heads, heads, dim = 9, 2, 4, 8
    q = torch.randn(heads, dim)
    keys = torch.randn(length, kv_heads, dim)
    values = torch.randn(length, kv_heads, dim)
    output, contribution, mass, positions = reference_statistics(q, keys, values, key_retention=1.0)
    # Full retention keeps every key, so the correction reproduces the dense
    # reference output exactly.
    dense = reference_statistics(q, keys, values, key_retention=1.0)[0]
    groups = heads // kv_heads
    corrected = sparse_correction(
        q,
        keys.permute(1, 0, 2).repeat_interleave(groups, dim=0),
        values.permute(1, 0, 2).repeat_interleave(groups, dim=0),
        mass,
        output,
        contribution,
    )
    assert positions.shape[1] == length
    assert torch.allclose(corrected, dense, atol=1e-5)
    assert torch.allclose(output, dense, atol=1e-5)


@pytest.mark.parametrize("retention, expected", [(0.5, 23.5), (1.0, 40.0)])
def test_paged_oracle_forces_correction_without_backend_shortcut(retention, expected):
    store = make_store(slots=1, layers=1, heads=1, head_dim=1, key_bucket=3)
    lease = store.reserve("r", 0, 1, generation=1)
    q = torch.ones(1, 1, 1)
    keys = torch.zeros(1, 4, 1, 1)
    values = torch.zeros_like(keys)
    keys[0, :3, 0, 0] = torch.log(torch.tensor([0.1, 0.3, 0.6]))
    values[0, :3, 0, 0] = torch.tensor([10.0, 20.0, 30.0])
    tables, lengths = torch.tensor([[0]], dtype=torch.int32), torch.tensor([3])
    capture = store.build_plan([(lease, 0)], [CAPTURE], key_retention=retention)
    # No dense callback: exercise the actual paged Torch oracle even at r=1.
    output = sparse_paged_attention(q, keys, values, tables, lengths, capture, layer=0)
    assert output.item() == pytest.approx(25.0)
    keys[0, :3, 0, 0] = torch.log(torch.tensor([1.0, 3.0, 1.0]))
    values[0, :3, 0, 0] = torch.tensor([100.0, 20.0, 40.0])
    consume = store.build_plan([(lease, 0)], [CONSUME], key_retention=retention)
    output = sparse_paged_attention(q, keys, values, tables, lengths, consume, layer=0)
    assert output.item() == pytest.approx(expected)
    assert store.uses.item() == 1
    store.release(lease)


# ---------------------------------------------------------------------------
# Store lifecycle (RFC #156 section 5)
# ---------------------------------------------------------------------------


def test_reserve_uses_one_slot_per_live_token():
    store = make_store()
    lease = store.reserve("r", 0, 3, generation=1)
    assert len(lease) == 3
    assert len(set(lease.slots)) == 3
    assert lease.incarnations == (1, 1, 1)
    assert store.outstanding == 3
    assert len(store.free) == 3


def test_release_requires_owner_and_retires_metadata():
    store = make_store()
    lease = store.reserve("r", 0, 2, generation=1)
    fill_payload(store, lease.slots)
    store.release(lease)
    for slot in lease.slots:
        # Readers are gated by counts/mass/valid. The retained index, output and
        # contribution buffers stay allocated and are overwritten by the next
        # capture's used prefix, so retiring never writes the whole bucket.
        assert int(store.counts[slot].sum()) == 0
        assert store.mass[slot].sum() == 0
        assert not bool(store.valid[slot].any())
    assert store.outstanding == 0
    # A retired lease cannot release its slots a second time.
    with pytest.raises(RuntimeError, match="different owner"):
        store.release(lease)


def test_accept_invalidates_rejected_candidates_and_descendants():
    store = make_store()
    lease = store.reserve("r", 0, 3, generation=1)
    fill_payload(store, lease.slots)
    store.accept([lease], torch.tensor([1]))
    accepted, middle, rejected = lease.slots
    # Rows 0..1 survive the accepted prefix; the rejected candidate and every
    # descendant stop being readable.
    assert int(store.counts[accepted].max()) > 0 and int(store.counts[middle].max()) > 0
    assert int(store.counts[rejected].max()) == 0
    assert store.mass[rejected].sum() == 0
    assert not bool(store.valid[rejected].any())
    assert int(store.accepted[accepted]) == 1
    rejected_plan = store.build_plan([(lease, 2)], [CONSUME], key_retention=0.5)
    with pytest.raises(RuntimeError, match="no captured reference"):
        store.read_reference(rejected_plan, layer=0, row=0)


def test_accept_keeps_root_when_every_candidate_is_rejected():
    store = make_store()
    lease = store.reserve("r", 0, 3, generation=1)
    fill_payload(store, lease.slots)
    store.accept([lease], torch.tensor([0]))
    assert store.mass[lease.slots[0]].sum() > 0
    for slot in lease.slots[1:]:
        assert store.mass[slot].sum() == 0


def test_capture_and_consume_mark_layer_state():
    store = make_store()
    lease = store.reserve("r", 0, 2, generation=1)
    plan = store.build_plan([(lease, 0), (lease, 1)], [CAPTURE, CONSUME], key_retention=0.5)
    assert not bool(store.valid[lease.slots[0], 0])
    store.capture(plan, layer=0)
    store.capture(plan, layer=1)
    assert bool(store.valid[lease.slots[0], 0]) and bool(store.valid[lease.slots[0], 1])
    assert not bool(store.valid[lease.slots[1], 0])
    store.consume(plan, layer=0)
    assert int(store.uses[lease.slots[1], 0]) == 1
    assert int(store.uses[lease.slots[0], 0]) == 0


def test_stale_plan_after_reincarnation_is_rejected():
    store = make_store()
    lease = store.reserve("r", 0, 2, generation=1)
    stale = store.build_plan([(lease, 0), (lease, 1)], [CAPTURE, CONSUME], key_retention=0.5)
    store.release(lease)
    # Same request and rows, a fresh token incarnation.
    renewed = store.reserve("r", 0, 2, generation=1)
    assert all(new > old for new, old in zip(renewed.incarnations, lease.incarnations))
    with pytest.raises(RuntimeError, match="stale"):
        store.capture(stale, layer=0)
    current = store.build_plan([(renewed, 0), (renewed, 1)], [CAPTURE, CONSUME], key_retention=0.5)
    store.capture(current, layer=0)


def test_outstanding_rounds_hold_disjoint_slots():
    store = make_store(slots=4)
    first = store.reserve("a", 0, 2, generation=1)
    second = store.reserve("b", 1, 2, generation=1)
    assert not set(first.slots) & set(second.slots)
    with pytest.raises(RuntimeError, match="no retired attention reference slots"):
        store.reserve("c", 2, 1, generation=1)
    store.release(first)
    third = store.reserve("c", 2, 1, generation=1)
    assert set(third.slots) <= set(first.slots)


def test_reincarnation_bumps_on_slot_reuse():
    store = make_store()
    lease = store.reserve("r", 0, 1, generation=1)
    slot = lease.slots[0]
    store.release(lease)
    again = store.reserve("r", 0, 1, generation=1)
    assert again.slots[0] == slot
    assert again.incarnations[0] == lease.incarnations[0] + 1
    assert int(store.incarnation[slot]) == again.incarnations[0]


def test_bytes_matches_layout():
    store = make_store(slots=3, layers=2, heads=2, head_dim=4, key_bucket=8)
    expected = 3 * 2 * 2 * (8 + 1 + 1 + 4 + 4) * 4 + 3 * 2 * (1 + 4) + 3 * (8 + 8 + 4)
    assert store.bytes == expected


def test_sparse_config_validates_serving_context_bound():
    assert SparseDecodeAttentionConfig(max_context_tokens=512).max_context_tokens == 512
    assert SparseDecodeAttentionConfig().max_context_tokens is None
    with pytest.raises(ValueError, match="max_context_tokens"):
        SparseDecodeAttentionConfig(max_context_tokens=0)


def test_device_metadata_carries_reference_plan():
    class StubCache:
        device = torch.device("cpu")

        def _require_live_batch(self, batch):
            return None

    builder = MetadataBuilder(StubCache())
    plan = SparseDecodePlan(
        modes=(CAPTURE, CONSUME),
        slots=(0, 1),
        incarnations=(1, 1),
        key_retention=0.5,
        key_bucket=4,
    )
    common = dict(
        allocations=(("r", object()),),
        position_ids=torch.zeros(2, dtype=torch.int64),
        write_blocks=torch.zeros(2, dtype=torch.int64),
        write_offsets=torch.zeros(2, dtype=torch.int64),
        block_tables=torch.zeros((2, 1), dtype=torch.int32),
        context_lengths=torch.zeros(2, dtype=torch.int32),
    )
    batch = builder.prepare_device(**common, reference_plan=plan)
    assert batch.reference_plan is plan
    padded = {**common, "position_ids": torch.zeros(3, dtype=torch.int64)}
    with pytest.raises(ValueError, match="one row mode per query row"):
        builder.prepare_device(**padded, reference_plan=plan)


# ---------------------------------------------------------------------------
# End-to-end decode through the model, the paged KV cache and attention
# ---------------------------------------------------------------------------


def step_metadata(cache, request_id, position, depth, plan=None):
    """Device-style recurrent metadata for one token at one loop depth."""
    allocation = cache._get_allocation(request_id)
    page = position // cache.block_size
    table = torch.tensor([allocation.block_tables[depth][: page + 1]], dtype=torch.int32)
    return cache.metadata_builder.prepare_device(
        allocations=((request_id, allocation),),
        position_ids=torch.tensor([position], dtype=torch.int64),
        write_blocks=table[:, page].long(),
        write_offsets=torch.tensor([position % cache.block_size], dtype=torch.int64),
        block_tables=table,
        context_lengths=torch.tensor([position + 1], dtype=torch.int32),
        reference_plan=plan,
    )


def decode(model, cache, request_id, tokens, *, store=None, lease=None, retention=1.0):
    """Serial greedy decode: prelude, every loop depth, coda, next token."""
    assert cache.allocate(request_id, len(tokens) + 1)
    logits = []
    for position, token in enumerate(tokens):
        hidden = model.prelude(torch.tensor([token]))
        for depth in range(model.config.total_ut_steps):
            plan = None
            mode = reference_mode(depth)
            if store is not None and mode != DENSE:
                plan = store.build_plan([(lease, 0)], [mode], key_retention=retention)
            metadata = step_metadata(cache, request_id, position, depth, plan)
            hidden, _ = model.recurrent_prepared(hidden, metadata, cache, compute_gate=False)
        logits.append(model.coda(hidden))
    return logits


def make_reference_store(model, **overrides):
    values = dict(
        slots=1,
        layers=model.config.num_hidden_layers,
        heads=model.config.num_attention_heads,
        head_dim=model.config.head_dim,
        key_bucket=model.config.max_position_embeddings,
        device="cpu",
    )
    values.update(overrides)
    return AttentionReferenceStore(**values)


def test_sparse_decode_matches_dense_through_model_and_kv():
    # Full retention is the RFC's control: the corrected output must reproduce
    # the dense path through the real model, paged KV cache and attention.
    model = tiny_ouro_model()
    tokens = [2, 3, 4, 5, 7]
    dense = decode(model, make_cpu_cache(model.config), "dense", tokens)
    store = make_reference_store(model)
    lease = store.reserve("sparse", 0, 1, generation=1)
    sparse = decode(
        model,
        make_cpu_cache(model.config),
        "sparse",
        tokens,
        store=store,
        lease=lease,
        retention=1.0,
    )
    assert store.uses.sum() > 0, "the sparse path never consumed a reference"
    for expected, actual in zip(dense, sparse):
        assert torch.allclose(actual, expected, atol=1e-4), float((actual - expected).abs().max())


def test_sparse_decode_diverges_as_retention_shrinks():
    model = tiny_ouro_model()
    tokens = [2, 3, 4, 5, 7, 9, 11]
    dense = decode(model, make_cpu_cache(model.config), "dense", tokens)
    errors = {}
    for retention in (1.0, 0.5, 0.25):
        store = make_reference_store(model)
        lease = store.reserve("sparse", 0, 1, generation=1)
        sparse = decode(
            model,
            make_cpu_cache(model.config),
            "sparse",
            tokens,
            store=store,
            lease=lease,
            retention=retention,
        )
        errors[retention] = max(
            float((actual - expected).abs().max()) for expected, actual in zip(dense, sparse)
        )
    assert errors[1.0] < 1e-4
    assert errors[0.25] > errors[1.0]
    assert errors[0.5] <= errors[0.25] * 4


def test_sparse_consume_reads_only_retained_keys(monkeypatch):
    model = tiny_ouro_model()
    tokens = [2, 3, 4, 5]
    read_counts = []
    original = sparse_module.gather_retained

    def recording(*args, **kwargs):
        keys, values = original(*args, **kwargs)
        read_counts.append(keys.shape[1])
        return keys, values

    monkeypatch.setattr(sparse_module, "gather_retained", recording)
    store = make_reference_store(model)
    lease = store.reserve("sparse", 0, 1, generation=1)
    decode(
        model,
        make_cpu_cache(model.config),
        "sparse",
        tokens,
        store=store,
        lease=lease,
        retention=0.5,
    )
    # Two consuming loops per token, each reading only the retained budget, and
    # every layer gathers its own retained keys.
    per_token = model.config.num_hidden_layers * 2
    expected = [
        retained_count(position + 1, 0.5)
        for position in range(len(tokens))
        for _ in range(per_token)
    ]
    assert read_counts == expected
    assert max(read_counts) < len(tokens)


# ---------------------------------------------------------------------------
# Async speculative integration (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_async_sparse_reference_plans_cover_capture_and_consume():
    model = tiny("cuda")
    e = make_speculative_engine(model, sparse=SparseDecodeAttentionConfig(enabled=True))
    store = e.reference_store
    assert store is not None
    observed = []
    original = store.build_plan

    def recording(owners, modes, **kwargs):
        plan = original(owners, modes, **kwargs)
        observed.append(plan.modes)
        return plan

    store.build_plan = recording
    bootstrap(e)
    drain(e)
    assert any(CAPTURE in modes for modes in observed)
    assert any(CONSUME in modes for modes in observed)
    # Dense steps never publish a plan.
    assert all(DENSE not in modes for modes in observed)
    assert store.outstanding == 0
    assert len(store.free) == store.slots


@pytest.mark.gpu
@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_async_sparse_reference_rejection_invalidates_rejected_rows(accepted, monkeypatch):
    model = tiny("cuda")
    e = make_speculative_engine(model, sparse=SparseDecodeAttentionConfig(enabled=True))
    store = e.reference_store
    bootstrap(e)
    calls = 0

    def controlled(hidden):
        nonlocal calls
        phase = calls % 4
        calls += 1
        logits = torch.full((len(hidden), model.config.vocab_size), -100.0, device="cuda")
        if phase < 3:
            logits[:, 10 + phase] = 100
        else:
            for row in range(4):
                logits[row, 10 + row if row < accepted else 20 + row] = 100
        return logits

    monkeypatch.setattr(e.speculative_runner, "_coda", controlled)

    observed = []
    original = store.accept

    def recording(leases, accepted_counts):
        # Pretend the reference loop published state, then let the device-side
        # retirement decide which rows keep it.
        for lease in leases:
            fill_payload(store, lease.slots)
        original(leases, accepted_counts)
        torch.cuda.synchronize()
        counts = [int(a) for a in accepted_counts.tolist()]
        for lease, count in zip(leases, counts):
            observed.append(
                (
                    count,
                    [float(store.mass[slot, 0, 0]) for slot in lease.slots],
                    [int(store.accepted[slot]) for slot in lease.slots],
                )
            )

    store.accept = recording
    e.step()
    e.step()
    e._collect_speculative()
    assert observed, "acceptance-driven retirement never ran"
    for count, masses, limits in observed:
        assert count == accepted
        assert all(limit == accepted for limit in limits)
        for row, mass in enumerate(masses):
            assert mass == (1.0 if row <= accepted else 0.0)
    e.abort_request("r")
    drain(e)
    assert store.outstanding == 0
    assert len(store.free) == store.slots


@pytest.mark.gpu
def test_async_sparse_reference_incarnations_advance_across_rounds():
    model = tiny("cuda")
    e = make_speculative_engine(model, sparse=SparseDecodeAttentionConfig(enabled=True))
    store = e.reference_store
    bootstrap(e)
    incarnations = []
    original = store.build_plan

    def recording(owners, modes, **kwargs):
        plan = original(owners, modes, **kwargs)
        incarnations.append(plan.incarnations)
        return plan

    store.build_plan = recording
    e.step()
    e.step()
    e._collect_speculative()
    assert len(incarnations) >= 2
    # Every new round is a new token incarnation, even at the same position.
    assert incarnations[-1][0] > incarnations[0][0]
    e.abort_request("r")
    drain(e)
    assert store.outstanding == 0


@pytest.mark.gpu
def test_sparse_engine_matches_dense_tokens_at_full_retention():
    # End-to-end through the serving engine: full retention must reproduce the
    # dense engine's tokens exactly, including async speculative rounds.
    def run(sparse):
        model = tiny("cuda")
        engine = make_speculative_engine(model, sparse=sparse)
        prompts = [[2, 3, 4], [5, 6]]
        for i, prompt in enumerate(prompts):
            engine.add_request(
                str(i), prompt, SamplingParams(max_tokens=12 - 4 * i, ignore_eos=True)
            )
        outputs = {}
        for _ in range(1000):
            if not engine.has_unfinished_requests():
                break
            for output in engine.step():
                outputs[output.request_id] = output
        tokens = [outputs[str(i)].token_ids for i in range(len(prompts))]
        outstanding = engine.reference_store.outstanding if engine.reference_store else 0
        engine.close()
        return tokens, outstanding

    dense, _ = run(None)
    sparse, outstanding = run(SparseDecodeAttentionConfig(enabled=True, key_retention=1.0))
    assert sparse == dense
    assert outstanding == 0


@pytest.mark.gpu
def test_accept_and_retire_do_not_allocate_bucket_sized_temporaries():
    # Reviewer-reported OOM: acceptance used to read-modify-write the retained
    # index buffer, materialising a bucket-sized temporary per batch. Retirement
    # must stay bounded by the rejected rows, not by slot count x bucket.
    torch.cuda.empty_cache()
    store = AttentionReferenceStore(
        slots=16, layers=24, heads=16, head_dim=128, key_bucket=32768, device="cuda"
    )
    lease = store.reserve("r", 0, 16, generation=1)
    store.counts[list(lease.slots)] = 1
    store.mass[list(lease.slots)] = 1.0
    store.output[list(lease.slots)] = 1.0
    store.contribution[list(lease.slots)] = 1.0
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    store.accept([lease], torch.tensor([0], device="cuda"))
    store.release(lease)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    bucket_bytes = 16 * 24 * 16 * 32768 * 4
    assert bucket_bytes > 512 * 1024**2, "test needs a bucket large enough to detect the bug"
    assert peak - baseline < bucket_bytes // 8, (
        f"retirement allocated {(peak - baseline) / 1024**2:.1f} MiB "
        f"for a {bucket_bytes / 1024**2:.1f} MiB bucket"
    )


@pytest.mark.gpu
def test_sparse_engine_bounds_reference_pool_by_serving_context():
    model = tiny("cuda")
    e = make_speculative_engine(
        model,
        sparse=SparseDecodeAttentionConfig(enabled=True, key_retention=1.0, max_context_tokens=64),
    )
    # The pool follows the supported serving bound, not the model's 128-token
    # positional limit.
    assert e.reference_store.key_bucket == 64
    e.close()


@pytest.mark.gpu
def test_sparse_engine_rejects_context_beyond_model_limit():
    model = tiny("cuda")
    with pytest.raises(ValueError, match="exceeds the model context"):
        make_speculative_engine(
            model, sparse=SparseDecodeAttentionConfig(enabled=True, max_context_tokens=4096)
        )


@pytest.mark.gpu
def test_sparse_async_overlapping_rounds_keep_isolated_references():
    # Reviewer-requested coverage: two submitted rounds of the same request are
    # in flight at once. They must hold disjoint reference slots, and retiring
    # the older round's rejected candidates must not disturb the newer round.
    model = tiny("cuda")
    e = make_speculative_engine(
        model, sparse=SparseDecodeAttentionConfig(enabled=True, key_retention=0.5)
    )
    store = e.reference_store
    bootstrap(e)
    accepted_rounds = []
    original = store.accept

    def recording(leases, accepted):
        accepted_rounds.append(tuple(lease.slots for lease in leases))
        return original(leases, accepted)

    store.accept = recording
    e.step()
    e.step()
    assert len(e._pending_speculative) == 2, "expected two overlapping rounds"
    assert len(accepted_rounds) >= 2
    first, second = accepted_rounds[0][0], accepted_rounds[1][0]
    assert first and second
    assert not set(first) & set(second), "outstanding rounds share reference slots"
    assert store.outstanding == len(first) + len(second)
    e._collect_speculative()
    # Only the older round retired; the newer round's references survive.
    assert store.outstanding == len(second)
    e.abort_request("r")
    drain(e)
    assert store.outstanding == 0
    assert len(store.free) == store.slots
    e.close()


@pytest.mark.gpu
def test_sparse_engine_sizes_key_buffer_from_retention():
    model = tiny("cuda")
    e = make_speculative_engine(
        model, sparse=SparseDecodeAttentionConfig(enabled=True, key_retention=0.25)
    )
    # None key_bucket derives the capacity from context x retention, not the
    # full model context (FlashLoop ceil rule at the worst-case key count).
    assert e.reference_store.key_bucket == math.ceil(model.config.max_position_embeddings * 0.25)


@pytest.mark.gpu
def test_sparse_engine_rejects_insufficient_key_bucket():
    model = tiny("cuda")
    with pytest.raises(ValueError, match="key_bucket"):
        make_speculative_engine(
            model,
            sparse=SparseDecodeAttentionConfig(enabled=True, key_retention=1.0, key_bucket=4),
        )


@pytest.mark.gpu
def test_sparse_engine_e2e_multiple_requests_abort_and_drain():
    # Engine-level lifecycle e2e: concurrent rounds across requests, an abort
    # in flight, and full slot reclamation without stale references.
    model = tiny("cuda")
    e = make_speculative_engine(
        model, sparse=SparseDecodeAttentionConfig(enabled=True, key_retention=0.5)
    )
    store = e.reference_store
    max_outstanding = 0
    original_reserve = store.reserve

    def counting_reserve(*args, **kwargs):
        nonlocal max_outstanding
        lease = original_reserve(*args, **kwargs)
        max_outstanding = max(max_outstanding, store.outstanding)
        return lease

    store.reserve = counting_reserve
    params = [SamplingParams(max_tokens=n, ignore_eos=True) for n in (5, 9, 50)]
    for i in range(3):
        e.add_request(str(i), [2, 3, 4], params[i])
    outputs = {}
    for _ in range(1000):
        if outputs.get("0") is not None and outputs["0"].finished:
            break
        for output in e.step():
            outputs[output.request_id] = output
    # Abort one request mid-flight; the rest must finish cleanly.
    if "2" not in outputs or not outputs["2"].finished:
        e.abort_request("2")
    outputs.update(drain(e))
    assert outputs["0"].finished and len(outputs["0"].token_ids) == 5
    assert outputs["1"].finished and len(outputs["1"].token_ids) == 9
    assert store.outstanding == 0
    assert len(store.free) == store.slots
    assert max_outstanding > 0
