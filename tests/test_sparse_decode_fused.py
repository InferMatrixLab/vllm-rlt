"""Opt-in fused sparse decode kernels versus the batched oracle-shaped path."""

import pytest
import torch

from tests.helpers import make_speculative_engine, tiny_ouro_model
from tests.test_sparse_decode_attention import drain, make_store
from vllm_rlt import SamplingParams, SparseDecodeAttentionConfig
from vllm_rlt.attention.sparse import (
    CAPTURE,
    CONSUME,
    reference_paged_attention,
    sparse_paged_attention,
)


def test_sparse_config_validates_implementation():
    assert SparseDecodeAttentionConfig().implementation == "batched"
    assert SparseDecodeAttentionConfig(implementation="fused").implementation == "fused"
    with pytest.raises(ValueError, match="implementation"):
        SparseDecodeAttentionConfig(implementation="native")


def test_fused_store_keeps_the_oracle_on_cpu(monkeypatch):
    calls = []
    import vllm_rlt.attention.sparse as sparse

    monkeypatch.setattr(sparse, "reference_paged_attention", lambda *a, **k: calls.append(1))
    store = make_store(slots=1, layers=1, heads=2, head_dim=4, implementation="fused")
    lease = store.reserve("cpu", 0, 1)
    plan = store.build_plan([(lease, 0)], [CAPTURE], key_retention=0.5)
    q = torch.ones(1, 2, 4)
    cache = torch.ones(1, 16, 2, 4)
    sparse.sparse_paged_attention(
        q, cache, cache, torch.zeros(1, 1, dtype=torch.int32), torch.tensor([3]), plan, layer=0
    )
    assert calls == [1]


@pytest.fixture(autouse=True)
def _check_fast_launches(monkeypatch):
    """Every reused launch must be the kernel Triton's own dispatch would pick."""
    try:
        from vllm_rlt.kernels import sparse_decode
    except ImportError:  # CPU-only installs have no Triton and no fused launches
        return
    monkeypatch.setattr(sparse_decode, "_LAUNCH_CHECK", True)


@pytest.mark.gpu
@pytest.mark.parametrize("context", [37, 1024, 4111])
def test_fast_launches_match_triton_dispatch_bit_for_bit(context, monkeypatch):
    from vllm_rlt.kernels import sparse_decode
    from vllm_rlt.kernels.paged_attention import TritonPagedAttention

    if not sparse_decode._FAST_LAUNCH:
        pytest.skip("fast launches are limited to Triton 3.8")
    rows, heads, kv_heads, dim, block_size, layers = 3, 4, 2, 64, 16, 18
    pages = -(-context // block_size)
    q, keys, values, tables = _paged(
        rows, heads, kv_heads, dim, pages, block_size, torch.bfloat16, 3
    )
    lengths = torch.tensor([context, max(1, context - 17), max(1, context // 2)], device="cuda")
    results = {}
    for fast in (False, True):
        monkeypatch.setattr(sparse_decode, "_FAST_LAUNCH", fast)
        store = make_store(
            slots=rows,
            layers=layers,
            heads=heads,
            head_dim=dim,
            key_bucket=context,
            device="cuda",
            implementation="fused",
        )
        lease = store.reserve("r", 0, rows)
        owners = [(lease, row) for row in range(rows)]
        outputs = []
        # Layers 0, 1, 16 and 17 differ in Triton's integer specialization.
        for _ in range(2):
            for layer in (0, 1, 2, 16, 17):
                for mode in (CAPTURE, CONSUME):
                    plan = store.build_plan(owners, [mode] * rows, key_retention=0.1)
                    outputs.append(
                        sparse_decode.fused_paged_attention(
                            q,
                            keys,
                            values,
                            tables,
                            lengths,
                            plan,
                            layer=layer,
                            dense_attention=TritonPagedAttention() if mode == CAPTURE else None,
                        )
                    )
        torch.cuda.synchronize()
        results[fast] = (
            outputs,
            store.selected.clone(),
            store.counts.clone(),
            store.mass.clone(),
            store.contribution.clone(),
            store.output.clone(),
        )
    slow, fast = results[False], results[True]
    for a, b in zip(slow[0], fast[0]):
        assert torch.equal(a, b)
    for a, b in zip(slow[1:], fast[1:]):
        assert torch.equal(a, b)


@pytest.mark.gpu
@pytest.mark.parametrize("dim", [8, 32, 128])
@pytest.mark.parametrize("folded", [False, True])
def test_fused_capture_rewrites_every_retained_index(dim, folded, monkeypatch):
    from vllm_rlt.kernels import sparse_decode
    from vllm_rlt.kernels.paged_attention import TritonPagedAttention

    monkeypatch.setattr(sparse_decode, "_PASS0_IN_SPLIT", folded)
    keys = torch.zeros(3, 2, 2, dim, device="cuda", dtype=torch.bfloat16)
    keys[..., 0] = torch.tensor(
        [-0.0064, 0.0029, -0.0073, 0.0038, -0.016, -0.0247], device="cuda"
    ).reshape(3, 2, 1)
    values = torch.ones_like(keys)
    q = torch.ones(1, 4, dim, device="cuda", dtype=torch.bfloat16)
    store = make_store(
        slots=1,
        layers=1,
        heads=4,
        head_dim=dim,
        key_bucket=6,
        device="cuda",
        implementation="fused",
    )
    # An old in-range index exposes a missed write without making this capture's
    # contribution kernel dereference an invalid address.
    store.selected.fill_(5)
    lease = store.reserve("r", 0, 1)
    plan = store.build_plan([(lease, 0)], [CAPTURE], key_retention=0.5)
    sparse_paged_attention(
        q,
        keys,
        values,
        torch.arange(3, device="cuda", dtype=torch.int32)[None],
        torch.tensor([6], device="cuda", dtype=torch.int32),
        plan,
        layer=0,
        dense_attention=TritonPagedAttention(),
    )
    assert bool((store.counts[0, 0] == 3).all())
    # Only channel 0 contributes to QK: positions 3, 1, 0 are the strict top 3.
    expected = torch.tensor([0, 1, 3], device="cuda").expand(4, -1)
    assert torch.equal(store.read_reference(plan, 0, 0)[0], expected)


@pytest.mark.gpu
def test_consume_scratch_follows_the_head_dimension(monkeypatch):
    """A stream's consume scratch must not be reused across head dimensions."""
    from vllm_rlt.kernels import sparse_decode
    from vllm_rlt.kernels.paged_attention import TritonPagedAttention

    def consume(dim):
        rows, heads, block_size, pages = 3, 4, 16, 5
        q, keys, values, tables = _paged(rows, heads, 2, dim, pages, block_size, torch.float32, 4)
        lengths = torch.tensor([7, 33, 80], device="cuda", dtype=torch.int32)
        store = make_store(
            slots=rows,
            layers=1,
            heads=heads,
            head_dim=dim,
            key_bucket=80,
            device="cuda",
            implementation="fused",
        )
        lease = store.reserve("r", 0, rows)
        owners = [(lease, row) for row in range(rows)]
        for mode in (CAPTURE, CONSUME):
            plan = store.build_plan(owners, [mode] * rows, key_retention=0.25)
            output = sparse_decode.fused_paged_attention(
                q,
                keys,
                values,
                tables,
                lengths,
                plan,
                layer=0,
                dense_attention=TritonPagedAttention() if mode == CAPTURE else None,
            )
        torch.cuda.synchronize()
        return output

    monkeypatch.setattr(sparse_decode, "CONSUME_SCRATCH", {})
    fresh = consume(128)
    sparse_decode.CONSUME_SCRATCH.clear()
    consume(16)
    # The same stream now holds 16-wide scratch; a 128-wide consume must regrow it.
    assert torch.equal(consume(128), fresh)


def _paged(rows, heads, kv_heads, dim, pages, block_size, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tables = torch.randperm(rows * pages, device="cuda", generator=generator)
    keys = torch.randn(
        rows * pages, block_size, kv_heads, dim, device="cuda", dtype=dtype, generator=generator
    )
    values = torch.randn(keys.shape, device="cuda", dtype=dtype, generator=generator)
    queries = torch.randn(rows, heads, dim, device="cuda", dtype=dtype, generator=generator)
    return queries, keys, values, tables.reshape(rows, pages).int()


@pytest.mark.gpu
@pytest.mark.parametrize("retention", [0.1, 0.5])
@pytest.mark.parametrize("ties", [False, True])
# The Triton backend takes the split capture; any other callable takes the fallback.
@pytest.mark.parametrize("split", [True, False])
def test_fused_capture_and_consume_track_batched_path(retention, ties, split):
    from vllm_rlt.kernels.paged_attention import TritonPagedAttention, triton_paged_attention

    backend = TritonPagedAttention() if split else (lambda *args: triton_paged_attention(*args))

    rows, heads, kv_heads, dim, block_size, pages = 6, 4, 2, 128, 16, 17
    # 10 and 30 sit on float rounding edges of ceil(length * 0.1) (30 * 0.1 > 3 in float64).
    lengths = torch.tensor([0, 10, 17, 30, 200, 257], device="cuda", dtype=torch.int32)
    stores = [
        make_store(
            slots=rows,
            layers=2,
            heads=heads,
            head_dim=dim,
            key_bucket=272,
            device="cuda",
            initial_key_capacity=1,
            implementation=implementation,
        )
        for implementation in ("batched", "fused")
    ]
    leases = [store.reserve("fused", 0, rows) for store in stores]
    owners = [[(lease, row) for row in range(rows)] for lease in leases]
    q, keys, values, tables = _paged(
        rows, heads, kv_heads, dim, pages, block_size, torch.float32, 1
    )
    if ties:
        # Every score equal: both paths must keep the smallest positions.
        q.zero_()
    captured = []
    for store, owner in zip(stores, owners):
        plan = store.build_plan(owner, [CAPTURE] * rows, key_retention=retention)
        captured.append(
            sparse_paged_attention(
                q,
                keys,
                values,
                tables,
                lengths,
                plan,
                layer=1,
                dense_attention=backend,
            )
        )
    batched, fused = stores
    # Capture rows return (and publish) the dense backend output unchanged. These
    # inputs are FP32, where some architectures (sm_90) round the split copy one
    # ulp from the dense kernel; the BF16 test below stays bit for bit.
    exact = 0 if not split else 1e-6
    torch.testing.assert_close(captured[1], captured[0], rtol=0, atol=exact)
    torch.testing.assert_close(fused.output[:, 1], batched.output[:, 1], rtol=0, atol=exact)
    torch.testing.assert_close(fused.counts[:, 1], batched.counts[:, 1], rtol=0, atol=0)
    # Both paths leave the same slots bound to their leases; availability is
    # decided by the retained count, which the comparison above already covers.
    assert set(fused._bindings) == set(batched._bindings)
    for row, length in enumerate(lengths.tolist()):
        slot_b, slot_f = leases[0].slots[row], leases[1].slots[row]
        count = int(batched.counts[slot_b, 1, 0])
        for head in range(heads):
            # Both paths write ascending positions; ties keep the smaller ones.
            expected = batched.selected[slot_b, 1, head, :count].tolist()
            assert fused.selected[slot_f, 1, head, :count].tolist() == expected
        torch.testing.assert_close(
            fused.mass[slot_f, 1], batched.mass[slot_b, 1], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            fused.contribution[slot_f, 1], batched.contribution[slot_b, 1], rtol=1e-4, atol=1e-5
        )
    # A later loop reads its own plane: fresh keys and queries, same pages.
    q2, keys2, values2, _ = _paged(rows, heads, kv_heads, dim, pages, block_size, torch.float32, 2)
    consumed = []
    for store, owner in zip(stores, owners):
        plan = store.build_plan(owner, [CONSUME] * rows, key_retention=retention)
        consumed.append(
            sparse_paged_attention(
                q2,
                keys2,
                values2,
                tables,
                lengths,
                plan,
                layer=1,
                dense_attention=backend,
            )
        )
    torch.testing.assert_close(consumed[1], consumed[0], rtol=1e-4, atol=1e-5)
    assert not consumed[1][0].any(), "an empty row stays zero"
    assert fused.outstanding > 0
    for store, lease in zip(stores, leases):
        store.release(lease)


@pytest.mark.gpu
@pytest.mark.parametrize("retention", [0.1, 0.37])
def test_fused_radix_selection_keeps_oracle_ties_on_repeated_scores(retention):
    from vllm_rlt.kernels.paged_attention import TritonPagedAttention

    # Keys cycle through three vectors, so every head sees three distinct scores
    # repeated across the context: the retained set is decided almost entirely
    # by the tie rule (smaller positions first), with negative scores included.
    # Small integers keep every dot product exact. With random reals the dense
    # Triton kernel can score one key a ULP apart at different tile positions
    # (seen on sm_89), so the "ties" would not be ties for either path.
    rows, heads, dim, block_size, pages = 3, 4, 64, 16, 40
    generator = torch.Generator(device="cuda").manual_seed(11)

    def integers(*shape):
        return torch.randint(-2, 3, shape, device="cuda", generator=generator).float()

    patterns = integers(3, heads, dim)
    tokens = torch.arange(rows * pages * block_size, device="cuda") % 3
    keys = patterns[tokens].reshape(rows * pages, block_size, heads, dim).contiguous()
    values = torch.randn(keys.shape, device="cuda", generator=generator)
    tables = torch.arange(rows * pages, device="cuda", dtype=torch.int32).view(rows, pages)
    q = integers(rows, heads, dim)
    lengths = torch.tensor([7, 300, 640], device="cuda", dtype=torch.int32)
    backend = TritonPagedAttention()
    stores = [
        make_store(
            slots=rows,
            layers=1,
            heads=heads,
            head_dim=dim,
            key_bucket=640,
            device="cuda",
            implementation=implementation,
        )
        for implementation in ("batched", "fused")
    ]
    for store in stores:
        lease = store.reserve("ties", 0, rows)
        plan = store.build_plan(
            [(lease, r) for r in range(rows)], [CAPTURE] * rows, key_retention=retention
        )
        sparse_paged_attention(
            q, keys, values, tables, lengths, plan, layer=0, dense_attention=backend
        )
    batched, fused = stores
    assert torch.equal(fused.counts, batched.counts)
    for slot in range(rows):
        for head in range(heads):
            count = int(batched.counts[slot, 0, head])
            assert (
                fused.selected[slot, 0, head, :count].tolist()
                == batched.selected[slot, 0, head, :count].tolist()
            )


@pytest.mark.gpu
@pytest.mark.parametrize("folded", [True, False])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("retention", [0.1, 0.37])
def test_fused_parallel_select_matches_batched_across_many_parts(
    ties, retention, folded, monkeypatch
):
    import vllm_rlt.kernels.sparse_decode as fused_module
    from vllm_rlt.kernels.paged_attention import TritonPagedAttention

    # The first radix pass runs inside the split kernel or as its own pass.
    monkeypatch.setattr(fused_module, "_PASS0_IN_SPLIT", folded)

    # Long rows split into up to 16 parts; with repeated integer scores the
    # retained ties span part boundaries, so offsets and tie quotas must carry.
    rows, heads, dim, block_size, pages = 4, 4, 64, 16, 320
    generator = torch.Generator(device="cuda").manual_seed(29)
    if ties:

        def integers(*shape):
            return torch.randint(-2, 3, shape, device="cuda", generator=generator).float()

        patterns = integers(3, heads, dim)
        tokens = torch.arange(rows * pages * block_size, device="cuda") % 3
        keys = patterns[tokens].reshape(rows * pages, block_size, heads, dim).contiguous()
        q = integers(rows, heads, dim)
    else:
        # Mostly distinct but still exact scores: a one-ULP difference between the
        # Triton and batched scores at this length could cross the boundary.
        keys = torch.randint(
            -8, 9, (rows * pages, block_size, heads, dim), device="cuda", generator=generator
        ).float()
        q = torch.randint(-8, 9, (rows, heads, dim), device="cuda", generator=generator).float()
    values = torch.randn(keys.shape, device="cuda", generator=generator)
    tables = torch.arange(rows * pages, device="cuda", dtype=torch.int32).view(rows, pages)
    lengths = torch.tensor([1, 129, 2049, 5000], device="cuda", dtype=torch.int32)
    backend = TritonPagedAttention()
    stores = [
        make_store(
            slots=rows,
            layers=2,
            heads=heads,
            head_dim=dim,
            key_bucket=5120,
            device="cuda",
            implementation=implementation,
        )
        for implementation in ("batched", "fused")
    ]
    leases = []
    for store in stores:
        lease = store.reserve("parts", 0, rows)
        leases.append(lease)
        plan = store.build_plan(
            [(lease, r) for r in range(rows)], [CAPTURE] * rows, key_retention=retention
        )
        sparse_paged_attention(
            q, keys, values, tables, lengths, plan, layer=1, dense_attention=backend
        )
    batched, fused = stores
    for row in range(rows):
        slot_b, slot_f = leases[0].slots[row], leases[1].slots[row]
        assert torch.equal(fused.counts[slot_f, 1], batched.counts[slot_b, 1])
        for head in range(heads):
            count = int(batched.counts[slot_b, 1, head])
            assert (
                fused.selected[slot_f, 1, head, :count].tolist()
                == batched.selected[slot_b, 1, head, :count].tolist()
            )
        torch.testing.assert_close(
            fused.mass[slot_f, 1], batched.mass[slot_b, 1], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            fused.contribution[slot_f, 1], batched.contribution[slot_b, 1], rtol=1e-4, atol=1e-5
        )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_split_capture_is_the_dense_backend_output_bit_for_bit(dtype):
    from vllm_rlt.kernels.paged_attention import TritonPagedAttention

    rows, heads, kv_heads, dim = 5, 4, 2, 128
    lengths = torch.tensor([0, 1, 33, 515, 1200], device="cuda", dtype=torch.int32)
    q, keys, values, tables = _paged(rows, heads, kv_heads, dim, 76, 16, dtype, 5)
    backend = TritonPagedAttention()
    expected = backend(q, keys, values, tables, lengths)
    store = make_store(
        slots=rows,
        layers=1,
        heads=heads,
        head_dim=dim,
        key_bucket=1216,
        device="cuda",
        implementation="fused",
    )
    lease = store.reserve("split", 0, rows)
    plan = store.build_plan([(lease, r) for r in range(rows)], [CAPTURE] * rows, key_retention=0.1)
    out = sparse_paged_attention(
        q, keys, values, tables, lengths, plan, layer=0, dense_attention=backend
    )
    # The split copy and the folded merge must not drift from the dense kernels:
    # bit for bit in BF16; in FP32 some architectures (sm_90) contract the split
    # copy's arithmetic differently and round it one ulp apart.
    atol = 0 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(out, expected, rtol=0, atol=atol)
    # O_ref is the output actually fed into the recurrence: rounded, then FP32.
    live = list(lease.slots)[1:]
    torch.testing.assert_close(store.output[live, 0], expected[1:].float(), rtol=0, atol=atol)
    assert all(slot in store._bindings for slot in lease.slots)


@pytest.mark.gpu
def test_fused_bf16_consume_matches_scalar_oracle_within_rounding():
    from vllm_rlt.kernels.paged_attention import triton_paged_attention

    rows, heads, kv_heads, dim = 4, 4, 4, 64
    lengths = torch.tensor([5, 64, 129, 300], device="cuda", dtype=torch.int32)
    q, keys, values, tables = _paged(rows, heads, kv_heads, dim, 19, 16, torch.bfloat16, 3)
    results = []
    for implementation, function in (
        ("fused", sparse_paged_attention),
        ("batched", reference_paged_attention),
    ):
        store = make_store(
            slots=rows,
            layers=1,
            heads=heads,
            head_dim=dim,
            key_bucket=304,
            device="cuda",
            implementation=implementation,
        )
        lease = store.reserve("bf16", 0, rows)
        owner = [(lease, row) for row in range(rows)]
        for modes in ([CAPTURE] * rows, [CONSUME] * rows):
            plan = store.build_plan(owner, modes, key_retention=0.1)
            out = function(
                q,
                keys,
                values,
                tables,
                lengths,
                plan,
                layer=0,
                dense_attention=triton_paged_attention,
            )
        results.append(out.float())
    # Different FP32 reduction order may move a BF16 rounding step.
    torch.testing.assert_close(results[0], results[1], rtol=2e-2, atol=2e-2)


@pytest.mark.gpu
def test_fused_consume_without_capture_is_nan_not_silent():
    rows, heads, dim = 2, 2, 64
    store = make_store(
        slots=rows, layers=1, heads=heads, head_dim=dim, device="cuda", implementation="fused"
    )
    lease = store.reserve("missing", 0, rows)
    plan = store.build_plan([(lease, 0), (lease, 1)], [CONSUME, CONSUME], key_retention=0.5)
    q, keys, values, tables = _paged(rows, heads, heads, dim, 2, 16, torch.float32, 4)
    lengths = torch.tensor([0, 9], device="cuda", dtype=torch.int32)
    out = sparse_paged_attention(q, keys, values, tables, lengths, plan, layer=0)
    assert not out[0].any() and torch.isnan(out[1]).all()


@pytest.mark.gpu
def test_fused_engine_rounds_abort_and_drain(monkeypatch):
    import vllm_rlt.kernels.sparse_decode as fused_module

    calls = {CAPTURE: 0, CONSUME: 0}
    original = fused_module.fused_paged_attention

    def counting(*args, **kwargs):
        calls[args[5].modes[0]] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fused_module, "fused_paged_attention", counting)
    # The engine's Triton backend adapter must reach the split capture path.
    monkeypatch.setattr(fused_module, "_capture_kernel", None)
    model = tiny_ouro_model("cuda")
    engine = make_speculative_engine(
        model,
        sparse=SparseDecodeAttentionConfig(enabled=True, key_retention=0.5, implementation="fused"),
    )
    store = engine.reference_store
    assert store.implementation == "fused"
    params = [SamplingParams(max_tokens=n, ignore_eos=True) for n in (5, 9, 50)]
    for i in range(3):
        engine.add_request(str(i), [2, 3, 4], params[i])
    outputs = {}
    for _ in range(1000):
        if "0" in outputs and outputs["0"].finished:
            break
        for output in engine.step():
            outputs[output.request_id] = output
    if "2" not in outputs or not outputs["2"].finished:
        engine.abort_request("2")
    outputs.update(drain(engine))
    assert outputs["0"].finished and len(outputs["0"].token_ids) == 5
    assert outputs["1"].finished and len(outputs["1"].token_ids) == 9
    assert calls[CAPTURE] > 0 and calls[CONSUME] > 0
    assert store.outstanding == 0 and len(store.free) == store.slots
    assert engine.cache_manager.num_used_blocks == 0
    engine.close()
