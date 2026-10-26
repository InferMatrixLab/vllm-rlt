import math

import pytest
import torch
import torch.nn.functional as F

# The WISE kernels are Triton; CPU-only installs (CI) skip this module.
pytest.importorskip("triton")

from tests.helpers import LocalLLM as LLM
from tests.helpers import tiny_ouro_config
from vllm_rlt import SamplingParams
from vllm_rlt.config import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.kernels.triton_attention import paged_attention
from vllm_rlt.kernels.wise_attention import (
    MODE_DENSE,
    MODE_DISCOVER,
    MODE_DISCOVER_UNION,
    MODE_SPARSE,
    block_mass,
    select_support,
    wise_paged_attention,
    wise_scratch,
)
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestPhase
from vllm_rlt.wise import WiseConfig

pytestmark = pytest.mark.gpu


def _pool(rows=6, heads=4, kv_heads=2, dim=64, page=16, length=400, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = math.ceil(length / page)
    k = torch.randn(
        (rows * pages + 3, page, kv_heads, dim), device="cuda", generator=g, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    q = torch.randn((rows, heads, dim), device="cuda", generator=g, dtype=torch.bfloat16)
    perm = torch.randperm(rows * pages, device="cuda", generator=g).to(torch.int32) + 3
    tables = perm.view(rows, pages).contiguous()
    lengths = torch.tensor([length - 37 * r for r in range(rows)], device="cuda", dtype=torch.int32)
    return q, k, v, tables, lengths


def _dense_probs(q, k, v, tables, lengths, row):
    n = int(lengths[row])
    pos = torch.arange(n, device="cuda")
    page = k.shape[1]
    blocks = tables[row, pos // page].long()
    keys = k[blocks, pos % page].float().repeat_interleave(q.shape[1] // k.shape[2], 1)
    values = v[blocks, pos % page].float().repeat_interleave(q.shape[1] // k.shape[2], 1)
    scores = torch.einsum("hd,thd->ht", q[row].float(), keys) / math.sqrt(q.shape[-1])
    return scores, values


def _support_reference(probabilities, block_size=32, eta=0.95):
    """vime_plugins/ouro/wise_ouro_reference._support for one query row (heads, 1, n)."""
    heads, n, _ = probabilities.shape
    blocks = math.ceil(n / block_size)
    padded = F.pad(probabilities.float(), (0, blocks * block_size - n, 0, blocks * block_size - n))
    mass = padded.reshape(heads, blocks, block_size, blocks, block_size).sum(-1).sum(2)
    counts = torch.full((blocks,), block_size, device=mass.device, dtype=mass.dtype)
    counts[-1] = n - (blocks - 1) * block_size
    mass = mass / counts[None, :, None]
    causal = torch.ones(blocks, blocks, device=mass.device, dtype=torch.bool).tril()
    ranked, order = torch.sort(mass.masked_fill(~causal, -torch.inf), descending=True, stable=True)
    keep = ranked.clamp_min(0).cumsum(-1).lt((eta * mass.sum(-1)).unsqueeze(-1)).sum(-1) + 1
    keep = torch.minimum(keep, torch.arange(1, blocks + 1, device=mass.device)).clamp_min(1)
    selected = torch.arange(blocks, device=mass.device) < keep.unsqueeze(-1)
    return (
        torch.zeros_like(causal).expand(heads, -1, -1).clone().scatter_(-1, order, selected)
        & causal
    )


def _launch(q, k, v, tables, lengths, mode, support=None, splits=None, num_warps=1):
    rows, heads = q.shape[:2]
    blocks = 1 << math.ceil(math.log2(math.ceil(tables.shape[1] * k.shape[1] / 32)))
    if support is None:
        support = torch.ones((rows, 1, heads, blocks), device="cuda", dtype=torch.uint8)
    modes = torch.full((rows,), mode, device="cuda", dtype=torch.int32)
    slots = torch.arange(rows, device="cuda", dtype=torch.int32)
    scratch = wise_scratch(rows, heads, q.shape[-1], support.shape[-1], "cuda")
    stats = torch.zeros(2, device="cuda", dtype=torch.int64)
    out = wise_paged_attention(
        q,
        k,
        v,
        tables,
        lengths,
        modes,
        slots,
        support,
        0,
        scratch,
        stats,
        splits=splits,
        num_warps=num_warps,
    )
    return out, scratch["mass"][0], scratch["mass"][1], support, stats


def test_split_kv_modes_agree_and_one_split_is_bitwise_the_dense_kernel():
    q, k, v, tables, lengths = _pool()
    expected = paged_attention(q, k, v, tables, lengths)
    # Same warps as the dense kernel: one split is the dense kernel bit for bit.
    assert torch.equal(
        _launch(q, k, v, tables, lengths, MODE_DENSE, splits=1, num_warps=4)[0], expected
    )
    for splits in (None, 4):
        dense = _launch(q, k, v, tables, lengths, MODE_DENSE, splits=splits)[0]
        torch.testing.assert_close(dense, expected, atol=2e-2, rtol=2e-2)
        for mode in (MODE_DISCOVER, MODE_DISCOVER_UNION, MODE_SPARSE):
            assert torch.equal(_launch(q, k, v, tables, lengths, mode, splits=splits)[0], dense), (
                mode
            )


def test_fused_selection_matches_softmax_block_mass_and_reference_support():
    q, k, v, tables, lengths = _pool()
    support = torch.full((q.shape[0], 1, q.shape[1], 16), 7, device="cuda", dtype=torch.uint8)
    _, mm, ms, support, stats = _launch(q, k, v, tables, lengths, MODE_DISCOVER, support, splits=4)
    mass, valid = block_mass(mm, ms, lengths.long())
    expected = select_support(mass, valid, 0.95)
    assert torch.equal(support[:, 0].bool(), expected)
    assert int(stats[0]) == int(expected.sum()) and int(stats[1]) == int(valid.sum()) * q.shape[1]
    for row in range(q.shape[0]):
        scores, _ = _dense_probs(q, k, v, tables, lengths, row)
        probs = scores.softmax(-1)
        n = probs.shape[-1]
        blocks = math.ceil(n / 32)
        sums = F.pad(probs, (0, blocks * 32 - n)).view(probs.shape[0], blocks, 32).sum(-1)
        torch.testing.assert_close(mass[row, :, :blocks], sums, atol=2e-6, rtol=1e-5)
        # Reference on a one-row query block: place the row last, pad earlier rows.
        full = torch.zeros((probs.shape[0], n, n), device="cuda")
        full[:, -1] = probs
        reference = _support_reference(full)[:, -1]
        assert torch.equal(expected[row, :, :blocks], reference)


def test_union_discovery_ors_into_the_first_support():
    q, k, v, tables, lengths = _pool(seed=1)
    q2 = torch.randn_like(q)
    _, mm, ms, first, _ = _launch(
        q,
        k,
        v,
        tables,
        lengths,
        MODE_DISCOVER,
        torch.zeros((q.shape[0], 1, q.shape[1], 16), device="cuda", dtype=torch.uint8),
    )
    a = select_support(*block_mass(mm, ms, lengths.long()), 0.95)
    before = first.clone()
    _, mm, ms, union, stats = _launch(q2, k, v, tables, lengths, MODE_DISCOVER_UNION, first)
    b = select_support(*block_mass(mm, ms, lengths.long()), 0.95)
    assert torch.equal(union[:, 0].bool(), a | b)
    assert torch.equal(before[:, 0].bool(), a)
    # Density counts the reused union: the union adds only the blocks it newly sets.
    assert stats.tolist() == [int((a | b).sum() - a.sum()), 0]


def test_sparse_rows_equal_masked_softmax_over_selected_blocks():
    q, k, v, tables, lengths = _pool(seed=3)
    rows, heads = q.shape[:2]
    g = torch.Generator(device="cuda").manual_seed(7)
    support = (torch.rand((rows, 1, heads, 16), device="cuda", generator=g) < 0.35).to(torch.uint8)
    support[..., 0] = 1
    out = _launch(q, k, v, tables, lengths, MODE_SPARSE, support=support)[0]
    for row in range(rows):
        scores, values = _dense_probs(q, k, v, tables, lengths, row)
        n = scores.shape[-1]
        keep = support[row, 0].bool().repeat_interleave(32, -1)[:, :n]
        weights = scores.masked_fill(~keep, -torch.inf).softmax(-1)
        expected = torch.einsum("ht,thd->hd", weights, values)
        torch.testing.assert_close(out[row].float(), expected, atol=2e-2, rtol=2e-2)


def _llm(model, wise, graphs=False):
    return LLM(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(num_blocks=512, block_size=16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=512, prefill_chunk_size=512
        ),
        execution_config=ExecutionConfig(cuda_graphs=graphs),
        wise_config=wise,
    )


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = tiny_ouro_config(
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_size=128,
        max_position_embeddings=1024,
    )
    return OuroForCausalLM(config).to("cuda", torch.bfloat16)


def test_engine_all_blocks_wise_matches_dense_engine_and_real_wise_prunes(model):
    prompts = [list((torch.arange(300 + 50 * i) * (i + 3)) % 60 + 2) for i in range(3)]
    prompts = [[int(t) for t in p] for p in prompts]
    params = SamplingParams(max_tokens=12, ignore_eos=True)
    dense = _llm(
        model, WiseConfig(max_model_len=1024, max_slots=8, support_override="dense")
    ).generate(prompts, params)
    forced = _llm(
        model, WiseConfig(max_model_len=1024, max_slots=8, support_override="all")
    ).generate(prompts, params)
    assert [o.token_ids for o in forced] == [o.token_ids for o in dense]
    llm = _llm(model, WiseConfig(max_model_len=1024, max_slots=8))
    real = llm.generate(prompts, params)
    assert all(len(o.token_ids) == 12 for o in real)
    assert llm.engine.cache_manager.wise.slots == {}


def test_cuda_graph_wise_replays_eager_wise(model):
    prompts = [[int(t) for t in (torch.arange(260 + 70 * i) * (i + 5)) % 60 + 2] for i in range(3)]
    params = SamplingParams(max_tokens=10, ignore_eos=True)
    eager = _llm(model, WiseConfig(max_model_len=1024, max_slots=8)).generate(prompts, params)
    llm = _llm(model, WiseConfig(max_model_len=1024, max_slots=8), graphs=True)
    graphed = llm.generate(prompts, params)
    assert [o.token_ids for o in graphed] == [o.token_ids for o in eager]
    assert llm.engine.model_runner.graphs.replays > 0
    dense = _llm(
        model, WiseConfig(max_model_len=1024, max_slots=8, support_override="dense"), graphs=True
    ).generate(prompts, params)
    forced = _llm(
        model, WiseConfig(max_model_len=1024, max_slots=8, support_override="all"), graphs=True
    )
    assert [o.token_ids for o in forced.generate(prompts, params)] == [o.token_ids for o in dense]


@pytest.mark.parametrize("graphs", [False, True])
def test_wise_draft_speculation_is_lossless_under_greedy(model, graphs):
    prompts = [[int(t) for t in (torch.arange(280 + 60 * i) * (i + 7)) % 60 + 2] for i in range(3)]
    params = SamplingParams(max_tokens=14, ignore_eos=True)
    dense = _llm(model, None).generate(prompts, params)
    llm = LLM(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(num_blocks=512, block_size=16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=512, prefill_chunk_size=512
        ),
        execution_config=ExecutionConfig(cuda_graphs=graphs),
        speculative_config=SpeculativeConfig(3, draft_attention="wise"),
        wise_config=WiseConfig(max_model_len=1024, max_slots=8),
    )
    drafted = llm.generate(prompts, params)
    assert [o.token_ids for o in drafted] == [o.token_ids for o in dense]
    stats = llm.engine.speculative_runner.stats
    assert stats.drafted_tokens > 0
    assert llm.engine.cache_manager.wise.mean_density() < 1


@pytest.mark.parametrize("graphs", [False, True])
def test_async_wise_matches_synchronous_wise(model, graphs):
    prompts = [[int(t) for t in (torch.arange(260 + 70 * i) * (i + 5)) % 60 + 2] for i in range(3)]
    params = SamplingParams(max_tokens=10, ignore_eos=True)

    def engine(asynchronous):
        return LLM(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(num_blocks=512, block_size=16),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4, max_num_batched_tokens=512, prefill_chunk_size=512
            ),
            execution_config=ExecutionConfig(async_scheduling=asynchronous, cuda_graphs=graphs),
            exit_config=ExitConfig(mode="ouro_delayed"),
            wise_config=WiseConfig(max_model_len=1024, max_slots=8),
        )

    sync = [o.token_ids for o in engine(False).generate(prompts, params)]
    llm = engine(True)
    assert [o.token_ids for o in llm.generate(prompts, params)] == sync
    assert llm.engine.cache_manager.wise.mean_density() < 1


def test_incomplete_discovery_falls_back_to_dense(model):
    llm = _llm(model, WiseConfig(max_model_len=1024, max_slots=8))
    wise = llm.engine.cache_manager.wise

    class Batch:
        allocations = (("x", object()),)

    def plan(depth, position):
        b = Batch()
        b.rows = ((b.allocations[0][1], depth, position),)
        b.context_lengths = torch.tensor([position + 1], device="cuda", dtype=torch.int32)
        b.block_tables = torch.zeros((1, 64), device="cuda", dtype=torch.int32)
        wise.mark(b, [True])
        return wise.plan(b)["modes"].tolist()

    # Depth 0 ran in a packed batch (no support recorded): depth 1 and the reuse
    # depths of that token attend densely; the next token discovers again.
    assert plan(1, 5) == [MODE_DENSE] and plan(2, 5) == [MODE_DENSE]
    assert plan(0, 6) == [MODE_DISCOVER] and plan(1, 6) == [MODE_DISCOVER_UNION]
    assert plan(2, 6) == [MODE_SPARSE]
    # Support released between discovery and reuse (KV offload) or another position's.
    wise.release("x")
    assert plan(3, 6) == [MODE_DENSE] and plan(2, 7) == [MODE_DENSE]
    assert wise.fallbacks == 4


def test_offload_between_discovery_and_reuse_resumes_densely(model):
    prompt = [int(t) for t in (torch.arange(300) * 3) % 60 + 2]
    # Every token runs all four depths, so suspension can fall between 1 and 2.
    params = SamplingParams(
        max_tokens=8, ignore_eos=True, exit_threshold=1.0, min_loops=4, max_loops=4
    )

    def engine(override):
        return LLM(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(num_blocks=512, block_size=16),
            scheduler_config=SchedulerConfig(
                max_num_seqs=1,
                max_num_batched_tokens=512,
                prefill_chunk_size=512,
                enable_preemption=True,
            ),
            wise_config=WiseConfig(max_model_len=1024, max_slots=8, support_override=override),
        ).engine

    dense = engine("dense")
    dense.add_request("r", prompt, params)
    while dense.has_unfinished_requests():
        outputs = [o for o in dense.step() if o.finished]
    reference = outputs[0].token_ids

    e = engine("all")
    e.add_request("r", prompt, params)
    request = e.scheduler.requests["r"]
    discovery = len(e.cache_manager.wise.config.discovery_depths)
    for _ in range(200):
        e.step()
        if (
            request.phase == RequestPhase.RECURRENT
            and request.loops_done == discovery
            and request.generated_token_ids
        ):
            break
    else:
        pytest.fail("request did not stop between discovery and reuse")
    e.add_request("pressure", [3, 4], SamplingParams(max_tokens=1, ignore_eos=True))
    assert e.scheduler.preempt(e.scheduler.requests["pressure"])
    assert e.preemption.snapshots["r"]["kv"] is not None
    e.abort_request("pressure")
    finished = []
    while e.has_unfinished_requests():
        finished += [o for o in e.step() if o.finished]
    assert e.preemption.resumptions == 1
    assert e.cache_manager.wise.fallbacks > 0
    assert [o.token_ids for o in finished] == [reference]


def _flash(generation, page):
    from vllm_rlt.kernels.flash_attention import FlashPagedAttention

    try:
        return FlashPagedAttention(
            torch.device("cuda"), torch.bfloat16, 64, page, f"flash_attn_{generation}"
        )
    except (ImportError, ValueError) as error:
        pytest.skip(str(error))


FLASH = [(2, 256), (3, 16), (4, 16)]


@pytest.mark.parametrize("generation,page", FLASH)
def test_flash_all_blocks_compaction_is_bitwise_dense_flash(generation, page):
    from vllm_rlt.kernels.wise_flash import compact_page_tables

    flash = _flash(generation, page)
    q, k, v, tables, lengths = _pool(page=page, length=900)
    rows, heads = q.shape[:2]
    blocks = 1 << math.ceil(math.log2(math.ceil(tables.shape[1] * page / 32)))
    support = torch.ones((rows, 1, heads, blocks), device="cuda", dtype=torch.uint8)
    modes = torch.full((rows,), MODE_SPARSE, device="cuda", dtype=torch.int32)
    slots = torch.arange(rows, device="cuda", dtype=torch.int32)
    out_t, out_l = compact_page_tables(
        tables,
        lengths,
        modes,
        slots,
        support,
        0,
        page,
        torch.empty_like(tables),
        torch.empty_like(lengths),
    )
    live = torch.arange(tables.shape[1], device="cuda")[None] * page < lengths[:, None]
    assert torch.equal(out_l, lengths) and torch.equal(out_t[live], tables[live])
    assert torch.equal(flash(q, k, v, out_t, out_l), flash(q, k, v, tables, lengths))


@pytest.mark.parametrize("generation,page", FLASH)
def test_flash_discovery_mass_selection_and_union_sparse_output(generation, page):
    from vllm_rlt.kernels.wise_flash import compact_page_tables, discover_support

    flash = _flash(generation, page)
    q, k, v, tables, lengths = _pool(page=page, length=900, seed=5)
    rows, heads = q.shape[:2]
    blocks = 1 << math.ceil(math.log2(math.ceil(tables.shape[1] * page / 32)))
    support = torch.zeros((rows, 1, heads, blocks), device="cuda", dtype=torch.uint8)
    slots = torch.arange(rows, device="cuda", dtype=torch.int32)
    discover = torch.full((rows,), MODE_DISCOVER, device="cuda", dtype=torch.int32)
    scratch = wise_scratch(rows, heads, q.shape[-1], blocks, "cuda")
    stats = torch.zeros(2, device="cuda", dtype=torch.int64)
    _, lse = flash(q, k, v, tables, lengths, return_lse=True)
    discover_support(q, k, tables, lengths, lse, discover, slots, support, 0, scratch, stats)
    mass = scratch["mass"][1]
    valid = torch.arange(blocks, device="cuda")[None] * 32 < lengths[:, None]
    normalized = mass.masked_fill(~valid[:, None], 0)
    normalized = normalized / normalized.sum(-1, keepdim=True)
    assert torch.equal(support[:, 0].bool(), select_support(normalized, valid, 0.95))
    for row in range(rows):
        scores, values = _dense_probs(q, k, v, tables, lengths, row)
        n = scores.shape[-1]
        nb = math.ceil(n / 32)
        sums = F.pad(scores.softmax(-1), (0, nb * 32 - n)).view(heads, nb, 32).sum(-1)
        torch.testing.assert_close(normalized[row, :, :nb], sums, atol=5e-3, rtol=5e-3)
    sparse = torch.full((rows,), MODE_SPARSE, device="cuda", dtype=torch.int32)
    out_t, out_l = compact_page_tables(
        tables,
        lengths,
        sparse,
        slots,
        support,
        0,
        page,
        torch.empty_like(tables),
        torch.empty_like(lengths),
    )
    out = flash(q, k, v, out_t, out_l)
    for row in range(rows):
        scores, values = _dense_probs(q, k, v, tables, lengths, row)
        n = scores.shape[-1]
        block_keep = support[row, 0].bool().any(0)
        page_keep = (
            block_keep.repeat_interleave(32)[: math.ceil(n / page) * page].view(-1, page).any(-1)
        )
        keep = page_keep.repeat_interleave(page)[:n]
        weights = scores.masked_fill(~keep[None], -torch.inf).softmax(-1)
        torch.testing.assert_close(
            out[row].float(), torch.einsum("ht,thd->hd", weights, values), atol=2e-2, rtol=2e-2
        )


@pytest.mark.parametrize("generation", [3, 4])
def test_flash_engine_all_blocks_matches_dense_and_wise_draft_is_lossless(model, generation):
    _flash(generation, 16)
    prompts = [[int(t) for t in (torch.arange(300 + 50 * i) * (i + 3)) % 60 + 2] for i in range(3)]
    params = SamplingParams(max_tokens=10, ignore_eos=True)

    def engine(wise, speculative=None):
        return LLM(
            model,
            attention_backend=f"flash_attn_{generation}",
            cache_config=CacheConfig(num_blocks=512, block_size=16),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4, max_num_batched_tokens=512, prefill_chunk_size=512
            ),
            speculative_config=speculative,
            wise_config=wise,
        )

    dense = [o.token_ids for o in engine(None).generate(prompts, params)]
    forced = engine(WiseConfig(max_model_len=1024, max_slots=8, support_override="all"))
    assert [o.token_ids for o in forced.generate(prompts, params)] == dense
    real = engine(WiseConfig(max_model_len=1024, max_slots=8))
    assert all(len(o.token_ids) == 10 for o in real.generate(prompts, params))
    assert real.engine.cache_manager.wise.mean_density() < 1
    drafted = engine(
        WiseConfig(max_model_len=1024, max_slots=8), SpeculativeConfig(3, draft_attention="wise")
    )
    assert [o.token_ids for o in drafted.generate(prompts, params)] == dense


@pytest.mark.parametrize("generation", [3, 4])
def test_triton_wise_kernel_beside_flash_prefill(model, generation):
    _flash(generation, 16)
    prompts = [[int(t) for t in (torch.arange(300 + 50 * i) * (i + 3)) % 60 + 2] for i in range(3)]
    params = SamplingParams(max_tokens=10, ignore_eos=True)

    def engine(override):
        return LLM(
            model,
            attention_backend=f"flash_attn_{generation}",
            cache_config=CacheConfig(num_blocks=512, block_size=16),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4, max_num_batched_tokens=512, prefill_chunk_size=512
            ),
            wise_config=WiseConfig(
                max_model_len=1024, max_slots=8, support_override=override, kernel="triton"
            ),
        )

    dense = engine("dense")
    assert dense.engine.cache_manager.wise.flash is None
    reference = [o.token_ids for o in dense.generate(prompts, params)]
    assert [o.token_ids for o in engine("all").generate(prompts, params)] == reference
    real = engine(None)
    assert all(len(o.token_ids) == 10 for o in real.generate(prompts, params))
    assert real.engine.cache_manager.wise.mean_density() < 1


@pytest.mark.parametrize("graphs", [False, True])
def test_mixed_packed_fa4_decode_rows_fall_back_to_dense(model, graphs):
    _flash(4, 16)
    # Staggered lengths: later prompts' chunks share packed batches with decode rows.
    prompts = [[int(t) for t in (torch.arange(200 + 150 * i) * (i + 3)) % 60 + 2] for i in range(3)]
    params = SamplingParams(max_tokens=10, ignore_eos=True)

    def engine(override):
        return LLM(
            model,
            attention_backend="flash_attn_4",
            cache_config=CacheConfig(num_blocks=512, block_size=16),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4, max_num_batched_tokens=96, prefill_chunk_size=64, mode="mixed"
            ),
            execution_config=ExecutionConfig(atomic_prefill=False, cuda_graphs=graphs),
            wise_config=WiseConfig(max_model_len=1024, max_slots=8, support_override=override),
        )

    reference = [o.token_ids for o in engine("dense").generate(prompts, params)]
    forced = engine("all")
    assert [o.token_ids for o in forced.generate(prompts, params)] == reference
    assert forced.engine.cache_manager.wise.fallbacks > 0
    real = engine(None)
    assert all(len(o.token_ids) == 10 for o in real.generate(prompts, params))
