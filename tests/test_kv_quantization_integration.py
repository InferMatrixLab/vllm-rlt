"""Quantized KV storage: packed pages plus a BF16 staging window (FlashLoop policy)."""

import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import (
    KVCacheManager,
    codec_page_scratch_bytes,
    codec_scratch_bytes,
    packed_page_bytes,
    staging_pages_bound,
)
from vllm_rlt.core.memory import plan_cache
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.kernels.kv_quantization import pack_kv, unpack_kv
from vllm_rlt.kernels.paged_attention import torch_paged_attention
from vllm_rlt.models import OuroForCausalLM


def make_cache(**overrides):
    values = dict(
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=32,
        block_size=4,
        max_loops=2,
        dtype=torch.bfloat16,
        kv_quantization="int4",
        kv_quantization_group_size=2,
        kv_quantization_window=4,
    )
    values.update(overrides)
    return KVCacheManager(**values)


def write_tokens(cache, request_id, depth, positions, generator, layers=None):
    shape = (len(positions), cache.num_kv_heads, cache.head_dim)
    written = {}
    for layer in range(cache.num_layers) if layers is None else layers:
        k = torch.randn(shape, generator=generator).to(cache.dtype)
        v = torch.randn(shape, generator=generator).to(cache.dtype)
        cache.write(layer, [request_id] * len(positions), [depth] * len(positions), positions, k, v)
        written[layer] = (k, v)
    return written


def fake_quantize(values, kind, format, group_size):
    return unpack_kv(pack_kv(values, kind=kind, format=format, group_size=group_size))


_PAGE_CASES = [("int4", 2, 4), ("nvfp4", 16, 16)]


@pytest.mark.parametrize("format,group_size,block_size", _PAGE_CASES)
def test_pages_pack_once_after_leaving_the_window(format, group_size, block_size):
    cache = make_cache(
        block_size=block_size,
        kv_quantization=format,
        kv_quantization_group_size=group_size,
        kv_quantization_window=block_size,
        kv_quantization_staging_blocks=8,
    )
    generator = torch.Generator().manual_seed(0)
    assert cache.allocate("r", 8 * block_size)
    tokens = 3 * block_size + 1
    exact = write_tokens(cache, "r", 0, list(range(tokens)), generator)
    assert cache.num_free_staging_blocks == 8 - 4
    # (tokens - window) // block_size = 2 pages have left the window.
    assert cache.ready_pages() == 2
    assert cache.pack_ready() == 2 and cache.ready_pages() == 0
    assert cache.num_free_staging_blocks == 8 - 2
    for layer, (k, v) in exact.items():
        read_k, read_v = cache.read(layer, "r", 0)
        for actual, values, kind in ((read_k, k, "key"), (read_v, v, "value")):
            for page in range(2):
                rows = slice(page * block_size, (page + 1) * block_size)
                expected = fake_quantize(values[rows], kind, format, group_size)
                assert torch.equal(actual[rows], expected)
            assert torch.equal(actual[2 * block_size :], values[2 * block_size :])
    before = cache.read(0, "r", 0)
    with pytest.raises(RuntimeError, match="packed KV page"):
        write_tokens(cache, "r", 0, [0], generator)
    # The next page packs only once it leaves the window; packed pages never change.
    write_tokens(cache, "r", 0, list(range(tokens, 4 * block_size)), generator)
    assert cache.pack_ready() == 1
    after = cache.read(0, "r", 0)
    packed = slice(0, 2 * block_size)
    assert torch.equal(after[0][packed], before[0][packed])
    assert torch.equal(after[1][packed], before[1][packed])


# Realistic sizes: tiny groups or heads make per-group metadata outweigh the payload.
# Bytes per group of parameters, and per (page, layer, kind) for a global scale.
@pytest.mark.parametrize(
    "format,group_size,block_size,group_bytes,global_bytes",
    [("int4", 64, 64, 4, 0), ("nvfp4", 16, 16, 1, 4)],
)
def test_storage_is_packed_and_staging_is_bounded(
    format, group_size, block_size, group_bytes, global_bytes
):
    cache = make_cache(
        head_dim=64,
        block_size=block_size,
        kv_quantization=format,
        kv_quantization_group_size=group_size,
        kv_quantization_staging_blocks=5,
    )
    # BF16 pages exist only for staging; every block id owns a packed slot.
    assert cache.key_cache.shape[0] == cache.value_cache.shape[0] == 5
    packed_bytes = sum(
        storage.numel() * storage.element_size()
        for pool in cache._packed_pools.values()
        for storage in pool.values()
    )
    expected = packed_page_bytes(
        format=format,
        group_size=group_size,
        block_size=block_size,
        num_kv_heads=2,
        head_dim=64,
        num_layers=2,
        dtype=torch.bfloat16,
    )
    assert packed_bytes == cache.num_blocks * expected == cache.num_blocks * cache.bytes_per_block
    # Per block id: 4-bit codes, group parameters, and any per-page global scale.
    elements = 2 * 2 * block_size * 2 * 64  # K/V x layers x tokens x heads x dim
    assert cache.bytes_per_block == (
        elements // 2 + elements // group_size * group_bytes + 2 * 2 * global_bytes
    )
    unquantized = make_cache(head_dim=64, block_size=block_size, kv_quantization=None)
    assert unquantized.bytes_per_block == elements * 2
    assert cache.bytes_per_block * 3 < unquantized.bytes_per_block
    # Staging never exceeds the logical page count.
    assert make_cache(kv_quantization_staging_blocks=1000).num_staging_blocks == 32


def test_staging_bound_covers_window_and_one_step():
    # 64-token pages and window: each request keeps 2 pages plus 1 partial page
    # for its share of the step; the 128-token step adds 2 pages per plane.
    bound = staging_pages_bound(block_size=64, window=64, tokens_per_step=128, requests=8, planes=4)
    assert bound == 4 * (8 * 3 + 2)
    # The step's tokens are shared by all requests, not counted once per request.
    large = staging_pages_bound(
        block_size=64, window=64, tokens_per_step=2048, requests=64, planes=4
    )
    assert large == 4 * (64 * 3 + 32)


def test_staging_bound_holds_for_every_split_of_a_step():
    block_size, window, step, requests = 4, 4, 9, 3
    bound = staging_pages_bound(
        block_size=block_size, window=window, tokens_per_step=step, requests=requests, planes=1
    )
    worst = 0
    # Worst case per request: the most unpacked tokens left after packing, then its share.
    retained = window + block_size - 1
    for first in range(step + 1):
        for second in range(step + 1 - first):
            shares = (first, second, step - first - second)
            worst = max(worst, sum(-(-(retained + t) // block_size) for t in shares))
    assert worst <= bound


def test_a_page_waits_for_every_layer_of_its_plane():
    cache = make_cache()
    generator = torch.Generator().manual_seed(1)
    assert cache.allocate("r", 32)
    write_tokens(cache, "r", 0, list(range(12)), generator, layers=[0])
    assert cache.ready_pages() == 0
    write_tokens(cache, "r", 0, list(range(8)), generator, layers=[1])
    # Layer 1 reaches 8 tokens: (8 - 4) // 4 = 1 page.
    assert cache.ready_pages() == 1
    write_tokens(cache, "r", 0, list(range(8, 12)), generator, layers=[1])
    assert cache.ready_pages() == 2


def test_each_request_and_plane_packs_independently():
    cache = make_cache()
    generator = torch.Generator().manual_seed(2)
    assert cache.allocate("a", 32) and cache.allocate("b", 32)
    write_tokens(cache, "a", 0, list(range(12)), generator)
    write_tokens(cache, "a", 1, list(range(8)), generator)
    write_tokens(cache, "b", 0, list(range(4)), generator)
    assert cache.pack_ready() == 2 + 1
    assert cache._allocations["a"].packed_pages == [2, 1]
    assert cache._allocations["b"].packed_pages == [0, 0]


def dense_pool(cache, request_ids, depths, layer):
    """Dense BF16 pool and tables rebuilt from read(), as unquantized pages would hold."""
    pages, tables = [], []
    for request_id, depth in zip(request_ids, depths):
        k, v = cache.read(layer, request_id, depth)
        count = -(-k.shape[0] // cache.block_size)
        pad = count * cache.block_size - k.shape[0]
        shape = (count, cache.block_size, cache.num_kv_heads, cache.head_dim)
        k = torch.cat([k, k.new_zeros(pad, *k.shape[1:])]).view(shape)
        v = torch.cat([v, v.new_zeros(pad, *v.shape[1:])]).view(shape)
        tables.append(list(range(len(pages), len(pages) + count)))
        pages.extend(zip(k, v))
    keys = torch.stack([k for k, _ in pages])
    values = torch.stack([v for _, v in pages])
    width = max(len(table) for table in tables)
    tables = torch.tensor([table + [0] * (width - len(table)) for table in tables])
    return keys, values, tables.to(torch.int32)


@pytest.mark.parametrize("format,group_size,block_size", _PAGE_CASES)
def test_attention_reads_packed_and_staged_pages(format, group_size, block_size):
    cache = make_cache(
        block_size=block_size,
        kv_quantization=format,
        kv_quantization_group_size=group_size,
        kv_quantization_window=block_size,
    )
    generator = torch.Generator().manual_seed(3)
    lengths = {"a": 3 * block_size + 2, "b": 2 * block_size - 1}
    for request_id, length in lengths.items():
        assert cache.allocate(request_id, 4 * block_size)
        for depth in range(2):
            write_tokens(cache, request_id, depth, list(range(length)), generator)
    assert cache.pack_ready() > 0
    rows = [("a", 0, 0), ("a", 1, block_size + 1), ("a", 0, lengths["a"] - 1), ("b", 1, 3)]
    ids, depths, positions = (list(column) for column in zip(*rows))
    q = torch.randn(len(rows), 4, 8, generator=generator).to(torch.bfloat16)
    for layer in range(cache.num_layers):
        actual = cache.attend(layer, ids, depths, positions, q)
        keys, values, tables = dense_pool(cache, ids, depths, layer)
        lengths_tensor = torch.tensor([p + 1 for p in positions], dtype=torch.int32)
        assert torch.equal(actual, torch_paged_attention(q, keys, values, tables, lengths_tensor))


def test_finalize_copies_between_staging_pages():
    cache = make_cache(kv_quantization_staging_blocks=8)
    generator = torch.Generator().manual_seed(4)
    assert cache.allocate("r", 32)
    for position in range(10):
        written = write_tokens(cache, "r", 0, [position], generator)
        cache.finalize_token("r", position, 0)
        for layer, (k, v) in written.items():
            deeper_k, deeper_v = cache.read(layer, "r", 1, position + 1)
            assert torch.equal(deeper_k[position], k[0])
            assert torch.equal(deeper_v[position], v[0])
        cache.pack_ready()
    assert cache._allocations["r"].packed_pages == [1, 1]
    # A finalized deeper plane packs to the same values as the exit plane.
    for layer in range(cache.num_layers):
        assert all(
            torch.equal(a, b) for a, b in zip(cache.read(layer, "r", 0), cache.read(layer, "r", 1))
        )


def test_staging_pages_are_returned_and_exhaustion_is_reported():
    cache = make_cache(kv_quantization_staging_blocks=2)
    generator = torch.Generator().manual_seed(5)
    assert cache.allocate("r", 32)
    write_tokens(cache, "r", 0, list(range(8)), generator)
    assert cache.num_free_staging_blocks == 0
    with pytest.raises(RuntimeError, match="staging pool is exhausted"):
        write_tokens(cache, "r", 0, [8], generator)
    cache.free("r")
    assert cache.num_free_staging_blocks == 2
    assert not any(cache._packed_page)


def test_truncation_and_snapshots_respect_packed_pages():
    cache = make_cache()
    assert cache.allocate("r", 32)
    write_tokens(cache, "r", 0, list(range(12)), torch.Generator().manual_seed(6))
    assert cache.pack_ready() == 2
    cache.truncate_suffix("r", 8)
    with pytest.raises(ValueError, match="packed KV"):
        cache.truncate_suffix("r", 7)
    with pytest.raises(ValueError, match="snapshots"):
        cache.snapshot("r")


@pytest.mark.parametrize(
    "overrides,message",
    [
        (dict(kv_quantization="fp8"), "unknown KV quantization"),
        (dict(kv_quantization_group_size=3), "multiple of kv_quantization_group_size"),
        (dict(kv_quantization="nvfp4", block_size=16, kv_quantization_group_size=8), "NVFP4"),
        (dict(kv_quantization_window=0), "window must be a positive"),
        (dict(kv_quantization_staging_blocks=0), "staging_blocks must be a positive"),
        (dict(layout="shared"), "requires last_exited"),
        (dict(enable_prefix_caching=True), "prefix caching"),
        (dict(recurrent_layers=[1]), "every cache layer"),
    ],
)
def test_rejects_unsupported_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_cache(**overrides)


def tiny_model(device="cpu", dtype=torch.float32):
    torch.manual_seed(123)
    return OuroForCausalLM(tiny_ouro_config()).to(device=device, dtype=dtype)


def make_engine(cache_config, *, device="cpu", dtype=torch.bfloat16, backend="torch", **options):
    options.setdefault(
        "scheduler_config", SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8)
    )
    return LLMEngine(
        tiny_model(device, dtype), cache_config=cache_config, attention_backend=backend, **options
    )


def run(engine, prompts, *, max_tokens=24):
    """Generate, keeping each request's KV as read just before it is freed."""
    captured = {}
    cache = engine.cache_manager
    original_free = cache.free

    def free(request_id):
        if cache.has_allocation(request_id):
            for depth in range(cache.max_loops):
                for layer in range(cache.num_layers):
                    k, v = cache.read(layer, request_id, depth)
                    captured[(request_id, depth, layer)] = (k.cpu(), v.cpu())
        return original_free(request_id)

    cache.free = free
    params = SamplingParams(max_tokens=max_tokens, exit_threshold=0.7, ignore_eos=True)
    for index, prompt in enumerate(prompts):
        engine.add_request(str(index), prompt, params)
    outputs = {}
    for _ in range(2000):
        if not engine.has_unfinished_requests():
            break
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output
    assert cache.num_used_blocks == 0
    return [outputs[str(index)] for index in range(len(prompts))], captured


def same_run(left, right):
    (left_outputs, left_kv), (right_outputs, right_kv) = left, right
    return (
        [o.token_ids for o in left_outputs] == [o.token_ids for o in right_outputs]
        and [o.exit_depths for o in left_outputs] == [o.exit_depths for o in right_outputs]
        and left_kv.keys() == right_kv.keys()
        and all(
            torch.equal(a, b)
            for key in left_kv
            for a, b in zip(left_kv[key], right_kv[key], strict=True)
        )
    )


def fake_quantized_engine(format, group_size, block_size, window, **options):
    """Unquantized pages, with each step's ready pages fake-quantized in place.

    This re-implements the packing policy on dense BF16 pages, independently of
    the manager's packed storage, staging pool and attention gather.
    """
    engine = make_engine(CacheConfig(num_blocks=256, block_size=block_size), **options)
    cache = engine.cache_manager

    def pack_kv_step():
        engine.model_runner.synchronize()
        for allocation in cache._allocations.values():
            for plane, layers in enumerate(allocation.written):
                frontier = min(written.prefix for written in layers)
                end = max(0, frontier - window) // block_size
                for page in range(allocation.packed_pages[plane], end):
                    block = allocation.block_tables[plane][page]
                    for layer in range(cache.num_layers):
                        for pool, kind in ((cache.key_cache, "key"), (cache.value_cache, "value")):
                            pool[block, layer] = fake_quantize(
                                pool[block, layer], kind, format, group_size
                            )
                allocation.packed_pages[plane] = max(allocation.packed_pages[plane], end)
        if cache.device.type == "cuda":
            torch.cuda.synchronize(cache.device)

    engine._pack_kv = pack_kv_step
    return engine


def quantized_config(format, group_size, block_size, window, **overrides):
    return CacheConfig(
        num_blocks=256,
        block_size=block_size,
        kv_quantization=format,
        kv_quantization_group_size=group_size,
        kv_quantization_window=window,
        **overrides,
    )


_ENGINE_CASES = [
    pytest.param("cpu", "torch", id="cpu-bf16-torch"),
    pytest.param("cuda", "torch", id="cuda-bf16-torch", marks=pytest.mark.gpu),
    pytest.param("cuda", "triton", id="cuda-bf16-triton", marks=pytest.mark.gpu),
]


@pytest.mark.parametrize("format,group_size,block_size", _PAGE_CASES)
@pytest.mark.parametrize("device,backend", _ENGINE_CASES)
def test_packed_engine_matches_fake_quantized_reference(
    device, backend, format, group_size, block_size
):
    # The production run never reads the pools between steps, so on CUDA only
    # _pack_kv's own synchronization orders packing against later reads.
    prompts = [list(range(2, 2 + 3 * block_size)), [5, 6, 7]]
    window = block_size
    options = dict(device=device, backend=backend)
    engine = make_engine(quantized_config(format, group_size, block_size, window), **options)
    cache = engine.cache_manager
    packed_pages = []
    original = cache.pack_ready

    def pack_ready():
        count = original()
        packed_pages.append(max(max(a.packed_pages) for a in cache._allocations.values()))
        return count

    cache.pack_ready = pack_ready
    production = run(engine, prompts)
    reference_engine = fake_quantized_engine(format, group_size, block_size, window, **options)
    reference = run(reference_engine, prompts)
    assert all(len(output.token_ids) == 24 for output in production[0])
    # The prompt alone packs (3 * block_size - window) // block_size = 2 pages,
    # so a third page was packed from decode tokens and attended afterwards.
    assert max(packed_pages) > 2
    # Without packing, two requests need more pages than the staging pool holds.
    assert cache.num_staging_blocks < cache.required_blocks(3 * block_size + 23) * 2
    assert same_run(production, reference)


def test_reference_comparison_detects_wrong_packed_pages():
    # Negative control: corrupting packed slots must break the comparison.
    format, group_size, block_size = "int4", 2, 4
    prompts = [list(range(2, 14)), [5, 6, 7]]
    reference = run(fake_quantized_engine(format, group_size, block_size, block_size), prompts)
    engine = make_engine(quantized_config(format, group_size, block_size, block_size))
    cache = engine.cache_manager
    original = cache.pack_ready

    def corrupt():
        count = original()
        for pool in cache._packed_pools.values():
            for storage in pool.values():
                storage.zero_()
        return count

    cache.pack_ready = corrupt
    assert not same_run(run(engine, prompts), reference)
    engine = make_engine(quantized_config(format, group_size, block_size, block_size))
    assert same_run(run(engine, prompts), reference)


@pytest.mark.parametrize("device,backend", _ENGINE_CASES)
def test_window_past_every_sequence_matches_unquantized(device, backend):
    # Nothing packs, so attention reads staged BF16 pages only.
    prompts = [list(range(2, 14)), [5, 6, 7]]
    options = dict(device=device, backend=backend)
    exact = run(make_engine(CacheConfig(num_blocks=256, block_size=4), **options), prompts)
    staged = run(make_engine(quantized_config("int4", 2, 4, 1000), **options), prompts)
    assert same_run(staged, exact)


def test_memory_plan_budgets_packed_blocks_and_staging():
    model = tiny_model(dtype=torch.bfloat16)
    config = model.config
    scheduler = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8)
    budget = 1 << 20
    plain = CacheConfig(block_size=64, kv_cache_memory_bytes=budget)
    quantized = CacheConfig(
        block_size=64,
        kv_cache_memory_bytes=budget,
        kv_quantization="int4",
        kv_quantization_window=64,
    )
    plain_blocks, plain_plan = plan_cache(model, plain, scheduler, ExecutionConfig(), "torch")
    blocks, plan = plan_cache(model, quantized, scheduler, ExecutionConfig(), "torch")
    packed = packed_page_bytes(
        format="int4",
        group_size=64,
        block_size=64,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_layers=config.num_hidden_layers,
        dtype=torch.bfloat16,
    )
    staging = staging_pages_bound(
        block_size=64, window=64, tokens_per_step=8, requests=2, planes=config.total_ut_steps
    )
    assert plan["bytes_per_block"] == packed
    assert plan["kv_staging_blocks"] == staging
    assert plan["kv_staging_bytes"] == staging * plain_plan["bytes_per_block"]
    assert blocks == (budget - plan["kv_staging_bytes"]) // packed
    assert packed * 2 < plain_plan["bytes_per_block"]
    assert blocks > plain_blocks
    # A budget below the staging bound: the manager stages at most one BF16
    # page per block id, so only those pages are charged.
    page = plain_plan["bytes_per_block"]
    tight = (staging // 2) * (packed + page) + packed
    small = CacheConfig(
        block_size=64,
        kv_cache_memory_bytes=tight,
        kv_quantization="int4",
        kv_quantization_window=64,
    )
    blocks, plan = plan_cache(model, small, scheduler, ExecutionConfig(), "torch")
    assert blocks == staging // 2
    assert plan["kv_staging_blocks"] == blocks
    assert plan["kv_staging_bytes"] == blocks * page
    assert blocks * packed + plan["kv_staging_bytes"] <= tight


class _PeakNewStorage(TorchDispatchMode):
    """Peak bytes of tensor storage allocated inside the block and still alive."""

    def __init__(self, existing):
        super().__init__()
        self.existing = {tensor.untyped_storage().data_ptr() for tensor in existing}
        self.live: dict[int, int] = {}
        self.current = self.peak = 0

    def _track(self, tensor):
        storage = tensor.untyped_storage()
        key, size = storage.data_ptr(), storage.nbytes()
        if not size or key in self.existing or key in self.live:
            return
        self.live[key] = size
        self.current += size
        self.peak = max(self.peak, self.current)

        def release():
            if self.live.pop(key, None) is not None:
                self.current -= size

        weakref.finalize(storage, release)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        for tensor in output if isinstance(output, (tuple, list)) else (output,):
            if isinstance(tensor, torch.Tensor):
                self._track(tensor)
        return output


@pytest.mark.parametrize(
    "format,group_size,heads,head_dim,layers,block_size",
    [
        # head_dim < G pads V channels 8x; a fixed bytes-per-element rate misses it.
        ("int4", 64, 2, 8, 24, 64),
        ("int4", 64, 1, 65, 2, 64),
        ("int4", 2, 2, 8, 2, 4),
        ("int4", 64, 4, 128, 2, 64),
        ("nvfp4", 16, 2, 8, 2, 16),
        ("nvfp4", 16, 4, 128, 2, 16),
    ],
)
def test_codec_scratch_covers_padded_groups(
    format, group_size, heads, head_dim, layers, block_size
):
    pages = 8  # One pack chunk.
    cache = make_cache(
        num_layers=layers,
        num_kv_heads=heads,
        head_dim=head_dim,
        block_size=block_size,
        max_loops=1,
        kv_quantization=format,
        kv_quantization_group_size=group_size,
        kv_quantization_window=block_size,
        kv_quantization_staging_blocks=pages + 2,
    )
    assert cache.allocate("r", (pages + 1) * block_size)
    write_tokens(
        cache, "r", 0, list(range((pages + 1) * block_size)), torch.Generator().manual_seed(0)
    )
    pools = [cache.key_cache, cache.value_cache]
    pools += [storage for pool in cache._packed_pools.values() for storage in pool.values()]
    per_page = codec_page_scratch_bytes(
        group_size=group_size, block_size=block_size, num_kv_heads=heads, head_dim=head_dim
    )
    with _PeakNewStorage(pools) as packing:
        assert cache.pack_ready() == pages
    assert packing.peak <= pages * layers * per_page
    blocks = torch.tensor(cache._allocations["r"].block_tables[0][:pages])
    for kind in ("key", "value"):
        with _PeakNewStorage(pools) as decoding:
            cache._decode_pages(kind, 0, blocks)
        assert decoding.peak <= pages * per_page
    chunk = codec_scratch_bytes(
        group_size=group_size,
        block_size=block_size,
        num_kv_heads=heads,
        head_dim=head_dim,
        num_layers=layers,
    )
    assert chunk >= max(64, pages * layers) * per_page


@pytest.mark.parametrize(
    "options,message",
    [
        (dict(execution_config=ExecutionConfig(async_scheduling=True)), "synchronous"),
        (dict(scheduler_config=SchedulerConfig(enable_preemption=True)), "preemption"),
    ],
)
def test_engine_rejects_unsupported_execution(options, message):
    with pytest.raises(ValueError, match=message):
        make_engine(quantized_config("int4", 2, 4, 4), **options)


def test_pd_is_rejected_when_the_worker_attaches_its_connector():
    from unittest.mock import Mock

    from tests.test_stage_engine import cpu_connector
    from vllm_rlt.engine.stage_config import StageConfig
    from vllm_rlt.engine.worker import StageWorker

    channel = Mock()
    options = dict(
        seed=0,
        dtype="float32",
        revision=None,
        model_config=None,
        engine=dict(cache_config=quantized_config("int4", 2, 2, 2)),
    )
    with pytest.raises(ValueError, match="PD transfer"):
        StageWorker(
            StageConfig(stage_id="p", role="prefill"),
            "cpu",
            channel,
            tiny_model(),
            options,
            connector_factory=cpu_connector,
        )
    assert not any(call.args[0].get("kind") == "ready" for call in channel.send.call_args_list)


def test_pd_is_rejected_when_the_connector_is_passed_to_the_engine():
    from tests.test_stage_engine import cpu_connector
    from vllm_rlt.engine.stage_config import StageConfig

    engine = StageEngine(tiny_model(), stage_config=StageConfig(role="prefill"))
    connector = cpu_connector(engine, notify=lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="PD transfer"):
        StageEngine(
            tiny_model(),
            stage_config=StageConfig(role="prefill"),
            cache_config=quantized_config("int4", 2, 2, 2),
            connector=connector,
        )
