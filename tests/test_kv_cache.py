import pytest
import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager, KVSnapshot


def make_cache(**kwargs):
    parameters = dict(
        num_layers=2, num_kv_heads=1, head_dim=4, num_blocks=18, block_size=2, max_loops=3
    )
    parameters.update(kwargs)
    return KVCacheManager(**parameters)


def test_admission_is_atomic_and_counts_all_depths():
    cache = make_cache(num_blocks=12)
    assert cache.required_blocks(3) == 6
    assert cache.allocate("a", 3)
    assert cache.num_used_blocks == 6
    assert not cache.allocate("b", 5)
    assert cache.num_used_blocks == 6
    with pytest.raises(ValueError, match="pool has"):
        cache.allocate("impossible", 9)
    with pytest.raises(ValueError, match="already owns"):
        cache.allocate("a", 3)
    assert cache.allocate("b", 4)
    assert cache.num_free_blocks == 0
    cache.free("never-admitted")
    cache.free("a")
    cache.free("a")
    assert cache.num_free_blocks == 6
    assert cache.allocate("c", 4)
    a_pages = {page for depth in range(3) for page in cache.get_block_table("c", depth)}
    b_pages = {page for depth in range(3) for page in cache.get_block_table("b", depth)}
    assert a_pages.isdisjoint(b_pages)


def test_last_exited_copies_every_layer_and_preserves_executed_depths():
    cache = make_cache()
    assert cache.allocate("a", 4)
    for layer in range(2):
        for depth in range(2):
            keys = torch.full((1, 1, 4), float(10 * layer + depth))
            cache.write(layer, ["a"], [depth], [0], keys, keys + 100)
    cache.finalize_token("a", 0, exit_depth=1)
    for layer in range(2):
        first_keys, first_values = cache.read(layer, "a", 0, 1)
        second_keys, second_values = cache.read(layer, "a", 1, 1)
        last_keys, last_values = cache.read(layer, "a", 2, 1)
        torch.testing.assert_close(first_keys, torch.full_like(first_keys, float(10 * layer)))
        torch.testing.assert_close(
            first_values, torch.full_like(first_values, float(10 * layer + 100))
        )
        torch.testing.assert_close(last_keys, second_keys)
        torch.testing.assert_close(last_values, second_values)
    # They are physical copies: a later write cannot mutate another depth.
    cache.write(0, ["a"], [1], [0], torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))
    torch.testing.assert_close(cache.read(0, "a", 2, 1)[0], torch.ones(1, 1, 4))


def test_early_exit_then_later_token_loops_deeper_across_block_boundary():
    cache = make_cache()
    assert cache.allocate("a", 3)
    for position, exit_depth in [(0, 0), (1, 1), (2, 2)]:
        for depth in range(exit_depth + 1):
            for layer in range(2):
                keys = torch.full((1, 1, 4), float(position * 10 + depth + layer))
                cache.write(layer, ["a"], [depth], [position], keys, keys + 100)
                attended = cache.attend(layer, ["a"], [depth], [position], torch.zeros(1, 2, 4))
                expected_values = torch.tensor(
                    [100 + p * 10 + min(p, depth) + layer for p in range(position + 1)]
                )
                torch.testing.assert_close(
                    attended, torch.full_like(attended, expected_values.float().mean())
                )
        cache.finalize_token("a", position, exit_depth)
    assert cache.read(0, "a", 2)[0].shape == (3, 1, 4)


def test_block_reuse_never_exposes_stale_request_history():
    cache = make_cache(num_blocks=6)
    assert cache.allocate("old", 4)
    keys = torch.full((4, 1, 4), 7.0)
    cache.write(0, ["old"] * 4, [0] * 4, list(range(4)), keys, keys)
    cache.free("old")
    assert cache.allocate("new", 4)
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache.attend(0, ["new"], [0], [0], torch.ones(1, 1, 4))
    cache.write(0, ["new"], [0], [0], torch.zeros(1, 1, 4), torch.ones(1, 1, 4))
    torch.testing.assert_close(
        cache.attend(0, ["new"], [0], [0], torch.zeros(1, 1, 4)), torch.ones(1, 1, 4)
    )
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache.read(0, "new", 0, 2)


def test_validation_happens_before_mutation():
    cache = make_cache()
    cache.allocate("a", 3)
    k = torch.ones(2, 1, 4)
    with pytest.raises(ValueError, match="position"):
        cache.write(0, ["a", "a"], [0, 0], [0, 3], k, k)
    assert cache.read(0, "a", 0)[0].shape[0] == 0
    with pytest.raises(ValueError, match="duplicate"):
        cache.write(0, ["a", "a"], [0, 0], [0, 0], k, k)
    with pytest.raises(RuntimeError, match="every layer"):
        cache.finalize_token("a", 0, 0)
    with pytest.raises(ValueError, match="one-dimensional integer"):
        cache.write(0, ["a"], [0], torch.tensor([0.5]), k[:1], k[:1])
    with pytest.raises(ValueError, match="depth"):
        cache.get_block_table("a", 3)
    with pytest.raises(ValueError, match="dtype"):
        cache.write(0, ["a"], [0], [0], k[:1].half(), k[:1].half())


def test_out_of_order_writes_only_expose_complete_prefix():
    cache = make_cache()
    cache.allocate("a", 4)
    k = torch.arange(16, dtype=torch.float32).reshape(4, 1, 4)
    cache.write(0, ["a", "a"], [0, 0], [3, 1], k[[3, 1]], k[[3, 1]])
    assert cache.read(0, "a", 0)[0].shape[0] == 0
    cache.write(0, ["a"], [0], [0], k[:1], k[:1])
    torch.testing.assert_close(cache.read(0, "a", 0)[0], k[:2])
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache.attend(0, ["a"], [0], [3], torch.zeros(1, 1, 4))
    cache.write(0, ["a"], [0], [2], k[2:3], k[2:3])
    torch.testing.assert_close(cache.read(0, "a", 0)[0], k)


def test_triton_is_explicit_and_rejects_cpu():
    with pytest.raises(ValueError, match="requires a CUDA"):
        make_cache(backend="triton")
    with pytest.raises(ValueError, match="backend"):
        make_cache(backend="automatic")


def fill_prefix(cache, request_id, positions, *, depths=(0,), value=1.0):
    """Write ``value`` at every layer for the given positions and depths."""
    for depth in depths:
        for layer in range(cache.num_layers):
            for position in positions:
                k = torch.full((1, cache.num_kv_heads, cache.head_dim), value)
                cache.write(layer, [request_id], [depth], [position], k, k)


def test_allocation_views_report_logical_facts_only():
    cache = make_cache(num_blocks=18, incremental_allocation=True)
    assert not cache.has_allocation("a")
    with pytest.raises(KeyError, match="no KV allocation"):
        cache.allocated_blocks("a")
    assert cache.allocate("a", 6, initial_tokens=2)
    assert cache.has_allocation("a")
    # One logical page across three LAST_EXITED planes.
    assert cache.allocated_blocks("a") == 3
    assert len(cache.plane_block_tables("a")) == 3
    assert cache.plane_block_tables("a")[1] == cache.get_block_table("a", 1)
    assert cache.ensure_capacity("a", 5)
    assert cache.allocated_blocks("a") == 9
    assert not cache.has_transfer_lease("a")
    cache.pin_transfer("a", "t1")
    assert cache.has_transfer_lease("a")
    cache.unpin_transfer("a", "t1")
    assert not cache.has_transfer_lease("a")
    cache.free("a")
    assert not cache.has_allocation("a")


def test_shared_layout_exposes_one_plane():
    cache = make_cache(layout="shared")
    assert cache.allocate("a", 4)
    assert len(cache.plane_block_tables("a")) == 1
    assert cache.allocated_blocks("a") == 2
    # Depth-indexed access still works for every loop depth.
    assert cache.get_block_table("a", 2) == cache.plane_block_tables("a")[0]


def test_exclusive_prefix_blocks_counts_cache_only_references():
    cache = make_cache(num_blocks=18, enable_prefix_caching=True)
    tokens = list(range(1, 6))
    assert cache.allocate("a", 6)
    fill_prefix(cache, "a", range(4), depths=range(3))
    cache.publish_prefix("a", tokens, 5)
    prefix = cache.lookup_prefix(tokens)
    assert len(prefix) == 2  # two complete pages of the four-token prefix
    # "a" still owns the pages, so the cache is not their only holder.
    assert cache.exclusive_prefix_blocks(prefix) == 0
    cache.free("a")
    # Now only the prefix cache references them: 2 pages x 3 planes.
    assert cache.exclusive_prefix_blocks(prefix) == 6
    assert cache.allocate("b", 6, prefix=prefix)
    assert cache.exclusive_prefix_blocks(prefix) == 0
    assert cache.exclusive_prefix_blocks(()) == 0
    # A shorter hit shares the first page with "b" while a second request
    # would still need to claim the second page from the cache.
    cache.free("b")
    assert cache.allocate("c", 2, prefix=prefix[:1])
    assert cache.exclusive_prefix_blocks(prefix) == 3


def test_token_written_requires_every_layer():
    cache = make_cache()
    assert cache.allocate("a", 3)
    k = torch.ones(1, 1, 4)
    cache.write(0, ["a"], [0], [0], k, k)
    assert not cache.token_written("a", 0, 0)
    cache.write(1, ["a"], [0], [0], k, k)
    assert cache.token_written("a", 0, 0)
    assert not cache.token_written("a", 0, 1)
    assert not cache.token_written("a", 1, 0)
    with pytest.raises(ValueError, match="depth"):
        cache.token_written("a", 0, 3)
    with pytest.raises(ValueError, match="position"):
        cache.token_written("a", 3, 0)


def test_mark_finalized_records_deeper_planes_without_copying():
    cache = make_cache()
    assert cache.allocate("a", 3)
    with pytest.raises(RuntimeError, match="every layer"):
        cache.mark_finalized("a", 0, 0)
    fill_prefix(cache, "a", [0], depths=[0], value=3.0)
    # Plant sentinels in the deeper planes; mark_finalized must leave them alone.
    fill_prefix(cache, "a", [0], depths=[1, 2], value=9.0)
    cache.mark_finalized("a", 0, 0)
    for depth in range(3):
        assert cache.token_written("a", 0, depth)
    torch.testing.assert_close(cache.read(0, "a", 0, 1)[0], torch.full((1, 1, 4), 3.0))
    for depth in (1, 2):
        torch.testing.assert_close(cache.read(0, "a", depth, 1)[0], torch.full((1, 1, 4), 9.0))
    # finalize_token remains the copying variant.
    fill_prefix(cache, "a", [1], depths=[0, 1], value=5.0)
    cache.finalize_token("a", 1, 1)
    torch.testing.assert_close(cache.read(0, "a", 2, 2)[0][1], torch.full((1, 4), 5.0))


def test_mark_finalized_at_the_deepest_loop_records_nothing_new():
    cache = make_cache()
    assert cache.allocate("a", 2)
    fill_prefix(cache, "a", [0], depths=range(3))
    cache.mark_finalized("a", 0, 2)
    assert all(cache.token_written("a", 0, depth) for depth in range(3))
    assert not cache.token_written("a", 1, 0)


def test_mark_finalized_is_a_no_op_for_shared_layout():
    cache = make_cache(layout="shared")
    assert cache.allocate("a", 3)
    fill_prefix(cache, "a", [0])

    def written_state():
        planes = cache._allocations["a"].written
        return [[(w.prefix, set(w.pending)) for w in plane] for plane in planes]

    before = written_state()
    # SHARED has one plane; without the early return this would index plane 1.
    cache.mark_finalized("a", 0, 0)
    assert written_state() == before
    assert cache.token_written("a", 0, 2)


def test_snapshot_and_restore_round_trip_pages_and_write_state():
    cache = make_cache(num_blocks=12)
    assert cache.allocate("a", 4)
    fill_prefix(cache, "a", [0, 1], depths=[0, 1], value=2.0)
    cache.finalize_token("a", 0, 1)
    before = {(d, layer): cache.read(layer, "a", d) for d in range(3) for layer in range(2)}
    snapshot = cache.snapshot("a")
    assert snapshot.pages == 2 and snapshot.max_tokens == 4
    assert snapshot.keys.shape[0] == 6  # 2 pages x 3 planes
    cache.free("a")
    # Disturb the pool so restored pages differ from the original ones.
    assert cache.allocate("filler", 2)
    assert cache.allocate("a", 4)
    with pytest.raises(ValueError, match="page count"):
        cache.restore("a", KVSnapshot(4, 1, snapshot.written, snapshot.keys, snapshot.values))
    with pytest.raises(ValueError, match="capacity"):
        cache.restore("a", KVSnapshot(3, 2, snapshot.written, snapshot.keys, snapshot.values))
    cache.restore("a", snapshot)
    for (d, layer), (keys, values) in before.items():
        restored_keys, restored_values = cache.read(layer, "a", d)
        torch.testing.assert_close(restored_keys, keys)
        torch.testing.assert_close(restored_values, values)
    # The snapshot is not aliased by the live allocation.
    cache.write(0, ["a"], [0], [2], torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))
    assert snapshot.written[0][0].prefix == 2
    with pytest.raises(RuntimeError, match="already holds"):
        cache.restore("a", snapshot)


def test_snapshot_and_restore_round_trip_shared_layout():
    cache = make_cache(layout="shared", num_blocks=6)
    assert cache.allocate("a", 3)
    fill_prefix(cache, "a", [0, 1, 2], value=4.0)
    snapshot = cache.snapshot("a")
    assert snapshot.keys.shape[0] == 2  # two pages, one plane
    cache.free("a")
    assert cache.allocate("filler", 2)
    assert cache.allocate("a", 3)
    assert set(cache.plane_block_tables("a")[0]).isdisjoint(cache.plane_block_tables("filler")[0])
    cache.restore("a", snapshot)
    keys, values = cache.read(1, "a", 2)
    torch.testing.assert_close(keys, torch.full((3, 1, 4), 4.0))
    torch.testing.assert_close(values, keys)


def test_failed_claim_leaves_prefix_entries_in_place():
    cache = make_cache(
        num_blocks=10,
        block_size=2,
        max_loops=2,
        enable_prefix_caching=True,
        incremental_allocation=True,
    )
    tokens = [1, 2, 3, 4, 5]
    assert cache.allocate("a", 4)
    cache.mark_imported_prefix("a", 4)
    cache.publish_prefix("a", tokens, 4)
    cache.free("a")
    assert len(cache.lookup_prefix(tokens)) == 2  # 2 pages x 2 planes evictable
    assert cache.allocate("b", 8, initial_tokens=2)
    assert cache.allocate("c", 2)
    assert cache.allocate("d", 2)
    assert cache.num_free_blocks == 4  # all of it is prefix-cache-held
    # Growing "b" to 8 tokens needs 6 blocks. Eviction could not satisfy it,
    # so the failure must not discard the cached pages.
    assert not cache.ensure_capacity("b", 8)
    assert len(cache.lookup_prefix(tokens)) == 2
    assert cache.num_free_blocks == 4
    assert cache.allocated_blocks("b") == 2
    # Once the claim can succeed, eviction proceeds as before.
    cache.free("c")
    assert cache.ensure_capacity("b", 6)
    assert cache.allocated_blocks("b") == 6
    assert cache.num_free_blocks == 2


def test_claim_skips_prefix_scan_when_free_blocks_suffice(monkeypatch):
    cache = make_cache(num_blocks=10, block_size=2, max_loops=2, enable_prefix_caching=True)
    tokens = [1, 2, 3, 4, 5]
    assert cache.allocate("a", 4)
    cache.mark_imported_prefix("a", 4)
    cache.publish_prefix("a", tokens, 4)
    cache.free("a")

    def scan(self):
        raise AssertionError("num_free_blocks scanned prefix entries")

    monkeypatch.setattr(KVCacheManager, "num_free_blocks", property(scan))
    assert cache.allocate("b", 2)
    assert len(cache.lookup_prefix(tokens)) == 2


def test_snapshot_refuses_leased_allocations():
    cache = make_cache()
    assert cache.allocate("a", 2)
    cache.pin_transfer("a", "t")
    with pytest.raises(RuntimeError, match="transfer"):
        cache.snapshot("a")
