"""Resident positions remain immutable; paged escape binds deferred addresses."""

import pytest
import torch

from tests.test_ascend import make_npu_model
from tests.test_fast_execution_plan import cache_for, metadata
from vllm_rlt.device import resolve_device
from vllm_rlt.worker.metadata import MetadataBuilder, stage_range_metadata

pytestmark = pytest.mark.npu


def deferred(cache, start, width):
    layouts = cache.metadata_builder.build_ranges(["a"], [start], [width], [(0,)])

    def reject_upload(*args):
        pytest.fail("owned metadata unexpectedly staged paged addresses")

    host = stage_range_metadata(layouts, reject_upload, defer_addresses=True)[0]
    assert host.position_ids is None
    return MetadataBuilder.bind(host, {})


@torch.inference_mode()
def test_owned_inputs_materialize_on_paged_escape():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    cache = cache_for(model)
    batch = deferred(cache, 0, 3)
    hidden = model.prelude(torch.tensor([3, 7, 11], device="npu:0"))
    model.recurrent_prepared(hidden, batch, cache)
    assert cache._key_cache is None and batch.write_blocks is None
    _ = cache.key_cache  # Ordered export before a legacy overwrite.
    assert cache._contiguous is None
    actual = model.recurrent_prepared(hidden, batch, cache)[0]
    full = cache.metadata_builder.materialize_paged(batch)
    assert full.write_blocks.device.type == "npu" and full.context_lengths is not None
    assert full.position_ids is batch.position_ids and batch.write_blocks is None
    reference = cache_for(model)
    expected = model.recurrent_prepared(hidden, metadata(reference, 0, 3)[0], reference)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    original = torch.npu.current_stream()
    foreign = torch.npu.Stream()
    with torch.npu.stream(foreign):
        foreign.wait_stream(original)
        other = cache.metadata_builder.materialize_paged(batch)
        assert other.write_blocks.data_ptr() != full.write_blocks.data_ptr()
        torch.testing.assert_close(other.write_blocks, full.write_blocks, atol=0, rtol=0)
    foreign.synchronize()
    cache.free("a")
    reference.free("a")


@torch.inference_mode()
def test_resident_position_views_survive_growth_and_eviction():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    cache = cache_for(model)
    batch = deferred(cache, 0, 3)
    original, snapshot = batch.position_ids, batch.position_ids.clone()
    for start, width in [(0, x) for x in range(1, 9)] + [(1, 2), (2, 2)]:
        current = deferred(cache, start, width)
        torch.testing.assert_close(
            current.position_ids.cpu(), torch.arange(start, start + width), atol=0, rtol=0
        )
    assert len(cache.metadata_builder._position_layouts) <= 8
    torch.testing.assert_close(original, snapshot, atol=0, rtol=0)
    cache.free("a")
