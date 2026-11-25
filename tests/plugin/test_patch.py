"""The Request patch, the entry point, and the carry buffer. Needs vLLM, CPU only."""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest
import tomllib

pytest.importorskip("vllm")

from vllm.sampling_params import SamplingParams  # noqa: E402

from vllm_rlt.plugin.patch import apply_request_patch  # noqa: E402
from vllm_rlt.plugin.request import LoopRequest  # noqa: E402


def test_patch_replaces_every_reference_and_is_idempotent():
    import vllm.v1.core.sched.scheduler as scheduler_module
    import vllm.v1.request as request_module

    apply_request_patch()
    assert request_module.Request is LoopRequest
    assert scheduler_module.Request is LoopRequest
    assert apply_request_patch() == 0  # second call changes nothing


def test_engine_builds_loop_requests_with_exit_params_from_xargs():
    apply_request_patch()
    import vllm.v1.request as request_module

    params = SamplingParams(
        max_tokens=4,
        extra_args={"rlt_exit_threshold": 0.9, "rlt_min_loops": 2, "rlt_max_loops": 3},
    )
    engine_request = SimpleNamespace(
        request_id="x",
        client_index=0,
        prompt_token_ids=[1, 2, 3],
        prompt_embeds=None,
        prompt_is_token_ids=None,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        priority=0,
        trace_headers=None,
        resumable=False,
        session_id=None,
        kv_hints=None,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
        abort_immediately=False,
    )
    request = request_module.Request.from_engine_core_request(engine_request, None)
    assert type(request) is LoopRequest
    assert request.exit_params.uses_gate and request.exit_params.limit(4) == 3
    assert not request.in_loop


def test_entry_point_is_declared_in_pyproject():
    root = pathlib.Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    points = data["project"]["entry-points"]["vllm.general_plugins"]
    assert points["vllm_rlt"] == "vllm_rlt.plugin:register"


def test_carry_delivers_once_and_forgets_dropped_requests():
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    from vllm_rlt.plugin.carry import Carry

    carry = Carry()
    cached = CachedRequestData(
        req_ids=["a"],
        new_token_ids=[[]],
        new_block_ids=[([4, 5],)],
        num_computed_tokens=[3],
        num_output_tokens=[1],
    )
    out = SchedulerOutput.make_empty()
    out.scheduled_cached_reqs = cached
    out.finished_req_ids = {"z"}
    carry.absorb(out)
    assert carry.take_blocks("a") == ([4, 5],) and carry.take_blocks("a") is None
    target = SchedulerOutput.make_empty()
    carry.drain_into(target)
    assert target.finished_req_ids == {"z"}
    again = SchedulerOutput.make_empty()
    carry.drain_into(again)
    assert not again.finished_req_ids  # delivered exactly once
    carry.absorb(out)
    carry.drop("a")
    assert carry.take_blocks("a") is None
