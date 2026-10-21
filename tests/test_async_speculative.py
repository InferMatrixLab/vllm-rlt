"""Cross-round device progress, reservation and lifetime contracts."""

import pytest
import torch

from tests.helpers import (
    bootstrap_speculation,
    make_speculative_engine,
    tiny_ouro_model,
)
from vllm_rlt import (
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.core.scheduler import Scheduler
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.request import Request, RequestPhase


def drain(engine):
    outputs = {}
    for _ in range(1000):
        if not engine.has_unfinished_requests():
            return outputs
        for output in engine.step():
            outputs[output.request_id] = output
    pytest.fail("async speculative engine failed to drain")


def test_scheduler_uses_upper_bound_and_releases_rejected_budget():
    scheduler = Scheduler(SchedulerConfig(), None, SpeculativeConfig(3))
    r = Request("r", [2, 3], SamplingParams(max_tokens=8), generated_token_ids=[4])
    r.num_output_placeholders = 4
    scheduled_request = scheduler._make_scheduled_request(r, RequestPhase.SPECULATIVE, 16)
    assert (scheduled_request.token_start, scheduled_request.token_count) == (6, 3)
    # One committed token instead of the four reserved: recover three slots.
    r.num_output_placeholders = 0
    r.generated_token_ids.append(5)
    scheduled_request = scheduler._make_scheduled_request(r, RequestPhase.SPECULATIVE, 16)
    assert (scheduled_request.token_start, scheduled_request.token_count) == (3, 4)


def test_async_spec_rejects_cpu_before_cache_allocation():
    with pytest.raises(ValueError, match="requires CUDA"):
        make_speculative_engine(tiny_ouro_model())


@pytest.mark.gpu
@pytest.mark.parametrize("k,budget", [(1, 1), (2, 7), (4, 16), (8, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_async_matches_sync_with_ragged_limits_and_prefix_reuse(k, budget, dtype):
    model = tiny_ouro_model("cuda", dtype)
    params = [SamplingParams(max_tokens=n, ignore_eos=True) for n in (1, 13, 8)]
    prompts = [[2, 3, 4, 5], [2, 3, 4, 7], [9]]
    results = []
    for asynchronous in (False, True):
        e = make_speculative_engine(
            model, asynchronous=asynchronous, k=k, budget=budget, prefix=True
        )
        for repetition in range(2):
            for i, (prompt, p) in enumerate(zip(prompts, params)):
                e.add_request(str(i), prompt, p)
            output = drain(e)
            results.append([output[str(i)].token_ids for i in range(3)])
            assert all(o.finished for o in output.values())
        if asynchronous:
            assert not e.speculative_runner.states
            assert all(not buffer.leased for buffer in e.speculative_runner.buffers)
    assert all(r == results[0] for r in results)


@pytest.mark.gpu
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_second_round_executes_before_cpu_commit_at_every_rejection(accepted, monkeypatch):
    model = tiny_ouro_model("cuda")
    e = make_speculative_engine(model)
    request = bootstrap_speculation(e)
    start = request.position
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

    monkeypatch.setattr(model, "coda", controlled)

    # A CPU read of candidate/acceptance values would fail during submission.
    def forbidden(*args, **kwargs):
        raise AssertionError("host read in cross-round GPU submission")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        e.step()
        e.step()
    assert len(e._pending_speculative) == 2
    assert len(request.generated_token_ids) == 1
    assert e.speculative_runner.submitted_before_collect == 1
    state = e.speculative_runner.states["r"]
    e.speculative_runner.synchronize()
    assert int(state.position) == start + 2 * (accepted + 1)
    # Collecting the older round must not rewind the newer device state.
    out = e._collect_speculative()[0]
    assert out.token_ids[1:] == list(range(10, 10 + accepted)) + [20 + accepted]
    assert int(state.position) == start + 2 * (accepted + 1)
    e.abort_request("r")
    drain(e)
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
def test_eos_discards_queued_round_and_request_id_can_be_reused(monkeypatch):
    model = tiny_ouro_model("cuda")
    e = make_speculative_engine(model)
    bootstrap_speculation(e, ignore_eos=False)
    coda = model.coda
    monkeypatch.setattr(
        model,
        "coda",
        lambda h: torch.zeros(
            len(h),
            model.config.vocab_size,
            device="cuda",
        ),
    )
    e.step()
    e.step()
    out = e._collect_speculative()[0]
    assert out.finished and out.token_ids[-1] == 0 and len(out.token_ids) == 2
    # Reuse while an old ticket still exists. It must never touch the new request.
    monkeypatch.setattr(model, "coda", coda)
    e.add_request("r", [7, 8], SamplingParams(max_tokens=5, ignore_eos=True))
    output = drain(e)["r"]
    assert len(output.token_ids) == 5
    assert e.speculative_runner.discarded_rounds == 1
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
def test_cancel_and_partial_submission_failure_reclaim_resources(monkeypatch):
    model = tiny_ouro_model("cuda")
    e = make_speculative_engine(model)
    bootstrap_speculation(e)
    e.step()
    e.step()
    assert e.abort_request("r").finish_reason == "abort"
    drain(e)
    assert not e.speculative_runner.states
    assert e.cache_manager.num_used_blocks == 0
    bootstrap_speculation(e)

    def fail(*args, **kwargs):
        raise RuntimeError("injected target failure")

    monkeypatch.setattr(model, "coda", fail)
    with pytest.raises(RuntimeError, match="injected"):
        e.step()
    assert not e.scheduler.requests
    assert not e.speculative_runner.states
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
def test_async_rejects_random_sampling():
    e = make_speculative_engine(tiny_ouro_model("cuda"))
    with pytest.raises(ValueError, match="greedy"):
        e.add_request("r", [2], SamplingParams(temperature=0.8))


@pytest.mark.gpu
def test_async_rejects_host_reading_attention():
    with pytest.raises(ValueError, match="device_context_lengths"):
        LLMEngine(
            tiny_ouro_model("cuda"),
            attention_backend="torch",
            execution_config=ExecutionConfig(async_scheduling=True),
            speculative_config=SpeculativeConfig(3),
        )


@pytest.mark.gpu
def test_prefill_arrivals_interleave_with_inflight_rounds():
    model = tiny_ouro_model("cuda", torch.bfloat16)
    e = make_speculative_engine(model, budget=5, prefix=True)
    bootstrap_speculation(e, max_tokens=17)
    e.step()
    e.step()
    assert len(e._pending_speculative) == 2
    prompts = [[2, 3, 4], [2, 3, 4, 8, 9], [7]]
    lengths = [17, 7, 9]
    for i in (1, 2):
        e.add_request(str(i), prompts[i], SamplingParams(max_tokens=lengths[i], ignore_eos=True))
    actual = drain(e)
    reference = make_speculative_engine(model, asynchronous=False, budget=5, prefix=True)
    for rid, prompt, length in zip(["r", "1", "2"], prompts, lengths):
        reference.add_request(rid, prompt, SamplingParams(max_tokens=length, ignore_eos=True))
    expected = drain(reference)
    assert {rid: o.token_ids for rid, o in actual.items()} == {
        rid: o.token_ids for rid, o in expected.items()
    }
    assert all(o.finished for o in actual.values())
    assert not e.speculative_runner.states
    assert all(not buffer.leased for buffer in e.speculative_runner.buffers)
    e.close()
    reference.close()


@pytest.mark.gpu
@pytest.mark.parametrize("multi_stream", [False, True])
def test_close_drains_both_submitted_rounds(multi_stream):
    e = make_speculative_engine(tiny_ouro_model("cuda"), multi_stream=multi_stream)
    bootstrap_speculation(e)
    e.step()
    e.step()
    assert len(e._pending_speculative) == 2
    e.close()
    assert not e.has_unfinished_requests()
    assert e.cache_manager.num_used_blocks == 0
    assert not e.speculative_runner.states
    assert all(not buffer.leased for buffer in e.speculative_runner.buffers)
    e.close()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_two_round_kv_matches_serial_replay_at_every_depth(dtype):
    model = tiny_ouro_model("cuda", dtype)
    e = make_speculative_engine(model)
    request = bootstrap_speculation(e)
    e.step()
    e.step()
    results = [ticket.collect()[0][0] for ticket in e._pending_speculative]
    tokens = request.prompt_token_ids + request.generated_token_ids
    tokens += [token for result in results for token in result.token_ids]
    # Last emitted token has not been forwarded; all earlier tokens have KV.
    tokens = tokens[:-1]
    oracle = make_speculative_engine(model, asynchronous=False).cache_manager
    assert oracle.allocate("serial", len(tokens))
    with torch.inference_mode():
        for position, token in enumerate(tokens):
            hidden = model.prelude(torch.tensor([token], device="cuda"))
            for depth in range(model.config.total_ut_steps):
                hidden, _ = model.recurrent(hidden, ["serial"], [depth], [position], oracle)
    allocation = e.cache_manager._get_allocation("r")
    pos = torch.arange(len(tokens), device="cuda")
    atol, rtol = (4e-5, 4e-5) if dtype == torch.float32 else (0.08, 0.04)
    for depth in range(model.config.total_ut_steps):
        pages = torch.tensor(allocation.block_tables[depth], device="cuda")
        blocks, offsets = pages[pos // 2], pos % 2
        for layer in range(model.config.num_hidden_layers):
            expected_k, expected_v = oracle.read(layer, "serial", depth, len(tokens))
            torch.testing.assert_close(
                e.cache_manager.key_cache[blocks, layer, offsets], expected_k, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                e.cache_manager.value_cache[blocks, layer, offsets],
                expected_v,
                atol=atol,
                rtol=rtol,
            )
    e.close()
    assert not e.has_unfinished_requests()
    assert e.cache_manager.num_used_blocks == 0
    assert not e.speculative_runner.states
    assert all(not buffer.leased for buffer in e.speculative_runner.buffers)
    e.close()


def test_active_page_tables_are_contiguous_and_do_not_touch_inactive_storage():
    from types import SimpleNamespace

    from vllm_rlt.core.kv_cache_manager import KVCacheManager
    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput
    from vllm_rlt.worker.async_speculative import AsyncSpeculativeRunner

    cache = KVCacheManager(1, 1, 4, 64, 2, 4)
    assert cache.allocate("short", 4) and cache.allocate("long", 10)
    runner = object.__new__(AsyncSpeculativeRunner)
    runner.cache = cache
    buffer = SimpleNamespace(
        leased=False,
        host_tables=torch.full((128,), -91, dtype=torch.int32),
        tables=torch.full((128,), -92, dtype=torch.int32),
    )
    runner.buffers = [buffer]
    requests = [Request(rid, [2], SamplingParams()) for rid in ("short", "long")]
    batch = SchedulerOutput(
        RequestPhase.SPECULATIVE,
        [
            ScheduledRequest(requests[0], 1, 2),
            ScheduledRequest(requests[1], 7, 2),
        ],
    )
    _, _, host, device = runner._prepare_round(batch)
    assert host.shape == device.shape == (2, 4, 5)
    assert host.is_contiguous() and device.is_contiguous()
    assert host.data_ptr() == buffer.host_tables.data_ptr()
    for i, rid in enumerate(("short", "long")):
        pages = 2 if i == 0 else 5
        for depth in range(4):
            expected = cache._get_allocation(rid).block_tables[depth][:pages]
            assert host[i, depth, :pages].tolist() == list(expected)
    assert not host[0, :, 2:].any()
    assert (buffer.host_tables[40:] == -91).all()
    assert (buffer.tables == -92).all()  # CPU preparation does not upload.
    with pytest.raises(RuntimeError, match="exhausted"):
        runner._prepare_round(batch)
    # Simulate retirement, then shrink both dimensions using the same storage.
    buffer.leased = False
    before = buffer.host_tables.clone()
    smaller = SchedulerOutput(RequestPhase.SPECULATIVE, [ScheduledRequest(requests[0], 0, 1)])
    _, _, host, device = runner._prepare_round(smaller)
    assert host.shape == device.shape == (1, 4, 1)
    assert host.is_contiguous() and device.is_contiguous()
    assert torch.equal(buffer.host_tables[4:], before[4:])


def test_device_metadata_rejects_foreign_and_reused_allocations_without_host_reads(monkeypatch):
    from vllm_rlt.core.kv_cache_manager import KVCacheManager

    cache = KVCacheManager(1, 1, 4, 16, 2, 4)
    foreign = KVCacheManager(1, 1, 4, 16, 2, 4)
    assert cache.allocate("r", 2) and foreign.allocate("r", 2)
    allocation = cache._get_allocation("r")
    arguments = dict(
        allocations=(("r", allocation),),
        position_ids=torch.tensor([0]),
        write_blocks=torch.tensor([allocation.block_tables[0][0]]),
        write_offsets=torch.tensor([0]),
        block_tables=torch.tensor([allocation.block_tables[0]], dtype=torch.int32),
        context_lengths=torch.tensor([1], dtype=torch.int32),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata creation read device values on CPU")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        batch = cache.metadata_builder.prepare_device(**arguments)
    assert not hasattr(batch, "owner")
    with pytest.raises(RuntimeError, match="stale"):
        foreign._require_live_batch(batch)
    cache.free("r")
    assert cache.allocate("r", 2)
    with pytest.raises(RuntimeError, match="stale"):
        cache._require_live_batch(batch)
    with pytest.raises(RuntimeError, match="stale"):
        cache.metadata_builder.prepare_device(**arguments)


@pytest.mark.gpu
def test_async_greedy_execution_never_acquires_a_generator(monkeypatch):
    # Object identity does not prove the asynchronous greedy path leaves the RNG
    # alone, so run a real request with a spy that fails on any acquire. This needs
    # a reserved device because AsyncSpeculativeRunner allocates CUDA streams; the
    # contract therefore stays pending for M7 3/3.
    e = make_speculative_engine(tiny_ouro_model("cuda"))

    def forbidden(*args, **kwargs):
        raise AssertionError("greedy speculative sampling must not acquire a generator")

    monkeypatch.setattr(e.rng, "acquire", forbidden)
    e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=6, ignore_eos=True))
    outputs = drain(e)
    assert outputs["r"].finish_reason == "length"
    assert e.rng.peek("r") is None


@pytest.mark.gpu
@pytest.mark.parametrize("multi_stream", [False, True])
def test_async_speculative_engine_shares_its_registry(multi_stream):
    # AsyncSpeculativeRunner allocates CUDA streams in its constructor, so this
    # wiring is only verifiable on a reserved device; a CPU construction or mock
    # would not exercise the object the engine actually holds.
    e = make_speculative_engine(tiny_ouro_model("cuda"), multi_stream=multi_stream)
    assert e.async_speculative
    assert e.speculative_runner.rng is e.rng
    assert e.model_runner.rng is e.rng
