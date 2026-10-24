"""Deployment owners share execution semantics, not scheduler or KV state."""

import pytest
import torch

from tests.helpers import RecordingConnector, make_stage_engine, tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestPhase


def drain(engine):
    outputs = {}
    for _ in range(500):
        if not engine.has_unfinished_requests():
            return outputs
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = (output.token_ids, output.exit_depths)
    pytest.fail("stage engine did not make progress")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_mixed_keeps_one_resource_owner_across_all_phases(asynchronous):
    engine = make_stage_engine(mixed=True, asynchronous=asynchronous)
    engine.add_request("r", [1, 2, 3, 4, 5], SamplingParams(max_tokens=3, ignore_eos=True))
    assert len(drain(engine)["r"][0]) == 3
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine.model_runner.state_slots


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_prefill_handoff_matches_local_execution_without_sampling(atomic, asynchronous):
    engine = make_stage_engine(role="prefill", atomic=atomic, asynchronous=asynchronous)
    local = make_stage_engine(atomic=atomic, asynchronous=asynchronous)
    connector = engine.connector = RecordingConnector()
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    for e in (engine, local):
        e.add_request("r", [1, 2, 3, 4, 5], params)
    request = engine.scheduler.requests["r"]
    original = local.scheduler.requests["r"]
    for e, r in ((engine, request), (local, original)):
        for _ in range(100):
            e.step()
            if r.num_prefilled_tokens == 5:
                break
        else:
            pytest.fail("prompt did not complete")
    assert request.phase == RequestPhase.TRANSFERRING
    assert original.phase == RequestPhase.CODA
    assert request.generated_token_ids == []
    torch.testing.assert_close(request.hidden_state, original.hidden_state)
    assert all(r is request for r, _ in connector.submissions)
    completed = [s for _, s in connector.submissions if s["progress"].chunk_complete]
    assert [s["start"] for s in completed] == [0, 3]
    assert completed[-1]["progress"].prompt_complete
    assert engine.step() == []  # Await remote acknowledgement; never sample on P.
    engine.close()
    local.close()


def test_connector_backpressure_preserves_other_requests_and_queue_membership():
    engine = make_stage_engine(role="prefill")
    connector = engine.connector = RecordingConnector()
    connector.blocked.add("a")
    for name in ("a", "b"):
        engine.add_request(name, [1, 2], SamplingParams(max_tokens=1))
    for _ in range(20):
        engine.step()
    assert engine.scheduler.requests["a"].num_preluded_tokens == 0
    assert engine.scheduler.requests["b"].phase == RequestPhase.TRANSFERRING
    connector.blocked.clear()
    for _ in range(20):
        engine.step()
    assert engine.scheduler.requests["a"].phase == RequestPhase.TRANSFERRING
    engine.close()


def test_cancelled_import_cannot_activate_reused_request_id():
    engine = make_stage_engine(role="decode")
    engine.reserve_transfer("r", [1, 2], SamplingParams(max_tokens=2), receiving=True)
    old = engine.scheduler.requests["r"]
    engine.release_transfer(old)
    engine.reserve_transfer("r", [3], SamplingParams(max_tokens=2), receiving=True)
    current = engine.scheduler.requests["r"]
    assert not engine.accept_import(old, torch.zeros(1))
    assert current.phase == RequestPhase.RECEIVING
    assert current.hidden_state is None
    engine.close()


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_independent_stage_resources_preserve_output_after_import(atomic, asynchronous):
    p = make_stage_engine(role="prefill", atomic=atomic, asynchronous=asynchronous)
    d = make_stage_engine(role="decode", atomic=atomic, asynchronous=asynchronous)
    local = make_stage_engine(atomic=atomic, asynchronous=asynchronous)
    p.connector = RecordingConnector()
    prompt = [1, 2, 3, 4, 5]
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    source = p.reserve_transfer("r", prompt, params, receiving=False)
    target = d.reserve_transfer("r", prompt, params, receiving=True)
    assert source is not target
    assert p.scheduler is not d.scheduler
    assert p.cache_manager is not d.cache_manager
    assert p.model_runner is not d.model_runner
    assert d.step() == []
    for _ in range(100):
        p.step()
        if source.phase == RequestPhase.TRANSFERRING:
            break
    else:
        pytest.fail("source prompt did not complete")
    # CPU transport simulation: copy payload only, never the live Request.
    # This checks import semantics, not NIXL or CUDA stream ordering.
    source_tables = p.cache_manager.plane_block_tables("r")
    target_tables = d.cache_manager.plane_block_tables("r")
    for left, right in zip(source_tables, target_tables):
        for src, dst in zip(left, right):
            d.cache_manager.key_cache[dst].copy_(p.cache_manager.key_cache[src])
            d.cache_manager.value_cache[dst].copy_(p.cache_manager.value_cache[src])
    assert d.accept_import(target, source.hidden_state.clone())
    assert d.scheduler.requests["r"] is target
    assert target.prefill_reclaimable_tokens == len(prompt)
    assert p.release_transfer(source)
    local.add_request("r", prompt, params)
    assert drain(d) == drain(local)
    assert p.cache_manager.num_used_blocks == d.cache_manager.num_used_blocks == 0
    for engine in (p, d, local):
        engine.close()


def test_transfer_reservation_reuses_identity_and_requires_retirement_before_id_reuse():
    engine = make_stage_engine(role="decode")
    params = SamplingParams(max_tokens=2)
    request = engine.reserve_transfer("generation-1", [1, 2], params, receiving=True)
    assert engine.cache_manager.has_transfer_lease(request.request_id)
    with pytest.raises(ValueError, match="duplicate request ID"):
        engine.reserve_transfer("generation-1", [1, 2], params, receiving=True)
    assert engine.scheduler.requests[request.request_id] is request
    assert engine.release_transfer(request)
    newer = engine.reserve_transfer("generation-1", [3], params, receiving=True)
    assert newer is not request
    assert not engine.accept_import(request, torch.zeros(1))
    assert not engine.release_transfer(request)
    assert engine.cache_manager.has_transfer_lease(newer.request_id)
    assert engine.release_transfer(newer)
    assert engine.cache_manager.num_used_blocks == 0
    engine.close()


def test_connector_submission_failure_keeps_remote_resources_pinned(monkeypatch):
    engine = make_stage_engine(role="prefill")
    engine.connector = RecordingConnector()
    request = engine.reserve_transfer("r", [1, 2], SamplingParams(max_tokens=1), receiving=False)

    def fail(batch):
        raise RuntimeError("submission failed")

    monkeypatch.setattr(engine.executor, "submit", fail)
    with pytest.raises(RuntimeError, match="submission failed"):
        engine.step()
    assert engine.scheduler.requests["r"] is request
    assert engine.cache_manager.has_transfer_lease("r")
    assert engine.release_transfer(request)  # Only after the owner stops remote access.
    engine.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_prefill_decode_stage_routes_per_request_and_keeps_local_decode_running(asynchronous):
    engine = make_stage_engine(mixed=True, asynchronous=asynchronous)
    connector = engine.connector = RecordingConnector()
    connector.exports = lambda request: request.request_id == "remote"
    connector.blocked.add("remote")
    for rid in ("remote", "local"):
        engine.add_request(rid, [1, 2, 3], SamplingParams(max_tokens=2, ignore_eos=True))
    local_outputs = []
    for _ in range(100):
        local_outputs.extend(engine.step())
        if "local" not in engine.scheduler.requests:
            break
    assert local_outputs[-1].finished
    assert local_outputs[-1].request_id == "local"
    assert engine.scheduler.requests["remote"].num_prefilled_tokens == 0
    assert not connector.submissions  # Local loop/chunk boundaries never export.
    connector.blocked.clear()
    for _ in range(100):
        engine.step()
        if engine.scheduler.requests["remote"].phase == RequestPhase.TRANSFERRING:
            break
    else:
        pytest.fail("remote request did not reach its handoff boundary")
    assert all(r.request_id == "remote" for r, _ in connector.submissions)
    assert engine.scheduler.requests["remote"].generated_token_ids == []
    engine.close()


def test_decode_capability_rejects_unreserved_prompt():
    engine = make_stage_engine(role="decode")
    with pytest.raises(ValueError, match="reserve_transfer"):
        engine.add_request("r", [1, 2], SamplingParams(max_tokens=1))
    assert not engine.has_request("r")
    assert not engine.cache_manager.has_allocation("r")
    engine.close()


def test_prefill_requires_explicit_export_route():
    engine = make_stage_engine(role="prefill")
    engine.connector = RecordingConnector()
    engine.connector.exports = lambda request: False
    engine.add_request("r", [1, 2], SamplingParams(max_tokens=1))
    with pytest.raises(ValueError, match="outgoing route"):
        engine.step()
    assert engine.scheduler.requests["r"].num_prefilled_tokens == 0
    engine.close()


def test_prefill_does_not_hide_a_lost_runnable_request():
    engine = make_stage_engine(role="prefill")
    engine.connector = RecordingConnector()
    engine.add_request("r", [1, 2], SamplingParams(max_tokens=1))
    engine.scheduler.queues[RequestPhase.WAITING].clear()
    with pytest.raises(RuntimeError, match="no progress"):
        engine.step()
    engine.close()


def test_prefill_admission_waits_for_transfer_to_release_kv():
    torch.manual_seed(123)
    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config()),
        stage_config=StageConfig(role="prefill"),
        cache_config=CacheConfig(4, 2),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2),
    )
    engine.connector = RecordingConnector()
    engine.add_request("first", [1, 2], SamplingParams(max_tokens=1))
    engine.step()
    first = engine.scheduler.requests["first"]
    assert first.phase == RequestPhase.TRANSFERRING
    engine.add_request("second", [1, 2], SamplingParams(max_tokens=1))
    assert engine.step() == []
    assert engine.scheduler.requests["second"].phase == RequestPhase.WAITING
    engine.release_transfer(first)
    engine.step()
    assert engine.scheduler.requests["second"].phase == RequestPhase.TRANSFERRING
    engine.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_spawned_mixed_worker_matches_local_without_connector(asynchronous):
    import multiprocessing

    from vllm_rlt.engine.worker import worker_main
    from vllm_rlt.models import ModelConfig

    local = make_stage_engine(mixed=True, asynchronous=asynchronous)
    options = dict(
        seed=123,
        dtype="float32",
        revision=None,
        model_config=ModelConfig(tiny_ouro_config()),
        engine=dict(
            cache_config=CacheConfig(128, 2),
            scheduler_config=SchedulerConfig(
                mode="mixed", max_num_seqs=3, max_num_batched_tokens=4, prefill_chunk_size=3
            ),
            execution_config=ExecutionConfig(async_scheduling=asynchronous, static_buffers=True),
            exit_config=ExitConfig("ouro_delayed" if asynchronous else "ouro"),
        ),
    )
    context = multiprocessing.get_context("spawn")
    channel, child = context.Pipe()
    process = context.Process(
        target=worker_main,
        args=(
            StageConfig(stage_id="local", role="prefill_decode"),
            "cpu",
            child,
            tiny_ouro_config(),
            options,
        ),
    )
    process.start()
    child.close()

    def receive():
        assert channel.poll(60), "stage worker did not respond"
        message = channel.recv()
        assert message["kind"] != "fatal", message.get("error")
        return message

    try:
        ready = receive()
        assert ready["kind"] == "ready"
        assert ready["role"] == "prefill_decode"
        assert "model" not in ready and "model_type" not in ready
        assert "info" not in ready  # No registered transport or PD handshake.
        params = SamplingParams(max_tokens=3, ignore_eos=True)
        for request_id, tokens in (("a", [1, 2, 3, 4, 5]), ("b", [2, 3])):
            local.add_request(request_id, tokens, params)
            channel.send(dict(kind="add", request_id=request_id, tokens=tokens, params=params))
        actual = {}
        while len(actual) < 2:
            message = receive()
            assert message["kind"] == "output"
            output = message["output"]
            assert message["request_id"] == output.request_id
            if output.finished:
                actual[output.request_id] = (output.token_ids, output.exit_depths)
        assert actual == drain(local)
        channel.send(dict(kind="stop"))
        stopped = receive()
        assert stopped["kind"] == "stopped"
        assert stopped["used_blocks"] == 0
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        channel.close()
        local.close()


def test_stage_client_owns_process_and_isolates_cancelled_generations():
    import os

    from vllm_rlt.engine.llm_engine import StageClient

    client = StageClient(
        tiny_ouro_config(),
        device="cpu",
        dtype="float32",
        seed=123,
        cache_config=CacheConfig(256, 2),
    )
    try:
        assert client.process.pid != os.getpid()
        assert client.model_config.model_type == "ouro"
        with pytest.raises(ValueError, match="prompt"):
            client.add_request("bad", [])
        assert client.request_ids() == ()
        assert client.process.is_alive()
        params = SamplingParams(max_tokens=60, ignore_eos=True)
        client.add_request("reuse", [1, 2, 3], params)
        client.abort_request("reuse")
        client.add_request("reuse", [4, 5], SamplingParams(max_tokens=2, ignore_eos=True))
        result = drain(client)
        assert len(result["reuse"][0]) == 2
        assert client.stats()["used_blocks"] == 0
        client.add_request("close-live", [2, 3], params)
    finally:
        client.close()
    assert not client.process.is_alive()
    assert client.stats() == dict(requests=0, used_blocks=0)


def test_stage_client_reports_worker_failure():
    from vllm_rlt.engine.llm_engine import StageClient

    client = StageClient(tiny_ouro_config(), dtype="float32", cache_config=CacheConfig(64, 2))
    try:
        client.process.terminate()
        client.process.join(10)
        with pytest.raises((RuntimeError, BrokenPipeError)):
            client.stats()
    finally:
        client.failure = "worker terminated by test"
        client.close()


def test_speculative_policy_schedules_newly_imported_coda():
    from vllm_rlt import SpeculativeConfig

    torch.manual_seed(123)
    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config()),
        stage_config=StageConfig(role="decode"),
        cache_config=CacheConfig(128, 2),
        speculative_config=SpeculativeConfig(3),
    )
    try:
        request = engine.reserve_transfer(
            "remote", [1, 2], SamplingParams(max_tokens=4), receiving=True
        )
        assert engine.step() == []
        engine.accept_import(request, torch.zeros(1, engine.model.config.hidden_size))
        batch = engine.scheduler.schedule()
        assert batch.phase == RequestPhase.CODA
        assert batch.requests[0].request is request
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("device", ["cuda", torch.device("cuda"), "cuda:0", 0])
def test_stage_client_cuda_async_generation_and_close(speculative, device):
    from vllm_rlt import SpeculativeConfig
    from vllm_rlt.engine.llm_engine import StageClient

    config = tiny_ouro_config(head_dim=64)
    client = StageClient(
        config,
        device=device,
        dtype="bfloat16",
        seed=123,
        cache_config=CacheConfig(128, 4),
        attention_backend="triton",
        speculative_config=SpeculativeConfig(3) if speculative else None,
        exit_config=ExitConfig("ouro" if speculative else "ouro_delayed"),
        execution_config=ExecutionConfig(async_scheduling=True, cuda_graphs=not speculative),
    )
    torch.manual_seed(123)
    local = StageEngine(
        OuroForCausalLM(config).to("cuda:0", torch.bfloat16),
        cache_config=CacheConfig(128, 4),
        attention_backend="triton",
    )
    try:
        params = SamplingParams(max_tokens=9, ignore_eos=True)
        for request_id, prompt in (("a", [1, 2, 3]), ("b", [4, 5])):
            for engine in (client, local):
                engine.add_request(request_id, prompt, params)
        assert drain(client) == drain(local)
        client.add_request("close-live", [1, 2, 3], params)
    finally:
        client.close()
        local.close()
    assert client.stats()["used_blocks"] == 0


@pytest.mark.gpu
def test_stage_client_cuda_random_sampling_reuses_external_id_with_a_new_generation():
    from vllm_rlt.engine.llm_engine import StageClient

    client = StageClient(
        tiny_ouro_config(head_dim=64),
        device="cuda",
        dtype="bfloat16",
        seed=123,
        cache_config=CacheConfig(128, 4),
        attention_backend="triton",
    )
    params = SamplingParams(max_tokens=3, temperature=0.8, top_k=5, seed=9, ignore_eos=True)
    long_params = SamplingParams(max_tokens=60, temperature=0.8, top_k=5, seed=9, ignore_eos=True)
    try:
        client.add_request("reuse", [1, 2, 3], params)
        first = drain(client)["reuse"]
        assert len(first[0]) == params.max_tokens

        # Let the worker advance this request's RNG slot before aborting. Aborting
        # must drop that slot in the worker, not only in the parent process.
        client.add_request("reuse", [1, 2, 3], long_params)
        old_generation = client.requests["reuse"]
        for _ in range(200):
            outputs = client.step()
            if outputs:
                break
        else:
            pytest.fail("worker produced no output before abort")
        # Aborting a finished request would make the replay assertion a
        # coincidence, so require a live, unfinished request here.
        assert client.has_request("reuse")
        assert not any(output.finished for output in outputs)
        client.abort_request("reuse")
        assert "reuse" not in client.requests

        client.add_request("reuse", [1, 2, 3], params)
        new_generation = client.requests["reuse"]
        # The external ID is reused, but the execution identity is not.
        assert new_generation != old_generation
        reused = drain(client)["reuse"]
        # The dropped slot restarts from the request seed instead of continuing
        # the aborted stream.
        assert reused == first
        assert client.stats()["used_blocks"] == 0
    finally:
        client.close()


class _RecordingChannel:
    """Capture worker messages without a pipe; no inbound commands are queued."""

    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)

    def poll(self, timeout=0):
        return False

    def recv(self):
        raise RuntimeError("no inbound messages")

    def close(self):
        pass


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_worker_release_drops_the_aborted_rng_entry(device):
    from vllm_rlt.engine.worker import StageWorker

    channel = _RecordingChannel()
    options = dict(
        seed=123,
        dtype="bfloat16" if device == "cuda" else "float32",
        revision=None,
        engine=dict(
            attention_backend="triton" if device == "cuda" else "torch",
            cache_config=CacheConfig(128, 4),
            scheduler_config=SchedulerConfig(max_num_seqs=2),
        ),
    )
    worker = StageWorker(
        StageConfig(stage_id="local"),
        device,
        channel,
        tiny_ouro_config(head_dim=64),
        options,
    )
    engine = worker.engine

    def step_until_sampled(request_id):
        for _ in range(200):
            request = engine.scheduler.requests.get(request_id)
            generator = engine.rng.peek(request_id)
            if request is not None and generator is not None and request.generated_token_ids:
                return generator
            for output in engine.step():
                worker.output(output)
        pytest.fail(f"request {request_id!r} did not sample a token")

    params = SamplingParams(max_tokens=60, temperature=0.8, top_k=7, seed=9, ignore_eos=True)
    try:
        worker.command(dict(kind="add", request_id="reuse", tokens=[1, 2, 3], params=params))
        old = step_until_sampled("reuse")
        assert old.device.type == torch.device(device).type
        assert engine.has_request("reuse")

        # Termination is owned by the worker. The parent StageClient cannot see
        # this registry, so assert the cleanup at the worker-side boundary.
        worker.command(dict(kind="abort", request_id="reuse"))
        assert not engine.has_request("reuse")
        assert engine.rng.peek("reuse") is None
        assert engine.cache_manager.num_used_blocks == 0

        # A new request reusing the same worker-side ID starts a fresh stream.
        worker.command(dict(kind="add", request_id="reuse", tokens=[1, 2, 3], params=params))
        fresh = step_until_sampled("reuse")
        assert fresh is not old
        worker.command(dict(kind="abort", request_id="reuse"))
        assert engine.rng.peek("reuse") is None
    finally:
        engine.close()


@pytest.mark.gpu
def test_public_llm_cuda_random_sampling_reaches_the_worker():
    from vllm_rlt import LLM

    torch.manual_seed(123)
    llm = LLM(
        tiny_ouro_config(head_dim=64),
        device="cuda",
        dtype="bfloat16",
        attention_backend="triton",
        cache_config=CacheConfig(128, 4),
    )
    params = SamplingParams(
        max_tokens=4, temperature=0.8, top_k=5, top_p=0.9, seed=7, ignore_eos=True
    )
    try:
        first = llm.generate([[1, 2, 3]], params)[0]
        second = llm.generate([[1, 2, 3]], params)[0]
        # StageClient maps external IDs to internal generations and back, so the
        # caller still sees its own request IDs on both outputs.
        assert (first.request_id, second.request_id) == ("0", "1")
        assert len(first.token_ids) == params.max_tokens
        assert len(second.token_ids) == params.max_tokens
        # The worker dropped the finished request's slot, so the same seed replays
        # its stream across two independent requests.
        assert second.token_ids == first.token_ids
        assert llm.engine.process.is_alive()
    finally:
        llm.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_connector_becoming_ready_after_selection_is_not_a_scheduler_stall(asynchronous):
    from unittest.mock import Mock

    engine = make_stage_engine(role="prefill", asynchronous=asynchronous)
    connector = engine.connector = RecordingConnector()
    # GPU events can turn ready immediately after the first query.
    connector.can_schedule = Mock(side_effect=[False, True, True, True])
    try:
        engine.add_request("r", [1, 2], SamplingParams(max_tokens=1))
        assert engine.step() == []
        assert connector.can_schedule.call_count == 1
        assert engine.scheduler.requests["r"].num_preluded_tokens == 0
        engine.step()
        assert engine.scheduler.requests["r"].num_preluded_tokens == 2
    finally:
        engine.close()


def cpu_connector(engine, *, notify):
    """Exercise the actual protocol adapter without remote memory access."""
    from unittest.mock import Mock

    from vllm_rlt.pd.config import PDConfig
    from vllm_rlt.pd.connector import PDStageConnector

    transport = Mock(
        info={"agent": "cpu-test"},
        pending=[],
        inflight_bytes=0,
        bytes_sent=0,
        transfers=0,
        transfer_seconds=0,
    )
    transport.poll.return_value = ([], [])
    return PDStageConnector(
        engine,
        torch.zeros(2, engine.model.config.hidden_size),
        transport,
        PDConfig(),
        notify=notify,
    )


def test_connected_stage_client_preserves_owner_after_failure():
    import time

    from vllm_rlt.engine.llm_engine import StageClient

    client = StageClient(
        tiny_ouro_config(),
        dtype="float32",
        stage_config=StageConfig(stage_id="p", role="prefill"),
        connector_factory=cpu_connector,
        wait_for_ready=False,
        cache_config=CacheConfig(64, 2),
    )
    try:
        client.wait_ready()
        assert client.ready and client.info["role"] == "prefill"
        client.send("connect", peers=[])
        deadline = time.monotonic() + 10
        while not client.connected:
            client.poll()
            assert time.monotonic() < deadline
        client.send("invalid-command")
        with pytest.raises(ValueError, match="unknown stage command"):
            while time.monotonic() < deadline:
                client.poll()
        assert client.process.is_alive()  # Keep receiver memory until writers stop.
        with pytest.raises(RuntimeError, match="coordinator"):
            client.close()
        assert client.process.is_alive()
    finally:
        client.terminate()
        client.join(5, force=True)
    assert client.closed and not client.process.is_alive()


def test_stage_client_controls_can_be_submitted_before_collecting():
    import time

    from vllm_rlt.engine.llm_engine import StageClient

    client = StageClient(tiny_ouro_config(), dtype="float32", cache_config=CacheConfig(64, 2))
    try:
        discarded = client.submit_control("stats")
        client.discard_control(discarded)
        calls = [client.submit_control("stats") for _ in range(2)]
        replies = {}
        deadline = time.monotonic() + 10
        while len(replies) < len(calls):
            assert time.monotonic() < deadline
            client.poll()
            for control_id in calls:
                reply = client.take_control_reply(control_id)
                if reply is not None:
                    replies[control_id] = reply
            time.sleep(0.001)
        assert all(r["result"] == dict(requests=0, used_blocks=0) for r in replies.values())
        assert not client.pending_controls and not client.replies
    finally:
        client.close()


@pytest.mark.parametrize(
    "options,expected_device,expected_dtype",
    [
        ({}, torch.device("cpu"), "float32"),
        ({"device": 3}, 3, "float32"),
        ({"device": "cuda:3", "dtype": torch.bfloat16}, "cuda:3", "bfloat16"),
        ({"device": "cuda:2"}, "cuda:2", "float32"),
        ({"dtype": "float16"}, torch.device("cpu"), "float16"),
    ],
)
def test_module_weight_source_preserves_explicit_deployment(
    monkeypatch, options, expected_device, expected_dtype
):
    from unittest.mock import Mock

    from vllm_rlt.engine import llm_engine

    launch = Mock(return_value=(Mock(), Mock()))
    monkeypatch.setattr(llm_engine, "start_stage_worker", launch)
    model = OuroForCausalLM(tiny_ouro_config())
    client = llm_engine.StageClient(model, wait_for_ready=False, **options)
    _, device, weights, settings = launch.call_args.args
    assert device == expected_device
    assert settings["dtype"] == expected_dtype
    assert not client.ready
    assert client.model_config.model_specific_config is model.config
    assert settings["model_config"] is client.model_config
    assert weights is not model
    assert next(weights.parameters()).device.type == "cpu"
    client.closed = True


@pytest.mark.parametrize("asynchronous", [False, True])
def test_backpressured_export_allows_queued_arrival_to_wait(asynchronous):
    from dataclasses import replace

    engine = make_stage_engine(role="prefill", asynchronous=asynchronous)
    engine.scheduler.config = replace(engine.scheduler.config, max_num_seqs=1)
    connector = engine.connector = RecordingConnector()
    engine.reserve_transfer("active", [1, 2], SamplingParams(max_tokens=1), receiving=False)
    engine.scheduler._admit()
    connector.blocked.add("active")
    engine.reserve_transfer("waiting", [3, 4], SamplingParams(max_tokens=1), receiving=False)
    assert engine.step() == []
    assert engine.scheduler.requests["waiting"].phase == RequestPhase.WAITING
    connector.blocked.clear()
    engine.step()
    assert engine.last_schedule is not None
    engine.close()


@pytest.mark.parametrize("receiving", [False, True])
@pytest.mark.parametrize("preemption", [False, True])
def test_transfer_reservation_preserves_active_prompt_growth(receiving, preemption):
    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config()),
        cache_config=CacheConfig(16, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=2,
            prefill_chunk_size=2,
            enable_preemption=preemption,
            wavefront_chunked_prefill=True,
        ),
        execution_config=ExecutionConfig(),
    )
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    try:
        engine.add_request("local", list(range(1, 9)), params)
        engine.scheduler._admit()
        assert engine.cache_manager.allocated_blocks("local") == 4
        assert engine.reserve_transfer("remote", [1, 2], params, receiving=receiving) is None
        assert not engine.has_request("remote")
        assert not engine.cache_manager.has_allocation("remote")
        assert all("remote" not in queue for queue in engine.scheduler.queues.values())
        assert len(drain(engine)["local"][0]) == 1
        # A rejected generation can retry after the local request releases its KV.
        remote = engine.reserve_transfer("remote", [1, 2], params, receiving=receiving)
        assert remote is not None
        engine.release_transfer(remote)
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine.close()


def test_waiting_import_retains_future_decode_growth_budget():
    engine = StageEngine(
        OuroForCausalLM(tiny_ouro_config()),
        cache_config=CacheConfig(16, 2, incremental_allocation=True),
    )
    try:
        request = engine.reserve_transfer(
            "import", [1, 2], SamplingParams(max_tokens=7), receiving=True
        )
        assert request is not None
        assert engine.cache_manager.allocated_blocks("import") == 4
        # Twelve currently free blocks are promised to this import's decode growth.
        assert (
            engine.reserve_transfer("other", [3, 4], SamplingParams(max_tokens=1), receiving=False)
            is None
        )
        engine.release_transfer(request)
        assert engine.cache_manager.num_used_blocks == 0
    finally:
        engine.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_decode_import_preemption_restores_outputs_and_rng(device):
    source_engine = make_stage_engine(role="prefill")
    source_engine.connector = RecordingConnector()
    params = SamplingParams(max_tokens=5, temperature=0.7, seed=41, ignore_eos=True)
    source = source_engine.reserve_transfer("source", [1, 2, 3, 4], params, receiving=False)
    while source.phase != RequestPhase.TRANSFERRING:
        source_engine.step()
    results = []
    try:
        for preempt in (False, True):
            torch.manual_seed(123)
            engine = StageEngine(
                OuroForCausalLM(tiny_ouro_config()).to(device),
                stage_config=StageConfig(role="decode"),
                cache_config=CacheConfig(128, 2),
                scheduler_config=SchedulerConfig(max_num_seqs=1, enable_preemption=True),
                execution_config=ExecutionConfig(async_scheduling=False),
            )
            try:

                def import_request(name):
                    target = engine.reserve_transfer(
                        name, source.prompt_token_ids, params, receiving=True
                    )
                    for left, right in zip(
                        source_engine.cache_manager.plane_block_tables("source"),
                        engine.cache_manager.plane_block_tables(name),
                    ):
                        for src, dst in zip(left, right):
                            engine.cache_manager.key_cache[dst].copy_(
                                source_engine.cache_manager.key_cache[src]
                            )
                            engine.cache_manager.value_cache[dst].copy_(
                                source_engine.cache_manager.value_cache[src]
                            )
                    engine.accept_import(target, source.hidden_state.to(device).clone())
                    return target

                victim = import_request("victim")
                while not victim.generated_token_ids:
                    engine.step()
                waiting = import_request("waiting")
                if preempt:
                    assert engine.scheduler.preempt(waiting)
                    assert victim.phase == RequestPhase.WAITING
                results.append(drain(engine))
                assert engine.preemption.resumptions == int(preempt)
                assert not engine.preemption.snapshots
                assert engine.cache_manager.num_used_blocks == 0
            finally:
                engine.close()
        assert results[0] == results[1]
    finally:
        source_engine.release_transfer(source)
        source_engine.close()
