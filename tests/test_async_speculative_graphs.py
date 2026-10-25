"""Device-owned positions and in-flight results must survive graph reuse."""

import pytest
import torch

from tests.helpers import bootstrap_speculation, tiny_ouro_config
from tests.test_async_speculative import drain
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM


def make_engine(backend, *, graphs, max_rows=32, max_graphs=16, multi_stream=True):
    torch.manual_seed(123)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
    return StageEngine(
        model,
        attention_backend=backend,
        cache_config=CacheConfig(128, 16, enable_prefix_caching=True, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=16, prefill_chunk_size=8
        ),
        execution_config=ExecutionConfig(
            async_scheduling=True,
            multi_stream=multi_stream,
            static_buffers=True,
            pad_to_power_of_two=True,
            cuda_graphs=graphs,
            cuda_graph_max_batch_size=max_rows,
            cuda_graph_max_graphs=max_graphs,
        ),
        speculative_config=SpeculativeConfig(3),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "flash_attn_4"])
@pytest.mark.parametrize("max_rows,max_graphs", [(32, 16), (2, 1)])
@pytest.mark.parametrize("multi_stream", [False, True])
def test_graphs_match_eager_with_ragged_rows_page_growth_and_reuse(
    backend, max_rows, max_graphs, multi_stream
):
    results = []
    for graphs in (False, True):
        engine = make_engine(
            backend,
            graphs=graphs,
            max_rows=max_rows,
            max_graphs=max_graphs,
            multi_stream=multi_stream,
        )
        try:
            rounds = []
            for reuse in range(2):
                for i, (prompt_length, output_length) in enumerate([(15, 23), (17, 9), (31, 1)]):
                    engine.add_request(
                        str(i),
                        [2 + i, 3, 4] * (prompt_length // 3) + [5] * (prompt_length % 3),
                        SamplingParams(max_tokens=output_length, ignore_eos=True),
                    )
                outputs = drain(engine)
                assert len(outputs) == 3
                rounds.append({rid: o.token_ids for rid, o in outputs.items()})
                assert not engine.speculative_runner.states
                assert all(not b.leased for b in engine.speculative_runner.buffers)
            runner = engine.speculative_runner
            assert runner.submitted_before_collect > 0
            if graphs:
                assert runner.graphs.captures > 0
                assert runner.graphs.replays > runner.graphs.captures
                assert runner.coda_graphs.captures > 0
                assert runner.coda_graphs.replays > runner.coda_graphs.captures
                if max_rows == 2:
                    assert runner.graphs.fallbacks > 0
                else:
                    assert any(key[0] > 3 for key in runner.graphs.entries)
            results.append(rounds)
        finally:
            engine.close()
    assert results[0] == results[1]


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "flash_attn_4"])
@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_two_graph_rounds_keep_device_frontier_and_rejection_mask(backend, accepted, monkeypatch):
    engine = make_engine(backend, graphs=True)
    try:
        request = bootstrap_speculation(engine)
        runner = engine.speculative_runner
        start = request.position
        allocation = engine.cache_manager._get_allocation("r")
        written_before = [[layer.prefix for layer in depth] for depth in allocation.written]
        calls = 0

        def controlled(hidden):
            nonlocal calls
            phase = calls % 4
            calls += 1
            logits = torch.full(
                (len(hidden), engine.model.config.vocab_size), -100.0, device="cuda"
            )
            if phase < 3:
                logits[:, 10 + phase] = 100
            else:
                for row in range(4):
                    logits[row, 10 + row if row < accepted else 20 + row] = 100
            return logits

        # Control acceptance independently of attention arithmetic; the real
        # recurrent graphs still execute every draft and verification loop.
        monkeypatch.setattr(runner, "_coda", controlled)

        def forbidden(*args, **kwargs):
            raise AssertionError("host read during device-owned graph submission")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "item", forbidden)
            patch.setattr(torch.Tensor, "tolist", forbidden)
            engine.step()
            engine.step()
        assert len(engine._pending_speculative) == 2
        assert len(request.generated_token_ids) == 1
        runner.synchronize()
        assert int(runner.states["r"].position) == start + 2 * (accepted + 1)
        assert [[layer.prefix for layer in depth] for depth in allocation.written] == written_before
        output = engine._collect_speculative()[0]
        assert output.token_ids[1:] == list(range(10, 10 + accepted)) + [20 + accepted]
        assert int(runner.states["r"].position) == start + 2 * (accepted + 1)
        assert runner.graphs.replays > runner.graphs.captures
        engine.abort_request("r")
        drain(engine)
        assert not runner.states and all(not b.leased for b in runner.buffers)
        # The same row shapes must replay without device-wide capture waits or
        # host reads, even though the request identity and KV addresses changed.
        bootstrap_speculation(engine, rid="next")
        calls = 0
        captures = runner.graphs.captures
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "synchronize", forbidden)
            patch.setattr(torch.Tensor, "item", forbidden)
            patch.setattr(torch.Tensor, "tolist", forbidden)
            engine.step()
            engine.step()
        assert len(engine._pending_speculative) == 2
        assert runner.graphs.captures == captures
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "flash_attn_4"])
def test_graph_eos_retires_later_round_before_id_reuse(backend, monkeypatch):
    engine = make_engine(backend, graphs=True)
    try:
        bootstrap_speculation(engine, ignore_eos=False)
        runner = engine.speculative_runner
        with monkeypatch.context() as patch:
            patch.setattr(
                runner,
                "_coda",
                lambda h: torch.zeros(len(h), engine.model.config.vocab_size, device=h.device),
            )
            engine.step()
            engine.step()
            assert len(engine._pending_speculative) == 2
            outputs = engine._collect_speculative()
            assert outputs[0].finished and outputs[0].token_ids[-1] == 0
        engine.add_request("r", [7, 8, 9], SamplingParams(max_tokens=9, ignore_eos=True))
        output = drain(engine)["r"]
        assert output.finished and len(output.token_ids) == 9
        assert not runner.states and all(not b.leased for b in runner.buffers)
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "flash_attn_4"])
def test_failed_graph_round_drains_readers_and_releases_buffers(backend, monkeypatch):
    engine = make_engine(backend, graphs=True)
    try:
        bootstrap_speculation(engine)
        engine.step()
        runner = engine.speculative_runner
        verify = runner._verify

        def fail_after_verify(*args, **kwargs):
            verify(*args, **kwargs)
            raise RuntimeError("failed after graph submission")

        monkeypatch.setattr(runner, "_verify", fail_after_verify)
        with pytest.raises(RuntimeError, match="failed after graph submission"):
            engine.step()
        assert not engine._pending_speculative and not engine._retiring_speculative
        assert not runner.states
        assert all(not b.leased and not b.keepalive for b in runner.buffers)
    finally:
        engine.close()


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "flash_attn_4"])
def test_pd_async_graphs_match_local_and_cancel_safely(backend):
    pytest.importorskip("nixl")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two reserved GPUs")
    from tests.test_pd import drain as drain_pd
    from vllm_rlt.pd.config import PDConfig
    from vllm_rlt.pd.engine import PDEngine

    reference = make_engine(backend, graphs=False)
    pd = PDEngine(
        tiny_ouro_config(head_dim=64),
        seed=123,
        pd_config=PDConfig(prefill_devices=(0,), decode_devices=(1,)),
        prefill_cache_config=CacheConfig(128, 16, enable_prefix_caching=True),
        decode_cache_config=CacheConfig(128, 16, enable_prefix_caching=True),
        prefill_scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=16, prefill_chunk_size=8
        ),
        decode_scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=16),
        execution_config=ExecutionConfig(
            async_scheduling=True,
            cuda_graphs=True,
            static_buffers=True,
            pad_to_power_of_two=True,
        ),
        speculative_config=SpeculativeConfig(3),
        attention_backend=backend,
    )
    try:
        params = SamplingParams(max_tokens=13, ignore_eos=True)
        for repeat in range(2):
            for i, prompt in enumerate([[2, 3, 4] * 7, [7, 8], [9] * 17]):
                for target in (reference, pd):
                    target.add_request(str(i), prompt, params)
            expected, actual = drain(reference), drain_pd(pd)
            assert {k: v.token_ids for k, v in actual.items()} == {
                k: v.token_ids for k, v in expected.items()
            }
        pd.add_request("cancel", [7, 8, 9], SamplingParams(max_tokens=60, ignore_eos=True))
        import time

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if any(len(o.token_ids) > 1 for o in pd.step()):
                break
        else:
            pytest.fail("PD speculative decode did not start")
        pd.abort_request("cancel")
        for target in (reference, pd):
            target.add_request("cancel", [2, 3], params)
        assert drain(reference)["cancel"].token_ids == drain_pd(pd)["cancel"].token_ids
        assert all(peer.slots == 0 and peer.reserved_blocks == 0 for peer in pd.peers.values())
    finally:
        pd.close()
        reference.close()
    assert all(m["used_blocks"] == 0 for m in pd.worker_metrics.values())
