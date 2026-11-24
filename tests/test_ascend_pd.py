"""Real same-host Ascend 2P2D coverage for the HCCL transport."""

import time

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import StageClient
from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.engine import PDEngine

pytestmark = pytest.mark.npu


def _params():
    return SamplingParams(
        max_tokens=4,
        min_loops=4,
        max_loops=4,
        temperature=0,
        ignore_eos=True,
    )


def _cache():
    return CacheConfig(num_blocks=64, block_size=16)


def _prefill_scheduler():
    return SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64, prefill_chunk_size=64)


def _decode_scheduler():
    return SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8)


def _drain(engine, expected):
    outputs = {}
    deadline = time.monotonic() + 180
    while len(outputs) < expected:
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output
        if time.monotonic() >= deadline:
            raise TimeoutError(f"only received {len(outputs)} of {expected} outputs")
    return outputs


def _make_pd(model):
    return PDEngine(
        model,
        pd_config=PDConfig(
            device_type="npu",
            prefill_devices=(0, 1),
            decode_devices=(2, 3),
            backend="HCCL",
            transfer_chunk_bytes=1 << 20,
            max_inflight_bytes=4 << 20,
            startup_timeout=180,
            shutdown_timeout=60,
        ),
        prefill_cache_config=_cache(),
        decode_cache_config=_cache(),
        prefill_scheduler_config=_prefill_scheduler(),
        decode_scheduler_config=_decode_scheduler(),
        execution_config=ExecutionConfig(async_scheduling=False),
        exit_config=ExitConfig("ouro"),
        attention_backend="ascend_eager",
        dtype="bfloat16",
    )


def _make_single(model):
    return StageClient(
        model,
        device="npu:0",
        stage_config=StageConfig(role="prefill_decode"),
        cache_config=_cache(),
        scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=64),
        execution_config=ExecutionConfig(async_scheduling=False),
        exit_config=ExitConfig("ouro"),
        attention_backend="ascend_eager",
        dtype="bfloat16",
    )


def test_ascend_2p2d_matches_single_stage_and_releases_resources():
    torch.manual_seed(41)
    config = tiny_ouro_config(hidden_size=256, intermediate_size=512, head_dim=64)
    prompts = {
        "r0": [3, 7, 2, 11, 5, 17, 8],
        "r1": [5, 9, 11, 13, 15, 19],
        "r2": [2, 4, 6, 8, 10],
        "r3": [1, 12, 14, 16, 18, 20, 22, 24],
    }

    torch.manual_seed(41)
    single = _make_single(OuroForCausalLM(config))
    try:
        for request_id, prompt in prompts.items():
            single.add_request(request_id, prompt, _params())
        expected = _drain(single, len(prompts))
    finally:
        single.close()

    torch.manual_seed(41)
    pd = _make_pd(OuroForCausalLM(config))
    try:
        for request_id, prompt in prompts.items():
            pd.add_request(request_id, prompt, _params())
        actual = _drain(pd, len(prompts))
        for request_id in prompts:
            assert actual[request_id].token_ids == expected[request_id].token_ids
            assert actual[request_id].exit_depths == expected[request_id].exit_depths
            assert actual[request_id].finish_reason == expected[request_id].finish_reason
    finally:
        pd.close()

    assert pd.closed
    assert all(metrics["used_blocks"] == 0 for metrics in pd.worker_metrics.values())
    assert sum(metrics.get("bytes_sent", 0) for metrics in pd.worker_metrics.values()) > 0


def test_ascend_2p2d_cancel_reuse_and_worker_failure():
    torch.manual_seed(41)
    config = tiny_ouro_config(hidden_size=256, intermediate_size=512, head_dim=64)
    pd = _make_pd(OuroForCausalLM(config))
    params = _params()
    try:
        pd.add_request("cancel", [3, 7, 2, 11, 5, 17, 8], params)
        deadline = time.monotonic() + 60
        while True:
            pd.step()
            transfer = next(iter(pd.transfers.values()), None)
            if transfer is not None and transfer.phase != "waiting":
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("cancel request was not admitted")
        pd.abort_request("cancel")
        while pd.transfers:
            pd.step()
            if time.monotonic() >= deadline:
                raise TimeoutError("cancelled HCCL transfer was not retired")

        pd.add_request("cancel", [5, 9, 11, 13], params)
        outputs = _drain(pd, 1)
        assert outputs["cancel"].finish_reason == "length"
        assert all(peer.reserved_blocks == 0 and peer.slots == 0 for peer in pd.peers.values())

        victim = next(iter(pd.peers.values()))
        victim.client.process.kill()
        victim.client.process.join()
        with pytest.raises(RuntimeError):
            pd.step()
    finally:
        pd.close()
