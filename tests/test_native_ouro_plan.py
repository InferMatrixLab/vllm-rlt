"""Native host dispatch keeps eager states, live weights and callback semantics."""

import pytest
import torch

from tests.test_ascend import make_npu_model
from tests.test_fast_execution_plan import cache_for, metadata
from tests.test_npu_async import drain
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.request import Operation
from vllm_rlt.worker.ascend import AscendOuroRuntime
from vllm_rlt.worker.executor import ForwardBatch

pytestmark = pytest.mark.npu


@torch.inference_mode()
def test_native_plan_matches_hidden_and_all_kv_layers():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    expected = None
    for native in [False, True]:
        cache = cache_for(model)
        runtime = AscendOuroRuntime(model, cache, native=native)
        actual = []
        try:
            for start, ids in [(0, [3, 7, 11]), (3, [13])]:
                forward = ForwardBatch(
                    Operation.RECURRENT,
                    hidden=model.prelude(torch.tensor(ids, device="npu:0")),
                    metadata=metadata(cache, start, len(ids)),
                )
                output = runtime.forward(forward)
                actual.append(output.hidden.clone())
            for depth in range(4):
                for layer in range(model.config.num_hidden_layers):
                    actual.extend(x.clone() for x in cache._contiguous.groups[depth, layer].pair)
            if expected is None:
                expected = actual
            else:
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                assert runtime.layer_executor.native_layers == 8 * model.config.num_hidden_layers
                assert not runtime.layer_executor.fallbacks
        finally:
            cache.free("a")


def engine_for(model, native, *, multi_stream=False):
    return StageEngine(
        model,
        attention_backend="ascend_eager",
        cache_config=CacheConfig(num_blocks=128, block_size=16, contiguous_kv=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=128, prefill_chunk_size=128
        ),
        execution_config=ExecutionConfig(
            async_scheduling=True,
            multi_stream=multi_stream,
            use_uva=False,
            coalesce_fixed_prelude=True,
            native_eager_plan=native,
        ),
        exit_config=ExitConfig("ouro_delayed"),
    )


@torch.inference_mode()
def test_native_plan_uses_assigned_weights_and_preserves_child_hooks():
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    native = engine_for(model, True)

    def once(engine):
        engine.add_request("a", [3, 7, 11], params)
        return drain(engine)

    try:
        once(native)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        state["model.layers.0.self_attn.q_proj.weight"].mul_(0.5)
        model.load_state_dict(state, assign=True)
        handle = model.model.layers[0].input_layernorm.register_forward_pre_hook(
            lambda m, args: (args[0] + 0.125,)
        )
        reference = engine_for(model, False)
        try:
            assert once(native) == once(reference)
            assert (
                native.model_runner.runtime.layer_executor.fallbacks["child_hooks_or_forward"] > 0
            )
        finally:
            handle.remove()
            reference.close()
        reference = engine_for(model, False)
        try:
            assert once(native) == once(reference)
        finally:
            reference.close()
        assert native.model_runner.runtime.layer_executor.native_layers > 0
    finally:
        native.close()


@pytest.mark.parametrize("multi_stream", [False, True])
@torch.inference_mode()
def test_native_plan_cancel_reuse_and_owned_outputs(multi_stream):
    resolve_device("npu")
    torch.npu.set_device(0)
    model = make_npu_model(torch.device("npu:0"))
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    reference = engine_for(model, False, multi_stream=multi_stream)
    for i, p in enumerate([[3, 7, 11], [13, 17, 19]]):
        reference.add_request(str(i), p, params)
    expected = drain(reference)
    reference.close()
    engine = engine_for(model, True, multi_stream=multi_stream)
    held = []
    handle = model.model.layers[0].register_forward_hook(
        lambda m, args, output: held.append(output)
    )
    try:
        for i, p in enumerate([[3, 7, 11], [13, 17, 19]]):
            engine.add_request(str(i), p, params)
        engine.step()
        # A consumer on another stream must first wait for the producer.
        torch.npu.synchronize()
        snapshot = held[0].clone()
        engine.step()
        engine.step()
        engine.abort_request("0")
        engine.add_request("0", [3, 7, 11], params)
        assert drain(engine) == expected
        torch.testing.assert_close(held[0], snapshot, atol=0, rtol=0)
        assert engine.model_runner.runtime.layer_executor.native_layers > 0
    finally:
        handle.remove()
        engine.close()
