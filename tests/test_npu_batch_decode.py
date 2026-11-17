"""Owned batch state and completion events preserve request lifetimes."""

from contextlib import nullcontext

import pytest
import torch

from tests.helpers import make_cpu_cache, tiny_ouro_model
from vllm_rlt import ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.request import Request
from vllm_rlt.worker.ascend import AscendModelRunner
from vllm_rlt.worker.model_runner import ModelRunner


def runner_cpu():
    model = tiny_ouro_model()
    return ModelRunner(
        model,
        make_cpu_cache(model.config),
        execution_config=ExecutionConfig(
            async_scheduling=True,
            multi_stream=False,
            static_buffers=False,
            pad_to_power_of_two=False,
            use_uva=False,
        ),
        scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=8),
    )


def requests():
    return [Request(x, [1], SamplingParams(max_tokens=3)) for x in ["a", "b"]]


def test_cpu_keeps_copy_semantics():
    runner = runner_cpu()
    reqs = requests()
    hidden = torch.arange(64.0).reshape(2, 32)
    runner._save_batch(reqs, hidden, None)
    expected = reqs[0].hidden_state.clone()
    hidden.fill_(99)
    assert torch.equal(reqs[0].hidden_state, expected)
    assert not hasattr(runner, "hidden_batch")


class Event:
    def __init__(self):
        self.recorded = []

    def record(self, stream):
        self.recorded.append(stream)


class Stream:
    def __init__(self):
        self.waited = []
        self.synchronized = 0

    def wait_event(self, event):
        self.waited.append(event)

    def synchronize(self):
        self.synchronized += 1


class DeviceOps:
    def __init__(self):
        self.created = []

    def stream(self, stream):
        return nullcontext()

    def Event(self):
        e = Event()
        self.created.append(e)
        return e


def write_depth(runner, reqs, depth, *, layers=None):
    cache = runner.cache_manager
    for r in reqs:
        assert cache.allocate(r.request_id, 4)
        r.loops_done = depth + 1
    c = runner.model.config
    data = torch.arange(
        len(reqs) * c.num_key_value_heads * c.head_dim, dtype=torch.float32
    ).reshape(len(reqs), c.num_key_value_heads, c.head_dim)
    for layer in range(cache.num_layers) if layers is None else layers:
        cache.write(
            layer,
            [r.request_id for r in reqs],
            [depth] * len(reqs),
            [0] * len(reqs),
            data,
            data + 1,
        )
    return data


def fake_events(runner, reqs):
    runner.boundary_stream = Stream()
    runner.device_ops = DeviceOps()
    source = Event()
    runner.events = {r.request_id: source for r in reqs}
    return source


def test_full_depth_finalize_reuses_event_and_still_validates():
    runner = runner_cpu()
    reqs = requests()
    write_depth(runner, reqs, 3)
    source = fake_events(runner, reqs)
    AscendModelRunner._finalize_npu_batch(runner, reqs)
    assert all(runner.events[r.request_id] is source for r in reqs)
    assert not runner.device_ops.created and not runner.boundary_stream.waited
    bad = runner_cpu()
    rs = requests()
    write_depth(bad, rs, 3, layers=[0])
    fake_events(bad, rs)
    with pytest.raises(RuntimeError, match="every layer"):
        AscendModelRunner._finalize_npu_batch(bad, rs)
    assert not bad.device_ops.created


def test_early_exit_finalize_copies_all_planes_then_shares_event():
    runner = runner_cpu()
    reqs = requests()
    data = write_depth(runner, reqs, 0)
    source = fake_events(runner, reqs)
    AscendModelRunner._finalize_npu_batch(runner, reqs)
    assert runner.boundary_stream.waited == [source]
    assert len(runner.device_ops.created) == 1
    done = runner.device_ops.created[0]
    assert done.recorded == [runner.boundary_stream]
    assert all(runner.events[r.request_id] is done for r in reqs)
    for i, r in enumerate(reqs):
        for depth in range(1, 4):
            for layer in range(runner.cache_manager.num_layers):
                k, v = runner.cache_manager.read(layer, r.request_id, depth, 1)
                assert torch.equal(k[0], data[i]) and torch.equal(v[0], data[i] + 1)


def test_partial_finalize_failure_drains_copies_before_cleanup():
    runner = runner_cpu()
    reqs = requests()
    write_depth(runner, reqs, 0)
    fake_events(runner, reqs)
    # Keep allocation live but make the second row lack a required layer.
    runner.cache_manager.truncate_suffix("b", 0)
    with pytest.raises(RuntimeError, match="every layer"):
        AscendModelRunner._finalize_npu_batch(runner, reqs)
    assert runner.boundary_stream.synchronized == 1
    assert not runner.device_ops.created


@pytest.mark.npu
@pytest.mark.parametrize("multi_stream", [False, True])
@pytest.mark.parametrize("coalesce", [False, True])
@torch.inference_mode()
def test_npu_owned_states_survive_future_tokens(multi_stream, coalesce):
    from tests.test_ascend import make_npu_model
    from tests.test_npu_async import drain, make_engine
    from vllm_rlt.device import resolve_device

    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    model = make_npu_model(device)
    baseline = make_engine(model, False, multi_stream)
    params = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    prompts = [[3, 7, 11] * 7, [2, 5, 13]]
    for i, p in enumerate(prompts):
        baseline.add_request(str(i), p, params)
    expected = drain(baseline)
    baseline.close()
    engine = make_engine(model, True, multi_stream)
    engine.scheduler.coalesce_fixed_decode = coalesce
    runner = engine.model_runner
    save = runner._save_batch
    gather = runner._gather
    held = []
    reuses = []

    def capture(reqs, hidden, routing):
        save(reqs, hidden, routing)
        if runner.hidden_batch is not None:
            held.append((reqs[0].hidden_state, reqs[0].hidden_state.clone()))

    def load(reqs, workspace, size):
        saved = runner.hidden_batch
        result = gather(reqs, workspace, size)
        if saved is not None and result is saved.hidden:
            reuses.append(True)
        return result

    runner._save_batch = capture
    runner._gather = load
    try:
        for repeat in range(2):
            for i, p in enumerate(prompts):
                engine.add_request(str(i), p, params)
            assert drain(engine) == expected
        assert held and reuses
        for actual, reference in held:
            assert torch.equal(actual, reference)
        assert runner.hidden_batch is None
    finally:
        engine.close()


@pytest.mark.npu
@torch.inference_mode()
def test_npu_retained_batch_records_delayed_consumer():
    from tests.test_ascend import make_npu_model
    from tests.test_npu_async import make_engine
    from vllm_rlt.device import resolve_device

    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    engine = make_engine(make_npu_model(device), True, True)
    runner = engine.model_runner
    producer = torch.npu.Stream(device=device)
    consumer = torch.npu.Stream(device=device)
    blocker = torch.npu.Stream(device=device)
    reqs = requests()
    with torch.npu.stream(producer):
        source = torch.full((2, 256), 7.0, device=device, dtype=torch.bfloat16)
    producer.synchronize()
    pointer = source.data_ptr()
    runner._save_batch(reqs, source, None)
    matrix = torch.ones((16384, 16384), device=device, dtype=torch.bfloat16)
    product = torch.empty_like(matrix)
    blocker.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(blocker):
        for _ in range(4):
            torch.mm(matrix, matrix, out=product)
        gate = torch.npu.Event()
        gate.record(blocker)
    consumer.wait_event(gate)
    try:
        with torch.npu.stream(consumer):
            gathered = runner._gather(reqs, None, 2)
            assert gathered is source
            actual = gathered + 0
        runner.hidden_batch = None
        for r in reqs:
            r.hidden_state = None
        del source, gathered
        with torch.npu.stream(producer):
            poison = [
                torch.full((2, 256), 99.0, device=device, dtype=torch.bfloat16) for _ in range(32)
            ]
        assert not gate.query(), "consumer delay expired before allocator probe"
        assert all(t.data_ptr() != pointer for t in poison)
        consumer.synchronize()
        assert actual.cpu().tolist() == [[7.0] * 256] * 2
    finally:
        producer.synchronize()
        consumer.synchronize()
        blocker.synchronize()
        engine.close()


@pytest.mark.npu
@pytest.mark.parametrize("multi_stream", [False, True])
@torch.inference_mode()
def test_npu_partial_depth_finalize_matches_per_request_path(multi_stream):
    # Serving still requires four loops. Test generic propagation directly.
    from tests.test_ascend import make_npu_model
    from tests.test_npu_async import make_engine
    from vllm_rlt.device import resolve_device

    device = resolve_device("npu:0")
    torch.npu.set_device(device)
    model = make_npu_model(device)
    results = []
    for optimized in [False, True]:
        engine = make_engine(model, True, multi_stream)
        runner, cache = engine.model_runner, engine.cache_manager
        reqs = [Request(str(i), [1], SamplingParams()) for i in range(3)]
        try:
            for r in reqs:
                assert cache.allocate(r.request_id, 4)
            with torch.npu.stream(runner.core_stream):
                for i, depth in enumerate([1, 1, 3]):
                    reqs[i].loops_done = depth + 1
                    shape = (1, model.config.num_key_value_heads, model.config.head_dim)
                    k = torch.full(shape, float(i + 2), device=device, dtype=torch.bfloat16)
                    for layer in range(cache.num_layers):
                        cache.write(layer, [str(i)], [depth], [0], k, k + 1)
                source = torch.npu.Event()
                source.record(runner.core_stream)
            runner.events = {r.request_id: source for r in reqs}
            if optimized:
                runner.finalize_many(reqs)
                assert runner.events["0"] is runner.events["1"]
                assert runner.events["2"] is source
            else:
                for r in reqs:
                    runner.finalize(r)
            runner.boundary_stream.synchronize()
            actual = {}
            for i, first in enumerate([1, 1, 3]):
                for depth in range(first, 4):
                    for layer in range(cache.num_layers):
                        actual[(i, depth, layer)] = tuple(
                            t.cpu() for t in cache.read(layer, str(i), depth, 1)
                        )
            results.append(actual)
        finally:
            for r in reqs:
                runner.release(r.request_id)
                cache.free(r.request_id)
            engine.close()
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        assert all(torch.equal(a, b) for a, b in zip(results[0][key], results[1][key]))
