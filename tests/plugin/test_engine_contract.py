import pickle
from collections import deque
from concurrent.futures import Future
from queue import Queue
from types import SimpleNamespace

import pytest

from vllm_rlt.core.loop import LoopOp, LoopSpan


def test_loop_plan_keeps_same_request_spans_after_serialization():
    pytest.importorskip("vllm")
    from vllm_rlt.core.output import LoopSchedulerOutput

    spans = (LoopSpan("a", 0, 16, 1), LoopSpan("a", 16, 16, 0))
    plan = LoopSchedulerOutput.for_step(LoopOp.CORE, spans)

    restored = pickle.loads(pickle.dumps(plan))
    assert restored.op is LoopOp.CORE
    assert restored.spans == spans
    assert restored.num_scheduled_tokens == {"a": 32}
    assert restored.total_num_scheduled_tokens == 32


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_engine_core_accepts_loop_steps_without_sampled_tokens(async_scheduling):
    pytest.importorskip("vllm")
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.outputs import ModelRunnerOutput

    from vllm_rlt.core.output import LoopSchedulerOutput

    plan_sequence = [
        LoopSchedulerOutput.for_step(op, [LoopSpan("a", 8, 1, depth)])
        for op, depth in (
            (LoopOp.PRELUDE, 0),
            (LoopOp.CORE, 0),
            (LoopOp.CORE, 1),
            (LoopOp.CODA, 1),
        )
    ]
    pending = deque(plan_sequence)
    completed = []
    launched = []

    def update_from_output(plan, result):
        assert result.req_ids == ["a"]
        completed.append(plan)
        if result.sampled_token_ids[0]:
            return {0: EngineCoreOutputs(outputs=[EngineCoreOutput("a", [42])])}
        return {}

    def execute_model(plan, *, non_block):
        assert non_block
        launched.append(pickle.loads(pickle.dumps(plan)))
        future = Future()
        future.set_result(None)
        return future

    def sample_tokens(grammar_output, *, non_block=False):
        assert grammar_output is None
        tokens = [42] if launched[-1].op is LoopOp.CODA else []
        result = ModelRunnerOutput(req_ids=["a"], sampled_token_ids=[tokens])
        if non_block:
            future = Future()
            future.set_result(result)
            return future
        return result

    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        has_requests=lambda: bool(pending),
        schedule=lambda _: pending.popleft(),
        get_grammar_bitmask=lambda _: None,
        update_from_output=update_from_output,
    )
    core.model_executor = SimpleNamespace(execute_model=execute_model, sample_tokens=sample_tokens)
    core.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_logging_iteration_details=False),
        gpu_iteration_timing=False,
    )
    core.log_stats = False
    core.num_executed_steps = 0
    core.prefill_schedule_interval = 1
    core.is_mm_encoder_only = False
    core.is_pooling_model = False
    core.batch_queue_size = 2
    core.batch_queue = deque(maxlen=2)
    core.deferred_step = None
    core.aborts_queue = Queue()
    core._weight_version = "test"

    step = core.step_with_batch_queue if async_scheduling else core.step
    tokens = []
    for _ in range(len(plan_sequence) + int(async_scheduling)):
        outputs, executed = step()
        if outputs:
            assert completed == plan_sequence
            tokens.extend(outputs[0].outputs[0].new_token_ids)
        if executed:
            assert launched[-1].total_num_scheduled_tokens == 1

    assert launched == plan_sequence
    assert completed == plan_sequence
    assert tokens == [42]
    assert core.num_executed_steps == len(plan_sequence)
    assert not core.batch_queue
