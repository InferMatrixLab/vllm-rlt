"""Staged attention-bearing prefill and request-local recurrent depths."""

import pytest
import torch

from tests.helpers import tiny_ouro_model
from tests.reference.dory import dense_dory_reference
from tests.test_dory import tiny_dory_config
from tests.test_huginn import tiny_huginn_config
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import DoryForCausalLM, HuginnForCausalLM
from vllm_rlt.request import Operation, RequestPhase
from vllm_rlt.worker.executor import ForwardBatch, ModelRuntime
from vllm_rlt.worker.model_runner import ModelRunner


def make_model(kind, loops=None):
    torch.manual_seed(42)
    if kind == "dory":
        return DoryForCausalLM(tiny_dory_config(n_recurrent_loops=loops or 2)).eval()
    return HuginnForCausalLM(tiny_huginn_config()).eval()


def run_requests(
    model, scheduler, prompts, targets=None, *, mix_coda=False, temperature=0.0, execution=None
):
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=128, block_size=2),
        scheduler_config=scheduler,
        execution_config=execution,
    )
    phases = []
    schedule = engine.scheduler.schedule

    def record_schedule(**kwargs):
        prompt = engine.scheduler.requests.get("1")
        if (
            mix_coda == "final"
            and engine.scheduler.has_ready(RequestPhase.CODA)
            and prompt is not None
            and prompt.phase
            in (
                RequestPhase.PREFILLING_PRELUDE,
                RequestPhase.PREFILLING_RECURRENT,
                RequestPhase.PREFILLING_CODA,
            )
        ):
            last_chunk = (len(prompt.prompt_token_ids) - 1) // scheduler.prefill_chunk_size
            ready_final = (
                prompt.phase == RequestPhase.PREFILLING_CODA
                and prompt.num_recurrent_prefilled_tokens == len(prompt.prompt_token_ids)
                and prompt.num_prefilled_tokens // scheduler.prefill_chunk_size == last_chunk
            )
            batch = engine.scheduler.take_phases(
                (RequestPhase.CODA, RequestPhase.PREFILLING_CODA)
                if ready_final
                else (prompt.phase,)
            )
        elif (
            mix_coda
            and engine.scheduler.has_ready(RequestPhase.CODA)
            and engine.scheduler.has_ready(RequestPhase.PREFILLING_CODA)
        ):
            batch = engine.scheduler.take_phases((RequestPhase.CODA, RequestPhase.PREFILLING_CODA))
        else:
            batch = schedule(**kwargs)
        if batch is not None:
            for item in batch.requests:
                if batch.phase_for(item) == RequestPhase.PREFILLING_CODA:
                    assert item.token_start == item.request.num_prefilled_tokens
                    assert (
                        item.token_start + item.token_count
                        <= item.request.num_recurrent_prefilled_tokens
                    )
            phases.append(batch)
        return batch

    engine.scheduler.schedule = record_schedule
    for index, prompt in enumerate(prompts):
        engine.add_request(
            str(index),
            prompt,
            SamplingParams(
                max_tokens=3,
                ignore_eos=True,
                temperature=temperature,
                target_loops=None if targets is None else targets[index],
            ),
        )
    outputs = {}
    for _ in range(300):
        if not engine.has_unfinished_requests():
            break
        if engine.scheduler.staged_prefill and model.requires_prefill_coda:
            for request in engine.scheduler.requests.values():
                assert (
                    request.prefill_reclaimable_tokens
                    <= request.num_prefilled_tokens
                    <= request.num_recurrent_prefilled_tokens
                    <= request.num_preluded_tokens
                    <= len(request.prompt_token_ids)
                )
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output
    else:
        pytest.fail("staged prefill did not finish")
    return engine, outputs, phases


@pytest.mark.parametrize("kind", ["dory", "huginn"])
@pytest.mark.parametrize("wavefront", [False, True])
def test_staged_prefill_matches_atomic(kind, wavefront):
    prompts = [[4, 7, 9, 3, 12, 6, 5], [8, 2, 10, 11, 4]]
    model = make_model(kind)
    config = SchedulerConfig(
        mode="mixed",
        wavefront_chunked_prefill=wavefront,
        max_num_seqs=2,
        max_num_batched_tokens=4,
        prefill_chunk_size=2,
    )
    engine, actual, phases = run_requests(model, config, prompts)
    assert any(batch.phase == RequestPhase.PREFILLING_CODA for batch in phases)
    assert set(actual) == {"0", "1"}
    for index, prompt in enumerate(prompts):
        _, expected, _ = run_requests(
            make_model(kind),
            SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4, prefill_chunk_size=2),
            [prompt],
        )
        assert actual[str(index)].token_ids == expected["0"].token_ids
        assert actual[str(index)].exit_depths == expected["0"].exit_depths
    assert len(engine.model_runner.batch_input_buffers) == 2
    assert (
        engine.model_runner.batch_input_buffers[0].hidden.data_ptr()
        != engine.model_runner.batch_input_buffers[1].hidden.data_ptr()
    )


@pytest.mark.parametrize("kind", ["dory", "huginn"])
@pytest.mark.parametrize("wavefront", [False, True])
def test_staged_prefill_with_static_padded_buffers_matches_atomic(kind, wavefront):
    prompts = [[4, 7, 9, 3, 12, 6, 5], [8, 2, 10, 11, 4], [6, 7, 3]]
    config = SchedulerConfig(
        mode="mixed",
        wavefront_chunked_prefill=wavefront,
        max_num_seqs=3,
        max_num_batched_tokens=6,
        prefill_chunk_size=2,
    )
    _, actual, _ = run_requests(
        make_model(kind),
        config,
        prompts,
        execution=ExecutionConfig(static_buffers=True, pad_to_power_of_two=True),
    )
    for index, prompt in enumerate(prompts):
        _, expected, _ = run_requests(
            make_model(kind),
            SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4, prefill_chunk_size=2),
            [prompt],
        )
        assert actual[str(index)].token_ids == expected["0"].token_ids
        assert actual[str(index)].exit_depths == expected["0"].exit_depths


@pytest.mark.parametrize("kind", ["dory", "huginn"])
def test_bfloat16_wavefront_matches_independent_atomic_requests(kind):
    prompts = [[4, 7, 9, 3, 12, 6, 5], [8, 2, 10, 11, 4]]
    targets = [1, 2] if kind == "dory" else None
    _, actual, _ = run_requests(
        make_model(kind).to(dtype=torch.bfloat16),
        SchedulerConfig(
            mode="mixed",
            wavefront_chunked_prefill=True,
            max_num_seqs=2,
            max_num_batched_tokens=4,
            prefill_chunk_size=2,
        ),
        prompts,
        targets,
    )
    for index, prompt in enumerate(prompts):
        target = targets[index] if targets is not None else None
        _, expected, _ = run_requests(
            make_model(kind, target).to(dtype=torch.bfloat16),
            SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4, prefill_chunk_size=2),
            [prompt],
        )
        assert actual[str(index)].token_ids == expected["0"].token_ids
        assert actual[str(index)].exit_depths == expected["0"].exit_depths


@pytest.mark.parametrize("wavefront", [False, True])
def test_dory_staged_mixed_fixed_depths_match_independent_models(wavefront):
    prompts = [[4, 7, 9, 3, 12, 6, 5], [8, 2, 10, 11, 4]]
    targets = [1, 2]
    config = SchedulerConfig(
        mode="mixed",
        wavefront_chunked_prefill=wavefront,
        max_num_seqs=2,
        max_num_batched_tokens=4,
        prefill_chunk_size=2,
    )
    _, actual, phases = run_requests(make_model("dory"), config, prompts, targets)
    assert any(
        len({item.request.target_loops for item in batch.requests}) > 1
        and batch.phase == RequestPhase.PREFILLING_RECURRENT
        for batch in phases
    )
    if wavefront:
        assert any(
            len({item.depth for item in batch.requests}) > 1
            and batch.phase == RequestPhase.PREFILLING_RECURRENT
            for batch in phases
        )
    for index, (prompt, target) in enumerate(zip(prompts, targets)):
        _, expected, _ = run_requests(
            make_model("dory", target),
            SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4, prefill_chunk_size=2),
            [prompt],
        )
        assert actual[str(index)].token_ids == expected["0"].token_ids
        assert actual[str(index)].exit_depths == [target] * 3


@pytest.mark.parametrize("kind", ["dory", "huginn"])
@pytest.mark.parametrize("wavefront", [False, True])
def test_prefill_coda_projects_only_selected_rows_and_matches_atomic_logits(kind, wavefront):
    prompt = [4, 7, 9, 3, 12, 6, 5]
    actual_model = make_model(kind)
    expected_model = make_model(kind)

    def capture(model):
        results = []
        masks = []
        calls = []
        coda = model.coda_prepared

        def wrapped(hidden, batch, cache, *, compute_logits=True):
            if isinstance(compute_logits, torch.Tensor):
                masks.append(compute_logits.clone())
            logits = coda(hidden, batch, cache, compute_logits=compute_logits)
            if logits is not None:
                results.append(logits.detach().clone())
            return logits

        model.coda_prepared = wrapped
        model.lm_head.register_forward_pre_hook(
            lambda module, inputs: calls.append(inputs[0].shape)
        )
        return results, masks, calls

    actual_logits, masks, head_calls = capture(actual_model)
    expected_logits, _, _ = capture(expected_model)
    staged = SchedulerConfig(
        mode="mixed",
        wavefront_chunked_prefill=wavefront,
        max_num_seqs=1,
        max_num_batched_tokens=2,
        prefill_chunk_size=2,
    )
    atomic = SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=2, prefill_chunk_size=2)
    _, actual, _ = run_requests(actual_model, staged, [prompt])
    _, expected, _ = run_requests(expected_model, atomic, [prompt])
    assert actual["0"].token_ids == expected["0"].token_ids
    assert len(actual_logits) == len(expected_logits) == len(head_calls) == 3
    assert any(not mask.any() for mask in masks)
    assert sum(mask.sum().item() for mask in masks) == 1
    assert all(shape[0] == 1 for shape in head_calls)
    for actual_row, expected_row in zip(actual_logits, expected_logits):
        torch.testing.assert_close(actual_row, expected_row, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("target", [1, 2])
@pytest.mark.parametrize("wavefront", [False, True])
def test_dory_fixed_depth_prefill_logits_match_dense_reference(target, wavefront):
    prompt = [4, 7, 9, 3, 12, 6, 5]
    model = make_model("dory")
    logits = []
    coda = model.coda_prepared

    def wrapped(hidden, batch, cache, *, compute_logits=True):
        result = coda(hidden, batch, cache, compute_logits=compute_logits)
        if result is not None:
            logits.append(result.detach().clone())
        return result

    model.coda_prepared = wrapped
    _, actual, _ = run_requests(
        model,
        SchedulerConfig(
            mode="mixed",
            wavefront_chunked_prefill=wavefront,
            max_num_seqs=1,
            max_num_batched_tokens=3,
            prefill_chunk_size=2,
        ),
        [prompt],
        [target],
    )
    reference = make_model("dory", target)
    tokens = list(prompt)
    for row, token in zip(logits, actual["0"].token_ids):
        expected = dense_dory_reference(reference, torch.tensor(tokens))["logits"][-1]
        torch.testing.assert_close(row[0], expected, rtol=3e-5, atol=3e-6)
        tokens.append(token)


@pytest.mark.parametrize("budget", [1, 3, 5])
@pytest.mark.parametrize("wavefront", [False, True])
def test_partial_chunks_and_three_fixed_depths(budget, wavefront):
    prompts = [[4, 7, 9, 3, 12], [8, 2, 10], [11, 4, 3, 9, 6, 5, 2]]
    targets = [1, 2, 3]
    _, actual, _ = run_requests(
        make_model("dory", 3),
        SchedulerConfig(
            mode="mixed",
            wavefront_chunked_prefill=wavefront,
            max_num_seqs=3,
            max_num_batched_tokens=budget,
            prefill_chunk_size=3,
        ),
        prompts,
        targets,
    )
    for index, (prompt, target) in enumerate(zip(prompts, targets)):
        _, expected, _ = run_requests(
            make_model("dory", target),
            SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=3, prefill_chunk_size=3),
            [prompt],
        )
        assert actual[str(index)].token_ids == expected["0"].token_ids
        assert actual[str(index)].exit_depths == [target] * 3


@pytest.mark.parametrize("kind", ["dory", "huginn"])
def test_mixed_coda_execution_routes_sparse_results(kind, monkeypatch):
    prompts = [[4], [8, 2, 10, 11, 4, 3, 9, 6]]
    targets = [1, 2] if kind == "dory" else None
    config = SchedulerConfig(
        mode="mixed",
        wavefront_chunked_prefill=True,
        max_num_seqs=2,
        max_num_batched_tokens=3,
        prefill_chunk_size=2,
    )
    original = ModelRunner._mixed_hidden_sources
    checked = []

    def check_sources(runner, batch, buffers):
        states = [span.request.hidden_state for span in batch.spans]
        original(runner, batch, buffers)
        if batch.operation == Operation.CODA:
            assert all(
                span.request.hidden_state is state for span, state in zip(batch.spans, states)
            )
            checked.append(True)

    monkeypatch.setattr(ModelRunner, "_mixed_hidden_sources", check_sources)
    _, actual, phases = run_requests(make_model(kind), config, prompts, targets, mix_coda=True)
    assert checked
    assert any(
        {batch.phase_for(item) for item in batch.requests}
        == {RequestPhase.CODA, RequestPhase.PREFILLING_CODA}
        for batch in phases
    )
    for index, prompt in enumerate(prompts):
        target = targets[index] if targets is not None else None
        _, expected, _ = run_requests(
            make_model(kind, target),
            SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=3, prefill_chunk_size=2),
            [prompt],
        )
        assert actual[str(index)].token_ids == expected["0"].token_ids


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_huginn_seed_is_invariant_to_chunking_and_mixed_prelude(temperature):
    prompts = [[4], [8, 2, 10, 11, 4, 3, 9]]
    _, actual, _ = run_requests(
        make_model("huginn"),
        SchedulerConfig(
            mode="mixed",
            wavefront_chunked_prefill=True,
            max_num_seqs=2,
            max_num_batched_tokens=4,
            prefill_chunk_size=2,
        ),
        prompts,
        mix_coda="final",
        temperature=temperature,
    )
    for index, prompt in enumerate(prompts):
        _, expected, _ = run_requests(
            make_model("huginn"),
            SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=3, prefill_chunk_size=3),
            [prompt],
            temperature=temperature,
        )
        assert actual[str(index)].token_ids == expected["0"].token_ids


@pytest.mark.parametrize("kind", ["dory", "huginn"])
def test_mixed_coda_samples_final_prompt_and_decode_in_one_batch(kind):
    prompts = [[4], [8, 2, 10, 11, 4, 3, 9]]
    targets = [1, 2] if kind == "dory" else None
    config = SchedulerConfig(
        mode="mixed",
        wavefront_chunked_prefill=True,
        max_num_seqs=2,
        max_num_batched_tokens=4,
        prefill_chunk_size=2,
    )
    _, actual, phases = run_requests(make_model(kind), config, prompts, targets, mix_coda="final")
    assert any(
        {batch.phase_for(item) for item in batch.requests}
        == {RequestPhase.CODA, RequestPhase.PREFILLING_CODA}
        and any(item.completes_prompt for item in batch.requests)
        for batch in phases
    )
    for index, prompt in enumerate(prompts):
        target = targets[index] if targets is not None else None
        _, expected, _ = run_requests(
            make_model(kind, target),
            SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4, prefill_chunk_size=2),
            [prompt],
        )
        assert actual[str(index)].token_ids == expected["0"].token_ids


def test_sparse_logits_mask_for_models_without_coda_attention():
    model = tiny_ouro_model()
    hidden = torch.randn(3, model.config.hidden_size)
    runtime = ModelRuntime(model, None)
    mask = torch.tensor([False, True, False])
    forward = ForwardBatch(Operation.CODA, hidden=hidden, compute_logits=mask, graphs=object())
    prepared = runtime.prepare(forward)
    assert prepared.graphs is None
    torch.testing.assert_close(runtime.forward(prepared).logits, model.coda(hidden[1:2]))
    assert (
        runtime.forward(
            ForwardBatch(
                Operation.CODA, hidden=hidden, compute_logits=torch.zeros(3, dtype=torch.bool)
            )
        ).logits
        is None
    )


@pytest.mark.parametrize("kind", ["dory", "huginn"])
@pytest.mark.parametrize("mask", [torch.ones(2), torch.ones(1, 3, dtype=torch.bool)])
def test_coda_rejects_invalid_logits_mask_before_writing_kv(kind, mask):
    model = make_model(kind)
    hidden = torch.randn(3, model.config.hidden_size)
    with pytest.raises(ValueError, match="boolean mask for every CODA row"):
        model.coda_prepared(hidden, None, None, compute_logits=mask)


def test_first_prefill_sample_can_finish_without_decode_coda():
    model = make_model("dory")

    def force_eos(state):
        logits = torch.full((len(state), model.config.vocab_size), -1.0, device=state.device)
        logits[:, model.config.eos_token_id] = 1.0
        return logits

    model.logits = force_eos
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=128, block_size=2),
        scheduler_config=SchedulerConfig(
            mode="mixed", max_num_seqs=1, max_num_batched_tokens=2, prefill_chunk_size=2
        ),
    )
    engine.add_request("request", [4, 7, 9], SamplingParams(max_tokens=3))
    outputs = []
    phases = []
    while engine.has_unfinished_requests():
        outputs.extend(engine.step())
        phases.append(engine.last_schedule.phase)
    assert [output.token_ids for output in outputs] == [[model.config.eos_token_id[0]]]
    assert outputs[0].finished
    assert RequestPhase.PREFILLING_CODA in phases
    assert RequestPhase.CODA not in phases


def test_staged_prefill_continuous_admission_and_completion():
    engine = LLMEngine(
        make_model("dory"),
        cache_config=CacheConfig(num_blocks=128, block_size=2),
        scheduler_config=SchedulerConfig(
            mode="mixed",
            wavefront_chunked_prefill=True,
            max_num_seqs=2,
            max_num_batched_tokens=2,
            prefill_chunk_size=2,
        ),
    )
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    engine.add_request("long", [4, 7, 9, 3, 12, 6, 5, 2, 8, 11, 10], params)
    engine.step()
    engine.add_request("short", [4], params)
    finished = []
    while engine.has_unfinished_requests():
        finished.extend(output.request_id for output in engine.step() if output.finished)
    assert finished == ["short", "long"]
    assert engine.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("wavefront", [False, True])
@pytest.mark.parametrize("complete_chunk", [False, True])
def test_abort_around_prefill_coda_releases_request_state(wavefront, complete_chunk):
    engine = LLMEngine(
        make_model("dory"),
        cache_config=CacheConfig(num_blocks=128, block_size=2),
        scheduler_config=SchedulerConfig(
            mode="mixed",
            wavefront_chunked_prefill=wavefront,
            max_num_seqs=1,
            max_num_batched_tokens=2,
            prefill_chunk_size=2,
        ),
    )
    params = SamplingParams(max_tokens=3, target_loops=1, ignore_eos=True)
    engine.add_request("reused", [4, 7, 9, 3, 12], params)
    request = engine.scheduler.requests["reused"]
    for _ in range(30):
        if request.phase == RequestPhase.PREFILLING_CODA:
            break
        engine.step()
    else:
        pytest.fail("request did not reach prefill CODA")
    if complete_chunk:
        engine.step()
        assert request.num_prefilled_tokens == 2
    engine.abort_request("reused")
    assert not engine.model_runner.state_slots
    assert engine.cache_manager.num_used_blocks == 0

    prompt = [5, 4, 3]
    engine.add_request("reused", prompt, params)
    actual = []
    while engine.has_unfinished_requests():
        actual.extend(output for output in engine.step() if output.finished)
    _, expected, _ = run_requests(
        make_model("dory", 1),
        SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=2, prefill_chunk_size=2),
        [prompt],
    )
    assert actual[0].token_ids == expected["0"].token_ids
    assert actual[0].exit_depths == expected["0"].exit_depths


@pytest.mark.parametrize(
    "execution", [ExecutionConfig(async_scheduling=True), ExecutionConfig(cuda_graphs=True)]
)
def test_boundary_staged_rejects_unqualified_execution(execution):
    with pytest.raises(ValueError, match="staged prefill requires synchronous eager"):
        LLMEngine(
            make_model("dory"),
            cache_config=CacheConfig(num_blocks=128, block_size=2),
            scheduler_config=SchedulerConfig(mode="mixed"),
            execution_config=execution,
        )
