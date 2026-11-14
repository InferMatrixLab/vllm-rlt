"""CPU contract and real-socket lifecycle tests for the serving frontend."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiohttp")
from aiohttp import ClientPayloadError
from aiohttp.test_utils import TestClient, TestServer

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.execution import WorkKey
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.serving.protocol import (
    CompletionRequest,
    IncrementalText,
    ServingError,
    ServingLimits,
)
from vllm_rlt.serving.server import WORKER, create_app
from vllm_rlt.serving.worker import EngineWorker


class TinyTokenizer:
    all_special_ids = [0]
    added_tokens_decoder = {}

    def encode(self, text):
        return [1 + ord(c) % 63 for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(64 + token) for token in ids if token != 0)

    def convert_ids_to_tokens(self, token_id):
        return chr(64 + token_id)


class PauseAt:
    """Hold one engine step while a test exercises HTTP or worker lifecycle."""

    def __init__(self, stage, *, fail=False):
        self.work, self.fail = stage, fail
        self.entered, self.release = threading.Event(), threading.Event()
        self.engine = None

    def __call__(self, engine):
        self.engine = engine
        submit = engine.executor.submit

        def pause(batch):
            if self.work in batch.works and not self.entered.is_set():
                self.entered.set()
                assert self.release.wait(5)
                if self.fail:
                    raise RuntimeError("injected engine failure")
            return submit(batch)

        engine.executor.submit = pause


def factory(*, seqs=8, blocks=256, hook=None):
    torch.manual_seed(123)
    engine = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()),
        cache_config=CacheConfig(num_blocks=blocks, block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=seqs, max_num_batched_tokens=3),
    )
    if hook:
        hook(engine)
    return engine, TinyTokenizer()


async def until(predicate):
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition did not become true")


@asynccontextmanager
async def client_for(load=factory, **kwargs):
    app = create_app(load, model="test-model", limits=ServingLimits(**kwargs))
    client = TestClient(TestServer(app, handler_cancellation=True))
    await client.start_server()
    try:
        yield client, app[WORKER]
    finally:
        await client.close()


def body(**kwargs):
    return {
        "model": "test-model",
        "prompt": "abc",
        "max_tokens": 4,
        "ignore_eos": True,
        **kwargs,
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"prompt": [1, 2]},
        {"prompt": None},
        {"prompt": "\ud800"},
        {"stream": 1},
        {"ignore_eos": "true"},
        {"n": 2},
        {"n": True},
        {"logprobs": 1},
        {"stop": "end"},
        {"max_tokens": 0},
        {"max_tokens": None},
        {"max_tokens": True},
        {"temperature": "0"},
        {"temperature": float("nan")},
        {"top_p": True},
        {"seed": -1},
        {"repetition_penalty": 1.1},
        {"unknown": None},
        {"stream_options": {}},
        {"stream": True, "stream_options": {"include_usage": 1}},
    ],
)
def test_validation(extra):
    with pytest.raises((ValueError, TypeError)):
        CompletionRequest.parse(body(**extra), "test-model")


def test_neutral_client_fields_and_extensions():
    spec = CompletionRequest.parse(
        body(
            stream=True,
            stream_options={"include_usage": True},
            repetition_penalty=1,
            logprobs=None,
            min_loops=2,
            max_loops=3,
            exit_threshold=0.7,
            top_k=9,
            top_p=0.8,
        ),
        "test-model",
    )
    assert spec.include_usage and spec.params.max_loops == 3 and spec.params.top_k == 9


def test_byte_decoder_unicode_special_tokens_and_final_flush(monkeypatch):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: i for i, token in enumerate(alphabet)}
    vocab["<eos>"] = len(vocab)
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>")
    tokenizer.add_tokens(["word🙂"])
    converted = []
    convert = tokenizer.convert_ids_to_tokens

    def record(token_id):
        converted.append(token_id)
        return convert(token_id)

    monkeypatch.setattr(tokenizer, "convert_ids_to_tokens", record)
    for text in ["你好🙂 café", "👩🏽‍💻", "hello 🌍", "a � b", "  a  b\n", "word🙂", "a" * 2048]:
        ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
        decoder = IncrementalText(tokenizer)
        converted.clear()
        deltas = [decoder.decode(token, i == len(ids) - 1) for i, token in enumerate(ids)]
        assert "".join(deltas) == text
        assert deltas[-1] == ""
        assert converted == ids[:-1]  # Each ordinary token is converted once.
    smile = tokenizer.encode("🙂")
    cases = [smile[:i] for i in range(1, len(smile))]
    cases += [[vocab[c]] for c in alphabet]  # All byte values, including invalid UTF-8.
    for ids in cases:
        decoder = IncrementalText(tokenizer)
        deltas = [decoder.decode(token, False) for token in ids]
        deltas.append(decoder.decode(tokenizer.eos_token_id, True))
        assert "".join(deltas) == tokenizer.decode(ids)


def test_incremental_text_uses_added_special_token_metadata():
    tokenizer = SimpleNamespace(
        all_special_ids=[0],
        # Transformers 5.14.1 omits added special tokens from all_special_ids.
        added_tokens_decoder={4: SimpleNamespace(special=True), 9: SimpleNamespace(special=False)},
        convert_ids_to_tokens=lambda token: "</think>" if token == 4 else "x",
    )
    decoder = IncrementalText(tokenizer)
    assert decoder.decode(4, False) == ""
    assert decoder.decode(9, True) == "x"


@pytest.mark.parametrize("byte_level", [False, True])
def test_load_engine_checks_decoder_before_loading_model(monkeypatch, byte_level, tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from vllm_rlt.entrypoints.serve import load_engine

    decoder = tokenizers.decoders.ByteLevel() if byte_level else tokenizers.decoders.WordPiece()
    tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(decoder=decoder))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    loaded = []

    def load_model(*args, **kwargs):
        loaded.append(True)
        return OuroForCausalLM(tiny_ouro_config())

    monkeypatch.setattr(
        "vllm_rlt.entrypoints.serve.StageClient", lambda *a, **k: LLMEngine(load_model())
    )
    args = SimpleNamespace(
        model=str(tmp_path),
        tokenizer=None,
        revision=None,
        tokenizer_revision=None,
        device="cpu",
        dtype="float32",
        num_blocks=16,
        block_size=2,
        max_num_seqs=1,
        max_num_batched_tokens=3,
        mode="refill",
        attention_backend="torch",
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "ouro"}))
    if byte_level:
        engine, actual = load_engine(args)
        assert actual is tokenizer and loaded
        assert not engine.has_unfinished_requests()
    else:
        with pytest.raises(ValueError, match="byte-level tokenizer"):
            load_engine(args)
        assert not loaded


def test_load_engine_allows_non_byte_level_tokenizer_for_nanbeige(monkeypatch, tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tests.helpers import tiny_nanbeige_config
    from vllm_rlt.entrypoints.serve import load_engine
    from vllm_rlt.models import NanbeigeForCausalLM

    decoder = tokenizers.decoders.WordPiece()
    tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(decoder=decoder))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    loaded = []

    def load_model(*args, **kwargs):
        loaded.append(True)
        return NanbeigeForCausalLM(tiny_nanbeige_config())

    monkeypatch.setattr(
        "vllm_rlt.entrypoints.serve.StageClient", lambda *a, **k: LLMEngine(load_model())
    )
    args = SimpleNamespace(
        model=str(tmp_path),
        tokenizer=None,
        revision=None,
        tokenizer_revision=None,
        device="cpu",
        dtype="float32",
        num_blocks=16,
        block_size=2,
        max_num_seqs=1,
        max_num_batched_tokens=3,
        mode="refill",
        attention_backend="torch",
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "nanbeige"}))
    engine, actual = load_engine(args)
    assert actual is tokenizer and loaded


def test_startup_readiness_failure_and_shutdown_during_load():
    async def run():
        loading, release = threading.Event(), threading.Event()

        def load():
            loading.set()
            assert release.wait(5)
            return factory()

        async with client_for(load) as (client, worker):
            await until(loading.is_set)
            assert (await client.get("/health")).status == 503
            assert (await client.post("/v1/completions", json=body())).status == 503
            release.set()
            await until(lambda: worker.ready)
            assert (await client.get("/health")).status == 200
            models = await (await client.get("/v1/models")).json()
            assert models["data"][0]["id"] == body()["model"]

        def fail():
            raise RuntimeError("load failed")

        async with client_for(fail) as (client, worker):
            await until(lambda: worker.failure is not None)
            assert (await client.get("/health")).status == 503
        loading.clear()
        release.clear()
        worker = EngineWorker(load)
        worker.start()
        await until(loading.is_set)
        closing = asyncio.create_task(worker.close())
        await asyncio.sleep(0)
        release.set()
        await closing
        assert not worker.ready and worker.engine is None

    asyncio.run(run())


def test_http_output_matches_direct_and_stream_has_exact_token_events():
    engine, tokenizer = factory()
    engine.add_request(
        "direct", tokenizer.encode("abc"), SamplingParams(max_tokens=4, ignore_eos=True)
    )
    outputs = []
    while engine.has_unfinished_requests():
        outputs.extend(engine.step())
    expected = tokenizer.decode(outputs[-1].token_ids)

    async def run():
        async with client_for() as (client, worker):
            await until(lambda: worker.ready)
            response = await client.post("/v1/completions", json=body())
            data = await response.json()
            assert response.status == 200
            assert data["choices"][0]["text"] == expected
            assert data["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
            response = await client.post(
                "/v1/completions",
                json=body(
                    stream=True,
                    stream_options={"include_usage": True},
                    repetition_penalty=1,
                    logprobs=None,
                ),
            )
            wire = await response.text()
            frames = wire.strip().split("\n\n")
            assert frames[-1] == "data: [DONE]"
            events = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-1]]
            tokens = [event for event in events if event["choices"]]
            assert len(tokens) == 4
            assert "".join(event["choices"][0]["text"] for event in tokens) == expected
            assert [event["choices"][0]["finish_reason"] for event in tokens] == [None] * 3 + [
                "length"
            ]
            assert events[-1]["choices"] == [] and events[-1]["usage"] == data["usage"]
            await until(lambda: not worker.channels)

    asyncio.run(run())


@pytest.mark.parametrize("expired", [False, True])
def test_shutdown_cleans_http_and_engine_state(caplog, expired):
    async def run():
        pause, cleaned = PauseAt(WorkKey.PROMPT_ATOMIC), asyncio.Event()
        app = create_app(
            lambda: factory(hook=pause),
            model="test-model",
            limits=ServingLimits(shutdown_timeout=0.01 if expired else 5),
        )

        async def cleanup(app):
            cleaned.set()

        app.on_cleanup.append(cleanup)
        client = TestClient(TestServer(app, handler_cancellation=True))
        await client.start_server()
        worker = app[WORKER]
        try:
            await until(lambda: worker.ready)
            channel = worker.submit(CompletionRequest.parse(body(), body()["model"]))
            await until(pause.entered.is_set)
            closing = asyncio.create_task(client.close())
            await until(lambda: worker.stopping)
            if not expired:
                pause.release.set()
            await asyncio.wait_for(closing, 1)
            assert cleaned.is_set() and not worker.ready
            if expired:
                assert not worker.task.done()
                assert "shutdown deadline exceeded" in caplog.text
        finally:
            pause.release.set()
            await asyncio.wait_for(worker.task, 5)
            await client.close()
        with pytest.raises(ServingError, match="shutting down"):
            await channel.receive()
        assert not worker.channels and pause.engine.cache_manager.num_used_blocks == 0

    asyncio.run(run())


def test_cancel_before_first_tick_skips_admission():
    async def run():
        worker = EngineWorker(factory)
        worker.start()
        try:
            await until(lambda: worker.ready)
            channel = worker.submit(CompletionRequest.parse(body(), body()["model"]))
            worker.release(channel)  # No yield: pending admission and cancellation share a tick.
            await until(lambda: not worker.channels)
            assert worker.engine.last_schedule is None
            assert not worker.decoders and worker.engine.cache_manager.num_used_blocks == 0
        finally:
            await worker.close()

    asyncio.run(run())


def test_http_invalid_admission_and_browser_requests_preserve_readiness():
    async def run():
        async with client_for() as (client, worker):
            await until(lambda: worker.ready)
            missing_model = body()
            del missing_model["model"]
            for payload in [
                missing_model,
                body(prompt=""),
                body(prompt="\ud800"),
                body(max_tokens=worker.engine.model.config.max_position_embeddings + 1),
                body(max_loops=worker.engine.model.config.total_ut_steps + 1),
            ]:
                assert (await client.post("/v1/completions", json=payload)).status == 400
            assert (await client.post("/v1/completions", json=body(model="missing"))).status == 404
            response = await client.post(
                "/v1/completions", data="{bad", headers={"Content-Type": "application/json"}
            )
            assert response.status == 400
            for path in ["/health", "/v1/models", "/v1/completions"]:
                response = await client.post(path, json=body(), headers={"Host": "attacker.test"})
                assert response.status == 403
            assert (await client.get("/health", headers={"Host": "localhost"})).status == 200
            for origin in ["https://example.com", "http://localhost:8000", "null"]:
                response = await client.post(
                    "/v1/completions", json=body(), headers={"Origin": origin}
                )
                assert response.status == 403
            response = await client.post("/v1/completions", data=json.dumps(body()))
            assert response.status == 415
            assert (await client.post("/v1/completions", json=body())).status == 200
            await until(lambda: not worker.channels)
            assert worker.ready and worker.engine.cache_manager.num_used_blocks == 0

    asyncio.run(run())


def test_eos_usage_and_non_usage_stream():
    def load():
        engine, tokenizer = factory()
        with torch.no_grad():
            engine.model.lm_head.weight.zero_()
        return engine, tokenizer

    async def run():
        async with client_for(load) as (client, worker):
            await until(lambda: worker.ready)
            response = await client.post("/v1/completions", json=body(ignore_eos=False))
            result = await response.json()
            assert result["choices"][0]["text"] == ""
            assert result["choices"][0]["finish_reason"] == "stop"
            assert result["usage"]["completion_tokens"] == 1
            response = await client.post(
                "/v1/completions", json=body(stream=True, ignore_eos=False)
            )
            frames = (await response.text()).strip().split("\n\n")
            assert len(frames) == 2 and "usage" not in json.loads(frames[0][6:])

    asyncio.run(run())


def test_owner_thread_dynamic_batching_cancellation_and_overload():
    async def run():
        blocked, release = threading.Event(), threading.Event()
        trace, owner_threads, engines = [], [], []

        def hook(engine):
            engines.append(engine)
            submit = engine.executor.submit

            def record(batch):
                owner_threads.append(threading.get_ident())
                trace.append(
                    (
                        batch.works,
                        [
                            scheduled_request.request.request_id
                            for scheduled_request in batch.requests
                        ],
                    )
                )
                if WorkKey.DECODE_RECURRENT in batch.works and not blocked.is_set():
                    blocked.set()
                    assert release.wait(5)
                return submit(batch)

            engine.executor.submit = record
            for name in ["add_request", "abort_request"]:
                method = getattr(engine, name)

                def check(*args, method=method):
                    owner_threads.append(threading.get_ident())
                    return method(*args)

                setattr(engine, name, check)

        worker = EngineWorker(lambda: factory(hook=hook), limits=ServingLimits(max_requests=2))
        worker.start()
        try:
            await until(lambda: worker.ready)
            first = worker.submit(CompletionRequest.parse(body(max_tokens=8), body()["model"]))
            await first.receive()
            await until(blocked.is_set)
            second = worker.submit(CompletionRequest.parse(body(max_tokens=8), body()["model"]))
            with pytest.raises(ServingError, match="capacity"):
                worker.submit(second.spec)
            release.set()
            await second.receive()
            await until(lambda: any(len(ids) == 2 for _, ids in trace))
            worker.release(first)
            while (await second.receive()).finish_reason is None:
                pass
            worker.release(second)
            await until(lambda: not worker.channels)
            assert engines[0].cache_manager.num_used_blocks == 0
            assert len(set(owner_threads)) == 1 and owner_threads[0] != threading.get_ident()
        finally:
            release.set()
            await worker.close()

    asyncio.run(run())


def test_slow_channel_does_not_block_other_requests():
    async def run():
        engines = []

        def load():
            engine, tokenizer = factory()
            engines.append(engine)
            return engine, tokenizer

        worker = EngineWorker(load, limits=ServingLimits(output_buffer=1, max_requests=2))
        worker.start()
        try:
            await until(lambda: worker.ready)
            slow = worker.submit(CompletionRequest.parse(body(max_tokens=20), body()["model"]))
            await until(lambda: slow.error is not None)
            assert slow.error.code == "slow_client"
            assert slow.request_id in worker.channels  # occupied until consumer releases it
            good = worker.submit(CompletionRequest.parse(body(max_tokens=1), body()["model"]))
            assert (await good.receive()).finish_reason == "length"
            worker.release(slow)
            worker.release(good)
            await until(lambda: not worker.channels)
            assert engines[0].cache_manager.num_used_blocks == 0
        finally:
            await worker.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure,stream,max_tokens",
    [("disconnect", True, 4), ("engine", True, 4), ("decode", False, 2), ("decode", True, 4)],
)
def test_request_failure_and_disconnect_cleanup(monkeypatch, failure, stream, max_tokens):
    from vllm_rlt.serving import worker as worker_module

    pause = PauseAt(WorkKey.DECODE_RECURRENT, fail=failure == "engine")

    class BadTokenizer(TinyTokenizer):
        calls = 0

        def convert_ids_to_tokens(self, token_id):
            self.calls += 1
            if self.calls == 2:
                raise ValueError("injected decode failure")
            return super().convert_ids_to_tokens(token_id)

    if failure == "decode":
        first = True

        def decoder(tokenizer):
            nonlocal first
            result = IncrementalText(BadTokenizer() if first else tokenizer)
            first = False
            return result

        monkeypatch.setattr(worker_module, "IncrementalText", decoder)

    async def run():
        async with client_for(lambda: factory(hook=pause)) as (client, worker):
            await until(lambda: worker.ready)
            bad = asyncio.create_task(
                client.post("/v1/completions", json=body(stream=stream, max_tokens=max_tokens))
            )
            try:
                await until(pause.entered.is_set)
                if stream:
                    response = await bad
                    assert json.loads((await response.content.readline())[6:])["choices"]
                assert (await client.get("/health")).status == 200
                good = asyncio.create_task(client.post("/v1/completions", json=body()))
                await until(lambda: len(worker.channels) == 2)
                if failure == "disconnect":
                    response.close()
                    await until(lambda: any(c.cancelled for c in worker.channels.values()))
                pause.release.set()
                if failure != "disconnect":
                    if stream:
                        with pytest.raises(ClientPayloadError):
                            await response.read()
                    else:
                        response = await bad
                        assert response.status == 400
                        assert "decode failure" in (await response.json())["error"]["message"]
                result = await good
                assert result.status == (503 if failure == "engine" else 200)
                if failure != "engine":
                    assert (await result.json())["usage"]["completion_tokens"] == 4
                await until(lambda: not worker.channels)
                assert (await client.get("/health")).status == (503 if failure == "engine" else 200)
                assert not worker.decoders and pause.engine.cache_manager.num_used_blocks == 0
            finally:
                pause.release.set()

    asyncio.run(run())


def test_http_capacity_body_limit_and_cancellation_during_tokenization():
    async def run():
        entered, release = threading.Event(), threading.Event()
        engines = []

        class BlockingTokenizer(TinyTokenizer):
            def encode(self, text):
                if text == "hold":
                    entered.set()
                    assert release.wait(5)
                return super().encode(text)

        def load():
            engine, _ = factory()
            engines.append(engine)
            return engine, BlockingTokenizer()

        async with client_for(load, max_requests=1, max_body_bytes=256) as (client, worker):
            await until(lambda: worker.ready)
            too_big = await client.post("/v1/completions", json=body(prompt="x" * 512))
            assert too_big.status == 413
            connection = asyncio.create_task(
                client.post("/v1/completions", json=body(prompt="hold"))
            )
            try:
                await until(entered.is_set)
                overloaded = await client.post("/v1/completions", json=body())
                assert overloaded.status == 429 and overloaded.headers["Retry-After"] == "1"
                connection.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await connection
                await until(lambda: any(c.cancelled for c in worker.channels.values()))
                # The owner has not yet observed cancellation, so its slot stays reserved.
                assert len(worker.channels) == 1
                release.set()
                await until(lambda: not worker.channels)
                assert engines[0].cache_manager.num_used_blocks == 0
                assert (await client.post("/v1/completions", json=body(max_tokens=1))).status == 200
            finally:
                release.set()

    asyncio.run(run())


def test_write_deadline_aborts_socket_without_blocking_other_clients(monkeypatch):
    from aiohttp import web

    async def run():
        original_write = web.StreamResponse.write
        held_id = None
        hold = asyncio.Event()

        async def stalled_write(response, data):
            nonlocal held_id
            if data.startswith(b"data: {"):
                request_id = json.loads(data[6:])["id"]
                if held_id is None:
                    held_id = request_id
                if request_id == held_id:
                    await hold.wait()
            return await original_write(response, data)

        monkeypatch.setattr(web.StreamResponse, "write", stalled_write)
        async with client_for(write_timeout=0.1) as (client, worker):
            await until(lambda: worker.ready)
            slow = await client.post("/v1/completions", json=body(stream=True))
            good = await client.post("/v1/completions", json=body(max_tokens=1))
            assert good.status == 200
            with pytest.raises(ClientPayloadError):
                await slow.read()
            await until(lambda: not worker.channels)
            assert worker.ready

    asyncio.run(run())


@pytest.mark.parametrize("value", [None, "", 1, [], True])
def test_invalid_trace_id(value):
    with pytest.raises(ValueError, match="trace_id"):
        CompletionRequest.parse(body(trace_id=value), "test-model")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_http_trace_selection_and_concurrent_reuse(asynchronous):
    from vllm_rlt.config import ExecutionConfig, ExitConfig

    async def run():
        completed = []

        def load():
            engine = LLMEngine(
                OuroForCausalLM(tiny_ouro_config()),
                cache_config=CacheConfig(num_blocks=256, block_size=2),
                exit_config=ExitConfig("trace", depths_by_request={"A": [4, 2, 3, 2]}),
                execution_config=ExecutionConfig(async_scheduling=asynchronous),
            )
            step = engine.step

            def record():
                outputs = step()
                completed.extend(o for o in outputs if o.finished)
                return outputs

            engine.step = record
            return engine, TinyTokenizer()

        async with client_for(load) as (client, worker):
            await until(lambda: worker.ready)
            for extra in ({}, {"trace_id": "missing"}, {"trace_id": "A", "max_tokens": 5}):
                response = await client.post("/v1/completions", json=body(**extra))
                assert response.status == 400
            responses = await asyncio.gather(
                *(client.post("/v1/completions", json=body(trace_id="A")) for _ in range(2))
            )
            assert all(r.status == 200 for r in responses)
            results = [await r.json() for r in responses]
            assert len({r["id"] for r in results}) == 2
            assert all(r["id"].startswith("cmpl-") for r in results)
            assert len(completed) == 2
            assert all(o.exit_depths == [4, 2, 3, 2] for o in completed)
            assert worker.engine.cache_manager.num_used_blocks == 0

        async with client_for() as (client, worker):
            await until(lambda: worker.ready)
            response = await client.post("/v1/completions", json=body(trace_id="A"))
            assert response.status == 400
            assert "trace exit mode" in (await response.json())["error"]["message"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "asynchronous,graphs,backend",
    [
        (False, False, "torch"),
        pytest.param(True, False, "triton", marks=pytest.mark.gpu),
        pytest.param(True, True, "triton", marks=pytest.mark.gpu),
        pytest.param(True, True, "flash_attn_4", marks=pytest.mark.gpu),
    ],
)
def test_speculative_http_and_stream_deliver_entire_committed_suffix(asynchronous, graphs, backend):
    from vllm_rlt import ExecutionConfig, SpeculativeConfig

    def load():
        torch.manual_seed(123)
        m = OuroForCausalLM(tiny_ouro_config(head_dim=64 if asynchronous else 8)).to(
            device="cuda" if asynchronous else "cpu",
            dtype=torch.bfloat16 if asynchronous else torch.float32,
        )

        # Force full acceptance of a printable token so this test exercises
        # multi-token chunks regardless of random tiny-model draft quality.
        def coda(hidden):
            logits = hidden.new_zeros((len(hidden), m.config.vocab_size))
            logits[:, 5] = 1
            return logits

        m.coda = coda
        return LLMEngine(
            m,
            speculative_config=SpeculativeConfig(4),
            execution_config=ExecutionConfig(async_scheduling=asynchronous, cuda_graphs=graphs),
            attention_backend=backend,
            cache_config=CacheConfig(128, 16 if asynchronous else 2),
        ), TinyTokenizer()

    direct, tokenizer = load()
    direct.add_request("r", tokenizer.encode("abc"), SamplingParams(max_tokens=12, ignore_eos=True))
    expected = None
    while direct.has_unfinished_requests():
        for out in direct.step():
            if out.finished:
                expected = tokenizer.decode(out.token_ids)

    async def run():
        async with client_for(load, output_buffer=2) as (client, worker):
            await until(lambda: worker.ready)
            response = await client.post("/v1/completions", json=body(max_tokens=12))
            result = await response.json()
            assert response.status == 200
            assert result["choices"][0]["text"] == expected
            assert result["usage"]["completion_tokens"] == 12
            response = await client.post(
                "/v1/completions",
                json=body(max_tokens=12, stream=True, stream_options={"include_usage": True}),
            )
            wire = await response.text()
            frames = [json.loads(f.removeprefix("data: ")) for f in wire.strip().split("\n\n")[:-1]]
            assert "".join(f["choices"][0]["text"] for f in frames if f["choices"]) == expected
            assert frames[-1]["usage"]["completion_tokens"] == 12
            assert worker.engine.speculative_runner.stats.accepted_tokens > 0
            if graphs:
                assert worker.engine.speculative_runner.graphs.replays > 0
                assert worker.engine.speculative_runner.coda_graphs.replays > 0

    asyncio.run(run())
