"""Unit coverage for collection control, rank routing and artifact processing."""

import asyncio
import csv
import gc
import json
import tarfile
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_rlt.profiling import ProfileConfig, Profiler
from vllm_rlt.profiling_artifacts import ProfileArtifactWorker, _package, parse_trace

TRACE = {
    "traceEvents": [
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::mm",
            "ts": 10,
            "dur": 5,
            "pid": 1,
            "tid": 2,
            "args": {"Input Dims": [[2, 4]], "Call stack": "caller"},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "gemm",
            "ts": 12,
            "dur": 20,
            "pid": 0,
            "tid": 7,
            "args": {"device": 0, "stream": 7},
        },
        {"ph": "i", "name": "[memory]", "args": {"Bytes": -16}},
    ]
}


class NativeProfilerStub:
    """Drive native callback boundaries without collecting devices or loading models."""

    def __init__(self, **options):
        self.options = options
        self.schedule = options["schedule"]
        self.index = 0
        self.stopped = False
        self.action = self.schedule(0) if self.schedule else None

    def start(self):
        self.owner = threading.get_ident()

    def step(self):
        assert threading.get_ident() == self.owner
        if self.action == torch.profiler.ProfilerAction.RECORD_AND_SAVE:
            self.options["on_trace_ready"](self)
        self.index += 1
        self.action = self.schedule(self.index) if self.schedule else None

    def stop(self):
        assert threading.get_ident() == self.owner
        if not self.stopped and (
            self.schedule is None
            or self.action
            in (torch.profiler.ProfilerAction.RECORD, torch.profiler.ProfilerAction.RECORD_AND_SAVE)
        ):
            self.options["on_trace_ready"](self)
        self.stopped = True

    def export_chrome_trace(self, path):
        assert threading.get_ident() == self.owner
        Path(path).write_text(json.dumps(TRACE))

    def events(self):
        assert threading.get_ident() == self.owner
        return [
            SimpleNamespace(
                name="aten::mm",
                device_type="CPU",
                thread=2,
                input_shapes=[[2, 4]],
                stack=["caller"],
                cpu_time_total=5,
                self_cpu_time_total=3,
                device_time_total=20,
                self_device_time_total=20,
                flops=64,
            )
        ]

    def export_stacks(self, path):
        Path(path).write_text("caller 3\n")

    def export_memory_timeline(self, path, device):
        Path(path).write_text(json.dumps([[10, 20], [[0, 8], [0, 16]]]))


@pytest.fixture
def native(monkeypatch):
    instances = []

    def create(**options):
        instance = NativeProfilerStub(**options)
        instances.append(instance)
        return instance

    monkeypatch.setattr(torch.profiler, "profile", create)
    monkeypatch.setattr(
        torch.profiler, "supported_activities", lambda: {torch.profiler.ProfilerActivity.CPU}
    )
    return instances


def config(tmp_path, **options):
    return ProfileConfig(output_dir=str(tmp_path), **options)


def read_archive(job):
    assert job["state"] == "ready", job
    assert not Path(job["staging"]).exists()
    with tarfile.open(job["archive"]) as archive:
        summary = json.load(archive.extractfile("summary.json"))
        inventory = json.load(archive.extractfile("inventory.json"))
    assert "trace.json" in inventory
    return summary


def test_core_options_artifacts_and_raw_cleanup(tmp_path, native):
    controller = Profiler("cpu")
    try:
        controller.start(
            config(
                tmp_path, record_shapes=True, with_stack=True, profile_memory=True, with_flops=True
            )
        )
        with pytest.raises(RuntimeError, match="already active"):
            controller.start(config(tmp_path))
        controller.stop()
        result = controller.wait(5)
        assert result["success"]
        assert read_archive(result["jobs"]["0"])["memory"]["freed_bytes"] == 16
        with tarfile.open(result["jobs"]["0"]["archive"]) as archive:
            memory = json.load(archive.extractfile("memory_summary.json"))
            assert memory["peak_visible_category_bytes"] == 16
            stack = json.load(archive.extractfile("stack_summary.json"))
            assert stack["self_cpu_time_us"] == 3
        assert native[0].options["activities"] == [torch.profiler.ProfilerActivity.CPU]
        assert native[0].options["schedule"] is None
    finally:
        controller.close()


def test_scheduled_completion_short_window_and_restart(tmp_path, native):
    controller = Profiler("cpu")
    try:
        first = controller.start(
            config(tmp_path, wait=1, warmup=1, active=2, repeat=2), scheduled=True
        )
        for _ in range(8):
            controller.step()
        assert not controller.recording
        status = controller.wait(5)
        assert len(status["jobs"]) == 2
        first_directory = Path(status["jobs"]["0"]["archive"]).parent
        assert first_directory.parent == tmp_path
        assert first_directory.name == f"rank-00000-{first['capture_time']}"
        assert (tmp_path / f"manifest-{first['capture_time']}.json").exists()
        assert all(not read_archive(j)["incomplete_window"] for j in status["jobs"].values())
        second = controller.start(config(tmp_path, wait=20), scheduled=True)
        controller.step()
        controller.stop()
        assert controller.wait(5)["empty"]
        assert second["session_id"] != first["session_id"]
        assert second["capture_time"] != first["capture_time"]
        assert first_directory.exists()
        assert (tmp_path / f"manifest-{second['capture_time']}.json").exists()
        controller.start(config(tmp_path, active=10), scheduled=True)
        controller.step()  # Warmup complete; stop before the full active window.
        controller.stop()
        assert read_archive(controller.wait(5)["jobs"]["0"])["incomplete_window"]
    finally:
        controller.close()


def test_background_processing_does_not_block_stop(tmp_path, native, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    owner = threading.get_ident()
    threads = []

    def blocked(*args):
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return _package(*args)

    monkeypatch.setattr("vllm_rlt.profiling_artifacts._package", blocked)
    controller = Profiler("cpu")
    try:
        controller.start(config(tmp_path))
        controller.stop()
        assert entered.wait(2)
        assert threads == [controller.current.artifacts.thread.ident] and threads[0] != owner
        assert not controller.status()["artifacts_complete"]
        assert not controller.current.artifacts_complete
        with pytest.raises(RuntimeError, match="still processing"):
            controller.start(config(tmp_path))
        with pytest.raises(TimeoutError):
            controller.wait(0)
    finally:
        release.set()
        controller.close()
    assert controller.current.artifacts_complete
    read_archive(controller.status()["jobs"]["0"])


@pytest.mark.parametrize("stage", ["parse_trace", "_package"])
def test_processing_failure_preserves_raw(tmp_path, native, monkeypatch, stage):
    def fail(*args):
        raise ValueError("injected failure")

    monkeypatch.setattr("vllm_rlt.profiling_artifacts." + stage, fail)
    controller = Profiler("cpu")
    try:
        controller.start(config(tmp_path))
        controller.stop()
        status = controller.wait(5)
        job = status["jobs"]["0"]
        assert not status["success"]
        assert job["state"] == "failed" and "injected failure" in job["error"]
        assert (Path(job["staging"]) / "trace.json").exists()
        assert not Path(job["archive"]).exists()
    finally:
        controller.close()


def test_same_second_captures_preserve_previous_artifacts(tmp_path, native, monkeypatch):
    from datetime import datetime, timezone

    instant = datetime(2026, 9, 30, 2, 10, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "vllm_rlt.profiling_artifacts.datetime", SimpleNamespace(now=lambda zone: instant)
    )
    controller = Profiler("cpu")
    archives = []
    try:
        for suffix in ("", "-2"):
            status = controller.start(config(tmp_path))
            assert status["capture_time"] == f"20260930T021015Z{suffix}"
            controller.stop()
            job = controller.wait(5)["jobs"]["0"]
            archives.append(Path(job["archive"]))
            assert archives[-1].parent.name == f"rank-00000-20260930T021015Z{suffix}"
        assert archives[0] != archives[1]
        assert all(path.is_file() for path in archives)
    finally:
        controller.close()


def test_rank_artifacts_are_distinct(tmp_path, native):
    capture_time = "20260930T021015Z"
    for rank in (0, 1):
        controller = Profiler("cpu")
        try:
            controller.start(
                config(tmp_path),
                session_id="shared",
                capture_time=capture_time,
                rank=rank,
                role="prefill" if rank == 0 else "decode",
                managed_manifest=False,
            )
            controller.stop()
            job = controller.wait(5)["jobs"]["0"]
            assert read_archive(job)["rank"] == rank
            assert Path(job["archive"]).parent == tmp_path / f"rank-{rank:05d}-{capture_time}"
            assert not (tmp_path / "shared").exists()
        finally:
            controller.close()


def test_configuration_validation_and_disabled_path(tmp_path, native):
    from vllm_rlt.entrypoints.runtime_args import profile_config_from_args

    assert profile_config_from_args(SimpleNamespace(profile=False)) is None
    with pytest.raises(ValueError, match="output_dir"):
        profile_config_from_args(SimpleNamespace(profile=True))
    assert profile_config_from_args(
        SimpleNamespace(profile=True, profile_dir=str(tmp_path))
    ) == config(tmp_path)
    for output_dir in (None, "", 123):
        with pytest.raises(ValueError, match="output_dir"):
            ProfileConfig(output_dir=output_dir)
    assert not list(tmp_path.iterdir()) and not native
    controller = Profiler("cpu")
    for options in (
        {"active": 0},
        {"wait": -1},
        {"repeat": True},
        {"activities": ("bogus",)},
        {"record_shapes": 1},
    ):
        with pytest.raises(ValueError):
            config(tmp_path, **options)
    with pytest.raises(ValueError, match="unsupported"):
        controller.start(config(tmp_path, activities=("cuda",)))
    assert not list(tmp_path.iterdir())


def test_stream_parser_handles_large_events(tmp_path):
    trace = json.loads(json.dumps(TRACE))
    trace["traceEvents"][0]["args"]["Call stack"] = "x" * 70000
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(trace))
    parse_trace(source, tmp_path, {"config": {"profile_memory": True}})
    with (tmp_path / "operators.csv").open() as output:
        rows = list(csv.DictReader(output))
    assert len(rows) == 2 and rows[0]["self_duration_us"] == ""
    assert rows[1]["attribution"] == "device"
    assert json.loads((tmp_path / "summary.json").read_text())["observed_duration_us"] == 22


def fake_pd():
    from vllm_rlt.pd.engine import PDEngine

    engine = object.__new__(PDEngine)
    engine.closed = False
    engine.peers = {"p": SimpleNamespace(name="p"), "d": SimpleNamespace(name="d")}
    engine.config = SimpleNamespace(startup_timeout=0.1, shutdown_timeout=0.1)
    engine._profile_session = None
    engine._profile_recording = False
    engine._profile_statuses = {}
    engine._profile_errors = []
    engine.transfers = {}
    return engine


def test_pd_acknowledgments_and_partial_failure_rollback(tmp_path):
    engine = fake_pd()
    sent = []
    fail_decode = False
    recording = {}

    from vllm_rlt.engine.llm_engine import StageClient

    queued = []

    def send(name, kind, **fields):
        assert kind == "control"
        sent.append((name, fields["method"], fields))
        queued.append((name, fields))

    for name, peer in engine.peers.items():
        client = StageClient.__new__(StageClient)
        client.pending_controls = set()
        client.replies = {}
        client.send = lambda kind, _name=name, **fields: send(_name, kind, **fields)
        peer.client = client

    def poll():
        # Both calls must be submitted before waiting for any worker response.
        assert len(queued) == len(engine.peers)
        for name, fields in queued:
            method = fields["method"]
            if fail_decode and name == "d" and method == "start_profile":
                reply = {"error": "cannot start"}
            else:
                if method != "profile_status":
                    recording[name] = method == "start_profile"
                reply = {
                    "result": {
                        "recording": recording.get(name, False),
                        "artifacts_complete": not recording.get(name, False),
                    }
                }
            client = engine.peers[name].client
            client.replies[fields["control_id"]] = dict(
                kind="reply",
                control_id=fields["control_id"],
                session_id=fields["session_id"],
                **reply,
            )
        queued.clear()

    engine._poll = poll
    status = engine.start_profile(config(tmp_path))
    assert status["recording"] and set(status["ranks"]) == {"0", "1"}
    assert all(not entry[2]["kwargs"]["scheduled"] for entry in sent)
    assert {entry[2]["kwargs"]["capture_time"] for entry in sent} == {status["capture_time"]}
    assert (tmp_path / f"manifest-{status['capture_time']}.json").exists()
    assert not (tmp_path / status["session_id"]).exists()
    engine.stop_profile()
    assert not engine.wait_for_profile_artifacts(1)["recording"]
    fail_decode = True
    with pytest.raises(RuntimeError, match="cannot start"):
        engine.start_profile(config(tmp_path))
    assert sent[-2][1] == sent[-1][1] == "stop_profile"
    assert "cannot start" in engine._profile_manifest()["errors"][0]
    assert all(
        not peer.client.pending_controls and not peer.client.replies
        for peer in engine.peers.values()
    )


def test_serving_control_runs_on_owner_thread(tmp_path):
    from vllm_rlt.serving.worker import EngineWorker

    threads = []
    engine = SimpleNamespace(start_profile=lambda *a, **kw: threads.append(threading.get_ident()))
    worker = EngineWorker(lambda: (engine, None))
    worker.engine, worker.ready = engine, True
    try:
        asyncio.run(worker.profile_control("start", config(tmp_path)))
        assert len(threads) == 1 and threads[0] != threading.get_ident()
    finally:
        worker.executor.shutdown(wait=True)


def test_completed_sessions_release_native_objects(tmp_path, native):
    controller = Profiler("cpu")
    previous = None
    try:
        for _ in range(3):
            controller.start(config(tmp_path, record_shapes=True, with_stack=True))
            if previous is not None:
                gc.collect()
                assert previous() is None
            profiler = weakref.ref(native.pop())
            previous = weakref.ref(controller.current)
            controller.stop()
            assert controller.current.profiler is None
            assert controller.wait(5)["success"]
            gc.collect()
            assert profiler() is None
    finally:
        controller.close()


def test_stop_failure_releases_native_profiler(tmp_path, native, monkeypatch):
    controller = Profiler("cpu")
    controller.start(config(tmp_path))

    def fail():
        raise RuntimeError("native stop failed")

    monkeypatch.setattr(native[0], "stop", fail)
    try:
        controller.stop()
        result = controller.wait(5)
        assert controller.current.profiler is None
        assert result["artifacts_complete"] and not result["success"]
        assert "native stop failed" in result["errors"]
    finally:
        controller.close()


def test_http_wait_after_partial_pd_start_failure(tmp_path):
    from vllm_rlt.pd.engine import PDEngine
    from vllm_rlt.serving.worker import EngineWorker

    engine = PDEngine.__new__(PDEngine)
    engine.peers = {"p": None, "d": None}
    engine._profile_session = "failed-start"
    engine._profile_timestamp = "20260930T021015Z"
    engine._profile_root = tmp_path
    engine._profile_recording = False
    engine._profile_errors = ["decode start failed"]
    engine._profile_statuses = {
        "p": {"recording": False, "artifacts_complete": False, "success": False},
        "d": Profiler("cpu").status(),
    }
    polls = []

    def status():
        polls.append(True)
        if len(polls) == 2:
            engine._profile_statuses["p"].update(artifacts_complete=True, success=True)
        return engine._profile_manifest()

    worker = EngineWorker(lambda: (None, None))
    worker.engine = SimpleNamespace(profile_status=status)
    worker.ready = True
    try:
        result = asyncio.run(worker.profile_control("wait", timeout=1))
        assert len(polls) == 2  # Wait for P's real work, never for unstarted D.
        assert result["artifacts_complete"] and not result["success"]
        assert result["errors"] == ["decode start failed"]
        assert result["ranks"]["1"]["state"] == "not_started"
    finally:
        worker.executor.shutdown(wait=True)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_pd_load_profile_failure_closes_engine(monkeypatch, cleanup_fails, tmp_path):
    from tokenizers.decoders import ByteLevel
    from transformers import AutoTokenizer

    from vllm_rlt.config import CacheConfig
    from vllm_rlt.entrypoints import pd_serve

    tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(decoder=ByteLevel()))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(pd_serve, "runtime_configs", lambda args: {"cache_config": CacheConfig()})
    original = RuntimeError("profile initialization failed")
    closed = []

    def fail(*args, **kwargs):
        raise original

    def close():
        closed.append(True)
        if cleanup_fails:
            raise RuntimeError("cleanup also failed")

    engine = SimpleNamespace(start_profile=fail, close=close)
    monkeypatch.setattr(pd_serve, "PDEngine", lambda *a, **k: engine)
    monkeypatch.setattr(
        pd_serve,
        "profile_config_from_args",
        lambda args: ProfileConfig(output_dir="/tmp/profiles"),
    )
    (tmp_path / "config.json").write_text('{"model_type": "ouro"}')
    args = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        tokenizer=None,
        tokenizer_revision=None,
        dtype="float32",
        prefill_devices=[0],
        decode_devices=[1],
        pd_transfer_chunk_bytes=1024,
        pd_max_inflight_bytes=4096,
        pd_max_transfer_descriptors=8,
        max_requests=8,
        request_timeout=1,
        pd_startup_timeout=1,
        shutdown_timeout=1,
        nixl_backend="UCX",
        pd_max_receiving_requests=2,
        pd_max_draining_requests=2,
        prefill_num_blocks=16,
        decode_num_blocks=16,
        prefill_max_num_seqs=2,
        prefill_max_num_batched_tokens=16,
        prefill_chunk_size=16,
        decode_max_num_seqs=2,
        decode_max_num_batched_tokens=16,
        mode="refill",
        min_coda_batch_size=1,
        attention_backend="torch",
    )
    with pytest.raises(RuntimeError) as error:
        pd_serve.load_engine(args)
    assert error.value is original
    assert closed == [True]


@pytest.mark.parametrize("entrypoint", ["serve", "pd_serve"])
def test_invalid_profile_config_fails_before_loading_model(monkeypatch, entrypoint):
    from importlib import import_module

    from transformers import AutoTokenizer

    module = import_module("vllm_rlt.entrypoints." + entrypoint)

    def unexpected_load(*args, **kwargs):
        pytest.fail("profiling configuration must be validated before loading resources")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", unexpected_load)
    engine_name = "StageClient" if entrypoint == "serve" else "PDEngine"
    monkeypatch.setattr(module, engine_name, unexpected_load)
    with pytest.raises(ValueError, match="output_dir"):
        module.load_engine(SimpleNamespace(profile=True))


def test_profile_cli_defaults_and_overrides(tmp_path, monkeypatch):
    import argparse

    from vllm_rlt.entrypoints.runtime_args import add_profile_args, profile_config_from_args

    monkeypatch.setattr(ProfileConfig, "warmup", 3)
    parser = argparse.ArgumentParser()
    add_profile_args(parser)
    args = parser.parse_args(["--profile", "--profile-dir", str(tmp_path)])
    assert args.profile_warmup == 3
    assert profile_config_from_args(args).warmup == 3
    assert (
        profile_config_from_args(SimpleNamespace(profile=True, profile_dir=str(tmp_path))).warmup
        == 3
    )
    args = parser.parse_args(
        [
            "--profile",
            "--profile-dir",
            str(tmp_path),
            "--profile-warmup",
            "5",
            "--profile-activities",
            "cpu,cuda",
            "--profile-with-flops",
        ]
    )
    result = profile_config_from_args(args)
    assert result.warmup == 5
    assert result.activities == ("cpu", "cuda")
    assert result.with_flops
    assert profile_config_from_args(SimpleNamespace()) is None


def test_finish_drains_cycles_beyond_queue_capacity(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    processed = []
    owner = threading.get_ident()

    def package(staging, archive):
        assert threading.get_ident() != owner
        entered.set()
        assert release.wait(5)
        processed.append(staging.name)
        return "checksum", None

    monkeypatch.setattr("vllm_rlt.profiling_artifacts._package", package)
    worker = ProfileArtifactWorker(
        tmp_path,
        dict(rank=0, session_id="test", capture_time="20260930T021015Z"),
        managed_manifest=False,
    )
    worker.start(1)
    try:
        worker.submit(0, tmp_path / "0", tmp_path / "0.tar.gz", incomplete_window=False)
        assert entered.wait(2)
        # Block the consumer so submission must overflow the bounded queue.
        for cycle in range(1, 21):
            worker.submit(
                cycle,
                tmp_path / str(cycle),
                tmp_path / f"{cycle}.tar.gz",
                incomplete_window=False,
            )
        worker.finish(2)
        status = worker.status()
        assert not status["recording"] and not status["artifacts_complete"]
        with pytest.raises(TimeoutError):
            worker.wait(0)
        release.set()
        status = worker.wait(5)
        assert status["success"] and status["artifacts_complete"]
        assert sorted(map(int, processed)) == list(range(21))
        persisted = json.loads((tmp_path / "status.json").read_text())
        assert persisted["jobs"] == status["jobs"]
        assert persisted["stopped_at_ns"] == 2 and not persisted["recording"]
    finally:
        release.set()
        worker.finish(2)
        worker.wait(5)


def test_worker_shared_control_validates_profile_session():
    from vllm_rlt.engine.stage_config import StageConfig
    from vllm_rlt.engine.worker import StageWorker

    worker = StageWorker.__new__(StageWorker)
    started, replies = [], []
    session = None

    def start_profile(config, **identity):
        nonlocal session
        started.append(identity)
        session = identity["session_id"]
        return dict(session_id=session, recording=True)

    stopped = []
    worker.engine = SimpleNamespace(
        stage_config=StageConfig(role="decode"),
        add_request=lambda *a: None,
        abort_request=lambda *a: None,
        stats=lambda: {},
        get_weight_version=lambda: 0,
        pause_generation=lambda: None,
        resume_generation=lambda: None,
        reset_prefix_cache=lambda: None,
        start_weight_update=lambda **kwargs: None,
        update_weights=lambda **kwargs: None,
        finish_weight_update=lambda: 1,
        start_profile=start_profile,
        stop_profile=lambda: stopped.append(True),
        profile_status=lambda: dict(session_id=session),
        wait_for_profile_artifacts=lambda: {},
    )
    worker.send = lambda kind, **fields: replies.append(dict(kind=kind, **fields))
    worker.command(
        dict(
            kind="control",
            control_id="start",
            method="start_profile",
            session_id="current",
            kwargs=dict(config={}, rank=1, managed_manifest=False),
        )
    )
    assert started == [dict(rank=1, managed_manifest=False, session_id="current", role="decode")]
    worker.command(
        dict(kind="control", control_id="stale", method="stop_profile", session_id="previous")
    )
    assert not stopped and "session ID mismatch" in replies[-1]["error"]
    worker.command(
        dict(kind="control", control_id="stop", method="stop_profile", session_id="current")
    )
    assert stopped == [True]
    assert all(reply["kind"] == "reply" for reply in replies)
    assert replies[-1]["session_id"] == "current"
