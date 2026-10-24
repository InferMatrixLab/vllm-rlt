"""One stage resource owner per process, independent of deployment topology.

The worker loads the model, drives StageEngine and handles generic control.
An optional connector owns transport resources and protocol-specific messages.
"""

import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, is_dataclass

import torch

from vllm_rlt.engine.connector import TransferConnector
from vllm_rlt.engine.stage_config import StageConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import MODEL_MAPPING, AutoModelForCausalLM


class StageWorker:
    def __init__(
        self,
        stage_config: StageConfig,
        device,
        channel,
        model_source,
        options,
        *,
        connector_factory: Callable[..., TransferConnector] | None = None,
        max_control_messages=32,
    ):
        if type(max_control_messages) is not int or max_control_messages < 1:
            raise ValueError("max_control_messages must be a positive integer")
        device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        torch.set_num_threads(1)
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(device)
        torch.manual_seed(options["seed"])
        self.channel = channel
        self.max_control_messages = max_control_messages
        dtype = getattr(torch, options["dtype"])
        if isinstance(model_source, torch.nn.Module):
            self.model = model_source.to(device=device, dtype=dtype)
        elif is_dataclass(model_source):
            model_cls = MODEL_MAPPING[model_source.to_dict()["model_type"]]
            self.model = model_cls(model_source).to(device=device, dtype=dtype)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_source, revision=options["revision"], device=str(device), dtype=dtype
            )
        self.engine = StageEngine(self.model, stage_config=stage_config, **options["engine"])
        self.connector = (
            connector_factory(self.engine, notify=self.send)
            if connector_factory is not None
            else None
        )
        self.engine.connector = self.connector
        self.running = True
        self.draining = False
        cache = self.engine.cache_manager
        active_limit = self.engine.scheduler.config.max_num_seqs
        info = dict(
            stage_id=stage_config.stage_id,
            role=stage_config.role,
            model=asdict(self.model.config),
            model_type=self.model.config.to_dict()["model_type"],
            num_blocks=cache.num_blocks,
            block_size=cache.block_size,
            depths=cache.storage_depths,
            max_active_requests=active_limit,
            watermark_blocks=cache.watermark_blocks,
        )
        if self.connector is not None:
            info.update(self.connector.startup_info())
        self.send("ready", **info)

    def send(self, kind, **fields):
        self.channel.send(dict(kind=kind, **fields))

    def command(self, command):
        kind = command["kind"]
        if kind == "control":
            # Only host-level operations cross this interface; never a loop RPC.
            methods = {
                "add_request": self.engine.add_request,
                "abort_request": self.engine.abort_request,
                "stats": self.engine.stats,
                "get_weight_version": self.engine.get_weight_version,
                "pause_generation": self.engine.pause_generation,
                "resume_generation": self.engine.resume_generation,
                "reset_prefix_cache": self.engine.reset_prefix_cache,
                "start_weight_update": self.engine.start_weight_update,
                "update_weights": self.engine.update_weights,
                "finish_weight_update": self.engine.finish_weight_update,
                "start_profile": self.engine.start_profile,
                "stop_profile": self.engine.stop_profile,
                "profile_status": self.engine.profile_status,
                "wait_for_profile_artifacts": self.engine.wait_for_profile_artifacts,
            }
            method = command["method"]
            session_id = command.get("session_id")
            admission_paused = method == "add_request" and (
                self.engine._paused
                or self.engine._update_version is not None
                or self.engine._update_failed
            )
            try:
                if session_id is not None and method in (
                    "stop_profile",
                    "profile_status",
                    "wait_for_profile_artifacts",
                ):
                    current = self.engine.profile_status().get("session_id")
                    if current is not None and current != session_id:
                        raise ValueError("profile session ID mismatch")
                kwargs = dict(command.get("kwargs", {}))
                if method == "start_profile" and session_id is not None:
                    kwargs.update(session_id=session_id, role=self.engine.stage_config.role)
                if method == "abort_request" and self.connector is not None:
                    self.connector.command(dict(kind="abort", request_id=command["args"][0]))
                if method == "abort_request" and not self.engine.has_request(command["args"][0]):
                    result = None
                else:
                    result = methods[method](*command.get("args", ()), **kwargs)
            except Exception as exc:
                if (
                    not isinstance(exc, (ValueError, KeyError))
                    and not admission_paused
                    and method
                    not in (
                        "get_weight_version",
                        "pause_generation",
                        "resume_generation",
                        "reset_prefix_cache",
                        "start_weight_update",
                        "update_weights",
                        "finish_weight_update",
                        "start_profile",
                        "stop_profile",
                        "profile_status",
                        "wait_for_profile_artifacts",
                    )
                ):
                    raise
                self.send(
                    "reply",
                    control_id=command["control_id"],
                    session_id=session_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            else:
                self.send(
                    "reply", control_id=command["control_id"], session_id=session_id, result=result
                )
            return
        if kind == "shutdown":
            if self.connector is not None:
                raise ValueError("connected stages must drain through the transfer protocol")
            for request_id in self.engine.request_ids():
                self.engine.abort_request(request_id)
            self.draining = True
            return
        if self.connector is not None and self.connector.command(command):
            return
        if kind == "add":
            self.engine.add_request(
                command["request_id"],
                command["tokens"],
                command["params"],
                trace_id=command.get("trace_id"),
            )
        elif kind == "abort":
            self.output(self.engine.abort_request(command["request_id"]))
        elif kind == "stop":
            if self.engine.has_unfinished_requests() or (
                self.connector is not None and self.connector.has_pending
            ):
                raise RuntimeError("stop requested before stage work drained")
            self.running = False
        else:
            raise ValueError(f"unknown stage command {kind}")

    def output(self, output):
        self.send("output", request_id=output.request_id, output=output)
        if output.finished and self.connector is not None:
            self.connector.output_finished(output.request_id)

    def run(self):
        while self.running:
            for _ in range(self.max_control_messages):
                if not self.channel.poll():
                    break
                self.command(self.channel.recv())
                if not self.running:
                    break
            if not self.running:
                break
            if self.connector is not None:
                self.connector.progress()
            if self.draining and not self.engine.has_unfinished_requests():
                break
            if self.engine.has_unfinished_requests():
                for output in self.engine.step():
                    self.output(output)
            if (
                not self.engine.has_unfinished_requests()
                and not (self.connector is not None and self.connector.has_pending)
                and not self.channel.poll()
            ):
                time.sleep(0.001)
        self.engine.close()
        metrics = self.connector.close() if self.connector is not None else {}
        self.send("stopped", used_blocks=self.engine.cache_manager.num_used_blocks, **metrics)


def worker_main(
    stage_config,
    device,
    channel,
    model_source,
    options,
    *,
    connector_factory=None,
    max_control_messages=32,
):
    # A connected worker retains all owners after a fatal error until the
    # deployment coordinator has stopped remote writers and terminates it.
    owner = None
    try:
        owner = StageWorker(
            stage_config,
            device,
            channel,
            model_source,
            options,
            connector_factory=connector_factory,
            max_control_messages=max_control_messages,
        )
        owner.run()
    except BaseException as exc:
        try:
            channel.send(
                dict(
                    kind="fatal",
                    error=traceback.format_exc(),
                    error_type=type(exc).__name__,
                    pid=os.getpid(),
                )
            )
            if connector_factory is not None:
                while True:
                    time.sleep(1)
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        channel.close()


def start_stage_worker(
    stage_config, device, model_source, options, *, connector_factory=None, max_control_messages=32
):
    """Launch the same resource owner for every deployment role."""
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=worker_main,
        args=(stage_config, device, child, model_source, options),
        kwargs=dict(connector_factory=connector_factory, max_control_messages=max_control_messages),
        name=f"vllm-rlt-{stage_config.stage_id}",
    )
    try:
        process.start()
    except BaseException:
        parent.close()
        raise
    finally:
        child.close()
    return process, parent
