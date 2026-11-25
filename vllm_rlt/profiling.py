"""Inference profiling configuration and capture lifecycle control."""

import logging
import os
import re
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from vllm_rlt import profiling_artifacts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfileConfig:
    output_dir: str
    activities: tuple[str, ...] | None = None
    record_shapes: bool = False
    with_stack: bool = False
    profile_memory: bool = False
    with_flops: bool = False  # Estimate floating-point operations for supported operators.
    wait: int = 0  # Scheduled mode: skip this many engine steps at the start of each cycle.
    warmup: int = 1  # Scheduled mode: warm up collection for this many steps without saving events.
    active: int = 10  # Scheduled mode: record this many engine steps per cycle.
    repeat: int = 1  # Scheduled mode: number of cycles; 0 repeats until explicitly stopped.

    def __post_init__(self):
        for name in ("record_shapes", "with_stack", "profile_memory", "with_flops"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("wait", "warmup", "active", "repeat"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "active" else 0):
                raise ValueError(f"invalid profile {name}")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("profiling requires output_dir")
        if self.activities is not None:
            values = tuple(self.activities)
            if (
                not values
                or len(set(values)) != len(values)
                or any(a not in ("cpu", "cuda") for a in values)
            ):
                raise ValueError("profile activities must be unique cpu/cuda values")
            object.__setattr__(self, "activities", values)

    def resolve_activities(self, device):
        if str(device).startswith("npu"):
            raise ValueError("NPU device profiling requires torch_npu.profiler")
        names = self.activities or (("cpu", "cuda") if str(device).startswith("cuda") else ("cpu",))
        values = [getattr(torch.profiler.ProfilerActivity, name.upper()) for name in names]
        if not set(values) <= torch.profiler.supported_activities():
            raise ValueError(f"unsupported profile activities: {names}")
        return values


class ProfileSession:
    """Own one native profiler and hand exported cycles to the artifact worker."""

    def __init__(
        self,
        config,
        device,
        *,
        scheduled=False,
        session_id=None,
        capture_time=None,
        rank=None,
        role="engine",
        managed_manifest=True,
    ):
        self.config = config
        activities = config.resolve_activities(device)
        self.session_id = session_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.session_id):
            raise ValueError("invalid profile session ID")
        if rank is None:
            rank = (
                torch.distributed.get_rank()
                if (torch.distributed.is_available() and torch.distributed.is_initialized())
                else int(os.environ.get("RANK", 0))
            )
        if type(rank) is not int or rank < 0:
            raise ValueError("profile rank must be a nonnegative integer")
        self.rank = rank
        self.capture_time = capture_time or profiling_artifacts.profile_timestamp(config.output_dir)
        if not re.fullmatch(r"\d{8}T\d{6}Z(?:-[1-9]\d*)?", self.capture_time):
            raise ValueError("invalid profile capture timestamp")
        self.root = Path(config.output_dir).expanduser().resolve()
        self.directory = self.root / f"rank-{rank:05d}-{self.capture_time}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(
            session_id=self.session_id,
            capture_time=self.capture_time,
            rank=rank,
            role=role,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            device=str(device),
            config=asdict(config),
            scheduled=scheduled,
            torch_version=torch.__version__,
            activities=[activity.name.lower() for activity in activities],
        )
        self.artifacts = profiling_artifacts.ProfileArtifactWorker(
            self.directory, self.metadata, managed_manifest=managed_manifest
        )
        self.steps = 0
        self.cycles = 0
        self.recording = False
        self.owner = threading.get_ident()
        self.scheduled = scheduled
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=config.wait, warmup=config.warmup, active=config.active, repeat=config.repeat
            )
            if scheduled
            else None,
            on_trace_ready=self._export,
            record_shapes=config.record_shapes,
            with_stack=config.with_stack,
            profile_memory=config.profile_memory,
            with_flops=config.with_flops,
            # Kineto needs verbose export to retain operator call stacks.
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
            if config.with_stack
            else None,
        )
        self.profiler.start()
        self.recording = True
        self.metadata["started_at_ns"] = time.time_ns()
        self.artifacts.start(self.metadata["started_at_ns"])

    def _check_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("profiler control must run on the engine owner thread")

    def step(self):
        self._check_owner()
        if self.recording:
            self.steps += 1
            self.profiler.step()
            length = self.config.wait + self.config.warmup + self.config.active
            if self.scheduled and self.config.repeat and self.steps >= length * self.config.repeat:
                self.stop()

    def _export(self, profiler):
        cycle = self.cycles
        self.cycles += 1
        staging = self.directory / f".cycle-{cycle:05d}.raw"
        archive = self.directory / f"cycle-{cycle:05d}.tar.gz"
        try:
            staging.mkdir()
            profiling_artifacts.export_profile(
                profiler, staging, self.config, self.metadata["device"]
            )
            expected = self.config.wait + self.config.warmup + self.config.active
            metadata = dict(
                self.metadata,
                cycle=cycle,
                steps=self.steps,
                incomplete_window=self.scheduled and self.steps < expected * (cycle + 1),
            )
            profiling_artifacts._write_json_atomic(staging / "metadata.json", metadata)
            self.artifacts.submit(
                cycle, staging, archive, incomplete_window=metadata["incomplete_window"]
            )
        except Exception as exc:
            self.artifacts.fail_export(cycle, staging, exc)
            logger.exception("profile export failed")

    @property
    def artifacts_complete(self) -> bool:
        return self.artifacts.artifacts_complete

    def status(self):
        return dict(self.artifacts.status(), steps=self.steps)

    def stop(self):
        self._check_owner()
        if self.recording:
            try:
                self.profiler.stop()
            except Exception as exc:
                self.artifacts.record_error(exc)
                logger.exception("profiler stop failed")
            finally:
                # Export has snapshotted everything needed by the artifact worker.
                # Drop native events (including shapes/stacks) on the owner thread.
                self.profiler = None
                self.recording = False
                self.artifacts.finish(time.time_ns())
        return self.status()

    def wait(self, timeout=None):
        self.artifacts.wait(timeout)
        return self.status()


class Profiler:
    """Owner-thread control retaining only the latest capture's status."""

    def __init__(self, device):
        self.device = device
        self.current = None

    @property
    def recording(self):
        return self.current is not None and self.current.recording

    def start(self, config, *, scheduled=False, **identity):
        if isinstance(config, dict):
            config = ProfileConfig(**config)
        if self.recording:
            raise RuntimeError("a profiling session is already active")
        if self.current is not None and not self.current.artifacts_complete:
            raise RuntimeError("previous profile artifacts are still processing")
        if self.current is not None:
            self.current.wait()  # Join the completed worker before releasing the session.
        self.current = ProfileSession(config, self.device, scheduled=scheduled, **identity)
        return self.status()

    def stop(self):
        return self.current.stop() if self.current else self.status()

    def status(self):
        return (
            self.current.status()
            if self.current
            else {
                "state": "not_started",
                "recording": False,
                "jobs": {},
                "artifacts_complete": True,
                "success": False,
            }
        )

    def wait(self, timeout=None):
        if self.current is not None:
            return self.current.wait(timeout)
        return self.status()

    def step(self):
        if self.recording:
            self.current.step()

    def close(self):
        self.stop()
        return self.wait()
