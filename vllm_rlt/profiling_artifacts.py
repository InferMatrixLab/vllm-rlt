"""Export, process, and track profiling artifacts through completion."""

import csv
import hashlib
import json
import logging
import os
import queue
import shutil
import tarfile
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def profile_timestamp(output_dir):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sequence = 1
    while True:
        name = stamp if sequence == 1 else f"{stamp}-{sequence}"
        try:
            # Reserve the shared manifest atomically, including across processes.
            with (root / f"manifest-{name}.json").open("x") as stream:
                json.dump({"capture_time": name, "ranks": {}}, stream)
            return name
        except FileExistsError:
            sequence += 1


def _write_json_atomic(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class _JSONStream:
    """Decode one JSON value at a time, keeping only the current event in memory."""

    def __init__(self, source):
        self.source, self.buffer = source, ""
        self.decoder = json.JSONDecoder()
        self.eof = False

    def fill(self):
        chunk = self.source.read(65536)
        self.buffer += chunk
        self.eof = not chunk

    def peek(self):
        self.buffer = self.buffer.lstrip()
        while not self.buffer and not self.eof:
            self.fill()
            self.buffer = self.buffer.lstrip()
        return self.buffer[:1]

    def take(self, character):
        if self.peek() != character:
            raise ValueError(f"malformed trace: expected {character!r}")
        self.buffer = self.buffer[1:]

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
                # A numeric value may extend into the next input chunk.
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                self.buffer = self.buffer[end:]
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.fill()


def _trace_events(path):
    with open(path) as source:
        stream = _JSONStream(source)
        stream.take("{")
        found = False
        while stream.peek() != "}":
            key = stream.value()
            stream.take(":")
            if key == "traceEvents":
                found = True
                stream.take("[")
                while stream.peek() != "]":
                    event = stream.value()
                    if not isinstance(event, dict):
                        raise ValueError("trace event must be an object")
                    yield event
                    if stream.peek() == "]":
                        break
                    stream.take(",")
                stream.take("]")
            else:
                stream.value()
            if stream.peek() == "}":
                break
            stream.take(",")
        stream.take("}")
        if not found or stream.peek():
            raise ValueError("missing traceEvents or trailing trace data")


def parse_trace(path, output, metadata):
    """Aggregate exported events, without claiming summed durations are wall time."""
    groups = {}
    counts = Counter()
    low, high = None, None
    memory = {"events": 0, "allocated_bytes": 0, "freed_bytes": 0}
    for event in _trace_events(path):
        category = str(event.get("cat", ""))
        counts[category] += 1
        args = event.get("args") or {}
        if event.get("name") == "[memory]":
            memory["events"] += 1
            size = args.get("Bytes", 0)
            if isinstance(size, (int, float)):
                memory["allocated_bytes" if size >= 0 else "freed_bytes"] += abs(size)
        if event.get("ph") != "X":
            continue
        start, duration = event.get("ts"), event.get("dur")
        if not isinstance(start, (int, float)) or not isinstance(duration, (int, float)):
            continue
        low = start if low is None else min(low, start)
        high = start + duration if high is None else max(high, start + duration)
        if category not in (
            "cpu_op",
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
            "cuda_runtime",
            "cuda_driver",
            "user_annotation",
            "python_function",
        ):
            continue
        shape = args.get("Input Dims")
        stack = args.get("Call stack")
        attribution = "device" if category in ("kernel", "gpu_memcpy", "gpu_memset") else "cpu"
        key = (
            event.get("name", ""),
            category,
            str(event.get("pid", "")),
            str(event.get("tid", "")),
            str(args.get("device", "")),
            str(args.get("stream", "")),
            json.dumps(shape),
            json.dumps(stack),
        )
        if key not in groups:
            groups[key] = dict(
                zip(
                    ("name", "category", "pid", "tid", "device", "stream", "input_shapes", "stack"),
                    key,
                )
            )
            groups[key].update(
                attribution=attribution,
                count=0,
                inclusive_duration_us=0,
                self_duration_us=None,
                flops=None,
            )
        row = groups[key]
        row["count"] += 1
        row["inclusive_duration_us"] += duration
        flops = args.get("flops", args.get("FLOPs"))
        if isinstance(flops, (int, float)):
            row["flops"] = (row["flops"] or 0) + flops
    fields = (
        "name",
        "category",
        "attribution",
        "pid",
        "tid",
        "device",
        "stream",
        "count",
        "inclusive_duration_us",
        "self_duration_us",
        "input_shapes",
        "stack",
        "flops",
    )
    with open(output / "operators.csv", "w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(groups.values())
    summary = dict(
        metadata,
        event_counts=dict(counts),
        observed_duration_us=None if low is None else high - low,
        memory=memory if metadata["config"]["profile_memory"] else None,
        notes=[
            "Durations are inclusive event times; their sum is not wall time.",
            "Self time and unavailable native event fields are null.",
            "Memory events describe visible allocations, not total device memory.",
        ],
    )
    # Older metadata may omit the resolved activities stored by ProfileSession.
    activities = metadata.get("activities", metadata["config"].get("activities")) or ()
    if "cuda" in activities and not any(counts[c] for c in ("kernel", "gpu_memcpy", "gpu_memset")):
        summary["notes"].append("No CUDA device events recorded; device coverage is unverified.")
    _write_json_atomic(output / "summary.json", summary)


def _parse_native_details(staging):
    """Aggregate immutable native event snapshots on the artifact thread."""
    metrics = {}
    path = staging / "events.jsonl"
    if path.exists():
        with path.open() as source:
            for line in source:
                event = json.loads(line)
                key = (
                    event["name"],
                    event["device_type"],
                    event["thread"],
                    json.dumps(event["input_shapes"]),
                    json.dumps(event["stack"]),
                )
                if key not in metrics:
                    metrics[key] = dict(
                        zip(("name", "device_type", "thread", "input_shapes", "stack"), key)
                    )
                    metrics[key].update(
                        count=0,
                        cpu_time_us=0,
                        self_cpu_time_us=0,
                        device_time_us=0,
                        self_device_time_us=0,
                        flops=None,
                    )
                row = metrics[key]
                row["count"] += 1
                for field in (
                    "cpu_time_us",
                    "self_cpu_time_us",
                    "device_time_us",
                    "self_device_time_us",
                    "flops",
                ):
                    if event[field] is not None:
                        row[field] = (row[field] or 0) + event[field]
        fields = (
            "name",
            "device_type",
            "thread",
            "input_shapes",
            "stack",
            "count",
            "cpu_time_us",
            "self_cpu_time_us",
            "device_time_us",
            "self_device_time_us",
            "flops",
        )
        with (staging / "operator_metrics.csv").open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metrics.values())
    memory = staging / "memory.json"
    if memory.exists():
        # Native memory timeline is [timestamps, category-size rows]. Read each
        # row separately; traces can contain millions of allocation changes.
        with memory.open() as source:
            stream = _JSONStream(source)
            stream.take("[")
            stream.take("[")
            samples = 0
            first, last = None, None
            while stream.peek() != "]":
                last = stream.value()
                first = last if first is None else first
                samples += 1
                if stream.peek() == "]":
                    break
                stream.take(",")
            stream.take("]")
            stream.take(",")
            stream.take("[")
            rows, peak = 0, 0
            while stream.peek() != "]":
                values = stream.value()
                peak = max(peak, sum(values))
                rows += 1
                if stream.peek() == "]":
                    break
                stream.take(",")
            stream.take("]")
            stream.take("]")
            if rows != samples or stream.peek():
                raise ValueError("invalid memory timeline")
        _write_json_atomic(
            staging / "memory_summary.json",
            dict(
                samples=samples,
                first_timestamp=first,
                last_timestamp=last,
                peak_visible_category_bytes=peak,
                note="Native timeline categories, not total device memory.",
            ),
        )
    stacks = staging / "stacks.txt"
    if stacks.exists():
        entries, weight = 0, 0.0
        with stacks.open() as source:
            for line in source:
                if line.strip():
                    _, value = line.rsplit(" ", 1)
                    weight += float(value)
                    entries += 1
        _write_json_atomic(
            staging / "stack_summary.json",
            dict(
                entries=entries,
                self_cpu_time_us=weight,
                note="An empty export may reflect native stack-collection limitations.",
            ),
        )


def _package(staging, archive):
    metadata = json.loads((staging / "metadata.json").read_text())
    parse_trace(staging / "trace.json", staging, metadata)
    _parse_native_details(staging)
    inventory = {
        p.name: {"bytes": p.stat().st_size, "sha256": _file_sha256(p)}
        for p in staging.iterdir()
        if p.is_file() and p.name != "inventory.json"
    }
    _write_json_atomic(staging / "inventory.json", inventory)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as bundle:
        for name in (*inventory, "inventory.json"):
            bundle.add(staging / name, arcname=name, recursive=False)
    with tarfile.open(temporary, "r:gz") as bundle:
        for name, item in inventory.items():
            result = hashlib.sha256()
            size = 0
            with bundle.extractfile(name) as member:
                for chunk in iter(lambda: member.read(1024 * 1024), b""):
                    result.update(chunk)
                    size += len(chunk)
            if size != item["bytes"] or result.hexdigest() != item["sha256"]:
                raise ValueError(f"archive verification failed: {name}")
    temporary.replace(archive)
    checksum = _file_sha256(archive)
    cleanup_error = None
    try:
        shutil.rmtree(staging)
    except OSError as exc:
        cleanup_error = str(exc)
    return checksum, cleanup_error


def export_profile(profiler, staging, config, device):
    """Export native data on the profiler owner thread before handing files to the worker."""
    profiler.export_chrome_trace(str(staging / "trace.json"))
    wait = getattr(profiler, "wait_for_exports", None)
    if wait is not None:
        wait()
    # Snapshot native events while this cycle still owns them. Native
    # post-processing is part of export; custom aggregation runs later
    # on the background thread, without a reference to the profiler.
    with (staging / "events.jsonl").open("w") as destination:
        for event in profiler.events():
            snapshot = dict(
                name=event.name,
                device_type=str(event.device_type),
                thread=event.thread,
                input_shapes=event.input_shapes if config.record_shapes else None,
                stack=event.stack if config.with_stack else None,
                cpu_time_us=event.cpu_time_total,
                self_cpu_time_us=event.self_cpu_time_total,
                device_time_us=event.device_time_total,
                self_device_time_us=event.self_device_time_total,
                flops=event.flops if config.with_flops else None,
            )
            destination.write(json.dumps(snapshot) + "\n")
    if config.with_stack:
        profiler.export_stacks(str(staging / "stacks.txt"))
    if config.profile_memory and config.record_shapes and config.with_stack:
        profiler.export_memory_timeline(str(staging / "memory.json"), device=device)


class ProfileArtifactWorker:
    """Process exported files without retaining the session or native profiler."""

    def __init__(self, directory, metadata, *, managed_manifest=True):
        self.directory = Path(directory)
        self.root = self.directory.parent
        self.metadata = dict(metadata)
        self.rank = metadata["rank"]
        self.capture_time = metadata["capture_time"]
        self.session_id = metadata["session_id"]
        self.managed_manifest = managed_manifest
        self.jobs = {}
        self.errors = []
        self.lock = threading.Lock()
        self.queue = queue.Queue(maxsize=16)
        self.done = threading.Event()
        self.completed = threading.Event()
        self.recording = False
        self.thread = threading.Thread(
            target=self._process, name=f"profile-rank-{self.rank}", daemon=False
        )

    def start(self, started_at_ns):
        with self.lock:
            self.recording = True
            self.metadata["started_at_ns"] = started_at_ns
        self.thread.start()
        self._write_status_safely()

    def submit(self, cycle, staging, archive, *, incomplete_window):
        with self.lock:
            self.jobs[cycle] = dict(
                state="queued",
                staging=str(staging),
                archive=str(archive),
                incomplete_window=incomplete_window,
            )
        try:
            self.queue.put_nowait(cycle)
        except queue.Full:
            pass  # Scan queued descriptors after consuming the bounded queue.

    def fail_export(self, cycle, staging, error):
        with self.lock:
            self.jobs[cycle] = dict(state="failed", staging=str(staging), error=str(error))

    def record_error(self, error):
        with self.lock:
            self.errors.append(str(error))

    def finish(self, stopped_at_ns):
        """Mark collection complete; queued files continue processing in the background."""
        with self.lock:
            self.recording = False
            self.metadata["stopped_at_ns"] = stopped_at_ns
            self.done.set()
        self._write_status_safely()

    def wait(self, timeout=None):
        if not self.done.is_set():
            raise RuntimeError("stop profiling before waiting for artifacts")
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError("profile artifact processing is still running")
        return self.status()

    def _process(self):
        while True:
            try:
                cycle = self.queue.get(timeout=0.1)
            except queue.Empty:
                with self.lock:
                    cycle = next((i for i, j in self.jobs.items() if j["state"] == "queued"), None)
                    finished = self.done.is_set()
                if cycle is None:
                    if finished:
                        break
                    continue
            with self.lock:
                job = self.jobs[cycle]
                if job["state"] != "queued":
                    continue
                job["state"] = "processing"
                staging, archive = Path(job["staging"]), Path(job["archive"])
            try:
                checksum, cleanup_error = _package(staging, archive)
                update = dict(state="ready", sha256=checksum, cleanup_error=cleanup_error)
            except Exception as exc:
                update = dict(state="failed", error=str(exc))
                logger.exception("profile processing failed: %s", staging)
            with self.lock:
                job.update(update)
            self._write_status_safely()
        self._write_status_safely()
        self.completed.set()

    @property
    def artifacts_complete(self) -> bool:
        return self.completed.is_set()

    def status(self):
        with self.lock:
            return dict(
                self.metadata,
                recording=self.recording,
                jobs={str(k): dict(v) for k, v in self.jobs.items()},
                errors=list(self.errors),
                artifacts_complete=self.artifacts_complete,
                empty=self.done.is_set() and not self.jobs,
                success=self.done.is_set()
                and bool(self.jobs)
                and not self.errors
                and all(
                    j["state"] == "ready" and not j.get("cleanup_error") for j in self.jobs.values()
                ),
            )

    def _write_status(self):
        # Serialize writers so their atomic temporary files cannot collide.
        with self.lock:
            status = dict(
                self.metadata,
                recording=self.recording,
                jobs={str(k): dict(v) for k, v in self.jobs.items()},
                errors=list(self.errors),
            )
            _write_json_atomic(self.directory / "status.json", status)
            if self.managed_manifest:
                world = int(os.environ.get("WORLD_SIZE", 1))
                statuses = {}
                for rank in range(world):
                    path = self.root / f"rank-{rank:05d}-{self.capture_time}" / "status.json"
                    statuses[str(rank)] = (
                        json.loads(path.read_text()) if path.exists() else {"state": "missing"}
                    )
                # Only rank zero owns the common manifest; all ranks retain status.json.
                if self.rank == 0:
                    _write_json_atomic(
                        self.root / f"manifest-{self.capture_time}.json",
                        dict(
                            session_id=self.session_id,
                            capture_time=self.capture_time,
                            ranks=statuses,
                        ),
                    )

    def _write_status_safely(self):
        try:
            self._write_status()
        except Exception as exc:
            logger.exception("cannot write profile status")
            with self.lock:
                self.errors.append(str(exc))
