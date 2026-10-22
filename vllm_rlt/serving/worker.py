"""Async request channels driving one synchronous engine owner thread."""

import asyncio
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from uuid import uuid4

from vllm_rlt.serving.protocol import IncrementalText, ServingError, ServingLimits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenEvent:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None


class RequestChannel:
    def __init__(self, spec, buffer_size):
        self.request_id = "cmpl-" + uuid4().hex
        self.spec = spec
        self.buffer_size = buffer_size
        self.events = deque()
        self.changed = asyncio.Event()
        self.error = None
        self.cancelled = False

    def fail(self, error):
        self.error = error
        self.events.clear()
        self.changed.set()

    async def receive(self):
        while True:
            if self.error is not None:
                raise self.error
            if self.events:
                return self.events.popleft()
            self.changed.clear()
            await self.changed.wait()


class EngineWorker:
    """Engine, tokenizer and KV are touched only by the executor's sole thread.

    Channels and admission state belong to the HTTP event loop. The driver
    awaits every tick, so no unbounded cross-thread callback queue is created.
    A channel occupies an admission slot until its HTTP handler releases it.
    """

    def __init__(self, factory, *, limits=ServingLimits()):
        self.factory = factory
        self.limits = limits
        self.channels = {}
        self.pending = deque()
        self.cancellations = set()
        self.wake = asyncio.Event()
        self.ready = False
        self.stopping = False
        self.failure = None
        self.task = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ouro-engine")
        # Owner-thread state:
        self.engine = None
        self.tokenizer = None
        self.decoders = {}

    def start(self):
        self.task = asyncio.create_task(self._run(), name="ouro-engine-driver")

    def submit(self, spec):
        if not self.ready or self.stopping:
            raise ServingError.not_ready()
        if len(self.channels) >= self.limits.max_requests:
            raise ServingError.overloaded()
        channel = RequestChannel(spec, self.limits.output_buffer)
        self.channels[channel.request_id] = channel
        self.pending.append(channel)
        self.wake.set()
        return channel

    def release(self, channel):
        # Keep the slot until the owner acknowledges cancellation, including
        # cancellation of an admission that is currently being tokenized.
        if channel.request_id in self.channels:
            channel.cancelled = True
            self.cancellations.add(channel.request_id)
            self.wake.set()

    async def close(self):
        self.stopping = True
        self.ready = False
        self.wake.set()
        try:
            if self.task is not None:
                await asyncio.wait_for(asyncio.shield(self.task), self.limits.shutdown_timeout)
        except asyncio.TimeoutError:
            logger.warning("shutdown deadline exceeded; a hung engine requires an outer supervisor")
        if self.task is None:
            self.executor.shutdown(wait=False, cancel_futures=True)

    async def _call(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    async def profile_control(self, action, config=None, *, scheduled=False, timeout=60):
        if not self.ready or self.stopping:
            raise ServingError.not_ready()
        if action == "start":
            return await self._call(lambda: self.engine.start_profile(config, scheduled=scheduled))
        if action == "stop":
            return await self._call(self.engine.stop_profile)
        if action == "status":
            return await self._call(self.engine.profile_status)
        if action != "wait":
            raise ValueError("unknown profile action")
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            status = await self._call(self.engine.profile_status)
            if status.get("recording"):
                raise RuntimeError("stop profiling before waiting for artifacts")
            ranks = status.get("ranks")
            complete = (
                all(r.get("artifacts_complete", False) for r in ranks.values())
                if ranks is not None
                else status.get("artifacts_complete", not status.get("jobs"))
            )
            if complete:
                return status
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("profile artifacts are still processing")
            await asyncio.sleep(0.05)

    def _load(self):
        self.engine, self.tokenizer = self.factory()

    def _tick(self, pending, cancellations):
        failures = []
        for request_id in cancellations:
            if self.engine.has_request(request_id):
                self.engine.abort_request(request_id)
            self.decoders.pop(request_id, None)
        for channel in pending:
            request_id = channel.request_id
            if request_id in cancellations:
                continue
            try:
                tokens = self.tokenizer.encode(channel.spec.prompt)
                options = {"trace_id": channel.spec.trace_id} if channel.spec.trace_id else {}
                self.engine.add_request(request_id, tokens, channel.spec.params, **options)
                self.decoders[request_id] = (IncrementalText(self.tokenizer), 0)
            except ValueError as exc:
                failures.append((request_id, ServingError.invalid_request(str(exc))))
        events = []
        closed = set()
        for output in self.engine.step():
            if output.request_id in closed:
                continue
            decoder, delivered = self.decoders[output.request_id]
            try:
                # A speculative round can commit several IDs. Decode every new
                # ID once; one event per round keeps backpressure independent of K.
                delta = "".join(
                    decoder.decode(token, output.finished and index == len(output.token_ids) - 1)
                    for index, token in enumerate(output.token_ids[delivered:], start=delivered)
                )
                self.decoders[output.request_id] = (decoder, len(output.token_ids))
            except ValueError as exc:
                if self.engine.has_request(output.request_id):
                    self.engine.abort_request(output.request_id)
                del self.decoders[output.request_id]
                closed.add(output.request_id)
                failures.append((output.request_id, ServingError.invalid_request(str(exc))))
                continue
            events.append(
                (
                    output.request_id,
                    TokenEvent(
                        delta,
                        len(output.prompt_token_ids),
                        len(output.token_ids),
                        output.finish_reason,
                    ),
                )
            )
            if output.finished:
                del self.decoders[output.request_id]
                closed.add(output.request_id)
        return failures, events, self.engine.has_unfinished_requests()

    def _unload(self):
        if self.engine is not None:
            for request_id in self.engine.request_ids():
                self.engine.abort_request(request_id)
            close = getattr(self.engine, "close", None)
            if close is not None:
                close()
            logger.info("engine cleanup: %s", self.engine.stats())
        self.decoders.clear()
        self.engine = None
        self.tokenizer = None

    async def _run(self):
        running = False
        try:
            await self._call(self._load)
            if not self.stopping:
                self.ready = True
                logger.info("model, tokenizer and engine ready")
            while not self.stopping:
                if not (running or self.pending or self.cancellations):
                    self.wake.clear()
                    await self.wake.wait()
                    continue
                pending = list(self.pending)
                self.pending.clear()
                cancellations = set(self.cancellations)
                self.cancellations.clear()
                failures, events, running = await self._call(self._tick, pending, cancellations)
                for request_id in cancellations:
                    channel = self.channels.get(request_id)
                    if channel is not None and channel.cancelled:
                        self.channels.pop(request_id)
                for request_id, error in failures:
                    channel = self.channels.get(request_id)
                    if channel is not None:
                        channel.fail(error)
                for request_id, event in events:
                    channel = self.channels.get(request_id)
                    if channel is None or channel.cancelled or channel.error is not None:
                        continue
                    if len(channel.events) >= channel.buffer_size:
                        channel.fail(
                            ServingError("client output buffer exhausted", 503, "slow_client")
                        )
                        self.cancellations.add(request_id)
                    else:
                        channel.events.append(event)
                        channel.changed.set()
        except Exception:
            self.failure = "engine initialization or execution failed"
            logger.exception(self.failure)
        finally:
            self.ready = False
            error = ServingError(
                self.failure or "server is shutting down", 503, "engine_unavailable"
            )
            for channel in self.channels.values():
                channel.fail(error)
            try:
                await self._call(self._unload)
            finally:
                self.channels.clear()
                self.pending.clear()
                self.cancellations.clear()
                # Keep the executor available for owner-thread cleanup after
                # a shutdown deadline, if the outstanding device call returns.
                self.executor.shutdown(wait=False, cancel_futures=True)
