"""Submission completion protects shared input storage before slot reuse."""

import gc
import weakref

import torch

from vllm_rlt.worker.model_runner import BatchInputBuffers
from vllm_rlt.worker.submission_queue import SubmissionQueue


class ConsumerEvent:
    def __init__(self, complete=False):
        self.complete = complete
        self.waits = 0
        self.before_wait = None

    def query(self):
        return self.complete

    def synchronize(self):
        if self.before_wait:
            self.before_wait()
        self.waits += 1
        self.complete = True


def test_submission_owner_retires_before_reset_releases_transfer_sources():
    buffers = [BatchInputBuffers(2, 4, torch.device("cpu"), torch.float32) for _ in range(2)]
    queue = SubmissionQueue(max_pending=2)
    assert not queue.begin()
    first = buffers[queue.slot]
    source = torch.ones(2, dtype=torch.int64)
    ref = weakref.ref(source)
    first.token_transfers.append(source, first.token_output[:2])
    del source
    pending = ConsumerEvent()
    queue.complete(pending)
    assert not queue.begin()
    assert buffers[queue.slot] is not first
    later = ConsumerEvent(True)  # Another stream may finish out of order.
    queue.complete(later)
    gc.collect()
    assert ref() is not None

    def before_wait():
        assert ref() is not None

    pending.before_wait = before_wait
    assert queue.begin()
    assert pending.waits == 1 and later.waits == 0
    assert buffers[queue.slot] is first
    first.reset()
    gc.collect()
    assert ref() is None


def test_failed_submission_gap_drains_before_slot_reuse():
    queue = SubmissionQueue()
    queue.begin()
    first = ConsumerEvent()
    queue.complete(first)
    queue.begin()  # Preparation fails after reserving the next slot.
    queue.drain()
    assert first.waits == 1 and not queue.events
    queue.begin()
    assert queue.slot == 0


def test_submission_slots_reuse_preallocated_prefill_storage():
    buffers = [BatchInputBuffers(2, 4, torch.device("cpu"), torch.float32) for _ in range(2)]
    queue = SubmissionQueue()
    queue.begin()
    first = buffers[queue.slot]
    first_ptr = first.segments.cpu.data_ptr()
    queue.begin()
    second = buffers[queue.slot]
    assert second.segments.cpu.data_ptr() != first_ptr
    queue.begin()
    assert buffers[queue.slot] is first
    first.reset()
    assert first.segments.cpu.data_ptr() == first_ptr
