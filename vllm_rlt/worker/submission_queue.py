"""Bound CPU submission lookahead before any reusable input is overwritten."""

from collections import deque

MAX_CONCURRENT_SUBMISSIONS = 2


class SubmissionQueue:
    """The execution owner, not individual tensor pools, retires GPU consumers.

    begin() must run before preparing the next batch. complete() receives an event
    covering every input consumer, even for batches that produce no host result.
    slot selects safe submission-owned scratch after begin() returns. Independent
    tensor pools with at most one publication per submission also remain safe
    when their depth is at least max_pending.
    """

    def __init__(self, max_pending=MAX_CONCURRENT_SUBMISSIONS):
        if max_pending < 1:
            raise ValueError("submission capacity must be positive")
        self.max_pending = max_pending
        self.events = deque()
        self.slot = 0
        self._next_slot = 0

    def begin(self):
        while self.events and self.events[0].query():
            self.events.popleft()
        waited = len(self.events) >= self.max_pending
        if waited:
            self.events[0].synchronize()
            self.events.popleft()
        # All submission-owned scratch uses this slot, including submissions
        # that do not need a particular input layout. Failed submissions drain
        # readers before the next begin, so gaps cannot cause unsafe reuse.
        self.slot = self._next_slot
        self._next_slot = (self.slot + 1) % self.max_pending
        return waited

    def complete(self, event):
        if event is not None:
            self.events.append(event)

    def drain(self):
        """Error recovery: retire every reader before arbitrary pool rotation resumes."""
        while self.events:
            self.events[0].synchronize()
            self.events.popleft()
