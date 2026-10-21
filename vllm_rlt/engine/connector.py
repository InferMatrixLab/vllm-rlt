"""Stage communication contract; no ownership of loop or sampling progress."""

from typing import Protocol

from vllm_rlt.core.scheduler import PrefillProgress
from vllm_rlt.request import Request, RequestPhase


class StageConnector(Protocol):
    def exports(self, request: Request) -> bool:
        """Whether this request has an explicit downstream prompt destination."""
        ...

    def can_schedule(self, request: Request) -> bool:
        """Nonblocking backpressure/readiness; must not mutate scheduler queues."""
        ...

    def prefill_submitted(
        self,
        request: Request,
        *,
        phase: RequestPhase,
        start: int | None,
        progress: PrefillProgress,
        event,
        stream,
    ) -> None:
        """Borrow submitted state under the transfer lease until delivery/abort.

        Logical progress is already applied. An event covers pending device
        writes; implementations must not interpret submission as completion.
        """
        ...


class TransferConnector(StageConnector, Protocol):
    """Control-plane lifecycle, driven by the deployment host, never by a loop.

    Transfer IDs identify generations. Implementations own transfer bookkeeping
    and staging, and ask StageEngine to mutate Request/compute/KV state. Transport
    completion, import activation and source retirement are distinct transitions.
    """

    def startup_info(self) -> dict:
        """Transport-specific handshake fields for the deployment coordinator."""
        ...

    @property
    def has_pending(self) -> bool:
        """Whether transfers or remote access still prevent owner shutdown."""
        ...

    def command(self, command: dict) -> bool:
        """Handle a protocol command; return False for generic worker commands."""
        ...

    def output_finished(self, request_id: str) -> None:
        """Release connector resources for a request retired by the engine."""
        ...

    def close(self) -> dict:
        """Close drained transport resources and return final transport metrics."""
        ...

    def progress(self) -> None:
        """Poll device/transport readiness and advance transfers without blocking."""
        ...

    def cancel(self, transfer_id: str) -> None:
        """Stop future submissions; retain resources still accessible by peers."""
        ...

    def cancel_safe(self, transfer_id: str) -> None:
        """Retire a cancelled import after the host proves remote writers stopped."""
        ...

    def commit(self, transfer_id: str, count: int) -> None:
        """Record submission count; activation also requires all completions."""
        ...

    def acknowledge(self, transfer_id: str) -> None:
        """Record downstream activation; local completion still gates retirement."""
        ...
