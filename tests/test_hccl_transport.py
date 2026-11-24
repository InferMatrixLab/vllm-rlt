"""CPU-side HCCL transport lifecycle checks without an NPU allocation."""

from collections import deque
from types import SimpleNamespace

import pytest

from vllm_rlt.pd.transport import HcclTransport, Transfer


class PendingFuture:
    def done(self):
        return False


class PendingWork:
    def get_future(self):
        return PendingFuture()


def test_hccl_send_timeout_is_fail_closed():
    transport = HcclTransport.__new__(HcclTransport)
    transport.config = SimpleNamespace(request_timeout=0)
    transport.sends = deque([Transfer((PendingWork(), None, PendingFuture()), "tid", 0, 16, 0)])
    transport.receives = {}
    transport.transfer_stats = {}
    with pytest.raises(TimeoutError, match="HCCL send timed out"):
        transport.poll()
    assert transport.transfer_stats["tid"]["status"] == "timeout"


def test_hccl_cancel_drops_unposted_receive_plan():
    transport = HcclTransport.__new__(HcclTransport)
    transport.receive_plans = {
        ("tid", 0): {"transfer_id": "tid"},
    }
    transport.receives = {}
    transport.sends = deque()
    assert transport.busy("tid")
    transport.cancel("tid")
    assert not transport.busy("tid")


def test_hccl_rejects_duplicate_receive_generation():
    transport = HcclTransport.__new__(HcclTransport)
    transport.receive_plans = {("tid", 0): {"transfer_id": "tid"}}
    transport.receives = {}
    with pytest.raises(RuntimeError, match="duplicate HCCL receive generation"):
        transport.prepare_receive("tid", "peer", (), 0, 0, 1, True)
