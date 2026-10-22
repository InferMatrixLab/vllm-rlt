"""Transfer protocol ordering and resource ownership without CUDA/NIXL."""

from unittest.mock import Mock

import pytest
import torch

from tests.helpers import RecordingConnector, make_stage_engine
from vllm_rlt import SamplingParams
from vllm_rlt.core.execution import WorkKey
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.connector import PDStageConnector
from vllm_rlt.request import RequestStatus


def make_connector(engine, slots=4):
    transport = Mock(inflight_bytes=0, pending=[])
    transport.poll.return_value = ([], [])
    transport.busy.return_value = False
    connector = PDStageConnector(
        engine,
        torch.zeros(slots, engine.model.config.hidden_size),
        transport,
        PDConfig(),
        notify=Mock(),
    )
    engine.connector = connector
    return connector


def reserve(connector, tid, *, receiving=True):
    connector.reserve(tid, [1, 2], SamplingParams(max_tokens=2), receiving=receiving, peer="peer")
    return connector.transfers[tid]


@pytest.mark.parametrize("commit_first", [False, True])
def test_import_requires_commit_all_sequences_and_capacity(commit_first):
    engine = make_stage_engine(role="decode")
    connector = make_connector(engine)
    transfers = [reserve(connector, f"r{i}") for i in range(4)]
    if commit_first:
        for transfer in transfers:
            connector.commit(transfer.tid, 2)
        connector.progress()
        assert not any(w.active for w in transfers)
    connector.transport.poll.return_value = ([], [("peer", w.tid, 0) for w in transfers])
    connector.progress()
    assert not any(w.active for w in transfers)
    connector.transport.poll.return_value = ([], [("peer", w.tid, 1) for w in transfers])
    connector.progress()
    if not commit_first:
        assert not any(w.active for w in transfers)
        for transfer in transfers:
            connector.commit(transfer.tid, 2)
        connector.progress()
    assert all(w.ready and not w.active for w in transfers)
    engine.scheduler._admit()
    connector.progress()
    assert sum(w.active for w in transfers) == 3
    blocked = next(w for w in transfers if not w.active)
    active = next(w for w in transfers if w.active)
    assert blocked.request.status == RequestStatus.RECEIVING
    assert engine.cache_manager.has_transfer_lease(blocked.tid)
    assert engine.scheduler.requests[active.tid] is active.request
    connector.cancel(active.tid)
    engine.scheduler._admit()
    connector.progress()
    assert blocked.active
    assert not engine.cache_manager.has_transfer_lease(blocked.tid)
    engine.close()


def test_cancelled_export_holds_slot_and_kv_until_local_and_remote_access_end():
    engine = make_stage_engine()
    connector = make_connector(engine, slots=1)
    transfer = reserve(connector, "source", receiving=False)
    event = transfer.last_event = Mock()
    event.query.return_value = False
    connector.cancel(transfer.tid)
    assert transfer.request.status == RequestStatus.TRANSFERRING
    assert transfer.tid not in engine.scheduler.waiting
    assert all(transfer.tid not in q for q in engine.scheduler.work_queues.values())
    connector.progress()
    assert not connector.free_slots
    assert engine.cache_manager.has_transfer_lease(transfer.tid)
    event.query.return_value = True
    connector.transport.busy.return_value = True
    connector.progress()
    assert transfer.tid in connector.transfers
    connector.transport.busy.return_value = False
    connector.progress()
    assert connector.free_slots == [0]
    assert not connector.transfers
    assert engine.cache_manager.num_used_blocks == 0
    event.synchronize.assert_not_called()
    engine.close()


def test_cancelled_import_waits_for_safe_notification_and_old_generation_is_ignored():
    engine = make_stage_engine(role="decode")
    connector = make_connector(engine, slots=1)
    old = reserve(connector, "generation-1")
    connector.cancel(old.tid)
    connector.commit(old.tid, 1)
    connector.transport.poll.return_value = ([], [("peer", old.tid, 0)])
    connector.progress()
    assert not connector.free_slots
    assert engine.cache_manager.has_transfer_lease(old.tid)
    connector.cancel_safe(old.tid)
    current = reserve(connector, "generation-2")
    connector.progress()  # Old completion cannot activate the reused staging slot.
    connector.commit(old.tid, 1)
    assert current.request.status == RequestStatus.RECEIVING
    assert current.received == set()
    assert engine.cache_manager.has_transfer_lease(current.tid)
    connector.cancel(current.tid)
    connector.cancel_safe(current.tid)
    engine.close()


def test_compute_release_checks_event_and_identity_without_releasing_kv():
    engine = make_stage_engine(role="prefill")
    engine.connector = RecordingConnector()
    request = engine.reserve_transfer("r", [1, 2], SamplingParams(max_tokens=1), receiving=False)
    while request.status != RequestStatus.TRANSFERRING:
        engine.step()
    event = Mock()
    event.query.return_value = False
    assert not engine.release_transfer_compute(request, ready_event=event)
    assert request.hidden_state is not None
    event.query.return_value = True
    assert engine.release_transfer_compute(request, ready_event=event)
    assert request.hidden_state is None
    assert engine.cache_manager.has_transfer_lease("r")
    engine.release_transfer(request)
    newer = engine.reserve_transfer("r", [2], SamplingParams(max_tokens=1), receiving=False)
    assert not engine.release_transfer_compute(request, ready_event=event)
    assert engine.cache_manager.has_transfer_lease(newer.request_id)
    event.synchronize.assert_not_called()
    engine.release_transfer(newer)
    engine.close()


def test_mixed_connector_handles_both_transfer_directions_by_request():
    engine = make_stage_engine(mixed=True)
    connector = make_connector(engine)
    outgoing = reserve(connector, "out", receiving=False)
    incoming = reserve(connector, "in")
    assert connector.exports(outgoing.request)
    assert not connector.exports(incoming.request)
    connector.commit(incoming.tid, 1)
    connector.transport.poll.return_value = ([], [("peer", incoming.tid, 0)])
    connector.progress()
    assert incoming.ready and not incoming.active
    engine.scheduler._admit()
    connector.progress()
    assert incoming.active
    assert engine.scheduler.pending_work(outgoing.request) == WorkKey.PROMPT_PRELUDE
    connector.cancel(outgoing.tid)
    connector.cancel(incoming.tid)
    connector.progress()
    assert not connector.transfers
    assert engine.cache_manager.num_used_blocks == 0
    engine.close()


def test_commit_validation_and_late_write_rejection():
    engine = make_stage_engine(role="decode")
    connector = make_connector(engine)
    transfer = reserve(connector, "r")
    with pytest.raises(RuntimeError, match="invalid transfer commit"):
        connector.commit("r", True)
    connector.commit("r", 1)
    with pytest.raises(RuntimeError, match="conflicting"):
        connector.commit("r", 2)
    connector.transport.poll.return_value = ([], [("peer", "r", 0)])
    connector.progress()
    assert transfer.ready and not transfer.active
    engine.scheduler._admit()
    connector.progress()
    assert transfer.active
    connector.progress()  # A repeated notification is harmless.
    connector.transport.poll.return_value = ([], [("peer", "r", 1)])
    with pytest.raises(RuntimeError, match="late KV write"):
        connector.progress()
    engine.close()


def test_ack_alone_cannot_release_unsubmitted_or_locally_pending_export():
    engine = make_stage_engine()
    connector = make_connector(engine, slots=1)
    transfer = reserve(connector, "out", receiving=False)
    engine.scheduler._admit()
    connector.acknowledge(transfer.tid)
    connector.progress()
    assert transfer.tid in connector.transfers
    engine.suspend_transfer(transfer.request)
    transfer.compute_done = True
    transfer.next_sequence = 1
    event = transfer.last_event = Mock()
    event.query.return_value = False
    transfer.chunks.append((event, 0, [], [], 1))
    connector.progress()
    assert not transfer.committed
    assert engine.scheduler.has_admission_slot(transfer.request)
    assert engine.cache_manager.has_transfer_lease(transfer.tid)
    event.query.return_value = True
    connector.transport.submit.return_value = True
    connector.transport.busy.return_value = True
    connector.progress()
    assert transfer.committed and not engine.scheduler.has_admission_slot(transfer.request)
    assert not connector.free_slots
    connector.transport.busy.return_value = False
    connector.progress()
    assert connector.free_slots == [0]
    assert engine.cache_manager.num_used_blocks == 0
    engine.close()


def test_connector_controls_protect_pending_remote_access():
    engine = make_stage_engine(role="decode")
    connector = make_connector(engine)
    transfer = reserve(connector, "remote")
    assert connector.has_pending
    with pytest.raises(ValueError, match="cancellation protocol"):
        connector.command(dict(kind="abort", request_id="remote"))
    with pytest.raises(RuntimeError, match="pending transfers"):
        connector.close()
    connector.transport.close.assert_not_called()
    assert connector.transfers["remote"] is transfer
    assert not connector.command(dict(kind="add", request_id="local"))
    connector.output_finished("local")
    assert connector.command(dict(kind="cancel", tid="remote"))
    assert connector.command(dict(kind="cancel_safe", tid="remote"))
    connector.progress()
    assert not connector.has_pending
    connector.close()
    connector.transport.close.assert_called_once()
    engine.close()


@pytest.mark.parametrize(
    "local_priority,remote_priority,winner", [(0, 5, "local"), (5, 0, "remote")]
)
def test_import_and_local_prompt_share_priority_admission(local_priority, remote_priority, winner):
    from dataclasses import replace

    engine = make_stage_engine(mixed=True)
    engine.scheduler.config = replace(engine.scheduler.config, policy="priority", max_num_seqs=1)
    connector = make_connector(engine)
    engine.add_request("local", [1, 2], SamplingParams(max_tokens=2, priority=local_priority))
    connector.reserve(
        "remote",
        [1, 2],
        SamplingParams(max_tokens=2, priority=remote_priority),
        receiving=True,
        peer="peer",
    )
    connector.commit("remote", 1)
    connector.transport.poll.return_value = ([], [("peer", "remote", 0)])
    connector.progress()
    assert engine.scheduler.requests["local"].status == RequestStatus.WAITING
    assert engine.scheduler.requests["remote"].status == RequestStatus.RECEIVING
    engine.scheduler._admit()
    assert engine.scheduler.requests[winner].status == RequestStatus.ACTIVE
    loser = "remote" if winner == "local" else "local"
    assert engine.scheduler.requests[loser].status in (
        RequestStatus.WAITING,
        RequestStatus.RECEIVING,
    )
    connector.cancel("remote")
    engine.close()


def test_export_holds_compute_admission_until_final_staging_read_finishes():
    from dataclasses import replace

    engine = make_stage_engine(mixed=True)
    engine.scheduler.config = replace(engine.scheduler.config, max_num_seqs=1)
    connector = make_connector(engine)
    transfer = reserve(connector, "export", receiving=False)
    engine.scheduler._admit()
    engine.suspend_transfer(transfer.request)
    engine.add_request("local", [2], SamplingParams(max_tokens=1))
    event = Mock()
    event.query.return_value = False
    assert not engine.release_transfer_compute(transfer.request, ready_event=event)
    engine.scheduler._admit()
    assert engine.scheduler.requests["local"].status == RequestStatus.WAITING
    event.query.return_value = True
    assert engine.release_transfer_compute(transfer.request, ready_event=event)
    engine.scheduler._admit()
    assert (
        engine.scheduler.pending_work(engine.scheduler.requests["local"])
        == WorkKey.PROMPT_PRELUDE
    )
    assert engine.cache_manager.has_transfer_lease("export")
    connector.cancel("export")
    connector.progress()
    engine.close()


@pytest.mark.parametrize(
    "local_priority,remote_priority,winner", [(0, 5, "local"), (5, 0, "remote")]
)
def test_export_reservation_obeys_shared_compute_admission(local_priority, remote_priority, winner):
    from dataclasses import replace

    engine = make_stage_engine(mixed=True)
    engine.scheduler.config = replace(engine.scheduler.config, policy="priority", max_num_seqs=1)
    connector = make_connector(engine)
    engine.add_request("local", [1, 2], SamplingParams(max_tokens=2, priority=local_priority))
    connector.reserve(
        "remote",
        [1, 2],
        SamplingParams(max_tokens=1, priority=remote_priority),
        receiving=False,
        peer="peer",
    )
    remote = engine.scheduler.requests["remote"]
    assert remote.status == RequestStatus.WAITING
    assert engine.cache_manager.has_transfer_lease("remote")
    engine.scheduler._admit()
    assert (
        engine.scheduler.pending_work(engine.scheduler.requests[winner])
        == WorkKey.PROMPT_PRELUDE
    )
    loser = "remote" if winner == "local" else "local"
    assert engine.scheduler.requests[loser].status == RequestStatus.WAITING
    # Reserving another export must not bypass an occupied compute slot either.
    reserve(connector, "queued", receiving=False)
    engine.scheduler._admit()
    assert engine.scheduler.requests["queued"].status == RequestStatus.WAITING
    connector.cancel("queued")
    connector.cancel("remote")
    engine.close()
