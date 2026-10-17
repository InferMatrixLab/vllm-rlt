"""Typed control-plane protocol between the PD coordinator and workers.

The Pipe wire format stays a plain dict of picklable values; these enums type
the send/receive boundaries without changing the serialized messages. Members
compare equal to their string values, so mixed string/enum comparisons keep
working during incremental adoption.
"""

from enum import Enum


class CommandKind(str, Enum):
    """Coordinator → worker commands."""

    CONNECT = "connect"
    RESERVE = "reserve"
    PREFILL = "prefill"
    CANCEL = "cancel"
    CANCEL_SAFE = "cancel_safe"
    COMMIT = "commit"
    ACK = "ack"
    STOP = "stop"
    PROFILE_START = "profile_start"
    PROFILE_STOP = "profile_stop"
    PROFILE_STATUS = "profile_status"


class EventKind(str, Enum):
    """Worker → coordinator events."""

    READY = "ready"
    CONNECTED = "connected"
    RESERVED = "reserved"
    PREFILL_STARTED = "prefill_started"
    PREFILL_COMPLETE = "prefill_complete"
    COMMIT = "commit"
    ACTIVATED = "activated"
    OUTPUT = "output"
    RELEASED = "released"
    REJECTED = "rejected"
    FATAL = "fatal"
    PROFILE_REPLY = "profile_reply"
    STOPPED = "stopped"


class ProfileAction(str, Enum):
    """Session-scoped profiling actions; never transfer-scoped (no tid)."""

    START = "start"
    STOP = "stop"
    STATUS = "status"


PROFILE_COMMANDS = {
    CommandKind.PROFILE_START: ProfileAction.START,
    CommandKind.PROFILE_STOP: ProfileAction.STOP,
    CommandKind.PROFILE_STATUS: ProfileAction.STATUS,
}
PROFILE_ACTION_COMMANDS = {action: kind for kind, action in PROFILE_COMMANDS.items()}


class PendingPhase(str, Enum):
    """Coordinator-side transfer lifecycle phases."""

    WAITING = "waiting"
    RESERVING = "reserving"
    PREFILL = "prefill"
    DECODE = "decode"
