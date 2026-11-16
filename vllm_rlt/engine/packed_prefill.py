"""Validate packed prefill requests and decide whether automatic packing is eligible."""


def packed_prefill_eligible(
    packed_prefill, *, device_type, layout, staged_prefill, sliding_window=False, loopcd=False
):
    """Validate explicit requests and return whether packing may be enabled.

    Staged prefill and sliding-window attention retain their existing metadata.
    An engine that reserves LoopCD keeps per-row prefill unless packing is
    requested explicitly; LoopCD admission then rejects guided requests.
    Backend support is checked separately by attention selection.
    """
    if packed_prefill and sliding_window:
        raise ValueError("packed_prefill does not support sliding-window attention")
    if packed_prefill and (device_type != "cuda" or layout != "last_exited"):
        raise ValueError("packed_prefill requires CUDA packed prefill with last_exited KV")
    if packed_prefill is not None:
        return bool(packed_prefill)
    return (
        device_type == "cuda"
        and layout == "last_exited"
        and not staged_prefill
        and not sliding_window
        and not loopcd
    )
