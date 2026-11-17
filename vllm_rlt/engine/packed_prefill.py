"""Validate packed prefill requests and decide whether automatic packing is eligible."""


def packed_prefill_eligible(
    execution,
    *,
    device_type,
    layout,
    staged_prefill,
    sliding_window=False,
    stage_role="prefill_decode",
    speculative=False,
    preemption=False,
):
    """Validate execution constraints before filtering by backend capability.

    Sliding-window packing is opt-in. Its current implementation requires host-planned
    metadata and ordinary synchronous prefill in a single stage.
    """
    requested = execution.packed_prefill
    if requested:
        if sliding_window and (
            execution.async_scheduling
            or execution.cuda_graphs
            or staged_prefill
            or speculative
            or preemption
            or stage_role != "prefill_decode"
        ):
            raise ValueError(
                "sliding-window packed prefill requires synchronous eager single-stage execution "
                "with ordinary prefill, without speculation or preemption"
            )
        if device_type != "cuda" or layout != "last_exited":
            raise ValueError("packed_prefill requires CUDA packed prefill with last_exited KV")
    if requested is not None:
        return requested
    return (
        device_type == "cuda"
        and layout == "last_exited"
        and not staged_prefill
        and not sliding_window
    )
