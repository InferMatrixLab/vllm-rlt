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

    Sliding-window packing requires host-planned metadata and ordinary
    synchronous prefill in a single stage.
    """
    requested = execution.packed_prefill
    if requested is False:
        return False
    if sliding_window and (
        execution.async_scheduling
        or execution.cuda_graphs
        or staged_prefill
        or speculative
        or preemption
        or stage_role != "prefill_decode"
    ):
        if requested is True:
            raise ValueError(
                "sliding-window packed prefill requires synchronous eager single-stage execution "
                "with ordinary prefill, without speculation or preemption"
            )
        return False
    if device_type != "cuda" or layout != "last_exited":
        if requested is True:
            raise ValueError("packed_prefill requires CUDA packed prefill with last_exited KV")
        return False
    return requested is True or not staged_prefill
