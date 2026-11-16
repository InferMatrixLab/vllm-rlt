"""Validate packed prefill requests and decide whether automatic packing is eligible."""

from vllm_rlt.platform import is_rocm


def packed_prefill_eligible(
    packed_prefill, *, device_type, layout, staged_prefill, sliding_window=False
):
    """Validate explicit requests and return whether packing may be enabled.

    Staged prefill and sliding-window attention retain their existing metadata.
    Backend support is checked separately by attention selection. ROCm stays
    opt-in: its BF16 packed attention can change greedy tokens (docs/rocm.md).
    """
    if packed_prefill and sliding_window:
        raise ValueError("packed_prefill does not support sliding-window attention")
    if packed_prefill and (device_type != "cuda" or layout != "last_exited"):
        raise ValueError("packed_prefill requires CUDA packed prefill with last_exited KV")
    if packed_prefill is not None:
        return bool(packed_prefill)
    return (
        device_type == "cuda"
        and not is_rocm()
        and layout == "last_exited"
        and not staged_prefill
        and not sliding_window
    )
