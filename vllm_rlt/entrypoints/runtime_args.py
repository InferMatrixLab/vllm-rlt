"""Map explicit RLT launch flags to the vLLM engine configuration."""

from argparse import ArgumentParser, BooleanOptionalAction, Namespace

from vllm_rlt.config import LoopRuntimeConfig

_BOOL_OPTIONS = {
    "trace-steps": "Log loop steps and Ouro's committed depths.",
    "wavefront-prefill": "Schedule Inkling prompt chunks at different CORE depths.",
    "packed-prefill": "Pack mixed Inkling prefill CORE rows into one ragged batch.",
    "decode-pipeline": "Start the next Inkling token before draft validation.",
    "mixed-core": "Share Inkling CORE projections and MoE across pass depths.",
    "compact-mixed-core": "Test shared Inkling CORE with compact MRv2 attention groups.",
    "self-speculation": "Enable Inkling's single-candidate greedy draft.",
    "merge-decode-core": "Run terminal and continuing Inkling decode rows together at each depth.",
    "merge-prefill-core": "Share each CORE with terminal rows and rows that continue past it.",
    "decode-state-slots": "Keep Inkling decode state in GPU buffers indexed by MRv2 request slot.",
    "batch-decode-coda": "Allow one ready terminal CORE before sampling a decode batch.",
    "finish-decode-token": "Finish ready CORE work before starting the next decode token.",
    "silu-and-mul": "Use the compiled RLT SiLU-and-Mul kernel in Ouro.",
}


def add_runtime_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("RLT")
    for option, help_text in _BOOL_OPTIONS.items():
        group.add_argument(f"--rlt-{option}", action=BooleanOptionalAction, help=help_text)
    group.add_argument(
        "--rlt-prefill-chunk-size", type=int, help="Maximum prompt tokens in a wavefront chunk."
    )


def apply_runtime_args(args: Namespace) -> None:
    options = dict(args.additional_config or {})
    for flag in (*_BOOL_OPTIONS, "prefill-chunk-size"):
        key = "rlt_" + flag.replace("-", "_")
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    LoopRuntimeConfig.from_additional_config(options)
    args.additional_config = options
    if args.scheduler_cls is None:
        args.scheduler_cls = (
            "vllm_rlt.core.async_scheduler.LoopAsyncScheduler"
            if args.async_scheduling
            else "vllm_rlt.core.loop_scheduler.LoopScheduler"
        )
    if args.worker_cls == "auto":
        args.worker_cls = "vllm_rlt.worker.loop_worker.LoopWorker"
