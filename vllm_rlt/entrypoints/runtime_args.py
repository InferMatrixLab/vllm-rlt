"""Shared offline/server flags for cache, scheduling and execution policies."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from vllm_rlt.config import CacheConfig, ExecutionConfig, ExitConfig, SpeculativeConfig
from vllm_rlt.profiling import ProfileConfig


def add_runtime_args(parser):
    add_profile_args(parser)
    parser.add_argument(
        "--speculative-tokens",
        type=int,
        help="Enable fixed-loop self-speculation with K draft tokens",
    )
    parser.add_argument("--draft-loops", type=int, default=2)
    parser.add_argument("--target-loops", type=int, default=4)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--incremental-kv", action="store_true")
    parser.add_argument("--kv-watermark", type=float, default=0.0)
    parser.add_argument("--scheduling-policy", choices=["fcfs", "priority"], default="fcfs")
    parser.add_argument("--enable-preemption", action="store_true")
    parser.add_argument(
        "--prefill-uva",
        action="store_true",
        help="Use packed prefill metadata staging (requires packed attention)",
    )
    parser.add_argument(
        "--no-uva",
        dest="use_uva",
        action="store_false",
        help="Copy metadata to the GPU instead of reading mapped host memory",
    )
    parser.add_argument(
        "--no-atomic-prefill",
        dest="atomic_prefill",
        action="store_false",
        help="Split prompt prefill into a staged prelude plus one boundary per recurrence depth",
    )
    parser.add_argument("--kv-layout", choices=["last_exited", "shared"], default="last_exited")
    parser.add_argument(
        "--kv-cache-dtype",
        choices=["auto", "fp8_e4m3"],
        default="auto",
        help="fp8_e4m3 stores KV as FP8 with per-vector scales (Torch/Triton attention)",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--memory-reserve-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument(
        "--exit-mode",
        choices=["ouro", "ouro_delayed", "random_lookahead", "trace"],
        default="ouro",
        help="ouro_delayed reuses the Ouro gate; random_lookahead uses an untrained head",
    )
    parser.add_argument("--exit-trace", help="JSON file with depths_by_request")
    parser.add_argument("--lookahead-seed", type=int, default=0)
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Capture decode recurrent cores and speculative coda; prefill/sampling stay eager",
    )
    parser.add_argument("--cuda-graph-max-batch-size", type=int, default=128)
    parser.add_argument("--cuda-graph-max-graphs", type=int, default=16)
    parser.add_argument("--cuda-graph-memory-reserve-bytes", type=int, default=1024**3)
    parser.add_argument(
        "--single-stream",
        action="store_true",
        help="Disable multi-stream CUDA execution independently of async scheduling",
    )
    parser.add_argument("--static-buffers", action="store_true")
    parser.add_argument("--pad-to-power-of-two", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--max-prefill-batches-before-decode", type=int, default=1)
    parser.add_argument("--admission-scan-limit", type=int, default=64)
    parser.add_argument("--max-admission-bypasses", type=int, default=8)
    parser.add_argument("--min-coda-batch-size", type=int, default=1)


def runtime_configs(args):
    # load_engine is also called directly with older argparse namespaces.
    parser = argparse.ArgumentParser(add_help=False)
    add_runtime_args(parser)
    args = SimpleNamespace(**(vars(parser.parse_args([])) | vars(args)))
    return dict(
        speculative_config=SpeculativeConfig(
            num_speculative_tokens=args.speculative_tokens,
            draft_loops=args.draft_loops,
            target_loops=args.target_loops,
        )
        if args.speculative_tokens is not None
        else None,
        cache_config=CacheConfig(
            enable_prefix_caching=args.enable_prefix_caching,
            incremental_allocation=args.incremental_kv,
            watermark_ratio=args.kv_watermark,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            layout=args.kv_layout,
            kv_cache_dtype=args.kv_cache_dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            memory_reserve_bytes=args.memory_reserve_bytes,
        ),
        exit_config=ExitConfig(
            mode=args.exit_mode,
            seed=args.lookahead_seed,
            depths_by_request=json.loads(Path(args.exit_trace).read_text())["depths_by_request"]
            if args.exit_trace
            else None,
        ),
        execution_config=ExecutionConfig(
            prefill_uva=args.prefill_uva,
            use_uva=args.use_uva,
            atomic_prefill=args.atomic_prefill,
            cuda_graphs=args.cuda_graphs,
            cuda_graph_max_batch_size=args.cuda_graph_max_batch_size,
            cuda_graph_max_graphs=args.cuda_graph_max_graphs,
            cuda_graph_memory_reserve_bytes=args.cuda_graph_memory_reserve_bytes,
            async_scheduling=args.async_scheduling,
            multi_stream=not args.single_stream,
            static_buffers=args.static_buffers,
            pad_to_power_of_two=args.pad_to_power_of_two,
        ),
    )


def add_profile_args(parser):
    group = parser.add_argument_group("PyTorch profiling")
    group.add_argument("--profile", action="store_true")
    group.add_argument("--profile-dir")
    group.add_argument(
        "--profile-activities",
        default=ProfileConfig.activities,
        help="comma-separated cpu,cuda",
    )
    for flag, field in (
        ("record-shapes", "record_shapes"),
        ("with-stack", "with_stack"),
        ("memory", "profile_memory"),
        ("with-flops", "with_flops"),
    ):
        group.add_argument(
            "--profile-" + flag, action="store_true", default=getattr(ProfileConfig, field)
        )
    for name in ("wait", "warmup", "active", "repeat"):
        group.add_argument("--profile-" + name, type=int, default=getattr(ProfileConfig, name))


def profile_config_from_args(args) -> ProfileConfig | None:
    """Build startup profiling settings, filling missing CLI fields from parser defaults."""
    parser = argparse.ArgumentParser(add_help=False)
    add_profile_args(parser)
    args = SimpleNamespace(**(vars(parser.parse_args([])) | vars(args)))
    if not args.profile:
        return None

    return ProfileConfig(
        output_dir=args.profile_dir,
        activities=tuple(args.profile_activities.split(",")) if args.profile_activities else None,
        record_shapes=args.profile_record_shapes,
        with_stack=args.profile_with_stack,
        profile_memory=args.profile_memory,
        with_flops=args.profile_with_flops,
        wait=args.profile_wait,
        warmup=args.profile_warmup,
        active=args.profile_active,
        repeat=args.profile_repeat,
    )
