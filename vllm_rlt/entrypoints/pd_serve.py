"""Serve Ouro using independent prefill/decode GPU worker pools and NIXL."""

import argparse
import logging
from dataclasses import asdict, replace
from functools import partial

from vllm_rlt.config import SchedulerConfig
from vllm_rlt.entrypoints.runtime_args import (
    add_runtime_args,
    profile_config_from_args,
    runtime_configs,
)
from vllm_rlt.models import ModelConfig, load_tokenizer, resolve_model_config, resolve_model_source
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.engine import PDEngine


def load_engine(args):
    from tokenizers.decoders import ByteLevel

    profiling = profile_config_from_args(args)
    source, revision, config = resolve_model_config(args.model, revision=args.revision)
    tokenizer = load_tokenizer(
        args.tokenizer or resolve_model_source(args.model),
        revision=(
            args.tokenizer_revision
            if args.tokenizer_revision is not None
            else (args.revision if args.tokenizer else revision)
        ),
    )
    if config["model_type"].lower() == "ouro":
        decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
        if not isinstance(decoder, ByteLevel):
            raise ValueError("serving requires the Ouro byte-level tokenizer")
    options = runtime_configs(args)
    cache = options.pop("cache_config")
    engine = PDEngine(
        source,
        model_config=ModelConfig.from_dict(config),
        revision=revision,
        dtype=args.dtype,
        pd_config=PDConfig(
            prefill_devices=tuple(args.prefill_devices),
            decode_devices=tuple(args.decode_devices),
            transfer_chunk_bytes=args.pd_transfer_chunk_bytes,
            max_inflight_bytes=args.pd_max_inflight_bytes,
            max_transfer_descriptors=args.pd_max_transfer_descriptors,
            max_pending_requests=args.max_requests,
            request_timeout=args.request_timeout,
            startup_timeout=args.pd_startup_timeout,
            shutdown_timeout=args.shutdown_timeout,
            backend=args.nixl_backend,
            max_receiving_requests=args.pd_max_receiving_requests,
            max_draining_requests=args.pd_max_draining_requests,
        ),
        prefill_cache_config=replace(cache, num_blocks=args.prefill_num_blocks),
        decode_cache_config=replace(cache, num_blocks=args.decode_num_blocks),
        prefill_scheduler_config=SchedulerConfig(
            policy=getattr(args, "scheduling_policy", "fcfs"),
            enable_preemption=getattr(args, "enable_preemption", False),
            max_num_seqs=args.prefill_max_num_seqs,
            max_num_batched_tokens=args.prefill_max_num_batched_tokens,
            prefill_chunk_size=args.prefill_chunk_size,
            wavefront_chunked_prefill=getattr(args, "wavefront_chunked_prefill", False),
        ),
        decode_scheduler_config=SchedulerConfig(
            policy=getattr(args, "scheduling_policy", "fcfs"),
            enable_preemption=getattr(args, "enable_preemption", False),
            max_num_seqs=args.decode_max_num_seqs,
            max_num_batched_tokens=args.decode_max_num_batched_tokens,
            mode=args.mode,
            min_coda_batch_size=args.min_coda_batch_size,
        ),
        attention_backend=args.attention_backend,
        **options,
    )
    try:
        if profiling is not None:
            engine.start_profile(profiling, scheduled=True)
    except BaseException:
        try:
            engine.close()
        except BaseException:
            logging.getLogger(__name__).exception("PD engine cleanup after initialization failed")
        raise
    return engine, tokenizer


def main():
    from aiohttp import web

    from vllm_rlt.serving.protocol import ServingLimits
    from vllm_rlt.serving.server import create_app

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", help="Local checkpoint path or HuggingFace repository ID", required=True
    )
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--served-model-name")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--attention-backend",
        choices=["triton", "flash_attn", "flash_attn_4"],
        default="flash_attn_4",
    )
    parser.add_argument("--prefill-devices", type=int, nargs="+", default=[0])
    parser.add_argument("--decode-devices", type=int, nargs="+", default=[1])
    parser.add_argument("--prefill-num-blocks", type=int)
    parser.add_argument("--decode-num-blocks", type=int)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prefill-max-num-seqs", type=int, default=8)
    parser.add_argument("--decode-max-num-seqs", type=int, default=128)
    parser.add_argument("--prefill-max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--decode-max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--mode", choices=["refill", "no_refill"], default="refill")
    parser.add_argument("--pd-max-receiving-requests", type=int, default=32)
    parser.add_argument("--pd-max-draining-requests", type=int, default=8)
    parser.add_argument("--nixl-backend", default="UCX")
    parser.add_argument(
        "--pd-transfer-chunk-bytes", type=int, default=PDConfig.transfer_chunk_bytes
    )
    parser.add_argument("--pd-max-inflight-bytes", type=int, default=PDConfig.max_inflight_bytes)
    parser.add_argument("--pd-max-transfer-descriptors", type=int, default=256)
    parser.add_argument("--pd-startup-timeout", type=float, default=300)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    for name, default in asdict(ServingLimits()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    add_runtime_args(parser)
    parser.add_argument("--no-async-scheduling", action="store_false", dest="async_scheduling")
    parser.set_defaults(
        num_blocks=None, exit_mode="ouro_delayed", prefill_chunk_size=2048, async_scheduling=True
    )

    args = parser.parse_args()
    limits = ServingLimits(**{name: getattr(args, name) for name in asdict(ServingLimits())})
    logging.basicConfig(level=logging.INFO)
    web.run_app(
        create_app(
            partial(load_engine, args),
            model=args.served_model_name or args.model,
            limits=limits,
            allowed_hosts=("localhost", args.host),
        ),
        host=args.host,
        port=args.port,
        handler_cancellation=True,
        shutdown_timeout=limits.shutdown_timeout,
    )


if __name__ == "__main__":
    main()
