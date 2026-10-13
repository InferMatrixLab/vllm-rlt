"""Launch one resident Ouro model behind an OpenAI completions endpoint."""

import argparse
import logging
from dataclasses import asdict
from functools import partial

import torch

from vllm_rlt.config import SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.entrypoints.runtime_args import (
    add_runtime_args,
    profile_config_from_args,
    runtime_configs,
)
from vllm_rlt.models import (
    AutoModelForCausalLM,
    load_tokenizer,
    resolve_model_config,
    resolve_model_source,
)


def load_engine(args):
    from tokenizers.decoders import ByteLevel

    profiling = profile_config_from_args(args)
    if profiling is not None:
        profiling.resolve_activities(args.device)
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
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=revision,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )
    engine = LLMEngine(
        model,
        **runtime_configs(args),
        scheduler_config=SchedulerConfig(
            policy=getattr(args, "scheduling_policy", "fcfs"),
            enable_preemption=getattr(args, "enable_preemption", False),
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            mode=args.mode,
            prefill_chunk_size=getattr(args, "prefill_chunk_size", 128),
            max_prefill_batches_before_decode=getattr(args, "max_prefill_batches_before_decode", 1),
            admission_scan_limit=getattr(args, "admission_scan_limit", 64),
            max_admission_bypasses=getattr(args, "max_admission_bypasses", 8),
            min_coda_batch_size=getattr(args, "min_coda_batch_size", 1),
        ),
        attention_backend=args.attention_backend,
    )
    if profiling is not None:
        engine.start_profile(profiling, scheduled=True)
    return engine, tokenizer


def main():
    from aiohttp import web

    from vllm_rlt.serving.protocol import ServingLimits
    from vllm_rlt.serving.server import create_app

    parser = argparse.ArgumentParser(description="Serve one Ouro model with OpenAI completions")
    parser.add_argument(
        "--model", help="Local checkpoint path or HuggingFace repository ID", required=True
    )
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--served-model-name")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--attention-backend",
        type=str.lower,
        choices=[
            "auto",
            "torch",
            "triton",
            "flashinfer",
            "flash_attn",
            "flash_attn_2",
            "flash_attn_3",
            "flash_attn_4",
        ],
        default="triton",
    )
    parser.add_argument("--mode", choices=["refill", "no_refill"], default="refill")
    parser.add_argument("--num-blocks", type=int, help="Override automatic KV sizing")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    defaults = asdict(ServingLimits())
    for name, default in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    add_runtime_args(parser)
    args = parser.parse_args()
    limits = ServingLimits(**{name: getattr(args, name) for name in defaults})
    if args.device == "cpu" and args.attention_backend not in ("auto", "torch"):
        parser.error("CPU execution requires --attention-backend torch or auto")
    torch.set_num_threads(args.cpu_threads)
    logging.basicConfig(level=logging.INFO)
    app = create_app(
        partial(load_engine, args),
        model=args.served_model_name or args.model,
        limits=limits,
        allowed_hosts=("localhost", args.host),
    )
    web.run_app(
        app,
        host=args.host,
        port=args.port,
        handler_cancellation=True,
        shutdown_timeout=limits.shutdown_timeout,
    )


if __name__ == "__main__":
    main()
