"""Run a reproducible same-host Ascend 2P2D PD smoke and write JSON metrics."""

import argparse
import json
import os
import time
from pathlib import Path

from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.engine import PDEngine

PROMPTS = {
    "r0": [3, 7, 2, 11, 5, 17, 8],
    "r1": [5, 9, 11, 13, 15, 19],
    "r2": [2, 4, 6, 8, 10],
    "r3": [1, 12, 14, 16, 18, 20, 22, 24],
}


def drain(engine, expected, timeout):
    outputs = {}
    deadline = time.monotonic() + timeout
    while len(outputs) < expected:
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output
        if time.monotonic() >= deadline:
            if os.getenv("VLLM_RLT_PD_DEBUG"):
                print(
                    "PD debug transfers="
                    + repr(
                        {
                            tid: {
                                "phase": transfer.phase,
                                "cancelled": transfer.cancelled,
                                "p": transfer.p,
                                "d": transfer.d,
                                "timings": transfer.timings,
                            }
                            for tid, transfer in engine.transfers.items()
                        }
                    ),
                    flush=True,
                )
                print(
                    "PD debug peers="
                    + repr(
                        {
                            name: {
                                "role": peer.role,
                                "slots": peer.slots,
                                "reserved_blocks": peer.reserved_blocks,
                                "ready": peer.client.ready,
                                "connected": peer.client.connected,
                                "failure": peer.client.failure,
                                "alive": peer.client.process.is_alive(),
                            }
                            for name, peer in engine.peers.items()
                        }
                    ),
                    flush=True,
                )
            raise TimeoutError(f"received {len(outputs)} of {expected} outputs")
    return outputs


def run(args):
    params = SamplingParams(
        max_tokens=args.max_tokens,
        min_loops=4,
        max_loops=4,
        temperature=0,
        ignore_eos=True,
    )
    cache = CacheConfig(num_blocks=args.num_blocks, block_size=args.block_size)
    engine = None
    result = None
    started = time.monotonic()
    try:
        engine = PDEngine(
            args.model,
            pd_config=PDConfig(
                device_type="npu",
                prefill_devices=tuple(args.prefill_devices),
                decode_devices=tuple(args.decode_devices),
                backend="HCCL",
                transfer_chunk_bytes=args.transfer_chunk_bytes,
                max_inflight_bytes=args.max_inflight_bytes,
                startup_timeout=args.startup_timeout,
                request_timeout=args.request_timeout,
                shutdown_timeout=args.shutdown_timeout,
            ),
            prefill_cache_config=cache,
            decode_cache_config=cache,
            prefill_scheduler_config=SchedulerConfig(
                max_num_seqs=2,
                max_num_batched_tokens=args.prefill_tokens,
                prefill_chunk_size=args.prefill_tokens,
            ),
            decode_scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=16),
            execution_config=ExecutionConfig(async_scheduling=False),
            exit_config=ExitConfig("ouro"),
            attention_backend="ascend_eager",
            dtype="bfloat16",
        )
        for request_id, prompt in PROMPTS.items():
            engine.add_request(request_id, prompt, params)
        outputs = drain(engine, len(PROMPTS), args.request_timeout)
        result = {
            "model": args.model,
            "topology": {
                "prefill_devices": args.prefill_devices,
                "decode_devices": args.decode_devices,
            },
            "elapsed_seconds": time.monotonic() - started,
            "outputs": {
                request_id: {
                    "token_ids": output.token_ids,
                    "exit_depths": output.exit_depths,
                    "finish_reason": output.finish_reason,
                }
                for request_id, output in outputs.items()
            },
            "request_metrics": list(engine.metrics),
        }
        return result
    finally:
        if engine is not None:
            engine.close()
            if result is not None:
                result["worker_metrics"] = engine.worker_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefill-devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--decode-devices", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--transfer-chunk-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-inflight-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--shutdown-timeout", type=float, default=90)
    parser.add_argument("--request-timeout", type=float, default=900)
    args = parser.parse_args()
    result = run(args)
    encoded = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
