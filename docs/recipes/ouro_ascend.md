# Ouro-1.4B on Ascend 910B3: initial synchronous support

This path reuses the native Ouro model and shared engine on one Ascend NPU.
Select **--attention-backend torch explicitly**. Weights, ordinary activations
and KV storage remain BF16; the existing Torch attention computes QK, softmax
and PV intermediates in FP32 through Torch-NPU. This is not a fused CANN
attention integration and makes no acceleration claim.

## Supported scope

- Ouro, fixed four-loop greedy inference, synchronous local execution,
  LAST_EXITED depth-aware paged KV, prefill and incremental decode.
- Explicit num_blocks or kv_cache_memory_bytes; automatic NPU memory sizing
  is not implemented.
- Cancellation, request ID reuse, EOS/length handling, HTTP completions and SSE
  reuse the existing engine and serving lifecycle.
- Async scheduling, CUDA graphs, speculation, preemption, prefix caching, PD,
  adaptive exit and stochastic sampling are rejected for this initial path.
- Qualification targets Ouro-1.4B on 910B3. Other checkpoints and Ascend devices
  require separate validation.

The shared scheduler, model and KV owner are retained. NPU submissions drain
before metadata buffers or request pages can be reused. CPU/CUDA execution
does not import the optional Torch-NPU package. Built-in profiling supports
CPU/CUDA; use torch_npu.profiler for NPU device traces.

## Environment and installation

The development environment is Linux aarch64, Python 3.11.13, Torch 2.9.0+cpu,
torch-npu 2.9.0, CANN 9.0.0, driver 26.1.0.b050 and Transformers 4.57.6.
Keep Torch and Torch-NPU matched; activate vendor CANN before the project Python.

~~~bash
source /path/to/cann/set_env.sh
source /path/to/isolated-npu-environment/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=4  # replace with your assigned physical NPU
python -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print(torch.npu.get_device_name(0))'
python -m pip install 'numpy>=1.24' 'safetensors>=0.4.3' 'huggingface-hub>=0.25' \
  'transformers>=4.54' 'aiohttp>=3.11,<4' setuptools wheel
python -m pip install --no-deps --no-build-isolation -e .
~~~

The no-deps install preserves the vendor Torch pair and avoids the project's
Linux Triton dependency. Neither Triton nor a native CUDA extension is required
for this explicit Torch backend. No InferRT or MindIE-SD integration is involved.

Use the complete ByteDance/Ouro-1.4B checkpoint at revision
574fa66cb8bf5abdc979642d01cf2b79b16bfab1. The model.safetensors SHA256 is
58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af.

## Start the service

~~~bash
python -m vllm_rlt.entrypoints.serve \
  --model /path/to/Ouro-1.4B --served-model-name ouro \
  --device npu --dtype bfloat16 --attention-backend torch \
  --num-blocks 64 --block-size 16 --max-num-seqs 2 \
  --max-num-batched-tokens 32 --port 8123
~~~

~~~bash
curl http://127.0.0.1:8123/health
curl http://127.0.0.1:8123/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ouro","prompt":"The capital of France is","max_tokens":4,"temperature":0,"min_loops":4,"max_loops":4,"exit_threshold":1}'
~~~

Add stream=true for SSE. Python callers use device="npu" and
attention_backend="torch"; the spawned worker registers Torch-NPU itself.
The assigned physical device becomes logical npu:0. Increase KV capacity for
longer prompts, output limits or concurrency. This small service command is a
smoke configuration, not a performance benchmark.

## Reproduce validation

~~~bash
python -m pip install pytest ruff pre-commit
pre-commit run --all-files
export VLLM_RLT_OURO_CHECKPOINT=/path/to/Ouro-1.4B
export VLLM_RLT_ASCEND_RESULTS=/path/to/validation-results
python -m pytest tests/test_ascend.py tests/test_ascend_checkpoint.py \
  tests/test_ascend_serving.py --run-npu -q
~~~

The NPU flag requires assigned ASCEND_RT_VISIBLE_DEVICES; CPU collection skips
device tests. Checks cover a CPU attention oracle, GQA and causal page
boundaries, incremental/full KV allocation, four-depth state ownership,
chunking, cancellation/reuse, EOS/length behavior, pinned checkpoint generation
and a real spawned HTTP/SSE worker.

Metadata-only allocation changes require exact hidden/logit/KV equality.
The BF16 diagnostic bounds remain relative L2 at most 2% and elementwise
atol=0.02, rtol=0.02. These are this validation's bounds, not a universal quality
guarantee. Task quality needs separately pinned paired evaluation against the
official code; short generation does not establish broader quality.
Performance characterization should retain input lengths, concurrency,
TTFT/TPOT/throughput, memory and repetitions; the existing
[serving benchmark guide](../serving.md#reuse-the-upstream-benchmark) describes
the compatible upstream client.

## Fused attention remains separate

An experimental BF16 paged npu_fused_infer_attention_score adapter is outside
this change. It failed frozen whole-model numerical checks and lost a correct
answer in a ten-question smoke. Promoting attention intermediates in
npu_fusion_attention improved local mathematical agreement but still failed
the original whole-model joint gate. Those failures have not been relaxed or
hidden; this recipe does not qualify that fused backend.

This contribution covers a bounded part of
[#69 Ascend support](https://github.com/InferMatrixLab/vllm-rlt/issues/69).
It does not complete CANN fused adaptation or other Ouro runtime features on
Ascend. Measured qualification results are recorded separately from this scope.
