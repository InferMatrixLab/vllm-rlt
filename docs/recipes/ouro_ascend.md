# Ouro-1.4B on Ascend 910B3: initial synchronous support

This path reuses the native Ouro model and shared engine on one Ascend NPU.
Select **--attention-backend torch explicitly**. Weights, ordinary activations
and KV storage remain BF16; the existing Torch attention computes QK, softmax
and PV intermediates in FP32 through Torch-NPU. This is not a fused CANN
attention integration and makes no acceleration claim.

**Quality status:** the selected same-device paired screen is HF 63/87 versus
native 61/87 (losses: 203 and 881; gains: none). The frozen no-net-loss gate
failed. This implementation remains experimental and is not ready for
submission as a quality-qualified backend.

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
python -m pip install 'numpy==1.26.4' 'safetensors==0.8.0' 'huggingface-hub==0.36.2' \
  'transformers==4.57.6' 'tokenizers==0.22.2' 'aiohttp==3.14.3' setuptools wheel
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
Ascend. Measured validation outcomes and remaining blockers are recorded below.

## Measured validation (2026-11-03)

Runtime source: 4734e1963bdf7fd4e0fd66b2612511d6d4173550, against base
989921ac1d3897c5c43e27ec8cddbc91fa953194. Later recipe updates do not change
the measured runtime sources.

- Ruff lint/format and the CPU regression hook passed.
- Opt-in real 910B3 tests: 10 passed in 44.41s, including actual HTTP/SSE.
- Incremental/preallocated real-weight KV differential: 216 finite states, all
  exactly equal; attention/chunking diagnostics retain their original joint bounds.
- Same-device paired selected GSM8K-87: official 63/87, native 61/87;
  losses [203, 881], gains []; 63/87 exact token sequences.
  Parse failures: {'hf_eager': 19, 'rlt_target': 19}; length limits: {'hf_eager': 0, 'rlt_target': 0};
  execution limits: {'hf_eager': 0, 'rlt_target': 0}.
- Each pair runs on the same physical 910B3; all 87 cases pool three cards.
  The original ten are regression smoke; 77 cases were held out. This is
  selected-case evidence, not full GSM8K or adaptive/concurrent quality.

The quality run uses official unmodified HF eager, Transformers 4.57.6,
Tokenizers 0.22.2, lm-eval 0.4.9.2 strict scoring, three-shot raw CoT,
greedy four-loop BF16, max 1024 output tokens and source-ranked seed0 IDs.
Dataset revision is 740312add88f781978c0658806c59bc2815b9866.

The table below is a separate **engine-only short-burst** characterization:
8 output tokens/request, ignore EOS, one shape-specific warmup and three
unprofiled repetitions. Values are medians of per-burst request means; output
throughput includes prefill and drain. TTFT starts before engine admission.
TPOT is first-to-last observed output time divided by seven.

| Input | Concurrency | Output tok/s (range) | TTFT ms | TPOT ms | Peak allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|
| 16 | 1 | 2.726 (2.700–2.901) | 1238.01 | 241.94 | 4.970 / 5.309 |
| 16 | 2 | 3.793 (3.657–3.810) | 2122.58 | 298.19 | 4.970 / 5.311 |
| 16 | 8 | 5.031 (5.016–5.086) | 7980.90 | 675.01 | 4.973 / 5.314 |
| 64 | 1 | 1.367 (1.342–1.372) | 4176.89 | 242.77 | 4.972 / 5.314 |
| 64 | 2 | 1.590 (1.569–1.599) | 7964.07 | 300.45 | 4.974 / 5.314 |
| 64 | 8 | 1.789 (1.786–1.793) | 23417.86 | 1756.95 | 4.984 / 5.314 |
| 256 | 1 | 0.370 (0.364–0.373) | 19472.72 | 307.45 | 4.978 / 5.318 |
| 256 | 2 | 0.372 (0.371–0.377) | 39959.74 | 431.61 | 4.988 / 5.318 |
| 256 | 8 | 0.379 (0.378–0.380) | 125114.88 | 6228.64 | 4.988 / 5.318 |

Configuration: 768 blocks, block size 16, max 8 sequences / 256 batched tokens,
incremental allocation, no async/graphs/UVA/prefix caching/speculation.
Measured requests: 99, output tokens: 792, failures: 0. Live logical KV blocks
after each completed burst: 0.
Torch allocator peaks include model/cache/workspace and exclude some driver
or runtime memory. Other cards ran independent quality shards; CPU/NUMA
affinity was not pinned. No steady-state throughput, HTTP/p99, accelerator
hardware-only comparison or acceleration claim is made.

Separate ops-only/full profiles capture a complete warmed second-token decode
iteration, with actual NPU device kernels. Full mode enables shapes, stacks
and memory; exported shapes, readable Python stacks and memory CSV rows
were checked separately from performance timing.
Full prefill plus all eight output tokens is outside these diagnostic captures.

The conservative Torch path has substantial prefill/dispatch overhead. Fused
attention optimization remains a later step and must retain the quality gates.

The no-net-loss quality criterion failed on this selected set. Case 203
changes a quantity from 7 to 5 and produces 27 instead of 31. Case 881 writes
'The answer is 10 + 6 = 16'; the unchanged strict scorer extracts 10. It remains
a failure under the frozen protocol. Neither the scorer nor the gate was
relaxed. A diagnostic HF-style BF16 attention alternative also failed the
original numerical bounds and was not incorporated.
