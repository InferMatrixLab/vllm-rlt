# Ouro-1.4B on Ascend

This experimental path runs native Ouro on one NPU with BF16, synchronous
greedy decoding and four recurrent loops. Select
`--attention-backend ascend_eager` explicitly. It batches causal queries by
request/depth through Torch-NPU/CANN MatMul and Softmax primitives and uses
native ScatterND for paged KV writes. It preserves released Ouro HF eager
rounding: BF16 QK/scale, FP32 softmax rounded to BF16, then BF16 PV.

The older explicit `torch` backend remains available with its original
FP32 attention intermediates and indexed KV writes. Its arithmetic is a
different reference contract. Automatic backend selection is unchanged.

## Supported scope

- Native Ouro-1.4B, BF16 parameters/ordinary activations/KV, fixed four loops,
  synchronous single-device greedy generation.
- Per-request/depth/layer paged KV; incremental/full allocation, causal prefill,
  decode, cancellation and request-ID reuse.
- Existing spawned HTTP/SSE serving lifecycle. Explicit KV capacity is required.
- Async scheduling, graphs, speculation, PD, preemption, prefix caching,
  adaptive depth and stochastic sampling are rejected for this initial path.
- Other checkpoints, Ascend models/devices and concurrent task quality need
  separate qualification. The quality regression below uses a full prompt
  forward; chunked generation can round differently.

The shared scheduler/model/KV owner are retained. NPU submissions drain before
metadata or pages can be reused. CPU/CUDA do not import Torch-NPU unconditionally.
Use `torch_npu.profiler` for NPU device profiling; built-in profiling targets
CPU/CUDA.

Hardware selection does not inspect or whitelist device model names.
The backend requires BF16 and the Torch-NPU/CANN operations used below.
The measurements in this recipe were collected on 910B3; other Ascend devices
can run the same opt-in checks without a model-name assertion. Their numerical
and performance results require validation on the actual device/runtime pair.

## Environment and installation

Measured environment: Linux aarch64, Python 3.11.13, Torch 2.9.0+cpu,
Torch-NPU 2.9.0, CANN 9.0.0, driver 26.1.0.b050, Transformers 4.57.6,
Tokenizers 0.22.2 and NumPy 1.26.4. Keep the vendor Torch pair matched.

~~~bash
source /path/to/cann/set_env.sh
source /path/to/isolated-npu-environment/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=4  # use your assigned physical NPU
python -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__); print(torch.npu.get_device_name(0))'
python -m pip install 'numpy==1.26.4' 'safetensors==0.8.0' 'huggingface-hub==0.36.2' \
  'transformers==4.57.6' 'tokenizers==0.22.2' 'aiohttp==3.14.3' setuptools wheel
python -m pip install --no-deps --no-build-isolation -e .
~~~

The no-deps install preserves vendor Torch and avoids the Linux Triton
dependency. No native CUDA extension, custom compiled kernel or InferRT is used.
Current upstream ScatterPA examples use an API/layout unavailable in this
Torch-NPU/CANN pair; the native writer retains the existing ND cache.

Use ByteDance/Ouro-1.4B revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1.
Weights SHA256:
`58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.

## Start and call the service

~~~bash
python -m vllm_rlt.entrypoints.serve \
  --model /path/to/Ouro-1.4B --served-model-name ouro \
  --device npu --dtype bfloat16 --attention-backend ascend_eager \
  --num-blocks 1024 --block-size 16 --max-num-seqs 1 \
  --max-num-batched-tokens 4096 --prefill-chunk-size 4096 --port 8123
~~~

~~~bash
curl http://127.0.0.1:8123/health
curl http://127.0.0.1:8123/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ouro","prompt":"The capital of France is","max_tokens":4,"temperature":0,"min_loops":4,"max_loops":4,"exit_threshold":1}'
~~~

Add `stream=true` for SSE. Python callers select `device="npu"` and
`attention_backend="ascend_eager"`; the spawned worker registers Torch-NPU.
The assigned physical NPU becomes logical `npu:0`. Capacity and token budgets
must fit your prompts/outputs/concurrency; this command is a reproduction
configuration, not certification of every 4096-token input.

Both `prefill_chunk_size` and `max_num_batched_tokens` must accommodate the
whole prompt to compare with official full-prefill. Raising only the latter
retains the scheduler's default 128-token chunks.

## Validation (2026-11-03)

Base 989921ac1d3897c5c43e27ec8cddbc91fa953194 rejects `--device npu`.
Runtime 45220081159627ff4fba36a6b0b3ac10787214c8 supplies the new backend.
Final runtime be037ed5cb9344b59830da2449b5590f3dd6c231 adds an arithmetic
regression and makes native writes exclusive to the new backend; selected
eager execution and original Torch execution were separately revalidated.
Later documentation changes preserve these runtime/test contents.

~~~bash
pre-commit run --all-files --verbose
export VLLM_RLT_OURO_CHECKPOINT=/path/to/Ouro-1.4B
export VLLM_RLT_ASCEND_RESULTS=/path/to/results
python -m pytest tests/test_ascend.py tests/test_ascend_checkpoint.py \
  tests/test_ascend_serving.py tests/test_ascend_kv.py \
  tests/test_ascend_eager.py --run-npu -q
~~~

- Ruff/pre-commit passed; CPU 926 passed / 602 skipped. Device cases were run
  separately:22 passed, comprising18 NPU and 4 CPU cases.
- Final eager vs unmodified official HF:216 selected states and2016 layer
  events exactly equal; restored Torch vs original Torch is also exact.
- The arithmetic regression distinguishes an analytical output16 in BF16
  eager from15.9375 in the original FP32-intermediate path, on CPU andNPU.
- Real spawned health/completions/SSE, termination/usage, disconnect recovery
  and shutdown pass for both explicit backends.
- At4522008,87 selected same-device HF/native pairs score63/87 each,
  losses[]/gains[], with87/87 complete token sequences exactly equal.
  Parse failures19 each; length/execution limits0 each. Every pair shares a
  physical910B3, pooled across cards4/5/6. These are reused regression cases,
  not a held-out/full-GSM8K estimate. No threshold or scorer was relaxed.
- Quality uses one full-prefill chunk4096 and4096 token budget. Chunked128,
  concurrent/adaptive task quality and other models remain unqualified.

Numerical bounds remain joint relative L2<=2% and atol=rtol=0.02.
The new explicit backend is checked against released HF BF16 eager.
It still fails 209/216 states against the old FP32-intermediate Torch reference;
that failure remains recorded and is not called passed.

The earlier63/87 HF vs61/87 native observation combined different arithmetic
with native default 128-token chunking. Changing either alone did not fix
question 203. Matching both restores203/881/526 and the full87 regression.
Manual question 203 replay has15 exact logits and192 exact populated KV hashes.

## Measured performance

Same physical910B3, profiler disabled, P64/C1 orC8, fixed4 loops, ignore EOS,
exactly 8 outputs/request,768 blocks,max 8 sequences / 256 batched tokens,explicit
128-token chunks. One warmup/shape,3repetitions/run; A/C/C/A gives 6 observations
per arm/shape. TTFT/TPOT are medians of per-burst request means, not pooled
individual-request percentiles. Throughput includes prefill/drain; no HTTP.

|Concurrency|Old Torch tok/s|New backend tok/s|Old/new TTFT ms|Old/new TPOT ms|
|---:|---:|---:|---:|---:|
|1|1.326|4.711|4266.35 /221.03|246.81 /210.99|
|8|1.720|12.778|24363.00 /702.72|1818.05 /583.61|

Pair-specific throughput ratios are 3.47–3.81x at C1 and 7.11–8.05x at C8,
with non-overlapping observed ranges. This combined backend/arithmetic change
does not isolate ScatterND. A KV-only A/B regressed 1.23% and 6.28%;
the final runtime therefore keeps original writes on the Torch path.
New backend native-writing optimality is not established.

Final runtime be037ed confirmation on the sameNPU gives medians 4.752tok/s
at C1 and13.705tok/s at C8 (three reps each, measured after ABBA).
Allocated peaks are about 4.97–4.99GiB; reserved peaks about 5.33GiB.
All bursts finish with 0 live logical KV blocks and no failed repetitions.

A separate 9-shape candidate characterization covers 16/64/256 input tokens and
concurrency 1/2/8. These are short bursts, not saturation/HTTP/SLO or hardware-only
claims. CPU/NUMA placement is not pinned; independent CPU/device checks overlap
part of the shared-host timing. Raw ranges and controls must accompany reuse.

Separate ops-only/full warmed decode traces confirm actual native ScatterND
dispatch and 96 softmax calls. Full mode has readable shapes, stacks and memory
exports. Complete warmed request traces additionally cover prefill and all 8
outputs. Profile durations are diagnostic and excluded from speed claims.

## Fused operators remain experimental

BF16 FIA and promoted FP32 FlashAttentionScore did not pass earlier whole-model
gates. Native RMSNorm changes also failed selected states despite local speed
gains; SwiGLU/RotaryMul candidates were not adopted.
The current backend uses native primitives and batching, not fused attention.

This is a bounded contribution toward
[#69](https://github.com/InferMatrixLab/vllm-rlt/issues/69), with task quality
qualified only for the stated regression configuration.
