# Ouro on Ascend

Run Ouro-1.4B on one Ascend NPU with BF16, greedy decoding and four recurrent
loops. The Python runtime uses Torch's `scaled_dot_product_attention` through
the `ascend_eager` backend, matching HF's SDPA entry point. Torch-NPU selects
the device kernel according to the input shape, mask and hardware.
The example enables the optional asynchronous engine plan.

## Prepare the environment and model

Use `main`, which includes Ascend support added after `v0.2.0`. Install CANN and
the matching driver and firmware for your Ascend hardware first.

### Install PyTorch and Torch-NPU

The following commands use CANN 9.0.0 with PyTorch 2.10.0 and Torch-NPU 2.10.0.
For other versions, consult the
[Ascend compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md#torchnpu-version-mapping-table).
Activate your Python environment and load the CANN environment.

```bash
source /path/to/npu-environment/bin/activate
source /path/to/cann/set_env.sh
```

On x86-64, install PyTorch from the CPU wheel index.

```bash
python -m pip install 'torch==2.10.0+cpu' --index-url https://download.pytorch.org/whl/cpu
```

On Arm64 (`aarch64`), use the following PyTorch command.

```bash
python -m pip install 'torch==2.10.0'
```

On either architecture, install `torch-npu` and check that PyTorch can see the NPU.
The package is imported as `torch_npu` in Python.

```bash
python -m pip install 'torch-npu==2.10.0' pyyaml setuptools
python -c 'import torch, torch_npu; assert torch.npu.is_available(), "Ascend NPU unavailable"'
```

### Install vllm-rlt and set the checkpoint path

Install the project dependencies in that environment. Use the checkpoint's
config and byte-level tokenizer.

```bash
git clone --branch main https://github.com/InferMatrixLab/vllm-rlt.git
cd vllm-rlt
python -m pip install --no-deps --no-build-isolation -e .
export MODEL=/path/to/prepared/Ouro-1.4B
export ASCEND_RT_VISIBLE_DEVICES=0  # Replace 0 with your assigned NPU.
```

## Serve fixed-depth requests

This example sizes KV for 32 resident requests with 128 input and 128 output
tokens each. Set `--num-blocks` explicitly and adjust it for the expected
sequence lengths, resident request count and available device memory. For
whole-cohort prefill, the chunk size must fit each prompt and the batched token
budget must fit the sum of prompt lengths in the cohort.

```bash
TASK_QUEUE_ENABLE=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m vllm_rlt.entrypoints.serve \
  --model "$MODEL" --tokenizer "$MODEL" --served-model-name ouro \
  --device npu --dtype bfloat16 --attention-backend ascend_eager \
  --mode refill --exit-mode ouro_delayed \
  --async-scheduling --single-stream --no-uva \
  --block-size 16 --num-blocks 2064 --max-num-seqs 32 \
  --max-num-batched-tokens 8192 --prefill-chunk-size 1024 \
  --max-prefill-batches-before-decode 32 --host 127.0.0.1 --port 8123
```

For synchronous eager execution, omit `--async-scheduling`.

Wait for `/health` to return HTTP 200, then send a completion. Add
`"stream": true` for server-sent events.

```bash
curl --fail http://127.0.0.1:8123/health
curl --fail http://127.0.0.1:8123/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ouro","prompt":"The capital of France is","max_tokens":8,"temperature":0,"min_loops":4,"max_loops":4,"exit_threshold":1}'
```

<a id="6-fixed-depth-asynchronous-engine-plan"></a>
## Fixed-depth asynchronous engine plan

- **Scheduling.** Stream events track completion and protect buffer and request
  lifetimes. Decode PRELUDE and each recurrent loop are separate submissions,
  so the scheduler can select a new batch at each boundary.
- **Reuse and batching.** Ascend uses the shared packed-prefill metadata by
  default with `ascend_eager`. Use `--no-packed-prefill` to select the per-query
  layout. Greedy sampling and compatible prefill are batched. Full prefill
  uses causal SDPA.
  Cached chunks and ragged decode pass boolean masks to preserve each row's
  visible KV range.
- **Paged KV.** Each loop writes to the page pool through PyTorch indexed
  assignment using its own block table.
  The Ascend backend gathers the visible KV rows for Torch SDPA.

The example uses private, fully reserved `last_exited` KV. Keep graphs,
static buffers, padding, prefix caching, incremental KV allocation, preemption,
speculation and prefill/decode disaggregation disabled in this configuration.
Greedy parity remains unqualified for arbitrary chunked or mixed-length inputs
and adaptive exit.

<a id="8-profile-guided-high-batch-scheduling"></a>
## Performance and correctness

A higher prefill quota lets more requests reach decode together and can delay
decode for existing requests. The scheduler default is 1.

### Full unified W1-W7 validation

This historical run used the C++ host executor and explicit eager attention
on both sides of the comparison. For the SDPA configuration above, compare
against HF SDPA with the same weights, input IDs, output lengths and concurrency.
Check the actual prefill and decode kernels on both sides with a short profile.
Measure throughput, TTFT and TPOT with the profiler disabled. Separate
synchronous and asynchronous vllm-rlt runs to measure the effect of scheduling.

Runtime `7b4537f` used the same budget 8192, chunk 1024 and prefill quota 32 for
all seven [unified execution workloads](../unified_execution.md#protocol-and-comparison-rules).
The comparison ran Ouro-1.4B BF16 with four fixed loops and greedy decoding on
one 64 GiB Ascend 910B3 per run, with CANN 9, Torch/Torch-NPU 2.10 and
Transformers 4.54.1. The reference used official HF eager at revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1` with a DynamicCache adapter.

All 42 runs completed with 1,422 successful requests and zero runtime failures.
Each case used three unprofiled repetitions per arm on the same card and CPU
affinity. Each repetition admitted closed-loop traffic for 60 seconds and then
fully drained. Throughput includes drain and excludes startup, compilation and
warmup. Timing uses the in-process engine actor. W1–W5 ignore EOS. W6/W7 use
natural EOS with a 512-token cap and a fixed ShareGPT selection.

| Workload | Input / output / clients | Residents per arm | HF eager tok/s | Engine tok/s | vs HF | Complete output matches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| W1 | 128 / 128 / 1 | 1 | 6.334 [5.284–6.496] | 9.464 [9.403–10.394] | +49.42% | 15/15 |
| W2 | 128 / 128 / 32 | 32 | 201.314 [200.099–204.437] | 305.496 [293.376–316.662] | +51.75% | 480/480 |
| W3 | 4096 / 128 / 1 | 1 | 6.254 [6.206–6.268] | 7.569 [6.110–9.215] | +21.03% | 12/12 |
| W4 | 4096 / 128 / 32 | 4 | 21.211 [21.012–21.580] | 31.312 [31.135–31.405] | +47.62% | 120/132 |
| W5 | 1024 / 512 / 8 | 8 | 50.885 [50.234–51.362] | 79.791 [76.843–80.666] | +56.80% | 48/48 |
| W6 | ShareGPT / ≤512 / 8 | 8 | 26.343 [26.060–26.385] | 37.992 [37.906–38.879] | +44.22% | 18/27 |
| W7 | ShareGPT / ≤512 / 32 | 8 | 36.341 [35.576–36.931] | 36.194 [35.808–36.679] | -0.41% | 45/99 |

Rates are medians [min–max]. Resident limits reflect single-device memory
capacity and apply equally to both arms. Complete output checks compare token IDs
and finish reasons against timed or supplemental HF references.

**W1, W2, W3 and W5 passed the covered-input output checks.** W4, W6 and W7
still differ from HF, so their throughput figures are diagnostic. W1–W5 used
owned KV and native dispatch throughout timing. W6/W7 used paged KV and Python
execution. W7 throughput ranges overlap, and its peak allocated memory was
30.06 GiB for the engine versus 19.54 GiB for HF. At W2, mean TPOT fell from
157.04 to 102.73 ms while TTFT p95 rose from 315.98 to 616.33 ms.

## Run regression checks

With an assigned NPU visible and `MODEL` set as above:

```bash
VLLM_RLT_OURO_CHECKPOINT="$MODEL" python -m pytest -q --run-npu \
  tests/test_ascend_eager.py tests/test_ascend_kv.py \
  tests/test_ascend_decode_scheduling.py \
  tests/test_ascend_checkpoint.py tests/test_ascend_serving.py
```

For broader evaluation, see [accuracy](../accuracy.md) and
[serving benchmarks](../serving.md).
