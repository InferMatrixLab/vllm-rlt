# Ouro on Ascend

Run Ouro on one Ascend NPU with BF16, synchronous greedy decoding and four
recurrent loops. Use the explicit `ascend_eager` attention backend.

## 1. Prepare the environment and model

Use a compatible Torch, Torch-NPU and CANN environment with the project's
Python dependencies installed. Keep the vendor Torch pair; CUDA/Triton
packages are not needed for this backend.

~~~bash
source /path/to/cann/set_env.sh
source /path/to/npu-environment/bin/activate
python -m pip install --no-deps --no-build-isolation -e .
export MODEL=/path/to/prepared/Ouro-1.4B
~~~

Use the checkpoint's own config and tokenizer. Model loading uses the shared
native Ouro loader, including its supported sharded checkpoint layout.

## 2. Start the server

Select an allocated physical NPU using `ASCEND_RT_VISIBLE_DEVICES`.

~~~bash
ASCEND_RT_VISIBLE_DEVICES=<allocated-device> \
python -m vllm_rlt.entrypoints.serve \
  --model "$MODEL" --tokenizer "$MODEL" --served-model-name ouro \
  --device npu --dtype bfloat16 --attention-backend ascend_eager \
  --num-blocks 1024 --max-num-seqs 1 \
  --max-num-batched-tokens 4096 --prefill-chunk-size 4096 \
  --host 127.0.0.1 --port 8123
~~~

Explicit KV capacity is required. Adjust it for the model, available memory,
prompt/output lengths and concurrency. To compare with full-prefill inference,
both the chunk size and batched-token budget must fit the whole prompt.

## 3. Send a request

~~~bash
curl --fail http://127.0.0.1:8123/health
curl --fail http://127.0.0.1:8123/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ouro","prompt":"The capital of France is","max_tokens":8,"temperature":0,"min_loops":4,"max_loops":4,"exit_threshold":1}'
~~~

Wait for HTTP 200 before sending requests. Add `"stream": true` for SSE.
The `ascend_eager` backend follows HF-style BF16 attention rounding;
`--attention-backend torch` selects the existing FP32-intermediate reference.

## 4. Run checks

With an assigned NPU visible to the test process:

~~~bash
pre-commit run --all-files
VLLM_RLT_OURO_CHECKPOINT="$MODEL" python -m pytest \
  tests/test_ascend.py tests/test_ascend_checkpoint.py \
  tests/test_ascend_eager.py tests/test_ascend_kv.py \
  tests/test_ascend_serving.py --run-npu -q
~~~

These checks cover causal attention, cross-page KV, request reuse and HTTP/SSE.
They do not establish model quality or performance; use the
[accuracy guide](../accuracy.md) and [serving benchmark guide](../serving.md)
for those evaluations.

Async scheduling, graphs, speculation, PD, preemption, prefix caching and
adaptive exit are currently unsupported on this path.
