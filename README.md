# vLLM-RLT

<img src="docs/assets/vllm-rlt-logo.png" alt="vLLM-RLT" width="360">

vLLM-RLT is a vLLM plugin for loop-level continuous batching in recurrent
language models. The plugin adds an operation-level scheduler and an MRv2
worker, using vLLM for serving, execution, and KV block allocation.

The first adapters are Inkling-2-Mini VarLoop and Ouro-1.4B. Inkling supports
request-level decode depth. Ouro uses its learned gate to exit on each decode
token and backfills the skipped KV depths.

## Install

Install CUDA-enabled vLLM at Git revision
`3146a9df41403b8fc30234259444d205974982dd` in a Python 3.11–3.14
environment. With that environment active, install the plugin:

```bash
python -m pip install --no-deps -e .
```

## Serve Inkling-2-Mini VarLoop

The project CLI uses the vLLM serve parser and selects the RLT worker and
scheduler. It also exposes RLT launch flags under `serve --help=all`.

```bash
vllm-rlt serve /path/to/inkling-varloop \
  --served-model-name inkling --language-model-only
```

The equivalent upstream entry point is:

```bash
VLLM_PLUGINS=rlt vllm serve /path/to/inkling-varloop \
  --served-model-name inkling \
  --language-model-only --enforce-eager --generation-config vllm \
  --async-scheduling \
  --scheduler-cls vllm_rlt.core.async_scheduler.LoopAsyncScheduler \
  --worker-cls vllm_rlt.worker.loop_worker.LoopWorker
```

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling","prompt":"The capital of France is","max_tokens":16,
       "temperature":0,"vllm_xargs":{"rlt_target_loops":1}}'
```

Inkling runs a full-depth prompt and first output token. The example sets each
subsequent decode token to one loop. Omit `rlt_target_loops` to use the
checkpoint's depth, which is three for Inkling-2-Mini VarLoop.

For synchronous execution, use `--no-async-scheduling` and
`--scheduler-cls vllm_rlt.core.loop_scheduler.LoopScheduler`. The verified
runtime configuration uses one data-parallel and one pipeline-parallel rank.

## Inkling execution options

`vllm-rlt serve` defaults to eager. To enable vLLM CUDA graph modes for
Inkling, pass an explicit compilation configuration:

```bash
vllm-rlt serve /path/to/inkling-varloop \
  --language-model-only \
  -cc '{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE","use_breakable_cudagraph":true,"cudagraph_capture_sizes":[1,2,4,8,16,128]}'
```

The vLLM worker warmup captures the selected Inkling shapes before serving.
`FULL` captures uniform decode operations. `PIECEWISE` captures the stable
parts of prefill and compatible mixed CORE operations with dynamic attention.
The scheduler can prepare multiple chunks from a request and fill spare CORE
capacity with ready work at other depths when the worker supports it. Use
`--rlt-prefill-chunk-size` to cap one prompt chunk and `--rlt-prefill-window`
to cap uncompleted prefill CORE rows. The default window holds two full
prefill batches across all CORE rounds. Use `--scheduling-policy fcfs` or
`--scheduling-policy priority` to select request ordering.

`--rlt-self-speculation --rlt-decode-pipeline` allows Inkling to start one
tentative token before verifying the prior proposal. These flags are opt-in.
The [development log](docs/plugin-development.md) records output comparisons,
including concurrent cases with token differences.

## Documentation

- [Development log and functional validation](docs/plugin-development.md)
- [Ouro early-exit and GSM8K-87 evaluation](docs/accuracy.md)
- [RLT SiLU-and-Mul build and selection](docs/kernels.md)
- [Contributing](CONTRIBUTING.md)
- [Release history](CHANGELOG.md)

The project WeChat group is listed in the [community QR code](docs/assets/wechat-group-qrcode.jpg).
