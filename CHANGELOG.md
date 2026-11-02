# Changelog

## 0.2.0 — 2026-11-01

This is a research and evaluation milestone, not a production-stability or
API-compatibility guarantee. Start with default synchronous execution and
validate optional configurations against your own workload.

### Changes since 0.1.0

- Unified stage/loop execution, with Scheduler-owned execution feedback,
  split prefill, wavefront chunked prefill, and compatible mixed batching.
- Native Huginn support and bounded Ouro-Thinking integration, alongside
  Ouro and Nanbeige. Checkpoint and feature-combination qualification remains
  scoped to the evidence in the model support matrix.
- Expanded asynchronous self-speculation, token pipelining and CUDA Graph
  execution. These remain opt-in rather than the default serving baseline.
- Optional native CUDA SiLU-and-Mul for Ouro, automatic build detection and
  Torch fallback when the extension is absent. Nanbeige and Huginn do not yet
  use this native operator.
- Fixes for async preemption token restoration (#109), async speculative
  request-failure isolation and resource cleanup (#115), and PD coordination.
- Automated CPU regression, lint and formatting checks.

### Migration and installation

- Use the tagged source or release distribution to pin the version rather
  than following `main`. Python 3.10+ and PyTorch 2.5+ remain required.
- `LLM` starts a spawned stage worker. Put construction inside a function and
  an `if __name__ == "__main__":` guard; use its context manager for cleanup.
  `LLM.engine` is a `StageClient`, not direct access to worker-local caches.
  See [the current Python example](docs/launching.md#python-api).
- Python-only wheels retain the Torch activation implementation. To build
  the optional native extension from source, install compatible CUDA-enabled
  PyTorch, a CUDA toolkit with `nvcc`, and a C++ compiler, then install with
  `--no-build-isolation` and the allocated GPU visible. A driver alone is not
  a compiler. Native binaries are specific to their Torch/CUDA/target GPU
  environment; see [kernel build instructions](docs/kernels.md).
- Model weights are downloaded separately. This GitHub release does not imply
  a PyPI publication or a bundled checkpoint.

### Validation boundaries

- Consult the release notes for checks executed on the final release source
  and the exact artifacts. CPU checks do not qualify CUDA execution.
- The merged native-kernel PR reports bounded B300 validation. Other hardware,
  FlashAttention combinations, autograd and `torch.compile` are not qualified
  by those results; native operator integration is Ouro-only.
- Existing serving benchmarks are diagnostic: full greedy output equivalence
  has not passed across all configurations. They are not equal-quality
  speedup claims or production acceptance. Historical results must not be
  presented as fresh measurements of this tag.
- Adaptive exit can change output quality. Each checkpoint, backend and
  advanced feature combination needs its own validation. The HTTP API exposes
  completions, not `/v1/chat/completions`; use a secured gateway for untrusted
  networks rather than exposing the unauthenticated server directly.
- Open work such as training integration, FP8 KV, FlashLoop, LoopCD and adaptive
  draft-length policy is not
  included merely because a proposal or pull request exists.

See [runtime documentation](docs/runtime.md),
[model support](README.md#model-support), and
[benchmark qualification](docs/unified_execution.md#measured-serving-performance).

## 0.1.0 — 2026-10-10

Initial experimental release. Its immutable source and historical validation
scope are documented in the [release notes](https://github.com/InferMatrixLab/vllm-rlt/releases/tag/v0.1.0).
The known #109 and #115 limitations described there were addressed after that
tag; do not infer that the fixes are present in 0.1.0 itself.
