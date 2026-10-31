# Documentation

New users should follow the [first-run user guide](launching.md), from creating
an environment to receiving the first generated response. The
[README](../README.md#getting-started) provides a brief installation reference.

| Guide | Topics |
| --- | --- |
| [Developer Must-Read](developer-guide.md) | Requirement understanding, correctness, performance analysis, refactoring rules, and completion criteria |
| [First-run user guide](launching.md) | Environment setup, model download, server startup, first request, CLI, Python API, and troubleshooting |
| [Unified architecture](unified_execution.md) | Stage process boundaries, component responsibilities, shared resources and loop-level execution |
| [Model and KV semantics](design.md) | PRELUDE/RECURRENT/CODA, request progress and depth-specific history |
| [Loop-level scheduling](scheduler.md) | Mixed batching, policy selection, admission budgets, progress and resource safety |
| [KV layout examples](kv_layout_computation.md) | SHARED and LAST_EXITED semantics and worked attention examples |
| [KV cache walkthrough](kv_cache_walkthrough.md) | Capacity, validity, shared metadata, transfer pins and snapshot/restore |
| [Runtime execution and configuration](runtime.md) | Batch contracts, preparation, transfers, execution options and compatibility |
| [Split prefill and WCPB](split-prefill-wavefront.md) | Chunk/depth scheduling, hidden-state ring, KV dependencies and validation limits |
| [Self-speculation](async-speculative.md) | Cross-round asynchronous block execution, token pipelines, and acceptance/rejection timelines |
| [Asynchronous scheduling](https://github.com/InferMatrixLab/vllm-rlt/pull/30) | CPU/GPU pipelining and single-stream or multi-stream execution |
| [FlashAttention](https://github.com/InferMatrixLab/vllm-rlt/pull/30) | FA2/FA3/FA4 installation, hardware selection, and constraints |
| [Cache and scheduling features](https://github.com/InferMatrixLab/vllm-rlt/pull/31) | Prefix reuse, incremental KV, priorities, and preemption |
| [Prefill/decode disaggregation](https://github.com/InferMatrixLab/vllm-rlt/pull/31) | Single-host GPU worker pools and NIXL transfer |
| [HTTP serving](serving.md) | Completions API, streaming, and service lifecycle |
| [Profiling](profiling.md) | Core profiler options, Python/HTTP controls, PD collection, and per-rank archives |
| [Ouro-Thinking recipe](recipes/ouro_thinking.md) | Shared 1.4B/2.6B setup, JSON/SSE client, budgets and validation limits |
| [Experimental deferred Ouro](recipes/ouro_deferred.md) | Fixed-interval deep batching, true per-loop KV, same-readout reference and unfinished qualification |
| [Accuracy evaluation](accuracy.md) | GSM8K regression setup and comparison methodology |

Earlier measurements and known limitations are retained in the
[historical validation record](runtime-history.md); they do not qualify the current revision.

## Support and Runtime Notes

Supported models are Ouro-1.4B, Nanbeige4.2-3B and Huginn-0125. See the
[checkpoint support matrix](../README.md#model-support) and the
[Thinking recipe](recipes/ouro_thinking.md) for model-specific coverage.
Huginn currently uses synchronous execution, LAST_EXITED KV, and fixed
recurrent depth; prefix caching, asynchronous execution, and speculative
decoding are rejected. Its CUDA Graphs capture the recurrent decode core
and coda, including the LM head.
CPU execution provides a Torch reference backend. FlashAttention hardware validation is currently documented
for FA4 on B300; FA2/FA3 require validation on their target devices. Disaggregated
serving currently targets multiple GPUs on a single host.

`ouro_delayed` reuses the trained Ouro gate with a one-loop delay; it changes
the exit policy. The `random_lookahead` mode uses an untrained head for runtime
experiments. Neither is a distilled lookahead predictor.
