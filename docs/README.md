# Documentation

New users should follow the [first-run user guide](launching.md), from creating
an environment to receiving the first generated response. The
[README](../README.md#getting-started) provides a brief installation reference.

| Guide | Topics |
| --- | --- |
| [Developer Must-Read](developer-guide.md) | Requirement understanding, correctness, performance analysis, refactoring rules, and completion criteria |
| [First-run user guide](launching.md) | Environment setup, model download, server startup, first request, CLI, Python API, and troubleshooting |
| [Engine design](design.md) | Model stages, scheduling, and KV ownership |
| [Scheduler walkthrough](scheduler_walkthrough.md) | Responsibilities, admission cases, control flow, and refactoring checklist |
| [KV layout examples](kv_layout_computation.md) | SHARED and LAST_EXITED semantics and worked attention examples |
| [KV cache walkthrough](kv_cache_walkthrough.md) | Allocation state, contracts, cross-module dependencies, and the M4 migration plan |
| [Runtime configuration](cdb_runtime.md) | Exit policies, KV layouts, execution options, and CUDA Graphs |
| [Asynchronous scheduling](https://github.com/hsliuustc0106/vllm-rlt/pull/30) | CPU/GPU pipelining and single-stream or multi-stream execution |
| [FlashAttention](https://github.com/hsliuustc0106/vllm-rlt/pull/30) | FA2/FA3/FA4 installation, hardware selection, and constraints |
| [Cache and scheduling features](https://github.com/hsliuustc0106/vllm-rlt/pull/31) | Prefix reuse, incremental KV, priorities, and preemption |
| [Prefill/decode disaggregation](https://github.com/hsliuustc0106/vllm-rlt/pull/31) | Single-host GPU worker pools and NIXL transfer |
| [PD refactoring draft](pd_refactoring.md) | M9 ownership, handoff contracts, migration dependencies, and validation plan for RFC #32 |
| [HTTP serving](serving.md) | Completions API, streaming, and service lifecycle |
| [Profiling](profiling.md) | Core profiler options, Python/HTTP controls, PD collection, and per-rank archives |
| [Ouro-Thinking recipe](recipes/ouro_thinking.md) | Shared 1.4B/2.6B setup, JSON/SSE client, budgets and validation limits |
| [Accuracy evaluation](accuracy.md) | GSM8K regression setup and comparison methodology |

## Support and Runtime Notes

Supported models are Ouro-1.4B, Nanbeige4.2-3B and Huginn-0125. See the
[checkpoint support matrix](../README.md#model-support) and the
[Thinking recipe](recipes/ouro_thinking.md) for model-specific coverage.
Huginn currently uses synchronous execution, LAST_EXITED KV, and fixed
recurrent depth; prefix caching, asynchronous execution, and speculative
decoding are rejected. Its CUDA Graphs capture the recurrent decode core
and coda, including the LM head.

CPU execution provides a Torch reference backend. FlashAttention hardware validation
is currently documented for FA4 on B300; FA2/FA3 require validation on their target
devices. Disaggregated serving currently targets multiple GPUs on a single host.

### Nanbeige4.2 PD validation recipe

The integration uses the existing Ouro PD runtime for fixed two-loop Nanbeige
execution with one prefill worker and one decode worker on distinct GPUs on the
same host. Models, worker execution, and the NIXL KV/final-hidden handoff are
unchanged. Multi-worker, adaptive exit, CUDA Graphs, prefix caching, Huginn, and
speculative PD are outside the Nanbeige validation scope.

Use Linux, CUDA-enabled PyTorch, Triton, and `nixl==1.4.1` with a CUDA-capable UCX
backend. Reserve two GPUs and select them with `CUDA_VISIBLE_DEVICES`; the tests
use logical devices 0 and 1, BF16, LAST_EXITED KV, and no CUDA Graphs. Test tools
are pytest and Ruff. Do not install dependencies into someone else's environment.

```bash
# CPU checks: no model download or NIXL required.
python -m pytest tests/test_nanbeige.py tests/test_pd.py -m 'not gpu' -v
python -m pytest -m 'not gpu and not benchmark'
python -m ruff check vllm_rlt/pd/engine.py tests/test_pd.py
python -m ruff format --check vllm_rlt/pd/engine.py tests/test_pd.py
git diff --check

# Real 1P1D/NIXL regression on a generated tiny Nanbeige checkpoint.
# Replace the device list with your reserved physical GPU IDs.
CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/test_pd.py --run-gpu -k 'nanbeige and gpu' -v

# Repeat on an explicitly provided, complete local Nanbeige4.2-3B checkpoint.
# Record the checkpoint revision and environment alongside actual results.
VLLM_RLT_NANBEIGE_CHECKPOINT=/path/to/local/checkpoint CUDA_VISIBLE_DEVICES=0,1 \
  python -m pytest tests/test_pd.py --run-gpu -k 'nanbeige and gpu' -v
```

Without the checkpoint variable, tests create small random weights locally;
no Hub download occurs. The reference and both PD workers load the same files.
Tests compare greedy and seeded-sampling outputs, a single-output handoff, finish
reason, and two-loop exit depths. Cancellation covers prefill-start, submitted
handoff before activation, decode-activation, and actual recurrent-decode output;
IDs are reused while old generations retire. Tests check credits, blocks, and
process exit. A controlled tiny checkpoint forces EOS after handoff. These tests
do not cover every instant inside a NIXL transfer. A skipped GPU test is not a pass.

Validation on two RTX 4080 SUPER 32GB GPUs (Linux, Python 3.12.3, torch 2.7.0+cu128,
Triton 3.3.0, NIXL 1.4.1/UCX):

- Generated tiny checkpoint: all six Nanbeige 1P1D tests passed.
- `Nanbeige/Nanbeige4.2-3B`, revision
  `b82e54bd609793562a75cbf9337970a93369eab5`: five official-checkpoint tests passed,
  including token/finish agreement, seeded sampling, and exit depth 2. The sixth
  EOS test uses controlled tiny weights. Official weights were downloaded through
  hf-mirror.com and checked against official file sizes and LFS SHA256.
- Existing two-GPU Ouro PD tests: eight passed.
- Full CPU regression before the final test-only additions: 411 passed, 29 skipped.
  Latest targeted CPU tests: 21 passed, 3 skipped. The final full Linux rerun could
  not be confirmed after the server disconnected. Ruff lint, formatting, and
  diff whitespace checks passed.

This evidence covers the engine/token-ID path above, not HTTP serving, arbitrary
workloads, every cancellation instant, or performance qualification.

`ouro_delayed` reuses the trained Ouro gate with a one-loop delay; it changes
the exit policy. The `random_lookahead` mode uses an untrained head for runtime
experiments. Neither is a distilled lookahead predictor.
