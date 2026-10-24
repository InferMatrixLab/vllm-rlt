# Runtime execution and configuration

This guide covers batch contracts, input preparation, execution features and
configuration constraints. Start with the [architecture](unified_execution.md)
for stage composition and component ownership, or the [scheduler guide](scheduler.md)
for selection policies and dependency timelines.

## Mechanisms and validation coverage

| Feature | Implementation | Verification |
| --- | --- | --- |
| SHARED KV | `core/kv_cache_manager.py`: one physical storage plane, overwrites at every loop; finalization validates without copying | Shared overwrite/read/free tests; chunk/batch invariance; CUDA comparison |
| Random lookahead head | `worker/model_runner.py`: separate frozen linear head, independent CPU seed; base checkpoint untouched | Same seed reproduces weights; CPU/CUDA RNG preservation; base state names unchanged |
| Delayed exit routing | `engine/stage_engine.py`: synchronous pending depth or async previous-submission signal | Signal at round 2 exits after round 3; final hidden and all depth bounds checked |
| CPU/GPU pipeline | `_step_async()` submits round r before collecting round r-1 signals | Submission-order test; GPU outputs/depths compared with sync |
| Multiple CUDA streams | `ModelRunner.submit/finalize/release`: core/boundary streams and per-request completion events | CUDA cancellation/reuse; event test demonstrates overlap with artificially extended core |
| Reusable execution buffers | `worker/buffers.py`, runner state pool and readback leases | Static/dynamic comparison; slot reuse; readback ownership; GPU tests |
| Power-of-two padding | Inactive rows have zero context, no KV write address, no sampling/routing | Free-page sentinels; padded/unpadded outputs; torch/Triton attention regression |
| Automatic KV capacity | `core/memory.py`: CUDA peak probe, memory fraction, headroom, overlap/static budget, useful-capacity cap | CPU byte-budget checks; actual CUDA profiling and admission tests |
| Admission fairness | `Scheduler._admit`: bounded scan and bounded bypass count | Short request admitted behind blocked long request; protection prevents further bypass |
| Chunked prefill policy | Per-request chunk ceiling, queue rotation and bounded prefill batches before decode | Short prompt completes before long prefill; existing decode starvation and chunk consistency tests |
| Exit-trace replay | `ExitConfig(mode="trace")` bypasses gate compute/readback | Variable-depth replay, gate hook rejects accidental execution, all replay variants match |

The checks above describe regression coverage, not results for a particular
revision. Dated measurements are retained in the
[validation history](runtime-history.md#historical-validation-2026-09-15).

## Configuration

`LLM` starts a spawned worker process. In Python scripts, create it inside
`main()` and use the `if __name__ == "__main__":` guard shown below. The context
manager closes the worker when inference finishes or raises.

```python
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, LLM, SchedulerConfig


def main():
    with LLM(
        "../models/Ouro-1.4B",
        device="cuda",
        cache_config=CacheConfig(layout="shared", gpu_memory_utilization=0.9),
        scheduler_config=SchedulerConfig(
            prefill_chunk_size=128,
            max_prefill_batches_before_decode=1,
            admission_scan_limit=64,
            max_admission_bypasses=8,
        ),
        exit_config=ExitConfig(mode="random_lookahead", seed=7),
        execution_config=ExecutionConfig(
            async_scheduling=True,
            static_buffers=True,
            pad_to_power_of_two=True,
        ),
        attention_backend="triton",  # Choose a backend supporting the requested features
    ) as llm:
        outputs = llm.generate(["Explain continuous depth batching."])


if __name__ == "__main__":
    main()
```

The default exit mode remains `ouro`, with the original cumulative hazard policy. For random lookahead, sigmoid is a direct score for exiting after one more loop, not a hazard/CDF. `SamplingParams.exit_threshold` compares this score directly. Threshold 1 disables adaptive exit; `max_loops` always caps execution. Adaptive lookahead cannot exit before round 2, although an explicitly configured hard maximum of 1 is honored. All prompt tokens still run the model's full depth.

Ordinary asynchronous execution requires `ouro_delayed`, `random_lookahead` or `trace`; speculative execution has the separate constraints below. It automatically enables reusable static inputs to avoid blocking device metadata transfers. CPU supports the same scheduling state machine for tests, with immediately completed operations; it does not simulate GPU parallel execution. `multi_stream` is an independent CUDA execution feature: `False` uses one stream for core, boundary and copies; `True` separates those streams, with or without async scheduling.

CLI flags are shared by offline and serving entrypoints:

```bash
OMP_NUM_THREADS=1 python -m vllm_rlt.entrypoints.cli --toy --dtype float32 \
  --max-tokens 4 --exit-mode random_lookahead --lookahead-seed 7 \
  --exit-threshold 0.5 --async-scheduling --static-buffers \
  --pad-to-power-of-two --kv-layout shared --prefill-chunk-size 2
```

For a task-assigned GPU, set `CUDA_VISIBLE_DEVICES` and add `--device cuda --attention-backend triton`. Use `--single-stream` for the single-stream pipeline. `--num-blocks` overrides automatic sizing, or specify `--kv-cache-memory-bytes`; these two overrides are mutually exclusive. CPU auto mode keeps a 256-block default. `StageEngine.memory_plan` records the worker-local sizing decision; it is not an attribute of the public `LLM.engine` client.

## Execution batches and mixed scheduling

Stage composition, component ownership and Request handoff are described in the
[architecture guide](unified_execution.md#inside-one-stage). This section starts
at the scheduler/executor boundary:

| Description | Contains | Used by |
| --- | --- | --- |
| `SchedulerOutput` / `ScheduledRequest` | Selected local Requests and operation/position/depth snapshots | StageEngine and Executor |
| `ExecutionBatch` | Bound operation, spans and recurrent loop count | Ordinary runner preparation |
| `ForwardBatch` | Prepared tensors, KV metadata, operation and graph selection | Shared ModelRuntime |
| `ForwardResult` | Hidden state, gate logits and vocabulary logits | Calling runner/controller |
| Submission ticket | Result storage, lifecycle snapshots and completion evidence | Engine collection and retirement |

These descriptions borrow stage-local state. They are neither network messages
nor replacements for the Request lifecycle. Speculative selections enter through
the same submission interface but retain their own round descriptors.

The runner shares input loading, loop execution and hidden writeback across
prompt and generation work. Homogeneous chunk storage keeps its segment kernels;
homogeneous generation keeps resident routing descriptors. Mixed storage uses
prepared source/target views and `TensorTransfer` gather/scatter descriptors. These are storage optimizations, not
separate stage execution paths. All model operations run through
`ModelRuntime.forward(ForwardBatch)`, including speculative PRELUDE, RECURRENT
and CODA. `ForwardBatch` binds tensors, KV metadata and the operation;
`ForwardResult` returns hidden state, gate logits and vocabulary logits.
Ordinary and speculative runners share the model runtime without sharing their
progress-control algorithms.

`ExecutionBatch.loop_count` describes the recurrence included in a submission.
PRELUDE can include subsequent recurrent loops; RECURRENT can advance several
depths while loading and storing hidden state once. Only the final gate is
returned from such a multi-loop call. A caller needing an exit decision after
each depth must yield at those boundaries. Atomic prefill runs every depth of a
prompt chunk locally through the same submission path.

Speculation with refill scheduling can use loop-yielding chunked prefill through
`ExecutionConfig(atomic_prefill=False)`. This does not allow speculative verification
rows in mixed batches; see [speculative constraints](#fixed-loop-self-speculative-decoding).

```python
scheduler_config = SchedulerConfig(
    mode="mixed", max_num_seqs=8, max_num_batched_tokens=128,
    prefill_chunk_size=64,
)
execution_config = ExecutionConfig(atomic_prefill=False)
```

CLI and serving accept `--mode mixed --no-atomic-prefill`. Mixed selects compatible recurrent work; PRELUDE and CODA remain separate operations.
See [batch selection and feedback](scheduler.md#detail-mixed-loop-selection-and-feedback)
for the selection order, shared budget and per-request progress.

This policy currently requires non-atomic prompt scheduling and LAST_EXITED KV;
it does not mix speculative rounds. Existing model restrictions still apply.
Mixed graphs and asynchronous device execution have regression cases marked
`gpu`; CPU tests validate the corresponding state machine, not GPU overlap or
performance. Mixed gather/scatter uses the shared transfer machinery; this
documentation update does not establish its end-to-end performance.

## Fixed-loop self-speculative decoding

Use shallow loops to draft tokens, then reuse their hidden states and KV to
verify them at full depth. Only verified output is committed; rejected suffixes
are discarded and corrected by the target. The feature is opt-in.

```python
from vllm_rlt import LLM, SamplingParams, SpeculativeConfig


def main():
    with LLM(
        "../models/Ouro-1.4B",
        device="cuda",
        attention_backend="flash_attn_4",
        speculative_config=SpeculativeConfig(
            num_speculative_tokens=4,
            draft_loops=2,
            target_loops=4,
        ),
    ) as llm:
        outputs = llm.generate(
            ["Explain why the sky is blue."],
            SamplingParams(max_tokens=64, max_loops=4, exit_threshold=1.0),
        )


if __name__ == "__main__":
    main()
```

Equivalent offline CLI (select an available GPU through `CUDA_VISIBLE_DEVICES`):

```bash
python -m vllm_rlt.entrypoints.cli \
  --model ../models/Ouro-1.4B --device cuda --dtype bfloat16 \
  --attention-backend flash_attn_4 --speculative-tokens 4 \
  --draft-loops 2 --target-loops 4 --max-loops 4 --exit-threshold 1.0 \
  --max-tokens 64 --prompt "Explain why the sky is blue."
```

`num_speculative_tokens` / `--speculative-tokens` is K, the maximum candidates
per round; K=4 here is an example, not a universal optimum. Draft depth defaults
to 2 and must be below target depth. Target depth must equal the full model depth
(4 for Ouro-1.4B). K shrinks to fit the token budget and remaining output length.

Requires `last_exited`, fixed-depth `ouro` exits and refill scheduling without
preemption. PD can configure speculation on its decode stage; imported prompts
enter the common admission path before CODA and local draft/verify rounds.
Mixed scheduling is not compatible with speculative rounds. Cross-round async
requires greedy sampling and CUDA attention with device-owned context lengths
(Triton or a compatible paged FlashAttention backend). CUDA Graphs can capture
its recurrent cores and coda; draft/verify control and acceptance stay eager.
See [async self-speculation](async-speculative.md) for graph limits and validation
scope. Synchronous speculation supports greedy and temperature/top-k/top-p sampling; sampling
preserves the target distribution in exact arithmetic, not identical text for
the same seed. BF16 batch shapes can introduce numerical differences.

The same flags work with the single-device server. A streaming event may contain
multiple committed tokens. See [RFC #43](https://github.com/InferMatrixLab/vllm-rlt/issues/43)
for the design.

## SHARED prefill semantics

SHARED retains one physical KV plane and completes every loop for a prompt
position before advancing that request. It is not interchangeable with packed,
depth-major LAST_EXITED prefill. SHARED requires atomic prompt execution; split
prefill and mixed scheduling require LAST_EXITED. See the
[prefill ordering example](kv_layout_computation.md#5-prefill-execution-order)
for the causal-history difference and its capacity implications.

## Preparation and execution boundaries

`Executor.submit()` routes a selected batch to the ordinary runner or the local
speculative controller. Ordinary `ModelRunner.submit()` follows:

```text
bound ExecutionBatch
  → prepare_on_cpu: plan rows, ranges, routing and host metadata
  → prepare_on_gpu: stage inputs/metadata and prepare graph calls
  → ModelRuntime.forward: execute PRELUDE / RECURRENT / CODA
  → submission: retain result storage and completion events
  → collect and apply per-request progress
```

The prepared `ForwardBatch` carries the inputs through preparation and execution;
there is no separate prefill metadata class. `KVMetadata` supports prompt,
decode and mixed rows. Device-resident speculative positions use
`DeviceKVMetadata` to avoid reading positions back merely to rebuild metadata.

Preparation moves decisions and result-independent indexing out of forward;
operations that depend on a newly produced hidden state still run after their
producer. Graph replay and eager forward remain alternative execution mechanisms
behind the shared runtime. Packed prompt layouts and persistent decode buffers
remain specialized storage paths.

A fused PRELUDE plus recurrent loops runs on the core stream. Standalone PRELUDE
and CODA use the boundary stream; chunking alone does not change their stream.
`multi_stream` controls stream separation independently of asynchronous scheduling.

## Submission lifetime and overlap

The [complete forward timeline](scheduler.md#2-asynchronous-execution-and-safe-buffer-reuse)
shows when GPU dependencies, host decisions and buffer reuse require waiting.
Here the relevant runtime contract is ownership: a submission retains result
storage while its ticket is live, and device completion is tracked independently
of logical Request progress. Request identity, token position and depth protect
against stale completion after reuse. Failed submissions drain owned readers
before their storage can be recycled.

Reusable input snapshots and separate readback buffers keep input preparation
independent of output collection. `UvaBufferPool` defaults to two slots; a slot
can grow when its first-dimension capacity is insufficient. Rotation itself is
not completion protection: the execution owner must retire earlier GPU readers
before reusing a slot. The bounded submission queue and completion events enforce
that ownership contract. Readback storage is released after consumption or safe
retirement, not merely after launching a copy.

UVA-capable kernels can read pinned host snapshots directly. Other paths upload
to reusable device storage. `TensorTransfer` batches compatible copy descriptors
for a common Triton transfer kernel; a caller can also explicitly request a whole
input copy with `UvaBufferPool.copy_to_gpu()`. Not every transfer is a Triton
operation, and pinned storage is not automatically safe to overwrite.

Static mode preallocates inputs, metadata, persistent request states and async readback buffers. Intermediate model activations still use PyTorch allocation; this is not CUDA-graph capture or a claim of an allocation-free forward. Padding affects the compute batch; only real rows write KV, produce signals and sample tokens. `last_effective_size` and `last_submitted_size` expose the distinction.

## Capacity and admission

CUDA automatic sizing profiles a synthetic maximum-row core/coda pass. It reserves the measured extra peak (doubled conservatively for overlap), static device buffers, metadata, long-context torch-reference scratch when relevant, and `memory_reserve_bytes`. The memory-utilization budget is also bounded by startup free memory. The resulting pool is capped at the blocks usable by all concurrent full-context requests, avoiding enormous useless pools for tiny models. Profiling includes temporary KV, so the estimate is deliberately conservative; it is not a promise against other processes changing GPU memory usage after startup.

For bounded admission scans, bypass fairness, incremental growth budgets and the
zero watermark in non-incremental mode, see [scheduler admission](scheduler.md#admission-available-memory-is-not-the-whole-budget).
Pool sizing establishes physical capacity; scheduler admission decides how much
of that capacity can safely be promised to new work.

## FP8 KV cache

`CacheConfig(kv_cache_dtype="fp8_e4m3")` (CLI `--kv-cache-dtype fp8_e4m3`) stores K and V as
float8 e4m3 codes with one float32 scale per (block, layer, token, KV head) vector
(`scale = amax / 448`). Values are quantized when written and dequantized by the attention
backend when read, so a block costs `head_dim + 4` bytes per vector instead of
`head_dim * element_size`: 51.6% of BF16 at `head_dim=128`, and automatic sizing or
`--kv-cache-memory-bytes` admits about 1.94x as many tokens. The default `"auto"` stores the
model dtype and leaves every existing path unchanged.

- Supported: Torch (reference) and Triton attention; synchronous and asynchronous scheduling;
  CUDA graphs (quantized writes are captured); LAST_EXITED finalize copies, including the
  asynchronous finalize kernel, copy codes and scales; preemption snapshots carry the scales;
  `KVCacheManager.read` returns dequantized values.
- Triton writes with one fused quantize-and-store kernel per layer. The Torch backend quantizes
  with Torch ops and is the reference; both writers produce the same codes and scales.
- Rejected: FlashAttention backends (the `fp8_kv` backend capability; `auto` selects Triton)
  and PD transfer, which does not yet move the scales.
- Each layer writes the current token's K/V before attending, so attention reads the current
  token in FP8 as well. Accuracy should be checked per model and workload; adaptive exits can
  change where a token exits because the gate sees slightly different attention.

## Replay and validation commands

Trace mode expects `depths_by_request`, with one depth per output: full prefill depth first, then each decode input's exit depth. Admission validates coverage and loop bounds. Request IDs must match the trace. The public engine replay fixes depth decisions; token generation still follows the configured sampler.

```python
ExitConfig(mode="trace", depths_by_request={"A": [4, 2, 4]})
```

The synthetic benchmark records prompts, output IDs, depths, model configuration/seed, dtype, layout and sampling configuration. It validates this fingerprint before replay and rejects output/depth mismatches. It compares refill/no-refill with eager, padded, async single-stream and async multi-stream execution. Initialization and a warmup are excluded from timing; per-step trace collection is included equally for all variants.

```bash
python -m benchmarks.cdb_runtime --device cpu --output /tmp/cdb-cpu.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.cdb_runtime --device cuda \
  --trace-out /tmp/cdb-trace.json --output /tmp/cdb-gpu.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.cdb_runtime --device cuda \
  --trace-in /tmp/cdb-trace.json --output /tmp/cdb-replay.json
CUDA_VISIBLE_DEVICES=0 python -m pytest -q --run-gpu
```

Use only a GPU available to the task. The overlap regression deliberately extends GPU core duration to test stream dependencies; it is not a throughput benchmark. The tiny replay benchmark also cannot establish real Ouro speedups. Random gate quality, real-model workload performance and the paper's accuracy results require separate evaluation.

## Reusing the trained Ouro gate: `ouro_delayed`

`ExitConfig(mode="ouro_delayed")` uses the checkpoint's existing
`model.early_exit_gate` directly, without creating or copying an auxiliary head.
Its per-round sigmoid output remains a hazard: after consuming round r,
`remaining_probability *= (1 - hazard_r)`. When `1 - remaining_probability`
reaches the threshold and r >= min_loops, execution exits after round r+1.
The hard max_loops still wins, and threshold 1 disables adaptive exits.
Prefill always runs full depth; probability state resets for every decode token.

For example, hazards 0.3, 0.3 give cumulative probabilities 0.3, 0.51.
With threshold 0.5 and min_loops 2, ordinary Ouro exits at round 2;
`ouro_delayed` exits at round 3. It preserves the original gate's probability
interpretation but changes the exit depth. This is a delayed original-gate
heuristic, not a trained predictor of the next round's exit quality.

Synchronous execution stores a pending exit depth. Async submits round r+1
before collecting round r's signal, accumulating each consumed hazard once.
Both modes use the same policy. Select both with `--exit-mode ouro_delayed`;
add `--async-scheduling` for async, which automatically enables static buffers.
Choose a backend supporting the requested async features. Padding is optional; use
`--static-buffers --pad-to-power-of-two` to enable it.

To try this policy, use the [serving setup](launching.md) and add
`--exit-mode ouro_delayed --async-scheduling` to the server command. Set
`min_loops`, `max_loops` and `exit_threshold` in the
[completion request](serving.md#completions-contract).

For a scheduling comparison keep exit mode, threshold, padding, cache and
workload identical, removing only `--async-scheduling` for sync. Comparing
ordinary `ouro` to async `ouro_delayed` also changes the exit policy, confounding
a scheduling speed comparison. A single request verifies functionality;
multiple concurrent requests are needed to exercise cross-request overlap.

Historical correctness results, including the unresolved BF16 multi-stream
output divergence, are retained in the [validation history](runtime-history.md#ouro-delayed-validation).
