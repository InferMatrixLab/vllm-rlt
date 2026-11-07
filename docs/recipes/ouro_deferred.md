# Experimental deferred Ouro decoding

This opt-in, synchronous runner groups Ouro's third and fourth loops across
response positions. It is intended for experiments by Ouro rollout developers
working with verl/vime, without importing either training framework.

## Execution and comparison

Prompt prefill runs all four loops and produces the first output. At interval
four, the next three response inputs read out after loop two. The fourth input
waits while all four retained loop-two states pass through loop three and then
loop four. The block-end loop-four state produces the next output. Every row
keeps its original token position and its own genuinely computed KV history at
each loop; shallow KV is never propagated into the pending deep planes.

The main execution variable is the deep batch size. For `N` active requests at
interval `B`, a complete aligned deep block contains `N * B` rows. Eager
execution uses `N` rows on each step. Both execute the same deep rows at a
completed boundary; deferral changes when and how those rows are grouped.
Unfinished deep tails are discarded when a response stops or reaches its limit.

Use three distinct comparisons:

- `--interval 1`: four-loop readout at every position, with full-depth histories.
- `--interval 4 --eager`: immediate deep writes under the mixed readout schedule.
- `--interval 4`: deferred deep writes under that same mixed readout schedule.

The eager/deferred pair isolates execution timing. The interval-one/mixed pair
changes the policy and needs a paired task-quality comparison. BF16 batch shapes
can also change scores and sampled trajectories; CPU causal equality does not
qualify BF16 GPU probability replay.

## Run an experiment

Install the normal repository dependencies and provide a local Ouro checkpoint.
The CLI uses BF16 on CUDA and FP32 on CPU. CUDA needs a scheduler-reserved device;
the selected backend must support the device and dtype.

```bash
python -m vllm_rlt.entrypoints.ouro_deferred /models/Ouro-1.4B \
  --prompt "What is 17 times 23?" --interval 4 --responses 4 \
  --max-tokens 4096 --temperature 1 --seed 42 --backend triton
```

Add `--eager` for the same-readout reference, or use `--interval 1` for native
four-loop readout. Repeat the same prompts, limits, seeds and sampling settings
for each comparison. The first token 0 or 2 ends a response. Outputs retain raw
token IDs, actual selected processed logprobs, readout depths, decoded text and
the `stop`/`length` finish reason. Length-limited answers remain in the results.

The Python entry point accepts token-ID prompts and independent per-answer seeds:

```python
from vllm_rlt.worker.deferred_ouro import generate_deferred

outputs = generate_deferred(
    model, [[3, 5, 7], [11, 4, 9]], [42, 43], interval=4,
    max_tokens=64, temperature=1.0, backend="auto",
)
```

Sampling uses the engine's `Sampler` and `RngRegistry` with full vocabulary,
positive temperature and one request-local random stream. Reported logprobs
are the logarithm of the actual processed sampling probability. They describe
the selected mixed readout and its KV history, rather than a separately
recomputed four-loop policy.

## State ownership and lifecycle

A runner owns one fixed batch and one four-plane paged cache. Weights must remain
unchanged until it closes. Retained hidden rows have independent storage, so a
finished neighbor cannot keep a larger active-batch tensor alive. `release()`
discards a request's unread deep tail and frees its pages. The context manager
and generation helper release the cohort on completion or error. A recurrent
execution failure closes the runner; create a fresh runner to retry.

KV reservation covers prompt and executed response inputs. The final sampled
token requires no extra forward position. Full capacity is reserved up front;
this is a batching experiment and does not compress or quantize KV.

## Validation and unfinished qualification

Run the causal and lifecycle checks with no GPU access:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest -q tests/test_deferred_ouro.py
pre-commit run --all-files
```

Tests compare selected hidden states and every populated KV plane against the
independent functional dense oracle across intervals 1/2/4/8 and grouped-query
attention configurations. They also check the actual deep row/call counts,
staggered request progress, termination, probability alignment, independent RNG,
bounded hidden storage, failure cleanup and the last executable position.

This experimental path uses the public native recurrent API and its attention
backend. The normal `LLM`/serving paths keep their existing feature selection.
The experiment does not integrate adaptive gating, speculative verification,
CUDA Graphs, PD, prefix reuse, preemption or weight publication. Those
combinations require separate implementations and validation.

Real-weight BF16 task-quality, memory and end-to-end throughput comparisons,
plus separate ops-only/full GPU profiles, remain required before qualifying an
acceleration claim. Measure the eager/deferred pair with the same readout policy
and count all GPU work, failure and length-limit costs. Compare the mixed policy
with K4 on the same tasks under an explicit quality budget. The synchronous
Python sampling/output path and per-loop metadata preparation add overhead;
profiling must determine whether larger deep batches cover those costs.

This supports an experimental execution path related to
[RFC #70](https://github.com/ThinkFlowLab/vllm-rlt/issues/70). It does not complete
training integration or differentiable current-policy probability replay.
