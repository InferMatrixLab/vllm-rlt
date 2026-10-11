# Initial LoopCD support for Ouro

Task 0 of [RFC #85](https://github.com/ThinkFlowLab/vllm-rlt/issues/85)
adds fixed-depth synchronous Ouro readout guidance from
[LoopCD, arXiv:2610.02185v1](https://arxiv.org/abs/2610.02185).
Guidance is off by default. The recurrent stack, normalized feedback, gate,
KV writes and sampling transforms retain their original behavior.

```text
Same token prefix -> original recurrent stack, updates 1 ... R
                               | b                   | R
                               v                     v
                       owned normalized h_b   normalized h_R
                               |                     |
                          original head          original head
                               |                     |
                              z_b                   z_R
                               `--------> z_R + w*(z_R-z_b) -> sampler

Reference owner = Request object + token position + output index + depth
Next token clears the old reference; completion/abort releases it before reuse.
```

The reference depth is one-based and must precede the actual prefill/decode
depth. `two_head` projects both normalized states and combines logits in FP32.
It is the enabled default. Optional `linear_fused` projects
`h_R + w*(h_R-h_b)` through the existing linear head, without another norm.
Its algebraic identity does not imply BF16 equality or identical low-margin
argmax decisions; it remains an explicit experimental choice.

```python
from vllm_rlt import ExecutionConfig, LoopCDParams, SamplingParams

execution = ExecutionConfig(loopcd=True, prefill_depth=4)
sampling = SamplingParams(
    min_loops=2,
    max_loops=2,
    exit_threshold=1.0,
    loopcd=LoopCDParams(reference_loop=1, strength=0.3),
)
```

This is fixed prefill P4/decode D2, not a P2/D2 run. The first emitted token
records depth 4 and subsequent tokens depth 2. `prefill_reference_loop` can
freeze a separate prompt reference. A zero strength follows the original
readout without reference allocation.

Nonzero guidance requires `ExecutionConfig(loopcd=True)` and synchronous
`last_exited` KV with fixed decode depth and the original `ouro` exit mode.
Prefix caching, prefill UVA, asynchronous scheduling, speculation and P/D
transfer are rejected before request allocation. Other model families are
rejected. Single-worker preemption is supported with eager execution only;
combining guidance, preemption and CUDA Graphs remains rejected until separate
qualification. Official-checkpoint eager/Graph readout qualification is tracked
in Task 1.

Automatic memory planning reserves owned references and conservative two-head
FP32 scratch before planning KV. Explicit KV budgets still require physical
device admission. Existing release-event ordering protects live references;
request-ID reuse cannot inherit another Request object's reference.

## Correctness scope

`tests/test_loopcd.py` checks the first readout against the repository's
independent functional dense Ouro oracle across prompt chunks and P/D depths.
Fixed P=D teacher-forced continuation checks all vocabulary logits, selected
log-probabilities, raw normalized state and top-1 decisions through abort and
request-ID reuse. It also covers zero-strength equivalence, refill/drain,
configuration rejection, stale owners and memory reservation. These use real
in-memory tiny FP32 CPU models. The affine FP32/FP64 test does not qualify BF16.

Official-checkpoint continuation and fixed-work results are tracked in
[Task 0's audited results](loopcd-scale-results.md), including its retained
BF16 independent-oracle failure and uncertain quality confirmation.
Preemption qualification is described below.


## Single-worker preemption contract

Enable the existing scheduler path with
`SchedulerConfig(enable_preemption=True)`. Suspension occurs only at the
existing safe scheduler boundaries; pending outputs and selected requests remain
ineligible. The Request object and its RNG stay owned by the scheduler.

At first-token CODA, or after the decode reference depth has been reached, the
owned normalized reference is copied to a private CPU tensor alongside the
existing KV, hidden-state and input-token snapshot. Its Request identity, token
position, output index and one-based reference depth are retained. Resume checks
these fields before KV allocation, then restores the reference on the cache
device before re-enqueuing the saved stage. It does not replay recurrent work or
change the guidance equation.

A partial prompt chunk, next-token PRELUDE, or decode before the reference depth
does not need the old reference: the resumed computation captures a fresh one.
A capacity-blocked resume keeps only the CPU snapshot. Abort discards both the
snapshot and runner-owned reference; a reused request ID gets a new Request
object and cannot inherit the old guidance state.

`tests/test_loopcd_preemption.py` covers suspend/resume logits and seeded
sampling, all relevant stage boundaries, asymmetric P4/D2, repeated preemption,
capacity-blocked restore, cancellation, ID reuse and rejection of corrupted
reference identity/metadata. Scheduler-driven priority and KV-pressure tests
also check output parity and resource cleanup on CPU and Triton FP32 CUDA.
The CUDA cases cover static buffers off and on. The CUDA tests use tiny random checkpoints and
require the `gpu` marker.

`tests/test_loopcd_preemption_continuation.py` uses the stateful
`OuroContinuation` oracle for P4D2/P4D3, with resumed CODA, PRELUDE and
RECURRENT stages, delayed KV allocation, cancellation and ID reuse. At each
readout it compares final hidden state, every historical K/V plane, guided
vocabulary logits and selected-token log probability. The original
`atol=rtol=1e-4` gates are unchanged. Reference-hidden error is recorded as a
separate diagnostic; saved references are checked bit-exactly.

Official Ouro-1.4B FP32/Triton passes all 12 combinations of decode depth 2/3,
prompt length 3/4/5 and guidance strength 0/0.3 in two fresh processes:
264 readouts including the reused requests. Native BF16 B1 priority resumption
also has bit-exact vocabulary logits with static buffers off/on in both
processes. This BF16 parity result does not qualify the retained independent
BF16 numerical failure in [Task 0's results](loopcd-scale-results.md).

Earlier official FP32 KV-pressure runs match tokens, exit depths and recurrent
work; score differences up to `2.4e-5` also occur in an uninterrupted batching
control. BF16 multi-request pressure remains unqualified.

### Fixed-work preemption cost

The RTX 5090 comparison uses FP32/eager Triton, P4D3, two_head/ref1/strength0.3,
B16, initial-total C32, P512/F128 and a common 18 GiB KV pool. The interrupted
arm suspends one request at completed-step CODA before output index 4, then
uses the native FCFS scheduler to restore it. Two reversed configuration orders
each run five warmups and five measurements per arm. The four-arm run retains
80 trials; the two guided arms have ten measured trials each.

| Guided arm | Mean tokens/s | Mean native request latency |
| --- | ---: | ---: |
| Uninterrupted | 224.60 | 13.67 s |
| One controlled preemption | 214.55 | 14.55 s |

All 20 trials per guided arm match request IDs, output tokens, exit depths and
finish state; model-row work is unchanged. Batch regrouping changes head calls
from 1670 to 1694. The snapshot contains 960 MiB of KV tensors and an 8 KiB
guidance reference, plus hidden state and written metadata. Actual successful
restore callbacks average 115.1 ms; suspension averages 546.3 ms.

The paused request's next-token delivery gap averages 7.63 s, reaching 7.74 s;
it waits at the back of the FCFS queue. Copy time alone therefore understates
its interruption. These timings include native scheduling, KV transfer and
batch regrouping; they do not isolate the reference-copy cost or estimate HTTP
SLOs. See [raw trials, source hashes and retained diagnostics](https://github.com/prettygirlisnotme/vllm-rlt/releases/tag/loopcd-preemption-evidence-20261010).
