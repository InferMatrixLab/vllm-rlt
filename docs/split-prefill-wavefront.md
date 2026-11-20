# Split prefill and wavefront chunked prefill

Split prefill limits how much prompt work a request contributes at once.
Wavefront Chunked Prefill Batching (WCPB) uses the same chunk boundary while
batching ready prompt work from different chunks and recurrent depths. Its main
target is better recurrent-core occupancy when prompt batches are sparse or
uneven, without making chunks larger and reducing scheduling opportunities.

![Split prefill compared with wavefront chunked prefill](assets/Split-prefill-vs-wavefront-chunked-prefill.png)

The current runner requires one operation per batch: PRELUDE, RECURRENT and
CODA ranges are submitted separately. Ordinary prefill can also execute its
selected range through all recurrent depths in one plan. Actual submission
counts depend on scheduler mode, token budget, and ready work; the numbered batches
in the figure should not be read as measured latency or a fixed runtime schedule.

## Execution model

The ordinary path selects a prompt range and runs the range through the
remaining recurrent depths before applying progress. In mixed scheduling, the
scheduler may choose a shorter execution boundary when ready decode work needs a
depth decision. `ExecutionBatch.loop_count` records the recurrent work in that
submission.

With `wavefront_chunked_prefill=True`, prompt work uses these phases:

| Phase | Work | Recurrent loops in the submission |
| --- | --- | ---: |
| `PREFILLING_PRELUDE` | Embed staged prompt ranges and prepare inputs | 0 |
| `PREFILLING_RECURRENT` | Run ready prompt ranges at their selected recurrent depths | 1 |
| `PREFILLING_CODA` | Run attention-bearing output layers on ready prompt ranges, when required by the model | 0 |

Huginn and Dory require the CODA phase to populate their output-attention KV.
All prompt rows run the output layers; only the last prompt row projects logits
and samples the first output. This phase is currently scheduled separately from
ordinary decode CODA. Ouro does not require prompt CODA.

Staged CODA passes a row-aligned boolean `ForwardBatch.compute_logits` mask:
only a final prompt row or a mixed decode row is selected for projection and
sampling. Sparse samples are mapped back to their request spans. Uniform boolean
values remain supported for the existing atomic and CUDA Graph paths; automatic
mixing of prompt and decode CODA is a separate scheduling change.

The scheduler can include several ranges from different chunks, requests, and
depths in one recurrent batch. It obeys the shared token budget, request-owner
limit, KV capacity, and each depth's readiness frontier. In mixed mode,
compatible decode and prompt work can share PRELUDE or RECURRENT batches. The
batch still contains one operation type.

WCPB is opt-in through `SchedulerConfig.wavefront_chunked_prefill` or the
`--wavefront-chunked-prefill` CLI flag. `prefill_chunk_size` controls the prompt
chunk size. The option works with refill or mixed scheduling; mixed scheduling
does not support speculative work.

## Progress and dependencies

WCPB relies on the depth-specific KV planes of `LAST_EXITED` layout. For
example, a later chunk's depth-0 work can use the completed depth-0 prefix while
an earlier chunk advances at a deeper recurrence depth. A deeper task also
needs its preceding depth's hidden state and the causal KV prefix at its own
depth. The scheduler uses per-depth prompt frontiers to admit only ready ranges.

Progress belongs to the request rather than to a persistent task table:

- `num_preluded_tokens` tracks the prompt range whose prelude has been staged.
- `prefill_depth_ranges` tracks the ready range at each recurrent depth.
- For attention-bearing CODA, `num_recurrent_prefilled_tokens` tracks the final
  recurrent frontier; `num_prefilled_tokens` tracks completed prompt CODA.
  For other models, `num_prefilled_tokens` tracks the final recurrent frontier.
- `prefill_reclaimable_tokens` marks the oldest resident chunk boundary. It
  advances after the chunk's final required stage completes: CODA for Huginn and
  Dory, otherwise the final recurrent depth.

This separation prevents a prelude or partial chunk from advancing the
reclaimable boundary. For models with prompt CODA, the CODA frontier cannot
overtake the final recurrent frontier and each chunk remains resident until
CODA consumes it.

## Hidden-state ring

Each request reserves `prefill_chunk_size * total_ut_steps` rows for staged
prompt hidden states. A chunk uses a ring slot derived from its index relative
to the cached-prefix origin and the request's prefill loop count. For a
chunk-local offset `j` and slot `s`, its row offset is:

```text
prefill_hidden_row_offset = s * prefill_chunk_size + j
```

`ScheduledRequest` and `StateSpan` carry this row offset. When it is `None`, the
runner derives the row from `token_start - chunk_start`. Intermediate outputs
are scattered to the ring for later depths or prompt CODA. Batch-local gathered
hidden scratch rotates with the submission's input descriptors and is reused
only after its last device consumer. Request events also protect ring rows from
reuse before dependent reads finish.

## Scheduling and recovery

WCPB advances one recurrent depth per recurrent submission so the scheduler can
regroup the next set of ready ranges. A boundary gives admission and mixed
scheduling a chance to select other work; it does not guarantee that another
request will run. Smaller chunks therefore improve scheduling granularity but
can reduce ordinary prefill occupancy. WCPB tries to recover some occupancy by
batching independent ready chunk/depth ranges.

The implementation uses the existing request queues, `ScheduledRequest`, and
per-depth frontiers; it does not introduce a generic `PrefillTask` registry.
Preemption rewinds staged work to the request's completed prompt frontier and
recomputes unfinished work after resume. PD transfer progress is reported for
completed chunks, and imported prompt state initializes the same frontiers.
Speculative execution is supported with compatible refill scheduling; mixed
mode rejects speculative configuration. For Huginn and Dory, staged prefill
currently requires synchronous eager generation without preemption, speculation,
prefix reuse, or PD; the separate CUDA streams remain event-ordered. Tiny-model
CPU and A100 CUDA regressions cover these staged paths. A short real Dory
checkpoint check compares mixed and WCPB prefill with atomic prefill; longer
contexts and broader backend qualification remain outstanding.

`SHARED` KV is rejected for WCPB. Its position-major dependencies do not permit
the same cross-chunk/depth ordering as `LAST_EXITED` without a different
scheduler dependency graph.

## Performance and correctness limits

Controlled A100 measurements show that ordinary small chunks can underfill the
recurrent batch. At chunk size 16, one-run WCPB measurements raised rows per
batch and throughput from 64.00 rows and 275.82 tok/s to 254.49 rows and
412.01 tok/s for four 8,064-token requests. For 32- and 128-request mixed
workloads, measured throughput changed from 337.51 to 628.45 tok/s and from
471.95 to 740.29 tok/s respectively. These are directional results from one
warmup and one measured run per variant, not multi-run medians or universal
speedup claims.

A separate HTTP matrix used Ouro-1.4B BF16 with Triton on two A100 80GB GPUs,
`max_num_seqs=8`, `max_num_batched_tokens=1024`, chunk sizes 32 and 128, and
fixed 512-token and variable 256/512/768-token prompts. Each serial, concurrent,
and arrival profile had 100 requests and one measured run. WCPB request
throughput was 0.8% to 22.6% higher than Split in the 12 matched profiles, but
some tail-latency metrics were worse. These measurements do not qualify decode
latency under arbitrary live traffic.

The prior GSM8K-87 BF16 run scored 63/87 for WCPB and 60/87 for Split and
Atomic, but WCPB differed from Split on 24/87 token sequences. The score
difference does not establish output equivalence or improved model quality.
Exact GPU token preservation remains an open validation issue.
