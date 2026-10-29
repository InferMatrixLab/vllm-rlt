# Architecture diagrams

Open `index.html` in a browser to switch between the diagrams and zoom.
No network connection is required.

## Overall architecture

`stage-architecture.svg` is the editable presentation source, with a fixed
1840 × 1240 layout. `stage-architecture.png` is its 2× browser-rendered export,
embedded in [`../unified_execution.md`](../unified_execution.md) for reliable
Markdown and VS Code preview.

The diagram centers one reusable stage, with faded optional upstream/downstream
stages at its sides. Its internal loop connects scheduling, execution, and
request/KV progress. Three colors track requests with different prefill/decode
phases and loop depths into one mixed recurrent batch, followed by independent
request progress. The scheduler is visually dominant: it owns scheduling decisions
and consumes progress feedback; the smaller execution module runs selected work.
Arrows inside execution show model operation order and recurrent repetition, not
a requirement to execute every operation in each submission. Edit the SVG and regenerate its PNG export together.

## Stage structure

[stage-components.svg](stage-components.svg) is the editable 2400 × 1840 source;
`stage-components.png` is its 2× export. It is embedded in the single
[architecture guide](../unified_execution.md) and available in `index.html`.
It expands the approved architecture overview with the caller/worker process
boundary, worker-driven service loop, engine/scheduler/executor calls, ordinary
and speculative dispatch, shared resources and optional connector. Calls,
returns and shared-reference/lifecycle lines are distinct. Model operation labels
do not imply that every submission executes all three operations.

## Scheduler overview

`scheduler-overview.svg` is the editable 1800 × 2600 source and
`scheduler-overview.png` its 2× export. The vertical A–F spine shows admission,
phase queues, policy/batch selection, execution, completion tracking and progress.
Read top to bottom; one return arrow on the left leads from F to B. A includes waiting, safe preemption and
restoration; KV growth pressure in C returns to the same preemption logic. C
separates prefill granularity from batching and shows supported combinations.
Thin connected branches show PD import/export, backpressure and safe retirement. They attach directly to A/C/F rather than using
detached letter badges. A pale D–F bypass distinguishes logical advance from host result
collection. The loop-scheduler figure expands the Mixed policy's execution cycle.
Both figures are embedded in [`../scheduler.md`](../scheduler.md).

## Loop-level scheduler

`loop-scheduler.svg` is the editable 1800 × 1320 presentation source;
`loop-scheduler.png` is its 2× browser-rendered export, embedded in the mixed-loop detail section of
[`../scheduler.md`](../scheduler.md). It follows three requests through loop
selection, mixed batching, execution, progress feedback and re-selection.
Loop labels are one-based for readers; code depths are zero-based. The next
recurrent batch is illustrative, not a promise that CODA/PRELUDE cannot intervene.

## Scheduler close-up views

These editable SVGs and their 2× PNG exports are embedded in `../scheduler.md`
and available as individual tabs in `index.html`.

| Source | Canvas | Focus |
| --- | --- | --- |
| [scheduler-preemption-detail.svg](scheduler-preemption-detail.svg) | 1800 × 1260 | Six data lifetimes across suspension and resumption |
| [scheduler-victim-selection.svg](scheduler-victim-selection.svg) | 2400 × 3160 | Complete preemption policy: two triggers, safety filtering, ranking, retry, restoration, cancellation and error exits |
| [scheduler-async-detail.svg](scheduler-async-detail.svg) | 2300 × 3160 | Full-token forward with explicit buffer waits; multi-request loop batching and batch pipelining; delayed-exit dependencies |
| [scheduler-stage-handoff-detail.svg](scheduler-stage-handoff-detail.svg) | 1900 × 1220 | Visual KV/hidden handoff: reserve, transfer, admit and safely release source storage |

Timing and outcomes are illustrative. Preserve safety conditions and distinguish
request progress, device completion and remote-access lifetime when editing.

## Synchronous versus asynchronous speculation

[async-speculative-timeline.svg](async-speculative-timeline.svg) is the editable
2700 × 1620 source; `async-speculative-timeline.png` is its 2× raster export.
It is embedded in [the speculation guide](../async-speculative.md#timeline-synchronous-versus-asynchronous-speculation)
and available in `index.html`. Both panels follow one request through two rounds
on the same illustrative time scale, with CPU, `compute_stream` and `copy_stream`
lanes. Purple blocks match CPU/GPU acceptance; blue Commit blocks show CPU output
and progress updates. The T guide aligns synchronous Round 1 completion with the
asynchronous timeline; an arrow marks the earlier asynchronous Round 1 commit.
The example assumes slightly shorter GPU acceptance, a small first-commit gap,
and accumulating overlap across rounds. These durations are not measurements.
CPU wait endpoints align with their result-copy completion. Later submissions and
buffer reuse are omitted. R0–R3 denote recurrent depths; Round 0 / 1
denote speculative iterations.

Start/end markers delimit CPU submission through CPU Commit (including async
ticket retirement); GPU/copy completion events remain separate. The bottom legend
defines these boundaries and notes the illustrative timing assumptions.
