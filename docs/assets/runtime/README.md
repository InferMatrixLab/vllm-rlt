# Original README animation

`rlt-loop-pipelines.gif` and `rlt-loop-pipelines.png` are original code-drawn assets for
vLLM-RLT. No source artwork, logos, stock imagery, or external fonts are bundled.
The four design colors are white, navy, blue, and yellow (antialiasing adds shades).

## Grounding

Implementation/documentation baseline: v0.2.0,
`c08b9d79834bb8b4ff283b15bae2c14c559f6b52`.

- [Stage/loop ownership and shared runtime](https://github.com/InferMatrixLab/vllm-rlt/blob/c08b9d79834bb8b4ff283b15bae2c14c559f6b52/docs/unified_execution.md#inside-one-stage)
- [Wavefront chunked prefill](https://github.com/InferMatrixLab/vllm-rlt/blob/c08b9d79834bb8b4ff283b15bae2c14c559f6b52/docs/runtime.md#wavefront-chunked-prefill)
- [Experimental pipelined self-speculation](https://github.com/InferMatrixLab/vllm-rlt/blob/c08b9d79834bb8b4ff283b15bae2c14c559f6b52/docs/async-speculative.md#pipelined-self-speculation)

The README layout reference is vLLM-Omni's current README (read November 1,
2026): centered header/navigation, About overview, architecture illustration,
Getting Started, contributor/community sections. Its current overview asset is
static, not an existing GIF to copy.

## Reading the diagram

The outer Stage boundary owns deployment/resources. StageEngine coordinates
Scheduler selection/progress and Executor submission; arrows are a simplified
work-flow, not a direct Scheduler-to-runner call graph. Shared model runtime and
state labels summarize resources, not additional exclusive owners.

The two phases are separate execution examples, NOT a claim that speculative
verification joins WCPB mixed recurrent batches. Four chunks/inputs and four
loops illustrate state transitions, not measured time, throughput or capacity.
Yellow marks current work; checks mark earlier completed recurrence. WCPB advances
ready chunks at different depths. Decode shows the all-accepted speculative path:
processing input x_i ultimately yields target output x_(i+1). Hidden states survive
submissions. Rejection discards dependent tokens; no quality equivalence to block
mode is asserted. Block is still the default. Pipeline is experimental and requires
greedy sampling and LAST_EXITED KV; backend/model configuration limits remain in
the linked documentation. Neither PD topology nor cross-host support is depicted.

## Reproduction

Requires Python 3 and Pillow, with DejaVu Sans TTF available at the standard Linux
paths used by `generate_pipelines.py` (adjust FONT/BOLD constants on other platforms).

```sh
python docs/assets/runtime/generate_pipelines.py --output-dir docs/assets
```

Outputs: animated GIF, static WCPB poster, and a decode QA frame. The renderer
uses no network or GPU. GIF frame coalescing is intentional: repeated frames are
stored as longer durations to keep the README asset small. The loop is 20 seconds.
Validation: load every GIF frame with Pillow, decode with FFmpeg, visually inspect
both phase posters, check duration/dimensions/infinite-loop metadata.
