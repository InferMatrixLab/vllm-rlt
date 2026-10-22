# Training rollout and full-weight publication

`LLM`, `StageClient`, and `LLMEngine` (the local `StageEngine` alias) expose the same full-update operations. `LLM` also accepts
an existing `LLMEngine` or `PDEngine`, so framework adapters use the same generation
and publication calls with either backend.

```python
llm.start_weight_update(version=1)
for chunk in converted_hf_weights:
    llm.update_weights(chunk)  # Iterable[(physical_parameter_name, torch.Tensor)]
llm.finish_weight_update()
outputs = llm.generate(prompts, SamplingParams(
    max_tokens=128, temperature=1.0, seed=42, logprobs=0,
))
```

Each output contains aligned selected-token `logprobs`, `exit_depths`, the
committed `weight_version`, and its effective `sampling_params`. `logprobs=None`
keeps the ordinary inference path. `logprobs=0` requests only the emitted token's
log-probability; top-N alternatives are not implemented.

`logprobs_mode` is an engine option of `LLM`, `StageClient`, `LLMEngine` and
`PDEngine`, not a request field. `"raw_logprobs"` is the default: FP32 log-softmax
of unscaled model logits. `"processed_logprobs"` scores the distribution the token
was drawn from, after positive temperature and top-k/top-p filtering. Greedy
requests return the raw score in either mode. Speculative outputs use the
verified **target** policy's score, including correction/bonus tokens; proposal or
rejection-residual probabilities are never exposed as the policy score. Explicit
`stop_token_ids` apply even with `ignore_eos=True`. `seed=None` resolves an
independent seed in `[0, 2**63)` when the engine admits the request. The
effective seed is returned in `sampling_params`, including for an abort before
sampling. The default remains `seed=0`. [Rollout outputs](launching.md#rollout-outputs)
states when replaying those parameters reproduces tokens and logprobs.

Publication requires an idle engine. Each chunk is checked for duplicate names,
unknown names and mismatched shapes before that chunk is copied. `finish_weight_update`
requires every physical parameter exactly once. Copies preserve parameter addresses
and captured graphs; a fresh transaction invalidates all cached prefix KV. If a
transfer fails, generation stays blocked until a fresh full transaction succeeds.
Starting again discards an unfinished transaction. `pause_generation` stops
admission while existing requests drain; `resume_generation` requires publication
to be complete. `abort_request` remains available to cancel requests before an update.

PDEngine waits for all transfers to release, broadcasts CPU-staged weight chunks
to every P/D worker, and commits a version only after every peer acknowledges it.
Recovery uses a version newer than **all attempted** PD versions because some
workers may already have committed. With `terminate_workers_on_failure=False`, fatal workers retain their registered
memory rather than receive forced process signals; normal `close()` remains
cooperative. No CUDA tensors change ownership over IPC;
NIXL's existing GPU KV path and registrations remain intact. Prefix invalidation
does not replace registered KV storage. PD worker control errors return replies
without terminating the worker.

Existing exit, async and speculative configuration rules still apply. Self-speculation
uses a full-depth target. Selected-token scores are supported by synchronous and
asynchronous decoding and by synchronous and cross-round asynchronous
self-speculation; the latter is greedy-only, so it reports raw scores in both
modes. PD follows the same rules on its decode workers. Publication
adds no such restriction to ordinary early-exit or async rollouts. Trainers must
replay full-depth prompt prefill and the returned decode depths rather than assume
the entire sequence used K.
