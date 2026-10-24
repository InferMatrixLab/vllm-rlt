"""Rollout logprobs: the FP32 log-probability of each sampled token.

Scoring consumes no RNG. Raw logprobs score the model distribution. Processed
logprobs score ``Sampler.processed_logits``, the logits ``Sampler.probabilities``
normalizes, so they score the distribution the token was actually drawn from.
This module imports ``worker/sampler.py``, never the reverse.
"""

from collections import defaultdict

import torch

from vllm_rlt.worker.sampler import Sampler

LOGPROBS_MODES = ("raw_logprobs", "processed_logprobs")


def check_logprobs_mode(mode):
    if mode not in LOGPROBS_MODES:
        raise ValueError(
            f"logprobs_mode must be one of {LOGPROBS_MODES}, got {mode!r}; "
            "raw_logits and processed_logits are not supported"
        )


def token_logprob(logits, params, token_id, mode):
    """0-dim FP32 log-probability of ``token_id`` (0-dim long tensor) for one logits row.

    raw_logprobs uses the model distribution. processed_logprobs uses the
    distribution actually sampled from; greedy rows have no temperature (vLLM).
    Consumes no RNG and keeps the result on the logits device.
    """
    if mode == "processed_logprobs" and params.temperature != 0:
        logits = Sampler.processed_logits(logits, params)
    return Sampler.raw_logprobs(logits[None], token_id.view(1))[0]


def truncates(params):
    """Whether this row is drawn from top-k/top-p truncated logits (see ``processed_logits``).

    Greedy rows never are: ``argmax`` ignores top-k and top-p.
    """
    return params.temperature != 0 and (params.top_k > 0 or params.top_p < 1)


def scaled_logprobs(logits, token_ids, temperature):
    """``raw_logprobs`` of ``logits / temperature``, divided as ``processed_logits`` divides."""
    scaled = logits if temperature == 1 else logits.float() / temperature
    return Sampler.raw_logprobs(scaled, token_ids)


def processed_logprobs(logits, params, token_ids, truncated):
    """FP32 [rows] processed logprobs of ``token_ids``, equal to per-row ``token_logprob``.

    ``logits`` [rows, vocab] are the CODA rows of ``params`` (one per row), at least one of
    which requests logprobs. ``truncated`` maps every requesting row that ``truncates`` to
    the 0-dim logprob ``Sampler.sample_with_logprob`` computed from the processed logits it
    drew from. Other requesting rows are grouped by temperature: greedy and T == 1 rows are
    scored by a raw ``log_softmax`` (``x / 1.0 == x``), each other temperature by
    ``log_softmax(logits.float() / T)`` with the Python scalar T, as ``processed_logits``
    divides. A group is scored in one pass over its row span ``[first, last]`` when that span
    is at most twice its rows, else row by row, so the scored rows never exceed twice the
    requesting rows, however many temperatures the batch mixes. The only group, holding at
    least half the rows, is scored over the whole batch and returned as is; otherwise one
    stack of 0-dim views assembles the rows. No host sync, per-row device write or per-row
    device metadata (spans are host-int slices). Rows that did not request logprobs hold
    arbitrary values.
    """
    groups = defaultdict(list)  # Temperature (greedy: 1) -> requesting untruncated rows.
    for row, row_params in enumerate(params):
        if row_params.logprobs is not None and row not in truncated:
            groups[1 if row_params.temperature == 0 else row_params.temperature].append(row)
    values = dict(truncated)
    for temperature, rows in groups.items():
        if not truncated and len(groups) == 1 and len(params) <= 2 * len(rows):
            return scaled_logprobs(logits, token_ids, temperature)  # One pass, no assembly.
        first, end = rows[0], rows[-1] + 1
        if end - first > 2 * len(rows):  # Sparse group: a span pass would mostly score others.
            for row in rows:
                one = slice(row, row + 1)
                values[row] = scaled_logprobs(logits[one], token_ids[one], temperature)[0]
            continue
        scored = scaled_logprobs(logits[first:end], token_ids[first:end], temperature)
        values.update((row, scored[row - first]) for row in rows)
    filler = next(iter(values.values()))
    return torch.stack([values.get(row, filler) for row in range(len(params))])
