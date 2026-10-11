"""Readout-only fixed LoopCD for Ouro; original recurrent feedback is preserved."""

from dataclasses import dataclass

import torch

from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Request, Stage


@dataclass(frozen=True)
class Reference:
    owner: Request
    position: int
    output_index: int
    loop: int
    hidden: torch.Tensor

    def check(self, request, position, output_index, loop):
        if self.owner is not request or (self.position, self.output_index, self.loop) != (
            position,
            output_index,
            loop,
        ):
            raise RuntimeError("LoopCD reference owner/token/loop mismatch")


def active(params):
    return params.loopcd is not None and params.loopcd.strength != 0


def validate(model, params, execution, cache, scheduler, exit_config, speculative):
    if not active(params):
        return
    if type(model) is not OuroForCausalLM:
        raise ValueError("LoopCD currently supports Ouro only")
    if not execution.loopcd:
        raise ValueError("LoopCD requires ExecutionConfig(loopcd=True) for memory reservation")
    if (
        execution.async_scheduling
        or execution.prefill_uva
        or cache.layout != "last_exited"
        or cache.enable_prefix_caching
        or speculative is not None
    ):
        raise ValueError("LoopCD requires synchronous last_exited without prefix/P-D/speculation")
    if scheduler.enable_preemption and execution.cuda_graphs:
        raise ValueError("LoopCD preemption currently requires eager execution")
    maximum = params.max_loops or model.config.total_ut_steps
    prefill = execution.prefill_depth or model.config.total_ut_steps
    if params.min_loops != maximum or params.exit_threshold != 1 or exit_config.mode != "ouro":
        raise ValueError("LoopCD requires fixed decode depth and ouro exit mode")
    cd = params.loopcd
    if not (1 <= cd.reference_loop < maximum and 1 <= cd.prefill_loop < prefill):
        raise ValueError("LoopCD requires 1 <= reference_loop < actual prefill/decode depth")


def preemption_reference(request, stage, reference):
    """Validate only the reference that the suspended token still needs.

    A partial prompt chunk and a next-token prelude produce a fresh reference.
    After the decode reference depth, or at readout, exact restoration is needed.
    """
    if not active(request.sampling_params):
        return None
    cd = request.sampling_params.loopcd
    needed = stage == Stage.CODA or (
        stage == Stage.RECURRENT and request.loops_done >= cd.reference_loop
    )
    if not needed:
        return None
    if reference is None:
        raise RuntimeError("LoopCD reference missing at preemption")
    index = request.num_scheduled_outputs
    loop = cd.prefill_loop if index == 0 else cd.reference_loop
    reference.check(request, request.position, index, loop)
    return reference


def extrapolate(final, reference, strength):
    return final + strength * (final - reference)


def reserved_bytes(config, scheduler, execution, element_size):
    if not execution.loopcd:
        return 0
    rows = scheduler.max_num_batched_tokens
    states = (scheduler.max_num_seqs + 3 * rows) * config.hidden_size * element_size
    # Two-head reference logits and FP32 combination coexist with normal logits.
    return states + 3 * rows * config.vocab_size * 4
