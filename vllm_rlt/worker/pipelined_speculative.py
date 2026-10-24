"""Experimental standalone pipelined self-speculative execution prototype.

Executes concurrent pipeline lanes across different recurrent depths in a single
continuous depth batch (CDB), reusing state bank slots and managing KV cache.

Note:
    This module is an experimental standalone prototype and is not yet integrated
    into StageEngine, ModelRunner, the scheduler, or the serving CLI.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.worker.speculative_decisions import (
    GreedyDecisionPolicy,
    StateSlotPool,
)
from vllm_rlt.worker.speculative_pipeline import (
    PipelineLane,
    PipelineResult,
    PipelineSchedule,
    continue_pipelined,
)


@dataclass
class PipelinedSpeculativeStats:
    cycles: int = 0
    committed_tokens: int = 0
    accepted_proposals: int = 0
    rejected_proposals: int = 0


class PipelinedSpeculativeRunner:
    """Runs pipelined self-speculative decoding on a recurrent model instance.

    Experimental standalone prototype with the following current limitations:
      - Standalone prototype: not wired into StageEngine, ModelRunner, KV leases/pins,
        or continuous batching.
      - Single-request: handles a single request_id per generate() call.
      - Greedy only: supports argmax decoding without sampling, temperature, or RNG.
      - No default EOS: stop conditions depend entirely on caller-provided stop_tokens.
      - No chunked prefill: prefill processes the prompt in full.
      - Architecture: Ouro-class models exposing prelude/recurrent_prepared/coda only;
        Huginn/Raven models with *_prepared variants are not yet supported.
      - Schedules: single-proposal schedules only (second_step=None); multi-proposal
        schedules (second_step != None) raise NotImplementedError.
    """

    def __init__(
        self,
        model: Any,
        cache: KVCacheManager,
        schedule: PipelineSchedule,
        stop_tokens: set[int] | None = None,
    ) -> None:
        if schedule.second_step is not None:
            raise NotImplementedError(
                f"second_step={schedule.second_step} (q2) is not supported with physical "
                "KV cache execution due to duplicate write collisions across sibling lanes. "
                "Set second_step=None."
            )

        total_ut_steps = getattr(model.config, "total_ut_steps", None)
        if callable(total_ut_steps):
            total_ut_steps = total_ut_steps()
        if total_ut_steps is not None and schedule.stages != total_ut_steps:
            raise ValueError(
                f"schedule.stages ({schedule.stages}) must match "
                f"model.config.total_ut_steps ({total_ut_steps})"
            )

        required_methods = ("prelude", "recurrent_prepared", "coda")
        missing_methods = [
            m for m in required_methods if not hasattr(model, m) or not callable(getattr(model, m))
        ]
        if missing_methods:
            raise NotImplementedError(
                f"model of type {type(model).__name__} is missing required method(s): "
                f"{', '.join(missing_methods)}. PipelinedSpeculativeRunner requires models with "
                "'prelude', 'recurrent_prepared', and 'coda' (e.g. OuroForCausalLM). "
                "Huginn/Raven models use *_prepared variants and are not yet supported."
            )

        self.model = model
        self.cache = cache
        self.schedule = schedule
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.stop_tokens = stop_tokens or set()

        self.slot_pool = StateSlotPool(schedule.max_rows)
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None and hasattr(model.config, "d_model"):
            hidden_size = model.config.d_model
        if hidden_size is None:
            raise ValueError("model config missing hidden_size or d_model")

        self.state_bank = torch.zeros(
            schedule.max_rows,
            hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.decisions = GreedyDecisionPolicy(schedule.max_rows, self.device)
        self.stats = PipelinedSpeculativeStats()

        self.request_id: str = ""
        self.prompt_len: int = 0

    def prefill(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        max_total_tokens: int = 2048,
    ) -> int:
        """Run dense prefill for the prompt and return the first decoded token."""
        self.request_id = request_id
        self.prompt_len = len(prompt_token_ids)
        self.cache.allocate(request_id, max_total_tokens)

        tokens = torch.tensor(prompt_token_ids, device=self.device, dtype=torch.long)
        hidden = self.model.prelude(tokens)

        positions = list(range(self.prompt_len))
        ids = [request_id] * self.prompt_len

        # Execute all recurrent loops for the prompt
        for depth in range(self.schedule.stages):
            batch = self.cache._prepare_batch(ids, [depth] * self.prompt_len, positions)
            hidden, _ = self.model.recurrent_prepared(hidden, batch, self.cache, compute_gate=False)

        last_hidden = hidden[-1:]
        logits = self.model.coda(last_hidden)
        first_token = int(logits.argmax(-1).item())
        return first_token

    def forward(self, lanes: Sequence[PipelineLane]) -> list[PipelineResult]:
        """Execute one forward cycle advancing all live lanes by schedule.step_size iterations."""
        self.stats.cycles += 1
        stride = self.schedule.step_size

        # 1. Initialize newly spawned lanes at stage 0
        new_lanes = [lane for lane in lanes if lane.stage == 0 and lane.state is None]
        if new_lanes:
            new_tokens = torch.tensor(
                [lane.token for lane in new_lanes],
                device=self.device,
                dtype=torch.long,
            )
            new_hidden = self.model.prelude(new_tokens)
            for i, lane in enumerate(new_lanes):
                slot = self.slot_pool.acquire()
                lane.state = slot
                self.state_bank[slot] = new_hidden[i]

        # 2. Gather hidden states for all active lanes
        active_slots = [lane.state for lane in lanes]
        hidden = self.state_bank[active_slots]

        # 3. Advance recurrent core by stride steps
        for step_offset in range(stride):
            depths = [lane.stage + step_offset for lane in lanes]
            positions = [self.prompt_len + lane.index - 1 for lane in lanes]
            ids = [self.request_id for _ in lanes]

            batch = self.cache._prepare_batch(ids, depths, positions)
            hidden, _ = self.model.recurrent_prepared(hidden, batch, self.cache, compute_gate=False)

        # 4. Write back updated hidden states to the state bank
        for i, lane in enumerate(lanes):
            self.state_bank[lane.state] = hidden[i]

        # 5. Identify lanes reaching readout endpoints (q1, q2, final)
        endpoints: dict[int, str] = {}
        for i, lane in enumerate(lanes):
            end_stage = lane.stage + stride - 1
            if end_stage in self.schedule.endpoint_kinds:
                endpoints[i] = self.schedule.endpoint_kinds[end_stage]

        read_decisions: dict[int, list[int]] = {}
        if endpoints:
            endpoint_indices = list(endpoints.keys())
            endpoint_hidden = hidden[endpoint_indices]
            logits = self.model.coda(endpoint_hidden)

            endpoint_mapping = {
                idx: endpoints[orig_i] for idx, orig_i in enumerate(endpoint_indices)
            }
            sub_lanes = [lanes[orig_i] for orig_i in endpoint_indices]

            decision_batch = self.decisions.queue(logits, endpoint_mapping, sub_lanes)
            sub_decisions = self.decisions.read(sub_lanes, endpoint_mapping, decision_batch)
            for idx, orig_i in enumerate(endpoint_indices):
                read_decisions[orig_i] = sub_decisions[idx]

        # 6. Build results for each lane
        results: list[PipelineResult] = []
        for i, lane in enumerate(lanes):
            if i in endpoints:
                outcome = self.decisions.resolve(
                    lane,
                    endpoints[i],
                    read_decisions[i],
                    None,
                    self.schedule,
                )
                if endpoints[i] == "final":
                    if outcome.accepted_stage is not None:
                        self.stats.accepted_proposals += 1
                    else:
                        self.stats.rejected_proposals += 1

                results.append(
                    PipelineResult(
                        state=lane.state,
                        prediction=outcome.prediction,
                        proposed=outcome.proposed,
                        accepted_stage=outcome.accepted_stage,
                        proposal_action=outcome.proposal_action,
                    )
                )
            else:
                results.append(
                    PipelineResult(
                        state=lane.state,
                        prediction=0,
                        proposed=False,
                    )
                )

        return results

    def retain_branches(
        self,
        lanes: Sequence[PipelineLane],
        accepted_stage: int | None = None,
    ) -> None:
        """Prune discarded state slots from the state bank.

        Note:
            Uncommitted KV cache suffix is not rolled back here per cycle.
            KV rollback/truncation to the committed frontier is performed at
            the end of generate() via cache.truncate_suffix().
        """
        live_slots = [lane.state for lane in lanes if lane.state is not None]
        self.slot_pool.retain(live_slots)

    def should_stop(self, tokens: Sequence[int]) -> bool:
        """Check stopping conditions (e.g. EOS token)."""
        return bool(tokens and tokens[-1] in self.stop_tokens)

    def generate(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int,
    ) -> list[int]:
        """Execute complete pipelined self-speculative generation."""
        max_total = len(prompt_token_ids) + max_new_tokens + 16
        first_token = self.prefill(request_id, prompt_token_ids, max_total_tokens=max_total)
        if self.should_stop([first_token]):
            self.stats.committed_tokens += 1
            self.retain_branches(())
            self.cache.truncate_suffix(request_id, self.prompt_len)
            return [first_token]

        committed = continue_pipelined(
            first_token=first_token,
            max_new_tokens=max_new_tokens,
            executor=self,
        )
        self.stats.committed_tokens += len(committed)

        # Truncate any speculative KV cache written beyond the committed frontier
        final_frontier = self.prompt_len + len(committed) - 1
        self.cache.truncate_suffix(request_id, final_frontier)
        return committed
