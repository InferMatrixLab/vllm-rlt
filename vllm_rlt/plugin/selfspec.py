"""Self-speculation: a loop model drafts with fewer passes and verifies with all.

This is vLLM's own speculative decoding with a custom speculator
(``register_speculative_method``). Everything around a draft is vLLM's:

* the scheduler opens a verify span of K+1 tokens (the last sampled token plus K
  drafts), allocates its KV, and rolls ``num_computed_tokens`` and the output
  placeholders back after a rejection (also under async scheduling);
* the model runner runs the span through the model at full depth and verifies it
  with vLLM's rejection sampler, then calls ``propose``;
* ``propose`` (here) produces the next K drafts by running the same model for
  ``draft_passes`` core passes, one token at a time. The draft writes KV for
  the shallow passes of the draft positions; the verify then rewrites those
  entries with the same values and fills the deeper passes.

Which checkpoints: Ouro (any pass count works, as the reference's early exit
shows) and Inkling *VarLoop* (``is_varloop`` true: trained to run any 1..N
passes, which is what makes a shallow draft usable). A Fixed Inkling checkpoint
was only trained at N passes, so its shallow output is not a draft; it is
refused.

Use::

    --speculative-config '{"method":"rlt_self","num_speculative_tokens":4,
                           "speculator_options":{"draft_passes":2}}'

Greedy drafting (draft = argmax), verified with vLLM's rejection sampler, so the
output distribution is the target's. Not run on a GPU.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig

METHOD = "rlt_self"
SPECULATOR = "vllm_rlt.plugin.selfspec.LoopSpeculator"


class LoopSelfSpecConfig(SpeculatorConfig):
    base_method = None  # a custom method, not based on a built-in one
    speculator_class = SPECULATOR
    supported_options = frozenset({"speculator_options"})

    @classmethod
    def configure(cls, spec) -> None:
        if spec.num_speculative_tokens is None:
            raise ValueError("rlt_self needs num_speculative_tokens")
        super().configure(spec)  # no draft checkpoint: the target is the draft

    @property
    def num_lookahead_tokens(self) -> int:
        # The draft writes KV up to K positions past the span vLLM allocated for.
        return self.num_speculative_tokens


def register_self_spec() -> None:
    from vllm.v1.worker.gpu.spec_decode.speculator_config import (
        register_speculative_method,
    )

    register_speculative_method(METHOD, LoopSelfSpecConfig)


class LoopSpeculator(BaseSpeculator):
    num_query_per_req = 1
    enable_adaptive_verification = False

    def __init__(self, vllm_config, device: torch.device) -> None:
        spec = vllm_config.speculative_config
        self.vllm_config, self.device = vllm_config, device
        self.num_speculative_steps = spec.num_speculative_tokens
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        options = spec.speculator_options or {}
        extra = vllm_config.additional_config
        extra = extra if isinstance(extra, dict) else {}
        self.requested_passes = int(options.get("draft_passes", extra.get("rlt_draft_passes", 0)))
        self.draft_passes = self.requested_passes
        self.runner = None
        self.draft_tokens = torch.zeros(
            self.max_num_reqs, self.num_speculative_steps, dtype=torch.int64, device=device
        )

    def bind(self, runner) -> None:
        """Called by the loop runner once it exists (the speculator is built
        inside vLLM's runner constructor, before the runner is usable)."""
        layout = runner.loop_layout
        if layout.num_passes < 2:
            raise ValueError("self-speculation needs a model that runs at least 2 passes")
        if not layout.has_gate and not layout.varloop:
            raise ValueError(
                "self-speculation needs an Ouro model or a VarLoop Inkling checkpoint "
                "(loop_config.is_varloop=true); a Fixed checkpoint only works at its "
                "own pass count, so a shallow pass is not a usable draft"
            )
        passes = self.requested_passes or max(1, layout.num_passes // 2)
        if not 1 <= passes < layout.num_passes:
            raise ValueError(f"draft_passes must be in [1, {layout.num_passes - 1}], got {passes}")
        self.draft_passes, self.runner = passes, runner

    # -- BaseSpeculator interface ----------------------------------------------

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        return  # draft steps run eagerly (see docs/vllm-plugin.md)

    def capture(self) -> None:
        return

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs=None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        n = input_batch.num_reqs
        k = self.num_speculative_steps if num_speculative_tokens is None else num_speculative_tokens
        out = self.draft_tokens[:n]
        out.zero_()
        if dummy_run or k == 0 or self.runner is None:
            return out
        runner = self.runner
        idx = input_batch.idx_mapping
        # After postprocess_sampled: ``last_sampled`` is the token sampled this
        # round and ``num_computed_tokens`` the position it will occupy.
        tokens = last_sampled[idx].reshape(-1).to(torch.int64)
        start = runner.req_states.num_computed_tokens.gpu[idx].to(torch.int64)
        limit = runner.max_model_len - 1
        for step in range(k):
            positions = torch.clamp(start + step, max=limit)
            logits = self._draft_logits(input_batch, idx, tokens, positions, step)
            tokens = logits.argmax(dim=-1)
            out[:, step] = tokens
        # A row that sampled nothing (a prompt chunk) has no draft to verify.
        out[num_sampled[:n] == 0] = -1
        return out

    # -- one draft step ---------------------------------------------------------

    def _draft_logits(self, input_batch, idx, tokens, positions, step) -> torch.Tensor:
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.v1.worker.gpu.attn_utils import (
            build_attn_metadata,
            build_slot_mappings_by_layer,
        )

        runner = self.runner
        n = idx.shape[0]
        qsl = torch.arange(n + 1, dtype=torch.int32, device=self.device)
        qsl_cpu = torch.arange(n + 1, dtype=torch.int32)
        seq_lens = (positions + 1).to(torch.int32)
        block_tables = runner.block_tables.gather_block_tables(
            idx,
            num_reqs_padded=n,
            num_computed_tokens=runner.req_states.num_computed_tokens.gpu,
        )
        slots = runner.block_tables.compute_slot_mappings(idx, qsl, positions, num_tokens_padded=n)
        upper = input_batch.seq_lens_cpu_upper_bound[:n] + 1 + step
        metadata = build_attn_metadata(
            attn_groups=runner.attn_groups,
            num_reqs=n,
            num_tokens=n,
            query_start_loc_gpu=qsl,
            query_start_loc_cpu=qsl_cpu,
            max_query_len=1,
            seq_lens=seq_lens,
            max_seq_len=runner.max_model_len,
            block_tables=block_tables,
            slot_mappings=slots,
            kv_cache_config=runner.kv_cache_config,
            seq_lens_cpu_upper_bound=upper,
            positions=positions,
        )
        model = runner.get_model()
        with set_forward_context(
            metadata,
            self.vllm_config,
            num_tokens=n,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=build_slot_mappings_by_layer(slots, runner.kv_cache_config),
        ):
            hidden = model.draft_hidden(tokens, positions, self.draft_passes)
        return model.compute_logits(hidden)
