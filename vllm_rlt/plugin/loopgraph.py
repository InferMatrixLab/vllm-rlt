"""CUDA graphs for loop ops, built on vLLM's own graph manager.

vLLM captures the whole model for a decode query length or a token count. A loop
step runs one op, so the plugin keeps one ``ModelCudaGraphManager`` per
(op, depth), each capturing ``model(...)`` with that op's loop context set, and
swaps it in for the step (``runner.cudagraph_manager``). Dispatch, padding, FULL
replay and the DP shape agreement stay vLLM's.

What a replay cannot do is run Python, so the state that flows between ops goes
through static buffers sized for the largest captured batch:

* ``inp[(op, depth)]``: the carried state the op reads. The runner copies the
  step's merged state into its first ``n`` tokens before the step.
* ``out[(op, depth)]``: a ``Sink`` the model writes the state after the op into,
  inside the graph. The runner reads its first ``n`` tokens after the step.

A graph for ``T`` tokens reads and writes views of the first ``T`` tokens of
those buffers, so the runner never needs to know which ``T`` vLLM picked.
Capture runs in program order, so an op's input buffer takes its shape from the
previous op's output.

Only uniform decode shapes are captured (``FULL_DECODE_ONLY``); anything else
runs eagerly, as in vLLM. A mixed-depth step never replays a graph.
"""

from __future__ import annotations

import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_rlt.plugin.context import LoopContext, Sink, state_copy_, state_like, state_view
from vllm_rlt.plugin.program import Op


class CaptureContext(LoopContext):
    """The context a graph is captured with: state and sink are static views."""

    def __init__(self, op: Op, depth: int, store: GraphStore, passes=None) -> None:
        super().__init__(op, depth, None, passes=passes)
        self.store = store

    def state_for(self, tokens: int):
        buf = self.store.inp.get((self.op, self.depth))
        return None if buf is None else state_view(buf, tokens)

    def emit(self, tokens: int, out) -> None:
        self.out = out
        self.store.sink(self.op, self.depth).write(out, tokens)


class GraphStore:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.inp: dict[tuple[Op, int], object] = {}
        self.out: dict[tuple[Op, int], Sink] = {}

    def sink(self, op: Op, depth: int) -> Sink:
        return self.out.setdefault((op, depth), Sink(self.max_tokens))

    def prepare_input(self, op: Op, depth: int, prev: tuple[Op, int] | None) -> None:
        """Allocate the input buffer of (op, depth) from the previous op's
        output shape. The first op of a token has no carried state."""
        if prev is None or (op, depth) in self.inp:
            return
        src = self.out.get(prev)
        if src is not None and src.state is not None:
            self.inp[(op, depth)] = state_like(src.state, self.max_tokens)


class LoopGraphs:
    """Per-(op, depth) managers, their static buffers, and the step hooks."""

    def __init__(self, runner) -> None:
        self.runner = runner
        self.managers: dict[tuple[Op, int], object] = {}
        self.store: GraphStore | None = None
        self.active: tuple[Op, int] | None = None
        self._saved = None
        self.replayed = 0

    @property
    def ready(self) -> bool:
        return bool(self.managers)

    def program(self) -> list[tuple[Op, int]]:
        layout = self.runner.loop_layout
        ops = [(Op.PRELUDE, 0)] if layout.has_prelude else []
        ops += [(Op.CORE, d) for d in range(layout.num_passes)]
        return [*ops, (Op.CODA, 0)]

    # -- capture -----------------------------------------------------------------

    def capture(self) -> int:
        """Capture every op's decode graphs. Returns bytes used (0 if skipped)."""
        runner = self.runner
        primary = runner.cudagraph_manager
        if primary is None or primary.cudagraph_mode.decode_mode() != CUDAGraphMode.FULL:
            return 0  # loop graphs are FULL decode graphs; nothing to capture otherwise
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

        free_before = torch.accelerator.get_memory_info()[0]
        sizes = [d.num_tokens for descs in primary._capture_descs.values() for d in descs]
        self.store = GraphStore(max(sizes, default=runner.max_num_reqs))
        model = runner.get_model()
        prev = None
        for op, depth in self.program():
            manager = ModelCudaGraphManager(
                runner.vllm_config,
                runner.device,
                CUDAGraphMode.FULL_DECODE_ONLY,
                decode_query_lens=runner.vllm_config.get_decode_cudagraph_query_lens(
                    runner.decode_query_len
                ),
                lora_capture_cases=runner.lora_capture_cases,
            )
            self.store.prepare_input(op, depth, prev)
            model.loop_ctx = CaptureContext(op, depth, self.store)
            try:
                manager.capture(
                    model,
                    runner.model_state,
                    runner.input_buffers,
                    runner.intermediate_tensors,
                    runner.block_tables,
                    runner.attn_groups,
                    runner.kv_cache_config,
                    use_aux_hidden_state_outputs=False,
                    progress_bar_desc=f"Capturing loop graphs {op.name}/{depth}",
                )
            finally:
                model.loop_ctx = None
            manager._graphs_captured = True
            self.managers[(op, depth)] = manager
            prev = (op, depth)
        return max(0, free_before - torch.accelerator.get_memory_info()[0])

    # -- per step ----------------------------------------------------------------

    def activate(self, ctx: LoopContext) -> bool:
        """Swap in the manager of this step's op. False: no graph for it."""
        # CODA's depth is the depth the token left at; the op itself is the same.
        key = (ctx.op, 0 if ctx.op == Op.CODA else ctx.depth)
        manager = None if ctx.mixed else self.managers.get(key)
        if manager is None:
            return False
        self._saved = self.runner.cudagraph_manager
        self.runner.cudagraph_manager = manager
        self.active = key
        return True

    def deactivate(self) -> None:
        if self._saved is not None:
            self.runner.cudagraph_manager = self._saved
        self._saved, self.active = None, None

    def stage(self, ctx: LoopContext, tokens: int) -> None:
        """Copy the step's merged input state into the op's static buffer, so a
        graph replay reads it. Harmless when the step then runs eagerly."""
        key = self.active
        if key is None or ctx.state is None or self.store is None:
            return
        buf = self.store.inp.get(key)
        if buf is not None:
            state_copy_(buf, ctx.state, tokens)

    def output(self, ctx: LoopContext, tokens: int):
        """The state after a replayed step (the model did not run Python)."""
        from vllm_rlt.plugin.context import state_slice

        if self.store is None or self.active is None:
            return None
        sink = self.store.out.get(self.active)
        if sink is None or sink.state is None:
            return None
        self.replayed += 1
        return state_slice(sink.state, 0, tokens)
