"""Copy the live rows of a merged CORE to its two continuation layouts."""

from dataclasses import dataclass

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d


@dataclass(slots=True)
class CoreRoutes:
    spans: tuple[list, list]
    lengths: tuple[int, int]
    offsets: torch.Tensor


def prepare_routes(spans, device):
    requests = ([], [])
    lengths = [0, 0]
    offsets = []
    source = 0
    for span in spans:
        lane = int(span.last_core)
        destination = lengths[lane]
        offsets.extend((source, destination, lane))
        requests[lane].append(span)
        source += span.token_count
        lengths[lane] += span.token_count
    return CoreRoutes(
        requests,
        tuple(lengths),
        async_tensor_h2d(offsets, device=device, dtype=torch.int32),
    )


@triton.jit
def _route_live_rows(
    SourceStack,
    SourcePrelude,
    SourceTerminal,
    SourceContinuation,
    SeqIdx,
    Routes,
    ContinueStack,
    ContinuePrelude,
    ContinueInputs,
    Bank,
    TerminalStack,
    TerminalPrelude,
    TerminalInputs,
    actual: tl.int32,
    continuing: tl.int32,
    continue_width: tl.int32,
    terminal: tl.int32,
    H: tl.constexpr,
    stack_blocks: tl.constexpr,
    prelude_blocks: tl.constexpr,
    completed: tl.constexpr,
    terminal_queries: tl.constexpr,
    previous_queries: tl.constexpr,
    src_stack_stride: tl.constexpr,
    src_terminal_stride: tl.constexpr,
    src_continuation_stride: tl.constexpr,
    tile: tl.constexpr,
):
    x = tl.program_id(0) * tile + tl.arange(0, tile)
    field = tl.program_id(1)
    valid = x < actual * H
    row = x // H
    req = tl.load(SeqIdx + row, valid, other=0)
    start = tl.load(Routes + 3 * req, valid, other=0)
    offset = tl.load(Routes + 3 * req + 1, valid, other=0)
    final = tl.load(Routes + 3 * req + 2, valid, other=0) != 0
    dst = (row - start + offset) * H + x % H
    to_coda = valid & final
    to_core = valid & ~final
    pad_dst = x - actual * H + continuing * H
    pad_core = ~valid & (pad_dst < continue_width * H)

    if field <= completed:
        old = tl.load(SourceStack + field * src_stack_stride + x, valid, other=0)
        tl.store(TerminalStack + field * terminal * H + dst, old, to_coda)
        if field <= prelude_blocks:
            if field == prelude_blocks:
                old = tl.load(SourcePrelude + x, to_core, other=0)
            tl.store(ContinueStack + field * continue_width * H + dst, old, to_core)
            tl.store(ContinueStack + field * continue_width * H + pad_dst, 0, pad_core)
        else:
            bank_block = field - prelude_blocks - 1
            tl.store(Bank + bank_block * continue_width * H + dst, old, to_core)
            tl.store(Bank + bank_block * continue_width * H + pad_dst, 0, pad_core)
    elif field == stack_blocks:
        saved = tl.load(SourcePrelude + x, valid, other=0)
        tl.store(TerminalPrelude + dst, saved, to_coda)
        tl.store(ContinuePrelude + dst, saved, to_core)
        tl.store(ContinuePrelude + pad_dst, 0, pad_core)
    elif field > stack_blocks:
        query = field - stack_blocks - 1
        if query < terminal_queries:
            terminal_value = tl.load(
                SourceTerminal + query * src_terminal_stride + x, to_coda, other=0
            )
            tl.store(TerminalInputs + query * terminal * H + dst, terminal_value, to_coda)
        if previous_queries and query < previous_queries:
            continuation_value = tl.load(
                SourceContinuation + query * src_continuation_stride + x, to_core, other=0
            )
            tl.store(ContinueInputs + query * continue_width * H + dst, continuation_value, to_core)
            tl.store(ContinueInputs + query * continue_width * H + pad_dst, 0, pad_core)


def route_live_rows(state, blocks, inputs, routes, seq_idx, continue_width=None):
    continuing, terminal = routes.lengths
    continue_width = continue_width or continuing
    capacity, _, hidden = state.stack.shape
    previous_inputs = state.inputs if state.bank is not None else None
    previous_queries = previous_inputs.shape[0] if previous_inputs is not None else 0
    cont_stack = state.stack.new_empty((capacity, continue_width, hidden))
    term_stack = state.stack.new_empty((capacity, terminal, hidden))
    cont_prelude = state.prelude.new_empty((continue_width, hidden))
    term_prelude = state.prelude.new_empty((terminal, hidden))
    term_inputs = inputs.new_empty((inputs.shape[0], terminal, hidden))
    cont_inputs = (
        previous_inputs.new_empty((previous_queries, continue_width, hidden))
        if previous_inputs is not None
        else None
    )
    bank = state.stack.new_empty((blocks - state.prelude_blocks, continue_width, hidden))
    tile = 2048
    _route_live_rows[
        (
            triton.cdiv((continue_width + terminal) * hidden, tile),
            capacity + 1 + max(inputs.shape[0], previous_queries),
        )
    ](
        state.stack,
        state.prelude,
        inputs,
        previous_inputs if previous_inputs is not None else inputs,
        seq_idx,
        routes.offsets,
        cont_stack,
        cont_prelude,
        cont_inputs if cont_inputs is not None else term_inputs,
        bank,
        term_stack,
        term_prelude,
        term_inputs,
        continuing + terminal,
        continuing,
        continue_width,
        terminal,
        hidden,
        capacity,
        state.prelude_blocks,
        blocks,
        inputs.shape[0],
        previous_queries,
        state.stack.stride(0),
        inputs.stride(0),
        previous_inputs.stride(0) if previous_inputs is not None else 0,
        tile,
    )
    return cont_stack, cont_prelude, cont_inputs, bank, term_stack, term_prelude, term_inputs
