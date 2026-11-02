"""FlashInfer FA2 paged decode for Ada (SM89); eager, CUDA Graphs and async steps.

The adapter builds the CSR metadata FlashInfer's paged kernel wants once per
traversal (``prepare_step``), then reuses it across every decoder layer through
``__call__``.  Plans are keyed by ``(rows, heads, width bucket)``, where the
width is rounded up to a power of two, so a growing eager context creates at
most one plan per bucket instead of one per page boundary.

Split-KV is disabled: with it, a row's reduction order depends on the planned
width and on batch composition, so eager and graph execution (padded rows,
full-width graph plans) disagree bit for bit.  Without splits the schedule is
per-row, and eager, graphs and any batch composition produce identical output.
The cost is low-batch long-context throughput (fewer CTAs); at batch 8 the
kernel cost is unchanged.

Only FlashInfer's CUDA-Graph-compatible tensor-core FA2 wrapper is used.  The
CUDA-core decode wrapper bakes its work schedule (request/tile indices) into
plan-time state, so a plan built for full capacity and replayed with a short
live context stalls on SM89; the tensor-core path takes the schedule from the
live CSR buffers and masks unused splits, which is exactly the graph contract.
The cost is FlashInfer's documented caveat that it does not switch kernels per
batch size or sequence length.

Construction is device-only: the per-row page counts, the compact page index
list and the last-page lengths are all derived with fixed-shape kernels, so a
step with device-owned context lengths never reads back to the host.

The CSR staging buffers stay allocated for as long as their metadata tensors
are alive (weak references).  CUDA Graph capture records the copy from those
buffers into the wrapper's plan buffers, so replay refreshes the plan buffers
from whatever ``prepare_step`` wrote before the replay; no Python runs inside
the graph.

Only the L20/L40S/RTX-40 class is accepted: FlashAttention-2 rejects the page
size 16 used by Ouro's cache, and FlashInfer XQA (PR #112) rejects SM89.
"""

import weakref
from importlib.metadata import PackageNotFoundError, version

import torch

FLASHINFER_FA2_PAGE_SIZE = 16
FLASHINFER_FA2_HEAD_DIM = 128
FLASHINFER_FA2_GROUP_SIZES = (1, 2, 4, 8)
# Live metadata sets are tiny (graph entries plus the current eager step). The
# bound only guards a caller that keeps many distinct tensors alive.
FLASHINFER_FA2_CSR_CAPACITY = 64
# Each (rows, heads, width) plan owns a device workspace. Reuse across the
# engine's batch sizes; captured plans are pinned and never evicted.
FLASHINFER_FA2_PLAN_CAPACITY = 32


def _version_tuple(value):
    core = value.split("+", 1)[0].split(".post", 1)[0]
    parts = []
    for component in core.split(".")[:3]:
        digits = "".join(character for character in component if character.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


class _CSR:
    """CSR sources owned by one (block_tables, context_lengths) pair.

    Metadata is referenced weakly so an eager step can release it; a graph entry
    keeps its metadata alive for the runner's lifetime, which keeps these sources
    and the raw pointers recorded by its capture valid.
    """

    __slots__ = ("indptr", "indices", "last_page_len", "block_tables", "context_lengths")

    def __init__(self, block_tables, context_lengths):
        rows, width = block_tables.shape
        device = block_tables.device
        self.indptr = torch.empty(rows + 1, dtype=torch.int32, device=device)
        self.indices = torch.empty(rows * width, dtype=torch.int32, device=device)
        self.last_page_len = torch.empty(rows, dtype=torch.int32, device=device)
        self.block_tables = weakref.ref(block_tables)
        self.context_lengths = weakref.ref(context_lengths)

    def owns(self, block_tables, context_lengths):
        return self.block_tables() is block_tables and self.context_lengths() is context_lengths


class _Plan:
    """One planned wrapper with the buffers its captured replays read.

    The wrapper keeps the CSR buffers by reference; ``prepare_step`` refreshes
    the staging copy and ``__call__`` copies it in, so replays see new metadata.
    """

    __slots__ = ("wrapper", "indptr", "indices", "last_page_len", "pinned")

    def __init__(self, wrapper, indptr, indices, last_page_len, pinned):
        self.wrapper = wrapper
        self.indptr = indptr
        self.indices = indices
        self.last_page_len = last_page_len
        self.pinned = pinned


class FlashInferFA2PagedAttention:
    def __init__(self, device, dtype, head_dim, block_size):
        if device.type != "cuda" or torch.version.hip:
            raise ValueError("flashinfer_fa2 requires an NVIDIA CUDA device")
        if dtype is not torch.bfloat16:
            raise ValueError("flashinfer_fa2 requires bfloat16")
        if block_size != FLASHINFER_FA2_PAGE_SIZE:
            raise ValueError(f"flashinfer_fa2 requires --block-size {FLASHINFER_FA2_PAGE_SIZE}")
        if head_dim != FLASHINFER_FA2_HEAD_DIM:
            raise ValueError(f"flashinfer_fa2 requires head_dim {FLASHINFER_FA2_HEAD_DIM}")
        capability = torch.cuda.get_device_capability(device)
        if capability != (8, 9):
            raise ValueError(
                f"flashinfer_fa2 is validated on SM89 only, got SM{capability[0]}{capability[1]}"
            )
        try:
            package_version = version("flashinfer-python")
            if not (0, 7, 2) <= _version_tuple(package_version) < (0, 8):
                raise ImportError(
                    "flashinfer_fa2 requires flashinfer-python>=0.7.2,<0.8; "
                    f"found {package_version}"
                )
            from flashinfer.decode import CUDAGraphBatchDecodeWithPagedKVCacheWrapper
        except (ImportError, PackageNotFoundError) as error:
            raise ImportError(
                "install flashinfer with `pip install 'flashinfer-python>=0.7.2,<0.8'` "
                "(CUDA >= 12.9 is required for its JIT kernels); no backend fallback "
                "was performed"
            ) from error
        self._wrapper_cls = CUDAGraphBatchDecodeWithPagedKVCacheWrapper
        self.device = torch.device(device)
        self.head_dim = head_dim
        self.page_size = block_size
        self._wrappers = {}
        self._column_index = {}
        self._csr = {}
        self._primed = {}
        self._plan_misses = 0
        self._plan_creations = 0
        self.info = dict(
            backend="flashinfer_fa2",
            package="flashinfer-python",
            version=package_version,
            kernel="fa2",
            tensor_cores=True,
            cuda_graph_wrapper=True,
            split_kv=False,
            page_size=block_size,
            head_dim=head_dim,
            sm="8.9",
            cuda_graphs=True,
            async_scheduling=True,
        )

    @property
    def plan_misses(self):
        return self._plan_misses

    @property
    def plan_creations(self):
        return self._plan_creations

    @property
    def plan_entries(self):
        return len(self._wrappers)

    def prepare_step(self, batches):
        """Build this traversal's CSR once per batch, without host synchronization.

        The graph paths call this with the graph entry's private metadata after
        copying the caller's tables in, so eager and graph execution share one
        contract.
        """
        self._primed.clear()
        for batch in batches:
            block_tables = getattr(batch, "block_tables", None)
            context_lengths = getattr(batch, "context_lengths", None)
            if block_tables is None or context_lengths is None:
                continue
            self._validate_metadata(block_tables, context_lengths)
            if block_tables.shape[0] == 0:
                continue
            csr = self._csr_entry(block_tables, context_lengths)
            self._write_csr(csr, block_tables, context_lengths)
            self._primed[(id(block_tables), id(context_lengths))] = csr

    @staticmethod
    def _validate_metadata(block_tables, context_lengths):
        if block_tables.ndim != 2 or context_lengths.ndim != 1:
            raise ValueError("flashinfer_fa2 expects block_tables [rows, width] and lengths [rows]")
        if block_tables.shape[0] != context_lengths.shape[0]:
            raise ValueError("flashinfer_fa2 block_tables and context_lengths disagree on rows")
        if block_tables.dtype != torch.int32 or context_lengths.dtype != torch.int32:
            raise ValueError("flashinfer_fa2 expects int32 block_tables and context_lengths")

    def _csr_entry(self, block_tables, context_lengths):
        key = (id(block_tables), id(context_lengths))
        csr = self._csr.get(key)
        if csr is not None and csr.owns(block_tables, context_lengths):
            return csr
        csr = _CSR(block_tables, context_lengths)
        self._csr[key] = csr
        if len(self._csr) > FLASHINFER_FA2_CSR_CAPACITY:
            for stale, entry in list(self._csr.items()):
                if entry.block_tables() is None or entry.context_lengths() is None:
                    del self._csr[stale]
        return csr

    def _columns(self, device, width):
        key = (device, width)
        columns = self._column_index.get(key)
        if columns is None:
            columns = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(0)
            self._column_index[key] = columns
        return columns

    def _write_csr(self, csr, block_tables, context_lengths):
        """Refresh CSR sources in place; every shape below is static.

        Valid pages of row ``r`` land in ``[indptr[r], indptr[r + 1])``; unused
        slots receive padding columns, which the kernel never reads because it
        stops at ``last_page_len``.
        """
        width = block_tables.shape[1]
        page = self.page_size
        pages = (context_lengths.clamp_min(0) + page - 1) // page
        # A Python scalar assignment to a CUDA slice stages through host memory
        # and synchronizes; fill_ keeps CSR construction device-only.
        csr.indptr[0].fill_(0)
        csr.indptr[1:].copy_(torch.cumsum(pages, dim=0))
        valid = self._columns(block_tables.device, width) < pages.unsqueeze(1)
        flat = valid.reshape(-1)
        packed = torch.cumsum(flat.to(torch.int32), dim=0) - 1
        tail = flat.sum(dtype=torch.int32) + torch.cumsum((~flat).to(torch.int32), dim=0) - 1
        order = torch.where(flat, packed, tail).to(torch.int64)
        csr.indices.scatter_(0, order, block_tables.reshape(-1))
        torch.where(
            pages == 0,
            torch.ones_like(pages),
            context_lengths - (pages - 1) * page,
            out=csr.last_page_len,
        )

    def _wrapper(self, rows, q_heads, kv_heads, width):
        # Round the table width up to a power of two: a growing eager context
        # then crosses at most one plan per bucket instead of one per page.
        width = max(1, width)
        width = 1 << (width - 1).bit_length()
        key = (rows, q_heads, kv_heads, width)
        plan = self._wrappers.get(key)
        if plan is not None:
            # Plans are usually created in capture warmup; pin them when the
            # graph later records them so eager churn cannot free their buffers.
            if not plan.pinned and torch.cuda.is_current_stream_capturing():
                plan.pinned = True
            return plan
        group = q_heads // kv_heads
        if q_heads % kv_heads or group not in FLASHINFER_FA2_GROUP_SIZES:
            raise ValueError(
                f"flashinfer_fa2 tensor-core decode needs a query/KV head ratio in "
                f"{FLASHINFER_FA2_GROUP_SIZES}, got {q_heads}/{kv_heads}"
            )
        device = self.device
        indptr = torch.zeros(rows + 1, dtype=torch.int32, device=device)
        indices = torch.zeros(rows * width, dtype=torch.int32, device=device)
        last_page_len = torch.full((rows,), self.page_size, dtype=torch.int32, device=device)

        def host(values):
            tensor = torch.as_tensor(values, dtype=torch.int32)
            return tensor.pin_memory() if torch.cuda.is_available() else tensor

        # Plan inputs live on the host: plan()/workspace_size() copy indptr and
        # last_page_len to CPU, and a device source would synchronize each time.
        capacity_indptr = host([row * width for row in range(rows + 1)])
        capacity_last = host([self.page_size] * rows)
        capacity_indices = torch.zeros(rows * width, dtype=torch.int32)
        if torch.cuda.is_available():
            capacity_indices = capacity_indices.pin_memory()
        probe = torch.empty(256, dtype=torch.uint8, device=device)
        wrapper = self._wrapper_cls(
            probe, indptr, indices, last_page_len, "NHD", use_tensor_cores=True
        )
        float_bytes, int_bytes = wrapper.workspace_size(
            capacity_indptr,
            capacity_indices,
            capacity_last,
            q_heads,
            kv_heads,
            self.head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            disable_split_kv=True,
        )
        wrapper.reset_workspace_buffer(
            torch.empty(max(float_bytes, 256), dtype=torch.uint8, device=device),
            torch.empty((int_bytes + 3) // 4, dtype=torch.int32, device=device),
        )
        wrapper.plan(
            capacity_indptr,
            capacity_indices,
            capacity_last,
            q_heads,
            kv_heads,
            self.head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            disable_split_kv=True,
        )
        resolved = wrapper.resolved_backend
        if resolved != "fa2":
            raise RuntimeError(
                f"flashinfer-python {self.info['version']} resolved the tensor-core "
                f"wrapper to {resolved!r}; flashinfer_fa2 requires the FA2 kernel"
            )
        self.info["resolved_backend"] = resolved
        if len(self._wrappers) >= FLASHINFER_FA2_PLAN_CAPACITY:
            # Captured plans are referenced by graph replays and must outlive
            # them; only eager plans are evictable.
            stale = next((name for name, entry in self._wrappers.items() if not entry.pinned), None)
            if stale is not None:
                del self._wrappers[stale]
        plan = _Plan(
            wrapper,
            indptr,
            indices,
            last_page_len,
            pinned=torch.cuda.is_current_stream_capturing(),
        )
        self._plan_creations += 1
        self._wrappers[key] = plan
        return plan

    def __call__(self, q, key_cache, value_cache, block_tables, context_lengths):
        if q.shape[0] == 0:
            return torch.empty_like(q)
        self._validate_metadata(block_tables, context_lengths)
        rows, width = block_tables.shape
        if rows != q.shape[0]:
            raise ValueError(
                f"flashinfer_fa2 needs one metadata row per query row, got {rows} rows "
                f"for {q.shape[0]} queries"
            )
        csr = self._primed.get((id(block_tables), id(context_lengths)))
        if csr is None or not csr.owns(block_tables, context_lengths):
            self._plan_misses += 1
            csr = self._csr_entry(block_tables, context_lengths)
            self._write_csr(csr, block_tables, context_lengths)
        plan = self._wrapper(rows, q.shape[1], key_cache.shape[2], width)
        # Copied per layer on purpose: under CUDA Graph capture these nodes are
        # recorded and re-read the refreshed staging buffers on every replay.
        # The plan's buffers cover the bucket width, so live CSR fits its prefix.
        plan.indptr.copy_(csr.indptr)
        plan.indices[: csr.indices.numel()].copy_(csr.indices)
        plan.last_page_len.copy_(csr.last_page_len)
        out = torch.empty_like(q)
        plan.wrapper.run(q, (key_cache, value_cache), out=out)
        out.mul_((context_lengths > 0).to(out.dtype).unsqueeze(-1).unsqueeze(-1))
        return out
