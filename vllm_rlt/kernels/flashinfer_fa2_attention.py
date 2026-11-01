"""FlashInfer FA2 paged decode for Ada (SM89); no packed prefill or graphs.

The adapter builds the CSR metadata FlashInfer's paged kernel wants once per
traversal (``prepare_step``), then reuses it across every decoder layer through
``__call__``.  The plan is created once per (rows, heads, table width) with a
synthetic full-capacity table, so later steps only refresh device buffers.

Only the L20/L40S/RTX-40 class is accepted: FlashAttention-2 rejects the page
size 16 used by Ouro's cache, and FlashInfer XQA (PR #112) rejects SM89.
"""

from importlib.metadata import PackageNotFoundError, version

import torch

FLASHINFER_FA2_PAGE_SIZE = 16
FLASHINFER_FA2_HEAD_DIM = 128
FLASHINFER_FA2_WORKSPACE_BYTES = 128 * 1024 * 1024


def _version_tuple(value):
    core = value.split("+", 1)[0].split(".post", 1)[0]
    parts = []
    for component in core.split(".")[:3]:
        digits = "".join(character for character in component if character.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


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
            from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
        except (ImportError, PackageNotFoundError) as error:
            raise ImportError(
                "install flashinfer with `pip install 'flashinfer-python>=0.7.2,<0.8'` "
                "(CUDA >= 12.9 is required for its JIT kernels); no backend fallback "
                "was performed"
            ) from error
        self._wrapper_cls = BatchDecodeWithPagedKVCacheWrapper
        self.device = torch.device(device)
        self.head_dim = head_dim
        self.page_size = block_size
        # Allocated on first execution so backend selection stays allocation-free.
        self._workspace = None
        self._wrappers = {}
        self._primed = {}
        self._plan_misses = 0
        self.info = dict(
            backend="flashinfer_fa2",
            package="flashinfer-python",
            version=package_version,
            kernel="fa2",
            page_size=block_size,
            head_dim=head_dim,
            sm="8.9",
            workspace_bytes=FLASHINFER_FA2_WORKSPACE_BYTES,
        )

    @property
    def plan_misses(self):
        return self._plan_misses

    @property
    def plan_entries(self):
        return len(self._wrappers)

    def prepare_step(self, batches):
        """Build the CSR metadata of this traversal once, on the current stream."""
        self._primed.clear()
        for batch in batches:
            block_tables = getattr(batch, "block_tables", None)
            context_lengths = getattr(batch, "context_lengths", None)
            if block_tables is None or context_lengths is None:
                continue
            if block_tables.ndim != 2 or context_lengths.ndim != 1:
                raise ValueError(
                    "flashinfer_fa2 expects block_tables [rows, width] and lengths [rows]"
                )
            if block_tables.shape[0] != context_lengths.shape[0]:
                raise ValueError("flashinfer_fa2 block_tables and context_lengths disagree on rows")
            if block_tables.dtype != torch.int32 or context_lengths.dtype != torch.int32:
                raise ValueError("flashinfer_fa2 expects int32 block_tables and context_lengths")
            if block_tables.shape[0] == 0:
                continue
            self._primed[id(block_tables)] = self._build_csr(block_tables, context_lengths)

    def _build_csr(self, block_tables, context_lengths):
        rows, width = block_tables.shape
        pages = (context_lengths.clamp_min(0) + self.page_size - 1) // self.page_size
        indptr = torch.zeros(rows + 1, dtype=torch.int32, device=block_tables.device)
        indptr[1:] = torch.cumsum(pages, dim=0)
        columns = torch.arange(width, dtype=torch.int32, device=block_tables.device).unsqueeze(0)
        indices = block_tables[columns < pages.unsqueeze(1)]
        last_page_len = torch.where(
            pages == 0,
            torch.ones_like(pages),
            context_lengths - (pages - 1) * self.page_size,
        ).to(torch.int32)
        return indptr, indices, last_page_len

    def _wrapper(self, rows, q_heads, kv_heads, width):
        key = (rows, q_heads, kv_heads, width)
        wrapper = self._wrappers.get(key)
        if wrapper is not None:
            return wrapper
        if self._workspace is None:
            self._workspace = torch.zeros(
                FLASHINFER_FA2_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
            )
        # Plan for the widest table this cache can present, so later steps only
        # refresh metadata. Actual shorter tables are bounded by last_page_len.
        indptr = torch.arange(rows + 1, dtype=torch.int32, device=self.device) * width
        indices = torch.zeros(rows * width, dtype=torch.int32, device=self.device)
        last_page_len = torch.full((rows,), self.page_size, dtype=torch.int32, device=self.device)
        wrapper = self._wrapper_cls(self._workspace, "NHD", backend="fa2", use_tensor_cores=False)
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            q_heads,
            kv_heads,
            self.head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        for name in (
            "_paged_kv_indptr_buf",
            "_paged_kv_indices_buf",
            "_paged_kv_last_page_len_buf",
        ):
            if not hasattr(wrapper, name):
                raise RuntimeError(
                    f"flashinfer-python {self.info['version']} no longer exposes {name}; "
                    "pin a supported version or update the adapter"
                )
        self._wrappers[key] = wrapper
        return wrapper

    def __call__(self, q, key_cache, value_cache, block_tables, context_lengths):
        if q.shape[0] == 0:
            return torch.empty_like(q)
        primed = self._primed.get(id(block_tables))
        if primed is None:
            self._plan_misses += 1
            primed = self._build_csr(block_tables, context_lengths)
        indptr, indices, last_page_len = primed
        rows, width = block_tables.shape
        wrapper = self._wrapper(rows, q.shape[1], key_cache.shape[2], width)
        wrapper._paged_kv_indptr_buf.copy_(indptr)
        wrapper._paged_kv_indices_buf[: indices.numel()].copy_(indices)
        wrapper._paged_kv_last_page_len_buf.copy_(last_page_len)
        out = torch.empty_like(q)
        wrapper.run(q, (key_cache, value_cache), out=out)
        out.mul_((context_lengths > 0).to(out.dtype).unsqueeze(-1).unsqueeze(-1))
        return out
