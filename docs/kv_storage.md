# VMM-backed KV storage (RFC #192 stage 2)

The KV pool can reserve its full address range once and map physical memory in
stages, so a CUDA graph captured against the K/V tensors stays valid while the
pool grows. This is the storage half of RFC #192: capacity is still decided by
`plan_cache`'s existing estimate, not by a warmup measurement.

## Enabling it

```bash
vllm-rlt serve <model> --kv-vmm-bootstrap-blocks 64
```

or in code:

```python
CacheConfig(block_size=16, vmm_bootstrap_blocks=64)
```

At startup the engine:

1. plans the block count as usual (`plan_cache`),
2. reserves one VMM range covering the K and V planes of that capacity,
3. maps only `vmm_bootstrap_blocks` blocks,
4. builds `KVCacheManager` against that pool,
5. maps the remaining blocks once the runners exist, then publishes the full
   planned capacity.

`stage.memory_plan` records `vmm`, `vmm_bootstrap_blocks`, `vmm_max_blocks`,
`vmm_committed_blocks` and `vmm_reserve_bytes`.

Without the option nothing changes: the cache allocates its tensors with
`torch.empty` as before, `max_blocks == committed_blocks == published_blocks`,
and `publish_capacity` is unavailable.

## Invariants

- `published_blocks <= committed_blocks <= max_blocks` at all times.
- Publishing requires committed physical memory, so an allocation can never
  address an unmapped block.
- The K/V tensors keep their `data_ptr`, shape and stride for the pool's whole
  lifetime; growth only appends mappings.
- `GPUKVStorage.close()` is the only release path; dropping tensors does not
  unmap anything.
- Only the committed prefix is addressable. Whole-tensor operations (`fill_`,
  `clone`, a full `copy_`) fault while the tail is unmapped, so anything that
  touches the K/V views must stay inside `committed_blocks`.

## Boundaries

- CUDA only, and only with automatic planning: `num_blocks` and
  `kv_cache_memory_bytes` keep their fixed-capacity semantics and reject the
  option.
- Not supported with a PD connector yet. NIXL registers the K/V planes once at
  startup and peers cache `num_blocks`, so growth needs a re-registration
  protocol that this stage does not implement.
- Growth happens once, during startup. There is no runtime expansion.

## Verification

```bash
python -m pytest tests/test_kv_storage.py -q                       # bookkeeping, CPU
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_vmm_pool.py --run-gpu
OURO_CHECKPOINT=/path/to/Ouro-1.4B CUDA_VISIBLE_DEVICES=0 \
    python -m pytest tests/test_vmm_real_checkpoint.py --run-gpu -s
```

`tests/test_vmm_pool.py` covers pointer/shape/stride stability, data survival
across growth, new-block access, graph replay after growth, release accounting,
the publish/commit bound, and an engine-level run whose output matches the
non-VMM path on both the torch and Triton attention backends.

`tests/test_vmm_real_checkpoint.py` runs the pinned Ouro snapshot and compares
the baseline `torch.empty` pool with a bootstrap-1 VMM pool: generated tokens,
KV bytes read back for the same positions in three layers, CUDA-graph capture,
prefix-cache hits, a clamped bootstrap, and repeated create/close. Driver
free-memory deltas are not a usable oracle on a device shared with other tenants,
so release is asserted through the pool's own counters and a 2 GiB mapping test.

