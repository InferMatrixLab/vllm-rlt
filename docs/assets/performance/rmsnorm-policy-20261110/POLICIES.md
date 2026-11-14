# Frozen policy candidates, before measurement

All candidates keep production arithmetic/alignment and use dynamic `H`, i.e. no 2048 specialization. For rows of H<=256, keep 32 threads. Otherwise define `work = ceil(H / (256 * (16 / sizeof(dtype))))`, the number of 16-byte vectors per 256-thread lane (before 1/2/4 cache bucket rounding). `sm` is `at::cuda::getDeviceProperties(device)->multiProcessorCount`; use 512 only when the device supports at least 512 threads/block. After deciding threads, recompute actual `packs=ceil(H/(threads*(16/sizeof(dtype))))`; round cached packs to 1/2/4 as before, or reread above 4.

- `nostatic` (diagnostic): old 32/256 and old cache rules; only eliminate static H branch.
- `conservative`: use 512 threads if `rows < ceil(sm/2)` and `work > 4`; if `rows >= 2*sm` and actual `packs > 2`, reread instead of caching.
- `balanced`: use 512 threads if `rows < sm` and `work >= 4`; if `rows >= sm` and actual `packs > 2`, reread instead of caching.
- `moderate`: use 512 threads if `rows < sm` and `work > 4`; if `rows >= 2*sm` and actual `packs > 2`, reread instead of caching.

No policy selects by model, exact hidden width, GPU vendor model, or hard-coded SM count. '2 packs' is 32 bytes of cached vectors per thread, not a universal hard limit: 4-pack caching is preserved for smaller grids. Arithmetic/rounding may differ when thread count changes the reduction tree; compare to the same FP32 Torch reference using predeclared dtype tolerances.
