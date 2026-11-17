# SHARED and LAST_EXITED KV: Semantics and Worked Examples

This note explains the `shared` and `last_exited` KV layouts in vllm-rlt and illustrates their attention computations with a numerical example. It describes the implementation reviewed on 2026-09-20.

## 1. Semantic distinction

The central question is which version of a historical token's KV a query should read at a given recurrent loop depth.

- **`last_exited`** retains KV separately for each loop. A query reads historical KV at the corresponding depth. When a historical token exits early, its final executed depth is copied into the remaining depths.
- **`shared`** retains one KV entry per token position and physical layer. Successive loops overwrite that entry. Subsequent tokens read the final retained KV of historical tokens at every loop depth.

Sharing applies across loops for the same request, token position, and physical layer. Requests, token positions, and layers retain distinct entries.

| Dimension | Meaning |
| --- | --- |
| Position | Token index within the sequence |
| Loop / depth | Recurrent pass through the shared Transformer |
| Layer | Physical Transformer layer within one pass |

Loop numbers in this note start at 1; depth indices in the code start at 0.

## 2. Worked example: two tokens and two loops

Consider tokens A and B, one attention layer, and two recurrent loops. Process A completely before B.

To make the arithmetic explicit, assume a single head with scalar Q, K, and V and a scaling factor of 1. Omit RoPE, residual connections, the MLP, output projection, and normalization. Specify A's KV and B's first-loop Q/K/V directly, then use the simplified update rule below for B's second loop. Both tokens execute both loops.

**This is a local example that isolates KV version selection, not a complete Ouro model or an accuracy experiment.** The supplied values need not arise from a single consistent set of model parameters; in particular, the second-loop update rule does not define how the initial values were generated.

The attention operation is:

$$
\operatorname{Attention}(q,K,V)=\operatorname{softmax}(qK^\top)V.
$$

### 2.1 KV retained after processing A

Assume A produces:

| Loop | K | V |
| --- | ---: | ---: |
| 1 | 1 | 10 |
| 2 | 2 | 20 |

The resulting storage is:

```text
last_exited:
  loop 1: (K=1, V=10)
  loop 2: (K=2, V=20)

shared:
  write (1,10), then overwrite it with (2,20)
  retained entry: (K=2, V=20)
```

### 2.2 B, loop 1: LAST_EXITED

Let B's first-loop values be:

```text
Q_B^1 = 1
K_B^1 = 1
V_B^1 = 30
```

Write B's KV before computing causal attention, which includes B's own position. The query reads:

| Entry | K | V |
| --- | ---: | ---: |
| A, loop 1 | 1 | 10 |
| B, loop 1 | 1 | 30 |

The scores, weights, and output are:

$$
s_A=1\times1=1,\qquad s_B=1\times1=1,
$$

$$
(w_A,w_B)=\operatorname{softmax}([1,1])=(0.5,0.5),
$$

$$
o_B^1=0.5\times10+0.5\times30=20.
$$

### 2.3 B, loop 1: SHARED

Writing B updates B's position; A's retained entry is unchanged. The query reads:

| Entry | K | V |
| --- | ---: | ---: |
| A, final retained loop 2 | 2 | 20 |
| B, current loop 1 | 1 | 30 |

Thus:

$$
s_A=1\times2=2,\qquad s_B=1\times1=1,
$$

$$
w_A=\frac{e^2}{e^2+e^1}\approx0.7311,\qquad
w_B=\frac{e^1}{e^2+e^1}\approx0.2689,
$$

$$
o_B^1=0.7311\times20+0.2689\times30\approx22.6894.
$$

The layouts already differ after the first loop:

| Layout | Historical entry read by B | Attention output |
| --- | --- | ---: |
| `last_exited` | A's loop-1 KV | 20 |
| `shared` | A's final loop-2 KV | 22.6894 |

### 2.4 B, loop 2: propagation of the difference

For illustration, use the same simplified update rule in both paths:

```text
h = B's first-loop attention output
Q = h / 20
K = h / 20
V = h
```

This is not Ouro's actual recurrent update.

For `last_exited`, the preceding output is 20, so B writes `(K=1, V=20)` and uses `Q=1`. It reads A's loop-2 entry:

```text
A: (2,20)
B: (1,20)
```

The scores are `[2,1]`. Both values are 20, so:

$$
o_B^2=20.
$$

The final stored entries are:

```text
loop 1: A=(1,10), B=(1,30)
loop 2: A=(2,20), B=(1,20)
```

For `shared`, the preceding output is approximately 22.6894:

```text
Q_B^2 ≈ 1.1345
K_B^2 ≈ 1.1345
V_B^2 ≈ 22.6894
```

B overwrites its first-loop KV. The cache becomes:

```text
A: (2,20)
B: (1.1345,22.6894)
```

Using unrounded intermediate values gives:

$$
s_A\approx2.2689,\qquad s_B\approx1.2870,
$$

$$
(w_A,w_B)\approx(0.7275,0.2725),
$$

$$
o_B^2\approx0.7275\times20+0.2725\times22.6894\approx20.73.
$$

Only the final entries remain:

```text
A: (2,20)
B: (1.1345,22.6894)
```

In the actual model, attention output also passes through projection, residual, MLP, and normalization operations. A difference in historical KV selection can therefore propagate to subsequent Q/K/V, gate scores, exit depth, and output logits.

## 3. Historical tokens that exit early

Suppose the maximum depth is four loops and A exits after loop 2:

```text
A, loop 1: (1,10)
A, loop 2: (2,20)
```

B reads A as follows:

| B's loop | `last_exited` | `shared` |
| --- | --- | --- |
| 1 | (1,10) | (2,20) |
| 2 | (2,20) | (2,20) |
| 3 | (2,20), copied during finalization | (2,20) |
| 4 | (2,20), copied during finalization | (2,20) |

For a completed historical token with exit loop `e` and query loop `d`, `last_exited` selects version `min(d,e)`, whereas `shared` selects version `e`.

These formulas describe version selection within each execution path. They do not imply numerically identical KV across the two paths. Even if later loops read the same supplied A values in this example, B's hidden state may already differ.

## 4. Implementation mapping

### 4.1 Storage and block tables

See initialization, `allocate()`, `_plane()`, and `get_block_table()` in [KVCacheManager](../vllm_rlt/core/kv_cache_manager.py). The mapping is equivalent to:

```python
storage_depths = max_loops if layout == "last_exited" else 1

def _plane(depth):
    return depth if layout == "last_exited" else 0
```

The key and value tensors each have shape:

```text
[num_blocks, num_layers, block_size, num_kv_heads, head_dim]
```

There is no explicit loop axis in the physical tensor. `last_exited` maps different depth-specific block tables to different physical blocks. `shared` maps every depth to the same storage plane.

A simplified write address is:

```python
plane = depth if layout == "last_exited" else 0
logical_page = position // block_size
offset = position % block_size
physical_block = allocation.block_tables[plane][logical_page]
key_cache[physical_block, layer, offset] = current_k
value_cache[physical_block, layer, offset] = current_v
```

### 4.2 Writes and attention reads

`_write_prepared()` writes the current token's KV. `_attend_prepared()` reads the context through the current token position using the corresponding block table.

Each `last_exited` attention operation selects one depth's KV plane. It does not concatenate all loop histories into a longer attention context. Context length is still determined by token position.

### 4.3 Finalization after early exit

After checking that all physical layers have written KV at the final executed depth, `finalize_token()` handles each layout as follows:

- `shared` returns immediately: the single entry already contains the final values.
- `last_exited` copies the token's K/V at every physical layer into each unexecuted depth and records those positions as valid.

The operation copies per-token, per-layer KV, rather than the final hidden state or an entire physical page. The asynchronous [ModelRunner](../vllm_rlt/worker/model_runner.py) can batch finalization through `finalize_many()` and `finalize_kernel`, with events enforcing execution ordering.

## 5. Prefill execution order

The shared [ModelRunner input preparation](../vllm_rlt/worker/model_runner.py)
uses `_prepare_loop_inputs()` for LAST_EXITED and `_prepare_shared_loop_inputs()`
for SHARED. Both enter the same submission and model-runtime interface; their
causal KV semantics require different execution orders.

`last_exited` can pack multiple prompt tokens within one chunk:

```text
A, B, C: loop 1
A, B, C: loop 2
A, B, C: loop 3
A, B, C: loop 4
```

Each loop reads the corresponding depth's KV with causal masking.

For `shared`, positions within a request advance sequentially so that each token reads the preceding tokens' final KV and the semantics remain independent of chunk boundaries:

```text
A: loop 1 → 2 → 3 → 4
B: loop 1 → 2 → 3 → 4
C: loop 1 → 2 → 3 → 4
```

Different requests can still be batched. Packing all positions of one prompt by loop would allow B's first loop to read A's first-loop KV instead of A's final KV, changing the current `shared` semantics.

## 6. Capacity and feature implications

For a request covering `T` token positions, block size `B`, and maximum loop count `D`, block requirements are:

```text
last_exited: ceil(T/B) × D physical blocks
shared:      ceil(T/B) physical blocks
```

| Property | `last_exited` | `shared` |
| --- | --- | --- |
| Historical versions | Separate entries per loop; unexecuted depths receive the exit-depth values | Final version only |
| Early-exit finalization copies | Required | Unnecessary |
| Packing prompt positions within a request | Supported | Positions currently execute sequentially |
| Prefix caching | Supported | Rejected by current configuration validation |
| Request block requirement for maximum depth D | D times the shared requirement | One storage plane |

The cache pool is preallocated using `num_blocks`. With the same `num_blocks`, both layouts allocate the same KV tensor bytes; `shared` allows that pool to cover more token positions. Early exit under `last_exited` does not automatically release deeper storage because those positions retain the copied history.

`shared` reduces block demand and eliminates finalization copies, but sequential prefill positions within a request can limit performance. The layout alone does not establish which implementation is faster. Because historical attention inputs differ, the layouts are not interchangeable as a lossless memory optimization, and this example does not establish an accuracy ranking.

## 7. Plane packing for partially recurrent models

Huginn runs a prelude and a coda once per token and repeats only the recurrent block. Prelude and coda attention use depth 0. A `last_exited` block in any deeper plane therefore holds live KV only in the recurrent layers' slots; the other layer slots are allocated but never written or read. For Huginn-0125 (2 prelude, 4 recurrent, 2 coda layers, 32 depths) 124 of 256 layer slots per logical page are dead.

When the recurrent layers are contiguous and occupy at most half of the layers, `CacheConfig(pack_recurrent_planes=True)` (the default) stores `k = num_layers // num_recurrent_layers` deeper planes in one physical block. Each packed plane uses its own slot range. Plane 0 still holds every layer:

```text
physical planes per page = 1 + ceil((D - 1) / k)       # Huginn-0125: 1 + ceil(31 / 2) = 17
plane 0 page entry       = physical_block * num_layers
plane d >= 1 page entry  = physical_block * num_layers + member * num_recurrent - first_recurrent
member                   = (d - 1) % k,  physical plane = 1 + (d - 1) // k
```

`get_block_table(request_id, depth)` returns these page entries. A layer's K/V for a row lives at page `entry + layer` of the layer-flattened cache `key_cache.view(num_blocks * num_layers, block_size, kv_heads, head_dim)`. Writes, attention, CUDA graphs, the device-resident routing tables and finalization (the Triton `finalize_kernel` copies `finalize_layers` slots from each entry with `entry_stride`) add the layer to the entry instead of indexing the layer axis. The stored values and the attention arithmetic are unchanged; only addresses move, so packed and unpacked execution produce bitwise-identical tokens and KV. Nothing is copied or recomputed.

`plane_block_tables` and the allocation views count physical planes, so `required_blocks` shrinks by `D / physical planes` (1.88x for Huginn-0125) and the same KV pool holds that many more tokens. `transfer_slots` reports each physical plane's live slot range. PD sends only those slots, packed or not: 132 of 256 layer slots per Huginn-0125 page, and the coordinator prices reservations in `blocks_per_page` physical blocks.

Models whose layers all recur (Ouro, Nanbeige) are unaffected: `k` is 1 and every path keeps the unpacked addressing. A non-recurrent layer may be accessed only at depth 0 in a packed cache; the manager rejects any other access because those slots belong to another plane.
