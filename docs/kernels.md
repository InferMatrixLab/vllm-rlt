# Native kernel development

Native kernels are optional and live alongside the existing Triton kernels.
Importing `vllm_rlt.kernels` loads the built library once and registers the Torch
fallbacks. It neither compiles code nor creates a CUDA context.

## Layout

| Location | Responsibility |
| --- | --- |
| `vllm_rlt/kernels/activation.py` | Python activation entry and Torch reference |
| `vllm_rlt/kernels/_extension.py` | One-time loading and missing-library detection |
| `csrc/ops.h` | Native operator declarations |
| `csrc/torch_bindings.cpp` | Operator schemas and CUDA dispatch registration |
| `csrc/activation/silu_and_mul.cu` | SiLU-and-Mul CUDA validation, launch and computation |
| `csrc/normalization/rms_norm.cu` | RMSNorm and RMSNorm-then-residual CUDA kernels |
| `vllm_rlt/kernels/normalization.py` | RMSNorm Python entry and Torch reference |
| `setup.py` | Optional build of `vllm_rlt._C` |
| `tests/test_activation.py` | Reference, CUDA numerics, layouts, tails and graph replay |
| `tests/test_normalization.py` | RMSNorm numerics, strides, alignment and graph replay |
| `benchmarks/kernels/silu_and_mul.py` | Matched Torch/native/vLLM accuracy and latency comparison |
| `benchmarks/kernels/rms_norm.py` | RMSNorm and residual accuracy and latency comparison |

Add future kernels to the same extension and group sources by operation family.
No backend registry or base class is required. Put future kernel performance
benchmarks under `benchmarks/kernels/`.

## SiLU-and-Mul contract and current status

`silu_and_mul_reference(gate, up)` computes `F.silu(gate) * up`. The inputs have
identical shape, dtype and device; broadcasting is rejected. Supported dtypes
are FP32, FP16 and BF16. The output owns new storage and preserves input shape,
dtype and device. Neither input is modified. The Torch reference preserves
the intermediate SiLU rounding of the existing MLP expression.

Separate inputs avoid concatenating Ouro and Nanbeige's separate projection
outputs. The reference also accepts Huginn's noncontiguous split-projection
views. The native entry requires both inputs to be contiguous; it neither pads
nor copies incompatible inputs.

`silu_and_mul(gate, up)` uses PyTorch dispatch: CUDA inputs use the native kernel
when the extension is installed; CPU inputs and installations without the
extension use the Torch reference. Import it from `vllm_rlt.kernels`. The model
call site does not check devices or extension availability. The native kernel
flattens contiguous inputs, using 16-byte
loads/stores for the aligned vector body and scalar accesses for the remaining
elements in the same kernel. Contiguous views with misaligned storage offsets
use scalar accesses. Empty inputs return an empty output without launching.

SiLU uses FP32 arithmetic and rounds to the input dtype before multiplying by
`up`, preserving the reference's intermediate rounding. FP16/BF16 vector paths
use `__hmul2`; FP32 uses separate round-to-nearest products. On SM103, forcing
`__fmul2_rn` increased the 16-byte vector kernel's registers/thread from 28 to
34 and reduced resident warps, hurting memory throughput. The scalar path uses
`__hmul` for FP16/BF16 and `__fmul_rn` for FP32, retaining the SiLU rounding
boundary without explicitly converting the rounded activation back to FP32.
The device guard follows the input device, and launches use that device's current
stream.

Blocks have 256 threads. The grid is capped at 4096 blocks, except for FP32
workloads with at least `16384 * 256` work items: these use up to 16384 blocks.
Each aligned work item processes four FP32 elements; an unaligned work item
processes one scalar. The threshold requires enough work to fill the larger
grid. This keeps small and medium shapes' launch geometry unchanged. Grid sizes,
16/32-byte transfers and scalar/packed FP32 products were compared on B300;
the selected configuration is not a universal hardware optimum.

Ouro's inference MLP calls this registered operator directly. Availability is
resolved once during module initialization; extension ABI or dependency errors
propagate rather than silently selecting another path. The native CUDA dispatch
reuses its C++ input checks; a missing extension does not mask native execution
errors when an installed kernel is selected.
Nanbeige and Huginn do not use this operator yet.

Autograd is unsupported: gradient-enabled inputs are rejected unless gradients
are disabled, consistent with Ouro's inference-only model. `torch.compile` has
no fake kernel registration. Tests cover the operator and Ouro MLP, including
CUDA Graph capture/replay.

## RMSNorm and residual contract

`rms_norm(hidden, weight, epsilon, residual=None)` returns a new tensor. Its
reference computes the mean square and inverse square root in FP32, multiplies
the input by that scale, rounds the normalized values to the input dtype, then
multiplies by `weight`. When supplied, the residual is added after the weight
multiplication. In Ouro, this fuses the existing `residual + RMSNorm(hidden)`
order; it does not implement `RMSNorm(residual + hidden)`. The weight is a
one-dimensional tensor of shape `[hidden_size]` and is shared across rows.

With the extension installed, the native inference path supports CUDA FP32,
FP16 and BF16 tensors of rank 1 through 4, matching one-dimensional contiguous
weights, and matching residuals. The last input and residual dimensions must
have stride 1. Outer dimensions may be strided, including views into a larger
row; input, weight and residual addresses may have independent offsets. The
output is contiguous and none of the inputs is modified. Empty tensors return
without a kernel launch. CPU tensors, autograd, mixed dtypes and other layouts
use the Torch reference. Nanbeige and Huginn continue to use their existing
Torch RMSNorm path.

The general CUDA kernel assigns one block per row. It sums squares in FP32,
uses a warp reduction for widths up to 256 and a block reduction otherwise,
then writes the scaled and weighted values, optionally with a residual. The
aligned body loads and stores 16 bytes at a time and uses paired FP16/BF16
arithmetic. Unaligned output heads and tails use scalar accesses; independently
shifted input, weight and residual vectors are reconstructed from safe aligned
reads, with scalar reads at view boundaries. Small rows retain their input
vectors in registers across the reduction; wider rows stream them again to bound
register usage. Aligned width 2048 has a specialized path. FP32 pairs use paired
arithmetic on SM100 and newer, with scalar FP32 operations on earlier
architectures. These launch choices are implementation details and are not
claimed optimal for every GPU.

The general native reduction can use a different floating-point summation order
from PyTorch, so its result need not be bitwise equal. Ouro uses a separate
2048-wide FP16/BF16 compatibility path that reproduces the tested PyTorch mean
reduction order, including its row-count-dependent block width. This path
fuses the residual, uses 8-byte vector loads and stores and paired arithmetic,
and keeps each thread's input values in registers across the reduction. It
supports two-dimensional inputs and residuals with 8-byte-aligned rows, including
strided views. Since the reduction tree is a PyTorch implementation detail,
this path is currently enabled only for PyTorch 2.13.0 on SM103; Ouro falls
back to the Torch reference for other versions, GPUs, shapes or alignments.
The general kernel remains available through `rms_norm(..., match_torch=False)`.
The tested compatibility path and the general path both retain the explicit
intermediate dtype roundings and residual order. CUDA Graph replay is supported;
a fake kernel for `torch.compile` is not registered.

## Build and reference checks

The extension builds automatically when the build environment has CUDA-enabled
PyTorch, a CUDA toolkit with `nvcc`, and a visible GPU. PyTorch locates the toolkit
and detects the visible GPUs' target architectures. No project build switch or
manual architecture setting is needed. A compatible C++ compiler is also required;
a driver alone is insufficient. Run these commands with the Python interpreter
used for inference, keeping your allocated GPU visible:

```bash
python -m pip install "torch>=2.5" setuptools wheel ninja
python -m pip install --no-build-isolation -e .
python -m pytest tests/test_activation.py -q
CUDA_VISIBLE_DEVICES=<allocated_gpu> python -m pytest \
  tests/test_activation.py tests/test_normalization.py --run-gpu -q
```

`pip` invokes setuptools, whose PyTorch `CUDAExtension`/`BuildExtension` compiles
the C++ bindings and CUDA sources with the host compiler and `nvcc`, then installs
`vllm_rlt._C`. `--no-build-isolation` uses the environment's existing PyTorch
headers and libraries; it avoids a separate build environment with a potentially
different Torch version. An isolated build without PyTorch cannot build the
extension and produces a Python-only package. Missing CUDA PyTorch, toolkit,
`nvcc`, or a GPU target also produces a Python-only package with a skip message.
Once native compilation is selected, compilation errors fail the installation.
No runtime JIT compilation or CMake is required.

For a wheel targeting the visible GPUs:

```bash
python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install dist/vllm_rlt-*.whl
```

Without a visible GPU, the deployment architecture cannot be detected. For
cross-compilation, PyTorch's standard `TORCH_CUDA_ARCH_LIST` override remains
available, for example `TORCH_CUDA_ARCH_LIST="8.0;10.3"` for SM80 and SM103.
Choose targets supported by the deployment hardware and toolkit. An installed
binary without compatible device code raises an error; there is no fallback
after a native launch failure. GPU kernel tests and the benchmark require the
native extension and fail if it is absent.

Restart Python after rebuilding the library. A successful build or passing CPU
reference tests establishes no CUDA compute correctness or speedup.

## Accuracy and performance comparison

With vLLM installed in the same environment:

```bash
CUDA_VISIBLE_DEVICES=<allocated_gpu> .venv/bin/python -m benchmarks.kernels.silu_and_mul \
  --output artifacts/silu-and-mul/results.json \
  --profile-dir artifacts/silu-and-mul/profiles
```

The benchmark checks matched random inputs against eager Torch and a staged
FP64 oracle that rounds SiLU before multiplication. It retains maximum absolute
and ULP errors, RMSE, mismatch fractions, and nonfinite mismatches. The default
matrix covers FP32/FP16/BF16, 1/16/128/1024/4096 rows, and widths 5632/5633.

Latency is measured without profiling, with three repetitions and shuffled
backend order. CUDA Graph timing amortizes replay over 32 calls; eager timing
includes Python, allocation, launch and device completion. `vllm_prepacked`
receives its packed input and preallocated output before timing. These operator
comparisons do not establish full-model throughput or task quality.

For a source-built standalone vLLM baseline, pass `--vllm-library <so>` and
`--vllm-namespace <namespace>`, plus `--vllm-source <checkout>` to record its
commit. The report retains library hashes, the local diff and raw measurements.
Optional ops-only/full profiles are captured separately after timing; full
capture enables shapes, memory and stacks. Large traces stay under ignored
`artifacts/`.

For RMSNorm, compare the Torch reference and both native paths without vLLM:

```bash
CUDA_VISIBLE_DEVICES=<allocated_gpu> python -m benchmarks.kernels.rms_norm \
  --output artifacts/rmsnorm/results.json \
  --profile-dir artifacts/rmsnorm/profiles
```

An optional source-built vLLM comparison can be supplied with
`--vllm-library <so> --vllm-namespace <namespace> --vllm-source-commit <sha>`.
Its different intermediate rounding is reported separately from the reference
semantics. The general native path can use a different reduction order than
PyTorch; use the `rlt_torch_order` measurements when assessing Ouro's actual
compatibility path.
