# RLT SiLU-and-Mul

The optional native implementation is in
[`csrc/activation/silu_and_mul.cu`](../csrc/activation/silu_and_mul.cu).
[`vllm_rlt/ops/silu_and_mul.py`](../vllm_rlt/ops/silu_and_mul.py) registers the
PyTorch operator, provides a Torch reference and adapts the packed vLLM MLP
input to its two-tensor interface.

## Use in Ouro

From the CUDA-enabled vLLM environment, install with the CUDA toolkit,
`nvcc`, and the target GPU visible to PyTorch:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

`TORCH_CUDA_ARCH_LIST` can specify a target when building without a visible
GPU. Verify the built operator with
`python -c 'from vllm_rlt.ops._extension import load_extension; load_extension()'`.

Use `--rlt-silu-and-mul` with `vllm-rlt serve`, or pass
`--additional-config '{"rlt_silu_and_mul":true}'` to the
[upstream Ouro serving command](accuracy.md#serve-ouro). To collect exit traces in the
same run, put `rlt_trace_steps` in the same JSON object. This option selects
the RLT activation in the unquantized Ouro MLP and requires the compiled
extension. With the option omitted, Ouro uses vLLM's default activation.

vLLM's gate/up projection produces one packed tensor. Unpacking it into
contiguous tensors may copy data for multiple rows. Run
`python -m benchmarks.kernels.silu_and_mul --help` for the two-input kernel
benchmark. End-to-end performance also depends on the projection layout and
the cost of those copies.
