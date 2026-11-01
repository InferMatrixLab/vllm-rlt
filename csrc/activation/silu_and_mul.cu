// SPDX-License-Identifier: Apache-2.0
#include "../ops.h"

#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace vllm_rlt {
namespace {

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

template <typename T>
struct Math;

template <>
struct Math<__half> {
  using Pair = __half2;
  __device__ static float to_float(__half x) { return __half2float(x); }
  __device__ static __half from_float(float x) { return __float2half_rn(x); }
  __device__ static __half mul(__half a, __half b) { return __hmul(a, b); }
  __device__ static Pair compute(Pair gate, Pair up) {
    float2 g = __half22float2(gate);
    Pair activated = __floats2half2_rn(silu(g.x), silu(g.y));
    return __hmul2(activated, up);
  }
};

template <>
struct Math<__nv_bfloat16> {
  using Pair = __nv_bfloat162;
  __device__ static float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ static __nv_bfloat16 from_float(float x) { return __float2bfloat16_rn(x); }
  __device__ static __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hmul(a, b);
  }
  __device__ static Pair compute(Pair gate, Pair up) {
    float2 g = __bfloat1622float2(gate);
    Pair activated = __floats2bfloat162_rn(silu(g.x), silu(g.y));
    return __hmul2(activated, up);
  }
};

template <>
struct Math<float> {
  using Pair = float2;
  __device__ static float to_float(float x) { return x; }
  __device__ static float from_float(float x) { return x; }
  __device__ static float mul(float a, float b) { return __fmul_rn(a, b); }
  __device__ static Pair compute(Pair gate, Pair up) {
    Pair activated = make_float2(silu(gate.x), silu(gate.y));
    // Packed FP32 products increased register allocation and reduced occupancy
    // in the SM103 benchmarks, unlike the FP16/BF16 packed products above.
    return make_float2(mul(activated.x, up.x), mul(activated.y, up.y));
  }
};

template <typename T>
struct alignas(16) Vector {
  typename Math<T>::Pair pairs[16 / sizeof(typename Math<T>::Pair)];
};

template <typename T>
__device__ __forceinline__ T compute_scalar(T gate, T up) {
  // Match F.silu(gate) * up: round the activation before multiplying.
  T activated = Math<T>::from_float(silu(Math<T>::to_float(gate)));
  return Math<T>::mul(activated, up);
}

template <typename T, bool vectorized>
__global__ void silu_and_mul_kernel(const T* gate, const T* up, T* out, int64_t count) {
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t tail_start = 0;
  if constexpr (vectorized) {
    constexpr int elements = 16 / sizeof(T);
    constexpr int pairs = 16 / sizeof(typename Math<T>::Pair);
    const int64_t vectors = count / elements;
    for (int64_t i = thread; i < vectors; i += stride) {
      Vector<T> g, u, result;
      *reinterpret_cast<uint4*>(&g) = *reinterpret_cast<const uint4*>(gate + i * elements);
      *reinterpret_cast<uint4*>(&u) = *reinterpret_cast<const uint4*>(up + i * elements);
#pragma unroll
      for (int j = 0; j < pairs; ++j) {
        result.pairs[j] = Math<T>::compute(g.pairs[j], u.pairs[j]);
      }
      *reinterpret_cast<uint4*>(out + i * elements) = *reinterpret_cast<uint4*>(&result);
    }
    tail_start = vectors * elements;
  }
  // Only valid scalar addresses are touched, including for count < one vector.
  for (int64_t i = tail_start + thread; i < count; i += stride) {
    out[i] = compute_scalar(gate[i], up[i]);
  }
}

template <typename T>
void launch(const at::Tensor& gate, const at::Tensor& up, at::Tensor& out) {
  const T* g = reinterpret_cast<const T*>(gate.const_data_ptr());
  const T* u = reinterpret_cast<const T*>(up.const_data_ptr());
  T* result = reinterpret_cast<T*>(out.mutable_data_ptr());
  // A contiguous storage-offset view is not necessarily 16-byte aligned.
  const bool vectorized = ((reinterpret_cast<uintptr_t>(g) |
                            reinterpret_cast<uintptr_t>(u) |
                            reinterpret_cast<uintptr_t>(result)) & 15) == 0;
  const int64_t count = gate.numel();
  const int64_t work = vectorized ? std::max<int64_t>(count / (16 / sizeof(T)), 1) : count;
  constexpr int threads = 256;
  int max_blocks = 4096;
  if constexpr (std::is_same_v<T, float>) {
    // More blocks improve large FP32 streaming workloads. Require enough work
    // to fill the enlarged grid, preserving smaller shapes' launch geometry.
    constexpr int streaming_blocks = 16384;
    if (work >= static_cast<int64_t>(streaming_blocks) * threads) {
      max_blocks = streaming_blocks;
    }
  }
  const int blocks = static_cast<int>(std::min<int64_t>((work + threads - 1) / threads, max_blocks));
  auto stream = c10::cuda::getCurrentCUDAStream(gate.get_device());
  if (vectorized) {
    silu_and_mul_kernel<T, true><<<blocks, threads, 0, stream>>>(g, u, result, count);
  } else {
    silu_and_mul_kernel<T, false><<<blocks, threads, 0, stream>>>(g, u, result, count);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

at::Tensor silu_and_mul_cuda(const at::Tensor& gate, const at::Tensor& up) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "silu_and_mul expects CUDA inputs");
  TORCH_CHECK(gate.device() == up.device(), "silu_and_mul expects inputs on the same device");
  TORCH_CHECK(gate.sizes() == up.sizes(), "silu_and_mul expects inputs with the same shape");
  TORCH_CHECK(gate.scalar_type() == up.scalar_type(),
              "silu_and_mul expects inputs with the same dtype");
  TORCH_CHECK(gate.scalar_type() == at::kFloat || gate.scalar_type() == at::kHalf ||
                  gate.scalar_type() == at::kBFloat16,
              "silu_and_mul supports float32, float16 and bfloat16");

  TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(),
              "silu_and_mul expects contiguous inputs");
  TORCH_CHECK(!at::GradMode::is_enabled() || (!gate.requires_grad() && !up.requires_grad()),
              "silu_and_mul is inference-only; use torch.no_grad() or silu_and_mul_reference");

  const c10::cuda::CUDAGuard guard(gate.device());
  auto out = at::empty(gate.sizes(), gate.options());
  if (gate.numel() == 0) {
    return out;
  }
  switch (gate.scalar_type()) {
    case at::kHalf: launch<__half>(gate, up, out); break;
    case at::kBFloat16: launch<__nv_bfloat16>(gate, up, out); break;
    case at::kFloat: launch<float>(gate, up, out); break;
    default: TORCH_CHECK(false, "unsupported dtype");
  }
  return out;
}

}  // namespace vllm_rlt
