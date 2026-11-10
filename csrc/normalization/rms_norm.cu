// SPDX-License-Identifier: Apache-2.0
#include "../ops.h"

#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace vllm_rlt {
namespace {

template <typename T>
struct alignas(16) Vector {
  T val[16 / sizeof(T)];
};

// Return one logical 16-byte vector without reading outside the input view.
// Aligned words are combined in registers when the logical start is shifted.
template <typename T>
__device__ __forceinline__ Vector<T> load_vector(const T* input, int index, int count) {
  constexpr int elements = 16 / sizeof(T);
  const int offset_bytes = reinterpret_cast<uintptr_t>(input + index) & 15;
  Vector<T> result;
  if (offset_bytes == 0) {
    *reinterpret_cast<uint4*>(&result) = *reinterpret_cast<const uint4*>(input + index);
    return result;
  }
  const int offset = offset_bytes / sizeof(T);
  const int lower = index - offset;
  Vector<T> head{}, tail{};
  if (lower >= 0) {
    *reinterpret_cast<uint4*>(&head) = *reinterpret_cast<const uint4*>(input + lower);
  } else {
#pragma unroll
    for (int j = 0; j < elements; ++j) {
      if (j >= offset) head.val[j] = input[lower + j];
    }
  }
  if (lower + 2 * elements <= count) {
    *reinterpret_cast<uint4*>(&tail) = *reinterpret_cast<const uint4*>(input + lower + elements);
  } else {
#pragma unroll
    for (int j = 0; j < elements; ++j) {
      if (j < offset) tail.val[j] = input[lower + elements + j];
    }
  }
  uint4 lo = *reinterpret_cast<const uint4*>(&head);
  uint4 hi = *reinterpret_cast<const uint4*>(&tail);
  if (offset_bytes & 8) {
    lo = make_uint4(lo.z, lo.w, hi.x, hi.y);
    hi = make_uint4(hi.z, hi.w, 0, 0);
  }
  if (offset_bytes & 4) {
    lo = make_uint4(lo.y, lo.z, lo.w, hi.x);
    hi.x = hi.y;
  }
  if constexpr (sizeof(T) == 2) {
    if (offset_bytes & 2) {
      lo = make_uint4(__funnelshift_r(lo.x, lo.y, 16), __funnelshift_r(lo.y, lo.z, 16),
                      __funnelshift_r(lo.z, lo.w, 16), __funnelshift_r(lo.w, hi.x, 16));
    }
  }
  *reinterpret_cast<uint4*>(&result) = lo;
  return result;
}

__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __fmul2_rn(a, b);
#else
  return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
#endif
}

__device__ __forceinline__ float2 add2(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __fadd2_rn(a, b);
#else
  return make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
#endif
}

template <typename T>
struct Arithmetic;

template <>
struct Arithmetic<__half> {
  using Pair = __half2;
  __device__ static float scalar(__half x) { return __half2float(x); }
  __device__ static __half cast(float x) { return __float2half_rn(x); }
  __device__ static float2 pair(Pair x) { return __half22float2(x); }
  __device__ static Pair cast2(float2 x) { return __floats2half2_rn(x.x, x.y); }
  __device__ static __half add(__half a, __half b) { return __hadd_rn(a, b); }
  __device__ static Pair add_pair(Pair a, Pair b) { return __hadd2_rn(a, b); }
};

template <>
struct Arithmetic<__nv_bfloat16> {
  using Pair = __nv_bfloat162;
  __device__ static float scalar(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ static __nv_bfloat16 cast(float x) { return __float2bfloat16_rn(x); }
  __device__ static float2 pair(Pair x) { return __bfloat1622float2(x); }
  __device__ static Pair cast2(float2 x) { return __floats2bfloat162_rn(x.x, x.y); }
  __device__ static __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd_rn(a, b);
  }
  __device__ static Pair add_pair(Pair a, Pair b) { return __hadd2_rn(a, b); }
};

template <>
struct Arithmetic<float> {
  using Pair = float2;
  __device__ static float scalar(float x) { return x; }
  __device__ static float cast(float x) { return x; }
  __device__ static float2 pair(Pair x) { return x; }
  __device__ static Pair cast2(float2 x) { return x; }
  __device__ static float add(float a, float b) { return __fadd_rn(a, b); }
  __device__ static Pair add_pair(Pair a, Pair b) { return add2(a, b); }
};

struct RowLayout {
  int64_t sizes[3];
  int64_t input_strides[3];
  int64_t residual_strides[3];
  int64_t input_linear_stride;
  int64_t residual_linear_stride;
  int hidden_size;
};

__device__ __forceinline__ int64_t row_offset(int row, const int64_t* sizes,
                                              const int64_t* strides) {
  const int64_t index2 = row % sizes[2];
  row /= sizes[2];
  const int64_t index1 = row % sizes[1];
  const int64_t index0 = row / sizes[1];
  return index0 * strides[0] + index1 * strides[1] + index2 * strides[2];
}

template <typename T>
__device__ __forceinline__ float2 accumulate(Vector<T> x, float2 accum) {
  using A = Arithmetic<T>;
  using Pair = typename A::Pair;
  constexpr int pairs = 16 / sizeof(Pair);
#pragma unroll
  for (int j = 0; j < pairs; ++j) {
    Pair p = reinterpret_cast<const Pair*>(&x)[j];
    float2 f = A::pair(p);
    accum = add2(accum, mul2(f, f));
  }
  return accum;
}

template <typename T, bool AddResidual>
__device__ __forceinline__ T finish_scalar(T x, T w, T residual, float scale) {
  using A = Arithmetic<T>;
  float normalized = __fmul_rn(A::scalar(x), scale);
  T result = A::cast(__fmul_rn(normalized, A::scalar(w)));
  if constexpr (AddResidual) result = A::add(residual, result);
  return result;
}

template <typename T, bool AddResidual>
__device__ __forceinline__ Vector<T> finish_vector(Vector<T> x, Vector<T> w,
                                                   Vector<T> residual, float scale) {
  using A = Arithmetic<T>;
  using Pair = typename A::Pair;
  constexpr int pairs = 16 / sizeof(Pair);
  Vector<T> result;
#pragma unroll
  for (int j = 0; j < pairs; ++j) {
    Pair x_pair = reinterpret_cast<const Pair*>(&x)[j];
    Pair w_pair = reinterpret_cast<const Pair*>(&w)[j];
    float2 normalized = mul2(A::pair(x_pair), make_float2(scale, scale));
    Pair value = A::cast2(mul2(normalized, A::pair(w_pair)));
    if constexpr (AddResidual) {
      Pair r_pair = reinterpret_cast<const Pair*>(&residual)[j];
      value = A::add_pair(r_pair, value);
    }
    reinterpret_cast<Pair*>(&result)[j] = value;
  }
  return result;
}

template <typename T, bool AddResidual, bool Aligned>
__device__ __forceinline__ void write_vector(T* out, const T* weight, const T* residual,
                                             Vector<T> x, int index, int H, float scale) {
  Vector<T> w, r{};
  if constexpr (Aligned) {
    *reinterpret_cast<uint4*>(&w) = *reinterpret_cast<const uint4*>(weight + index);
    if constexpr (AddResidual) {
      *reinterpret_cast<uint4*>(&r) = *reinterpret_cast<const uint4*>(residual + index);
    }
  } else {
    w = load_vector(weight, index, H);
    if constexpr (AddResidual) r = load_vector(residual, index, H);
  }
  Vector<T> result = finish_vector<T, AddResidual>(x, w, r, scale);
  *reinterpret_cast<uint4*>(out + index) = *reinterpret_cast<const uint4*>(&result);
}

template <typename T, int Threads, bool AddResidual, bool Aligned, int StaticH, int CachePacks>
__global__ void rms_norm_kernel(T* out, const T* input, const T* weight,
                                const T* residual, RowLayout layout, float epsilon) {
  constexpr int elements = 16 / sizeof(T);
  const int H = StaticH ? StaticH : layout.hidden_size;
  const int row = blockIdx.x;
  const T* x = input + (layout.input_linear_stride >= 0
      ? static_cast<int64_t>(row) * layout.input_linear_stride
      : row_offset(row, layout.sizes, layout.input_strides));
  const T* r = nullptr;
  if constexpr (AddResidual) {
    r = residual + (layout.residual_linear_stride >= 0
        ? static_cast<int64_t>(row) * layout.residual_linear_stride
        : row_offset(row, layout.sizes, layout.residual_strides));
  }
  T* y = out + static_cast<int64_t>(row) * H;
  int prefix = 0;
  if constexpr (!Aligned) {
    prefix = min(H, static_cast<int>(((16 - (reinterpret_cast<uintptr_t>(y) & 15)) & 15) /
                                     sizeof(T)));
  }
  const int vectors = (H - prefix) / elements;
  const int tail_start = prefix + vectors * elements;
  Vector<T> cached[CachePacks > 0 ? CachePacks : 1];
  float2 accum = make_float2(0.0f, 0.0f);
  T head_value{}, tail_value{};
  float edge_sum = 0.0f;
  if (threadIdx.x < prefix) {
    head_value = x[threadIdx.x];
    float f = Arithmetic<T>::scalar(head_value);
    edge_sum = __fmul_rn(f, f);
  }
  if (tail_start + threadIdx.x < H) {
    tail_value = x[tail_start + threadIdx.x];
    float f = Arithmetic<T>::scalar(tail_value);
    edge_sum = __fadd_rn(edge_sum, __fmul_rn(f, f));
  }
  if constexpr (CachePacks > 0) {
#pragma unroll
    for (int k = 0; k < CachePacks; ++k) {
      int index = threadIdx.x + k * Threads;
      if (index < vectors) {
        int j = prefix + index * elements;
        if constexpr (Aligned) {
          *reinterpret_cast<uint4*>(&cached[k]) = *reinterpret_cast<const uint4*>(x + j);
        } else {
          cached[k] = load_vector(x, j, H);
        }
        accum = accumulate(cached[k], accum);
      }
    }
  } else {
    for (int index = threadIdx.x; index < vectors; index += Threads) {
      int j = prefix + index * elements;
      Vector<T> value;
      if constexpr (Aligned) {
        *reinterpret_cast<uint4*>(&value) = *reinterpret_cast<const uint4*>(x + j);
      } else {
        value = load_vector(x, j, H);
      }
      accum = accumulate(value, accum);
    }
  }
  float sum = __fadd_rn(__fadd_rn(accum.x, accum.y), edge_sum);
  float scale;
  if constexpr (Threads == 32) {
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
      sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, delta));
    }
    scale = threadIdx.x == 0 ? rsqrtf(sum / H + epsilon) : 0.0f;
    scale = __shfl_sync(0xffffffff, scale, 0);
  } else {
    using Reduce = cub::BlockReduce<float, Threads>;
    __shared__ typename Reduce::TempStorage storage;
    __shared__ float shared_scale;
    sum = Reduce(storage).Sum(sum);
    if (threadIdx.x == 0) shared_scale = rsqrtf(sum / H + epsilon);
    __syncthreads();
    scale = shared_scale;
  }
  if (threadIdx.x < prefix) {
    int j = threadIdx.x;
    T rv{};
    if constexpr (AddResidual) rv = r[j];
    y[j] = finish_scalar<T, AddResidual>(head_value, weight[j], rv, scale);
  }
  if (tail_start + threadIdx.x < H) {
    int j = tail_start + threadIdx.x;
    T rv{};
    if constexpr (AddResidual) rv = r[j];
    y[j] = finish_scalar<T, AddResidual>(tail_value, weight[j], rv, scale);
  }
  if constexpr (CachePacks > 0) {
    // Constant indices let nvcc keep each saved vector in registers.
#pragma unroll
    for (int k = 0; k < CachePacks; ++k) {
      int index = threadIdx.x + k * Threads;
      if (index < vectors) {
        int j = prefix + index * elements;
        write_vector<T, AddResidual, Aligned>(y, weight, r, cached[k], j, H, scale);
      }
    }
  } else {
    for (int index = threadIdx.x; index < vectors; index += Threads) {
      int j = prefix + index * elements;
      Vector<T> xv;
      if constexpr (Aligned) {
        *reinterpret_cast<uint4*>(&xv) = *reinterpret_cast<const uint4*>(x + j);
      } else {
        xv = load_vector(x, j, H);
      }
      write_vector<T, AddResidual, Aligned>(y, weight, r, xv, j, H, scale);
    }
  }
}

template <typename T, int Threads, bool AddResidual, bool Aligned>
void launch_layout(at::Tensor& out, const at::Tensor& input, const at::Tensor& weight,
                   const std::optional<at::Tensor>& residual, RowLayout layout, int rows,
                   float epsilon, cudaStream_t stream) {
  const T* x = reinterpret_cast<const T*>(input.const_data_ptr());
  const T* w = reinterpret_cast<const T*>(weight.const_data_ptr());
  const T* r = residual ? reinterpret_cast<const T*>(residual->const_data_ptr()) : nullptr;
  T* y = reinterpret_cast<T*>(out.mutable_data_ptr());
  constexpr int elements = 16 / sizeof(T);
#define RUN_RMS(H, K) \
  rms_norm_kernel<T, Threads, AddResidual, Aligned, H, K><<<rows, Threads, 0, stream>>>( \
      y, x, w, r, layout, epsilon)
  if constexpr (Aligned && Threads == 256) {
    if (layout.hidden_size == 2048) {
      RUN_RMS(2048, (2048 + Threads * elements - 1) / (Threads * elements));
      return;
    }
  }
  // Cache small rows; stream very wide rows to avoid unbounded register pressure.
  const int packs = (layout.hidden_size + Threads * elements - 1) / (Threads * elements);
  if (packs <= 1) { RUN_RMS(0, 1); }
  else if (packs <= 2) { RUN_RMS(0, 2); }
  else if (packs <= 4) { RUN_RMS(0, 4); }
  else { RUN_RMS(0, 0); }
#undef RUN_RMS
}

template <typename T, bool AddResidual>
void launch(at::Tensor& out, const at::Tensor& input, const at::Tensor& weight,
            const std::optional<at::Tensor>& residual, float epsilon) {
  constexpr int elements = 16 / sizeof(T);
  RowLayout layout{};
  for (int i = 0; i < 3; ++i) layout.sizes[i] = 1;
  const int outer = input.dim() - 1;
  // A singleton penultimate dimension can have an arbitrary stride even when
  // the tensor is contiguous, so a contiguous row advances by hidden_size.
  layout.input_linear_stride = input.is_contiguous()
      ? (outer ? input.size(-1) : 0)
      : (input.dim() <= 2 ? input.stride(-2) : -1);
  if constexpr (AddResidual) {
    layout.residual_linear_stride = residual->is_contiguous()
        ? (outer ? residual->size(-1) : 0)
        : (residual->dim() <= 2 ? residual->stride(-2) : -1);
  }
  bool aligned = ((reinterpret_cast<uintptr_t>(input.const_data_ptr()) |
                   reinterpret_cast<uintptr_t>(weight.const_data_ptr()) |
                   reinterpret_cast<uintptr_t>(out.mutable_data_ptr())) & 15) == 0;
  if constexpr (AddResidual) {
    aligned = aligned && (reinterpret_cast<uintptr_t>(residual->const_data_ptr()) & 15) == 0;
  }
  for (int d = 0; d < outer; ++d) {
    int target = 3 - outer + d;
    layout.sizes[target] = input.size(d);
    layout.input_strides[target] = input.stride(d);
    aligned = aligned && (input.size(d) == 1 || input.stride(d) % elements == 0);
    if constexpr (AddResidual) {
      layout.residual_strides[target] = residual->stride(d);
      aligned = aligned && (input.size(d) == 1 || residual->stride(d) % elements == 0);
    }
  }
  layout.hidden_size = static_cast<int>(input.size(-1));
  aligned = aligned && layout.hidden_size % elements == 0;
  const int rows = static_cast<int>(input.numel() / layout.hidden_size);
  auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  if (layout.hidden_size <= 256) {
    if (aligned) launch_layout<T, 32, AddResidual, true>(out, input, weight, residual, layout, rows, epsilon, stream);
    else launch_layout<T, 32, AddResidual, false>(out, input, weight, residual, layout, rows, epsilon, stream);
  } else {
    if (aligned) launch_layout<T, 256, AddResidual, true>(out, input, weight, residual, layout, rows, epsilon, stream);
    else launch_layout<T, 256, AddResidual, false>(out, input, weight, residual, layout, rows, epsilon, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

at::Tensor rms_norm_cuda(const at::Tensor& input, const at::Tensor& weight, double epsilon,
                         const std::optional<at::Tensor>& residual) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "rms_norm expects CUDA inputs");
  TORCH_CHECK(input.device() == weight.device(), "rms_norm expects inputs on the same device");
  TORCH_CHECK(input.dim() >= 1 && input.dim() <= 4, "rms_norm expects rank 1 through 4");
  TORCH_CHECK(input.stride(-1) == 1, "rms_norm expects a contiguous last dimension");
  TORCH_CHECK(weight.dim() == 1 && weight.is_contiguous() && weight.size(0) == input.size(-1),
              "rms_norm expects contiguous weight with shape [hidden_size]");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "rms_norm expects matching dtypes");
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "rms_norm supports float32, float16 and bfloat16");
  if (residual) {
    TORCH_CHECK(residual->is_cuda() && residual->device() == input.device(),
                "rms_norm expects residual on the input device");
    TORCH_CHECK(residual->sizes() == input.sizes() && residual->stride(-1) == 1 &&
                    residual->scalar_type() == input.scalar_type(),
                "rms_norm expects matching residual shape, dtype and contiguous last dimension");
  }
  TORCH_CHECK(!at::GradMode::is_enabled() ||
                  (!input.requires_grad() && !weight.requires_grad() &&
                   (!residual || !residual->requires_grad())),
              "rms_norm is inference-only; use torch.no_grad() or the Torch reference");
  TORCH_CHECK(input.size(-1) <= std::numeric_limits<int>::max(), "hidden size is too large");
  const c10::cuda::CUDAGuard guard(input.device());
  auto out = at::empty(input.sizes(), input.options());
  if (input.numel() == 0) return out;
  TORCH_CHECK(input.numel() / input.size(-1) <= std::numeric_limits<int>::max(),
              "too many rms_norm rows");
#define LAUNCH_RMS(T) \
  if (residual) launch<T, true>(out, input, weight, residual, epsilon); \
  else launch<T, false>(out, input, weight, residual, epsilon)
  switch (input.scalar_type()) {
    case at::kHalf: LAUNCH_RMS(__half); break;
    case at::kBFloat16: LAUNCH_RMS(__nv_bfloat16); break;
    case at::kFloat: LAUNCH_RMS(float); break;
    default: TORCH_CHECK(false, "unsupported dtype");
  }
#undef LAUNCH_RMS
  return out;
}

}  // namespace vllm_rlt
