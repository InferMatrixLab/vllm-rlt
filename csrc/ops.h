// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>

namespace vllm_rlt {

at::Tensor silu_and_mul_cuda(const at::Tensor& gate, const at::Tensor& up);

}  // namespace vllm_rlt
