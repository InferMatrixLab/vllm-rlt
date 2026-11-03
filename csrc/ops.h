// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>

namespace vllm_rlt {

at::Tensor silu_and_mul_cuda(const at::Tensor& gate, const at::Tensor& up);
at::Tensor rms_norm_cuda(const at::Tensor& input, const at::Tensor& weight, double epsilon,
                         const std::optional<at::Tensor>& residual, bool match_torch);

}  // namespace vllm_rlt
