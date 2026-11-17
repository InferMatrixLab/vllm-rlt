// SPDX-License-Identifier: Apache-2.0
// Host-side eager execution only: every operation below dispatches separately
// through ATen to the existing Torch-NPU implementation. No graph or fusion.
#include <torch/extension.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <vector>

using at::Tensor;

static Tensor rms(const Tensor& x, const Tensor& weight, double eps) {
    auto value = x.to(at::kFloat);
    auto variance = value.pow(2).mean(-1, true);
    value = value * at::rsqrt(variance + eps);
    return weight * value.to(x.scalar_type());
}

static Tensor rotate_half(const Tensor& x) {
    auto halves = x.chunk(2, -1);
    return at::cat({-halves[1], halves[0]}, -1);
}

std::vector<Tensor> layer_forward(
    const Tensor& hidden, const Tensor& cosine, const Tensor& sine,
    const std::vector<Tensor>& weights, const std::vector<Tensor>& history,
    const Tensor& mask, int64_t rows, int64_t width, int64_t start,
    int64_t head_dim, int64_t kv_heads, double eps) {
    c10::InferenceMode guard(true);
    TORCH_CHECK(weights.size() == 11, "Ouro layer plan requires eleven weights");
    TORCH_CHECK(rows > 0 && width > 0 && start >= 0 && hidden.size(0) == rows * width,
                "invalid eager traversal geometry");
    TORCH_CHECK(history.size() == (start ? 2 : 0), "invalid eager KV history");
    auto normalized = rms(hidden, weights[0], eps);
    auto q = at::linear(normalized, weights[4]).view({rows * width, -1, head_dim});
    auto k = at::linear(normalized, weights[5]).view({rows * width, -1, head_dim});
    auto v = at::linear(normalized, weights[6]).view({rows * width, -1, head_dim});
    q = q * cosine + rotate_half(q) * sine;
    k = k * cosine + rotate_half(k) * sine;
    auto incoming_k = k.reshape({rows, width, kv_heads, head_dim}).transpose(1, 2);
    auto incoming_v = v.reshape({rows, width, kv_heads, head_dim}).transpose(1, 2);
    Tensor key, value;
    if (!start) {
        key = incoming_k.clone();
        value = incoming_v.clone();
    } else {
        key = at::cat({history[0], incoming_k}, 2);
        value = at::cat({history[1], incoming_v}, 2);
    }
    // Published cache tensors own their bytes; attention's head expansion is
    // temporary and must not replace the original KV-head representation.
    auto stored_key = key;
    auto stored_value = value;
    auto groups = q.size(1) / kv_heads;
    TORCH_CHECK(groups > 0 && q.size(1) % kv_heads == 0, "invalid grouped heads");
    if (groups != 1) {
        key = at::repeat_interleave(key, groups, 1);
        value = at::repeat_interleave(value, groups, 1);
    }
    auto query = width == 1
        ? q.contiguous().reshape({rows, q.size(1), 1, head_dim})
        : q.reshape({rows, width, q.size(1), head_dim}).transpose(1, 2).contiguous();
    auto scores = at::matmul(query, key.transpose(-2, -1)) * (1.0 / std::sqrt(double(head_dim)));
    scores = scores + mask.unsqueeze(0).unsqueeze(0);
    auto probabilities = at::softmax(scores, -1, at::kFloat).to(q.scalar_type());
    Tensor attention;
    if (start && width > 1 && rows > 1) {
        std::vector<Tensor> outputs;
        outputs.reserve(rows);
        for (int64_t i = 0; i < rows; ++i) {
            outputs.push_back(at::matmul(probabilities.slice(0, i, i + 1),
                                        value.slice(0, i, i + 1)));
        }
        attention = at::cat(outputs, 0);
    } else {
        attention = at::matmul(probabilities, value);
    }
    attention = width == 1 ? attention.squeeze(2)
                          : attention.transpose(1, 2).reshape({rows * width, q.size(1), head_dim});
    attention = at::linear(attention.reshape({rows * width, -1}), weights[7]);
    auto state = hidden + rms(attention, weights[1], eps);
    auto mlp_input = rms(state, weights[2], eps);
    auto gate = at::linear(mlp_input, weights[8]);
    auto up = at::linear(mlp_input, weights[9]);
    auto mlp = at::linear(at::silu(gate) * up, weights[10]);
    auto output = state + rms(mlp, weights[3], eps);
    return {output, stored_key, stored_value};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("layer_forward", &layer_forward,
               pybind11::call_guard<pybind11::gil_scoped_release>());
}
