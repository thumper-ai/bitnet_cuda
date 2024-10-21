#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward declarations of our CUDA kernels
torch::Tensor bitnet_quantize_weights_cuda(torch::Tensor weights, torch::Tensor scales);
torch::Tensor bitnet_fused_quantize_matmul_cuda(torch::Tensor input, torch::Tensor weights, torch::Tensor input_scale, torch::Tensor weight_scale);
torch::Tensor bitnet_layernorm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps);

// C++ wrapper for the quantize_weights CUDA kernel
torch::Tensor quantize_weights(torch::Tensor weights, torch::Tensor scales) {
    TORCH_CHECK(weights.device().is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(scales.device().is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(weights.dim() == 2, "weights must be 2-dimensional");
    TORCH_CHECK(scales.dim() == 1, "scales must be 1-dimensional");
    TORCH_CHECK(scales.size(0) == weights.size(0), "scales must have the same first dimension as weights");

    return bitnet_quantize_weights_cuda(weights, scales);
}

// C++ wrapper for the fused_quantize_matmul CUDA kernel
torch::Tensor fused_quantize_matmul(torch::Tensor input, torch::Tensor weights, torch::Tensor input_scale, torch::Tensor weight_scale) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weights.device().is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(input_scale.device().is_cuda(), "input_scale must be a CUDA tensor");
    TORCH_CHECK(weight_scale.device().is_cuda(), "weight_scale must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2-dimensional");
    TORCH_CHECK(weights.dim() == 2, "weights must be 2-dimensional");
    TORCH_CHECK(input_scale.dim() == 1, "input_scale must be 1-dimensional");
    TORCH_CHECK(weight_scale.dim() == 1, "weight_scale must be 1-dimensional");
    TORCH_CHECK(input.size(1) == weights.size(0), "input and weights dimensions must match for multiplication");

    return bitnet_fused_quantize_matmul_cuda(input, weights, input_scale, weight_scale);
}

// C++ wrapper for the layernorm CUDA kernel
torch::Tensor layernorm(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(gamma.device().is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.device().is_cuda(), "beta must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2-dimensional");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1-dimensional");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1-dimensional");
    TORCH_CHECK(gamma.size(0) == input.size(1), "gamma must have the same size as input's last dimension");
    TORCH_CHECK(beta.size(0) == input.size(1), "beta must have the same size as input's last dimension");

    return bitnet_layernorm_cuda(input, gamma, beta, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_weights", &quantize_weights, "BitNet quantize weights (CUDA)");
    m.def("fused_quantize_matmul", &fused_quantize_matmul, "BitNet fused quantize matmul (CUDA)");
    m.def("layernorm", &layernorm, "BitNet layer normalization (CUDA)");
}