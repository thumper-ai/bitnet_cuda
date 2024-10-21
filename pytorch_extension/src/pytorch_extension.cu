// src/pytorch_extension.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "bitnet_cuda.h"  // Include our main BitNet CUDA header

// Declare the CUDA kernels (implemented in other files)
__global__ void bitnet_quantize_weights_kernel(const float* weights, PackedTernary* quantized_weights, float* scales, int size);
__global__ void bitnet_dequantize_weights_kernel(const PackedTernary* quantized_weights, float* weights, const float* scales, int size);
__global__ void bitnet_matmul_kernel(const PackedTernary* A, const half* B, half* C, const float* A_scale, const float* B_scale, int M, int N, int K);

// Helper function to check CUDA errors
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Wrapper for quantize_weights kernel
torch::Tensor bitnet_quantize_weights_cuda(torch::Tensor weights) {
    CHECK_INPUT(weights);
    
    auto size = weights.numel();
    auto quantized = torch::empty({(size + 15) / 16}, torch::dtype(torch::kInt32).device(weights.device()));
    auto scales = torch::empty({1}, torch::dtype(torch::kFloat32).device(weights.device()));
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    bitnet_quantize_weights_kernel<<<blocks, threads>>>(
        weights.data_ptr<float>(),
        reinterpret_cast<PackedTernary*>(quantized.data_ptr<int32_t>()),
        scales.data_ptr<float>(),
        size
    );
    
    return std::make_tuple(quantized, scales);
}

// Wrapper for dequantize_weights kernel
torch::Tensor bitnet_dequantize_weights_cuda(torch::Tensor quantized_weights, torch::Tensor scales, int original_size) {
    CHECK_INPUT(quantized_weights);
    CHECK_INPUT(scales);
    
    auto weights = torch::empty({original_size}, torch::dtype(torch::kFloat32).device(quantized_weights.device()));
    
    const int threads = 256;
    const int blocks = (original_size + threads - 1) / threads;
    
    bitnet_dequantize_weights_kernel<<<blocks, threads>>>(
        reinterpret_cast<const PackedTernary*>(quantized_weights.data_ptr<int32_t>()),
        weights.data_ptr<float>(),
        scales.data_ptr<float>(),
        original_size
    );
    
    return weights;
}

// Wrapper for matmul kernel
torch::Tensor bitnet_matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor A_scale, torch::Tensor B_scale) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(A_scale);
    CHECK_INPUT(B_scale);
    
    int M = A.size(0);
    int K = A.size(1) * 16;  // Packed dimension
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, torch::dtype(torch::kHalf).device(A.device()));
    
    const dim3 threads(16, 16);
    const dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    
    bitnet_matmul_kernel<<<blocks, threads>>>(
        reinterpret_cast<const PackedTernary*>(A.data_ptr<int32_t>()),
        B.data_ptr<at::Half>(),
        C.data_ptr<at::Half>(),
        A_scale.data_ptr<float>(),
        B_scale.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}

// Python module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_weights", &bitnet_quantize_weights_cuda, "BitNet weight quantization (CUDA)");
    m.def("dequantize_weights", &bitnet_dequantize_weights_cuda, "BitNet weight dequantization (CUDA)");
    m.def("matmul", &bitnet_matmul_cuda, "BitNet matrix multiplication (CUDA)");
}

// BitNetLinear layer implementation
class BitNetLinearFunction : public torch::autograd::Function<BitNetLinearFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        torch::Tensor weight_scale
    ) {
        auto quantized_weight = bitnet_quantize_weights_cuda(weight);
        auto output = bitnet_matmul_cuda(std::get<0>(quantized_weight), input.t(), std::get<1>(quantized_weight), torch::ones_like(std::get<1>(quantized_weight)));
        
        if (bias.defined()) {
            output += bias.unsqueeze(0).expand_as(output);
        }
        
        ctx->save_for_backward({input, weight, bias, weight_scale});
        return output;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];
        auto weight_scale = saved[3];
        
        auto grad_output = grad_outputs[0];
        
        // Implement backward pass
        auto quantized_weight = bitnet_quantize_weights_cuda(weight);
        auto grad_input = bitnet_matmul_cuda(std::get<0>(quantized_weight).t(), grad_output, std::get<1>(quantized_weight), torch::ones_like(std::get<1>(quantized_weight)));
        
        auto grad_weight = bitnet_matmul_cuda(grad_output.t(), input, torch::ones_like(std::get<1>(quantized_weight)), torch::ones_like(std::get<1>(quantized_weight)));
        grad_weight = bitnet_dequantize_weights_cuda(std::get<0>(grad_weight), std::get<1>(grad_weight), weight.numel());
        
        torch::Tensor grad_bias;
        if (bias.defined()) {
            grad_bias = grad_output.sum(0);
        }
        
        return {grad_input, grad_weight, grad_bias, torch::Tensor()};
    }
};

torch::Tensor bitnet_linear(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor weight_scale
) {
    return BitNetLinearFunction::apply(input, weight, bias, weight_scale);
}

// Register the custom op
static auto registry = 
    torch::RegisterOperators("mynamespace::bitnet_linear", &bitnet_linear);


//     This implementation includes:
// CUDA kernel wrappers for weight quantization, dequantization, and matrix multiplication.
// A PyTorch extension module that exposes these CUDA operations to Python.
// A custom BitNetLinearFunction that implements the forward and backward passes for a BitNet linear layer.
// Registration of a custom PyTorch operator for the BitNet linear layer.
// To use this extension in your PyTorch project:
// Compile this file as a PyTorch extension (you'll need to set up a setup.py file).
// Import the extension in your Python code:
// python
// import torch
// from torch.utils.cpp_extension import load

// bitnet_cuda = load(name="bitnet_cuda", sources=["pytorch_extension.cu"])

// # Use the custom ops
// quantized_weight, scale = bitnet_cuda.quantize_weights(weight)
// output = bitnet_cuda.matmul(quantized_weight, input, scale, torch.ones_like(scale))

// # Or use the custom BitNet linear layer
// output = bitnet_cuda.bitnet_linear(input, weight, bias, weight_scale)

// This implementation provides a seamless integration of BitNet operations with PyTorch, allowing you to use BitNet layers in your models with automatic differentiation support. Remember to compile the extension with the appropriate CUDA flags and ensure that all necessary CUDA headers and BitNet-specific headers are available during compilation.