// include/pytorch_extension.h:
// BitNetPyTorchExtension class

// include/pytorch_extension.h

#pragma once

#include <torch/extension.h>
#include "bitnet_cuda.h"

class BitNetPyTorchExtension {
private:
    std::unique_ptr<BitNetCUDA> bitnet_cuda;

public:
    BitNetPyTorchExtension() : bitnet_cuda(std::make_unique<BitNetCUDA>()) {}

    torch::Tensor quantize_weights(torch::Tensor weights) {
        TORCH_CHECK(weights.is_cuda(), "Input tensor must be on CUDA device");
        TORCH_CHECK(weights.dim() == 2, "Input tensor must be 2-dimensional");

        int rows = weights.size(0);
        int cols = weights.size(1);

        auto quantized = torch::empty({rows, (cols + 15) / 16}, torch::dtype(torch::kInt32).device(weights.device()));
        auto scales = torch::empty({rows}, torch::dtype(torch::kFloat32).device(weights.device()));

        bitnet_cuda->quantize_weights(
            weights.data_ptr<float>(),
            reinterpret_cast<PackedTernary*>(quantized.data_ptr<int32_t>()),
            scales.data_ptr<float>(),
            rows, cols
        );

        return std::make_tuple(quantized, scales);
    }

    torch::Tensor dequantize_weights(torch::Tensor quantized, torch::Tensor scales, int original_cols) {
        TORCH_CHECK(quantized.is_cuda(), "Quantized tensor must be on CUDA device");
        TORCH_CHECK(scales.is_cuda(), "Scales tensor must be on CUDA device");
        TORCH_CHECK(quantized.dim() == 2, "Quantized tensor must be 2-dimensional");
        TORCH_CHECK(scales.dim() == 1, "Scales tensor must be 1-dimensional");

        int rows = quantized.size(0);
        int packed_cols = quantized.size(1);

        auto dequantized = torch::empty({rows, original_cols}, torch::dtype(torch::kFloat32).device(quantized.device()));

        bitnet_cuda->dequantize_weights(
            reinterpret_cast<const PackedTernary*>(quantized.data_ptr<int32_t>()),
            dequantized.data_ptr<float>(),
            scales.data_ptr<float>(),
            rows, original_cols
        );

        return dequantized;
    }

    torch::Tensor bitnet_linear(torch::Tensor input, torch::Tensor quantized_weight, torch::Tensor weight_scales) {
        TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
        TORCH_CHECK(quantized_weight.is_cuda(), "Quantized weight tensor must be on CUDA device");
        TORCH_CHECK(weight_scales.is_cuda(), "Weight scales tensor must be on CUDA device");

        int batch_size = input.size(0);
        int in_features = input.size(1);
        int out_features = quantized_weight.size(0);

        auto output = torch::empty({batch_size, out_features}, torch::dtype(torch::kFloat32).device(input.device()));

        bitnet_cuda->bitnet_linear(
            input.data_ptr<float>(),
            reinterpret_cast<const PackedTernary*>(quantized_weight.data_ptr<int32_t>()),
            output.data_ptr<float>(),
            weight_scales.data_ptr<float>(),
            batch_size, in_features, out_features
        );

        return output;
    }

    torch::Tensor bitnet_matmul(torch::Tensor a, torch::Tensor b) {
        TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Input tensors must be on CUDA device");
        TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Input tensors must be 2-dimensional");
        TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions of matrices must match");

        int m = a.size(0);
        int k = a.size(1);
        int n = b.size(1);

        auto c = torch::empty({m, n}, torch::dtype(torch::kFloat32).device(a.device()));

        bitnet_cuda->bitnet_matmul(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            m, n, k
        );

        return c;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BitNetPyTorchExtension>(m, "BitNetPyTorchExtension")
        .def(py::init<>())
        .def("quantize_weights", &BitNetPyTorchExtension::quantize_weights)
        .def("dequantize_weights", &BitNetPyTorchExtension::dequantize_weights)
        .def("bitnet_linear", &BitNetPyTorchExtension::bitnet_linear)
        .def("bitnet_matmul", &BitNetPyTorchExtension::bitnet_matmul);
}