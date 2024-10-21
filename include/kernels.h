// include/kernels.h:
// Declarations for all CUDA kernels and device functions
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Custom data types
typedef int8_t ternary_t;

struct PackedTernary {
    uint32_t data;
};

// Device function declarations
__device__ float gelu(float x);
__device__ float rms_norm(float x, float epsilon);
__device__ ternary_t quantize_weight(float w, float scale);
__device__ int8_t quantize_activation(float x, float scale);
__device__ float dequantize_activation(int8_t x, float scale);
__device__ uint32_t pack_ternary(const ternary_t* values);
__device__ ternary_t unpack_ternary(uint32_t packed, int index);
__device__ float warp_reduce_sum(float val);

// Kernel declarations
__global__ void bitnet_quantize_weights_kernel(
    const float* __restrict__ weights,
    PackedTernary* __restrict__ quantized_weights,
    float* __restrict__ scales,
    int size
);

__global__ void bitnet_dequantize_weights_kernel(
    const PackedTernary* __restrict__ quantized_weights,
    float* __restrict__ weights,
    const float* __restrict__ scales,
    int size
);

__global__ void bitnet_matmul_kernel(
    const PackedTernary* __restrict__ A,
    const int8_t* __restrict__ B,
    int8_t* __restrict__ C,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    float* __restrict__ C_scale,
    int M, int N, int K
);

__global__ void bitnet_rmsnorm_activation_kernel(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ input_scale,
    float* __restrict__ output_scale,
    int N, int hidden_size
);

__global__ void bitnet_linear_kernel(
    const int8_t* __restrict__ input,
    const PackedTernary* __restrict__ weight,
    int8_t* __restrict__ output,
    const float* __restrict__ input_scale,
    const float* __restrict__ weight_scale,
    float* __restrict__ output_scale,
    int batch_size, int in_features, int out_features
);

__global__ void bitnet_persistent_kernel(
    const int8_t* __restrict__ global_input,
    int8_t* __restrict__ global_output,
    const PackedTernary* __restrict__ global_weight,
    const float* __restrict__ global_scales,
    int* __restrict__ global_work_counter,
    int total_work_items,
    int batch_size, int in_features, int out_features
);

__global__ void bitnet_fused_quantize_matmul_kernel(
    const float* __restrict__ A,
    const int8_t* __restrict__ B,
    int8_t* __restrict__ C,
    float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    float* __restrict__ C_scale,
    int M, int N, int K
);

__global__ void bitnet_mixed_precision_matmul_kernel(
    const PackedTernary* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    int M, int N, int K
);

__global__ void bitnet_adaptive_quantize_kernel(
    const float* __restrict__ weights,
    PackedTernary* __restrict__ quantized_weights,
    float* __restrict__ scales,
    int size,
    float* __restrict__ running_mean,
    float* __restrict__ running_var
);

__global__ void bitnet_fused_quantize_layernorm_kernel(
    const float* __restrict__ input,
    PackedTernary* __restrict__ quantized_output,
    float* __restrict__ scales,
    float* __restrict__ gamma,
    float* __restrict__ beta,
    int size,
    int hidden_size
);

__global__ void bitnet_warp_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
);

// Tensor Core optimized kernel (for compatible architectures)
#if __CUDA_ARCH__ >= 700
__global__ void bitnet_tensor_core_matmul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    int M, int N, int K
);
#endif