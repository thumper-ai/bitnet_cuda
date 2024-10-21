// kernels/normalization_kernels.cu:
// bitnet_rmsnorm_activation_kernel
// fused_quantize_layernorm_kernel

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "bitnet_types.h"

// Helper function for GELU activation
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void bitnet_rmsnorm_activation_kernel(
    const quant_t* __restrict__ input,
    quant_t* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ input_scale,
    float* __restrict__ output_scale,
    int N, int hidden_size
) {
    extern __shared__ float s_sum[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * hidden_size + tid;
    
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = dequantize_activation(input[idx + i], *input_scale);
        thread_sum_sq += x * x;
    }
    
    // Warp-level reduction
    thread_sum_sq = warp_reduce_sum(thread_sum_sq);
    
    // Block-level reduction
    if (tid % 32 == 0) {
        s_sum[tid / 32] = thread_sum_sq;
    }
    __syncthreads();
    
    if (tid < 32) {
        thread_sum_sq = (tid < (blockDim.x + 31) / 32) ? s_sum[tid] : 0.0f;
        thread_sum_sq = warp_reduce_sum(thread_sum_sq);
        if (tid == 0) {
            s_sum[0] = thread_sum_sq;
        }
    }
    __syncthreads();
    
    float rms = rsqrtf(s_sum[0] / hidden_size + 1e-6f);
    
    // Compute max absolute value for output quantization
    float max_abs = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = dequantize_activation(input[idx + i], *input_scale);
        float normalized = x * rms * weight[i];
        float activated = gelu(normalized);
        max_abs = fmaxf(max_abs, fabsf(activated));
    }
    
    // Warp-level reduction for max_abs
    max_abs = warp_reduce_max(max_abs);
    
    // Block-level reduction for max_abs
    if (tid == 0) {
        atomicMax((int*)output_scale, __float_as_int(max_abs));
    }
    __syncthreads();
    
    // Quantize and store output
    float out_scale = *output_scale / 127.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = dequantize_activation(input[idx + i], *input_scale);
        float normalized = x * rms * weight[i];
        float activated = gelu(normalized);
        output[idx + i] = quantize_activation(activated, out_scale);
    }
}

__global__ void fused_quantize_layernorm_kernel(
    const float* __restrict__ input,
    quant_t* __restrict__ quantized_output,
    float* __restrict__ scales,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batch_size, int hidden_size
) {
    extern __shared__ float s_data[];
    float* s_mean = s_data;
    float* s_var = &s_data[blockDim.x];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * hidden_size + tid;
    
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = input[idx + i];
        thread_sum += x;
        thread_sum_sq += x * x;
    }
    
    // Warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);
    thread_sum_sq = warp_reduce_sum(thread_sum_sq);
    
    // Block-level reduction
    if (tid % 32 == 0) {
        s_mean[tid / 32] = thread_sum;
        s_var[tid / 32] = thread_sum_sq;
    }
    __syncthreads();
    
    if (tid < 32) {
        thread_sum = (tid < (blockDim.x + 31) / 32) ? s_mean[tid] : 0.0f;
        thread_sum_sq = (tid < (blockDim.x + 31) / 32) ? s_var[tid] : 0.0f;
        thread_sum = warp_reduce_sum(thread_sum);
        thread_sum_sq = warp_reduce_sum(thread_sum_sq);
        if (tid == 0) {
            s_mean[0] = thread_sum / hidden_size;
            s_var[0] = thread_sum_sq / hidden_size - s_mean[0] * s_mean[0];
        }
    }
    __syncthreads();
    
    float mean = s_mean[0];
    float inv_std = rsqrtf(s_var[0] + 1e-6f);
    
    // Compute max absolute value for quantization
    float max_abs = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = input[idx + i];
        float normalized = (x - mean) * inv_std;
        float scaled_shifted = gamma[i] * normalized + beta[i];
        max_abs = fmaxf(max_abs, fabsf(scaled_shifted));
    }
    
    // Warp-level reduction for max_abs
    max_abs = warp_reduce_max(max_abs);
    
    // Block-level reduction for max_abs
    if (tid == 0) {
        scales[bid] = max_abs / 127.0f;
    }
    __syncthreads();
    
    // Quantize and store output
    float scale = scales[bid];
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = input[idx + i];
        float normalized = (x - mean) * inv_std;
        float scaled_shifted = gamma[i] * normalized + beta[i];
        quantized_output[idx + i] = quantize_activation(scaled_shifted, scale);
    }
}

// Helper function for warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Helper function for warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}


// This implementation includes:
// bitnet_rmsnorm_activation_kernel: This kernel performs RMSNorm, followed by GELU activation, and then quantizes the result. It uses shared memory and warp-level reductions for efficient computation of the normalization statistics.
// fused_quantize_layernorm_kernel: This kernel fuses layer normalization with quantization. It computes the mean and variance, normalizes the input, applies scaling and shifting, and then quantizes the result.
// Both kernels use warp-level reductions for efficiency and compute the quantization scale dynamically based on the maximum absolute value of the output.
// Key optimizations and features:
// Use of shared memory for efficient computation of statistics
// Warp-level reductions to minimize synchronization points
// Fused operations to reduce memory bandwidth usage
// Dynamic computation of quantization scales
// Use of inline CUDA math functions for better performance
// To use these kernels, you would typically launch them with appropriate grid and block dimensions, and with the correct amount of shared memory allocated. For example:
// cuda
// int block_size = 256;
// int grid_size = N;
// size_t shared_mem_size = 2 * sizeof(float) * ((block_size + 31) / 32);

// bitnet_rmsnorm_activation_kernel<<<grid_size, block_size, shared_mem_size>>>(
//     input, output, weight, input_scale, output_scale, N, hidden_size);

// fused_quantize_layernorm_kernel<<<batch_size, block_size, shared_mem_size>>>(
//     input, quantized_output, scales, gamma, beta, batch_size, hidden_size);

// Remember to profile these kernels on your specific hardware and with your typical data sizes to ensure they provide the expected performance improvements. You may need to adjust parameters like block sizes or shared memory usage for optimal performance on different GPU architectures.