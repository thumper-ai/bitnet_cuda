// kernels/linear_kernels.cu:
// bitnet_linear_kernel

// src/kernels/linear_kernels.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "bitnet_types.h"

__global__ void bitnet_linear_kernel(
    const quant_t* __restrict__ input,
    const PackedTernary* __restrict__ weight,
    quant_t* __restrict__ output,
    const float* __restrict__ input_scale,
    const float* __restrict__ weight_scale,
    float* __restrict__ output_scale,
    int batch_size, int in_features, int out_features
) {
    extern __shared__ quant_t shared_input[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int row = blockIdx.y;

    float thread_sum = 0.0f;
    float local_input_scale = input_scale[0];
    float local_weight_scale = weight_scale[0];

    // Load input into shared memory
    for (int i = tid; i < in_features; i += blockDim.x) {
        shared_input[i] = input[row * in_features + i];
    }
    __syncthreads();

    // Compute dot product
    for (int i = 0; i < in_features; i += 16) {
        uint32_t packed_weight = weight[(bid * in_features + i) / 16].data;
        
        #pragma unroll
        for (int j = 0; j < 16 && (i + j) < in_features; ++j) {
            ternary_t w = unpack_ternary(packed_weight, j);
            float x = dequantize_activation(shared_input[i + j], local_input_scale);
            thread_sum += w * x;
        }
    }

    // Apply scaling
    thread_sum *= local_weight_scale;

    // Quantize output
    quant_t quantized_output = quantize_activation(thread_sum, *output_scale);

    // Write output
    if (row < batch_size && bid < out_features) {
        output[row * out_features + bid] = quantized_output;
    }
}

// Helper function to launch the kernel
void launch_bitnet_linear_kernel(
    const quant_t* input,
    const PackedTernary* weight,
    quant_t* output,
    const float* input_scale,
    const float* weight_scale,
    float* output_scale,
    int batch_size, int in_features, int out_features,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((out_features + block.x - 1) / block.x, batch_size);
    size_t shared_mem_size = in_features * sizeof(quant_t);

    bitnet_linear_kernel<<<grid, block, shared_mem_size, stream>>>(
        input, weight, output, input_scale, weight_scale, output_scale,
        batch_size, in_features, out_features
    );
}

// This implementation includes several optimizations:
// Shared Memory Usage: The input is loaded into shared memory to reduce global memory accesses.
// Packed Ternary Weights: The weights are stored in a packed format (16 ternary values per 32-bit integer) to reduce memory usage and bandwidth.
// Vectorized Memory Access: The packed weights are loaded as 32-bit integers for efficient memory access.
// Loop Unrolling: The inner loop is unrolled for better instruction-level parallelism.
// Quantization: The input is dequantized, and the output is quantized using the provided scaling factors.
// Flexible Dimensions: The kernel can handle arbitrary batch sizes, input features, and output features.
// CUDA Stream Support: The launch function accepts a CUDA stream for asynchronous execution.
// To use this kernel, you would typically call the launch_bitnet_linear_kernel function from your main CUDA code. For example:
// cpp
// launch_bitnet_linear_kernel(
//     d_input, d_weight, d_output, d_input_scale, d_weight_scale, d_output_scale,
//     batch_size, in_features, out_features, stream
// );

// Remember to error-check the kernel launch and to profile the kernel performance on your specific hardware. You may need to adjust the block size or other parameters for optimal performance on different GPU architectures.