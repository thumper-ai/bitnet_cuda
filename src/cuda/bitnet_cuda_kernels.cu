#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for quantizing weights
__global__ void quantize_weights_kernel(
    const float* __restrict__ weights,
    int8_t* __restrict__ quantized_weights,
    const float* __restrict__ scales,
    int size, int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int row = idx / hidden_dim;
        float scale = scales[row];
        float w = weights[idx];
        quantized_weights[idx] = (w > 0.5f / scale) ? 1 : ((w < -0.5f / scale) ? -1 : 0);
    }
}

// CUDA kernel for fused quantize and matrix multiplication
__global__ void fused_quantize_matmul_kernel(
    const float* __restrict__ input,
    const int8_t* __restrict__ weights,
    float* __restrict__ output,
    const float* __restrict__ input_scale,
    const float* __restrict__ weight_scale,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float in_val = input[row * K + k] * input_scale[0];
            int8_t w_val = weights[k * N + col];
            sum += in_val * (float)w_val;
        }
        output[row * N + col] = sum * weight_scale[0];
    }
}

// CUDA kernel for layer normalization
__global__ void layernorm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int M, int N, float eps
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ float mean_s, var_s;

    for (int m = 0; m < M; ++m) {
        float sum = 0.0f, sq_sum = 0.0f;
        for (int i = tid; i < N; i += stride) {
            float val = input[m * N + i];
            sum += val;
            sq_sum += val * val;
        }

        // Reduce sum and sq_sum
        atomicAdd(&mean_s, sum);
        atomicAdd(&var_s, sq_sum);

        __syncthreads();

        if (tid == 0) {
            mean_s /= N;
            var_s = var_s / N - mean_s * mean_s;
            var_s = rsqrtf(var_s + eps);
        }

        __syncthreads();

        for (int i = tid; i < N; i += stride) {
            float val = input[m * N + i];
            val = (val - mean_s) * var_s;
            output[m * N + i] = val * gamma[i] + beta[i];
        }

        __syncthreads();

        if (tid == 0) {
            mean_s = 0.0f;
            var_s = 0.0f;
        }

        __syncthreads();
    }
}

// C++ wrapper for the quantize_weights kernel
torch::Tensor bitnet_quantize_weights_cuda(torch::Tensor weights, torch::Tensor scales) {
    const auto size = weights.numel();
    const auto hidden_dim = weights.size(1);
    auto quantized_weights = torch::empty_like(weights, torch::kInt8);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    quantize_weights_kernel<<<blocks, threads>>>(
        weights.data_ptr<float>(),
        quantized_weights.data_ptr<int8_t>(),
        scales.data_ptr<float>(),
        size, hidden_dim
    );

    return quantized_weights;
}

// C++ wrapper for the fused_quantize_matmul kernel
torch::Tensor bitnet_fused_quantize_matmul_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor input_scale,
    torch::Tensor weight_scale
) {
    const auto M = input.size(0);
    const auto K = input.size(1);
    const auto N = weights.size(1);

    auto output = torch::empty({M, N}, torch::kFloat32);

    const dim3 threads(16, 16);
    const dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    fused_quantize_matmul_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weights.data_ptr<int8_t>(),
        output.data_ptr<float>(),
        input_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        M, N, K
    );

    return output;
}

// C++ wrapper for the layernorm kernel
torch::Tensor bitnet_layernorm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    const auto M = input.size(0);
    const auto N = input.size(1);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    layernorm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        M, N, eps
    );

    return output;
}

