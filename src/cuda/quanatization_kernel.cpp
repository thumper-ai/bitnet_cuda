__global__ void quantize_weights_kernel(
    const float* __restrict__ weights,
    PackedTernary* __restrict__ quantized_weights,
    float* __restrict__ scales,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ float shared_sum[32];
    float local_sum = 0.0f;

    // Compute the average absolute value (scale)
    for (int i = tid; i < size; i += stride) {
        local_sum += fabsf(weights[i]);
    }

    // Parallel reduction to compute the sum
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        shared_sum[warp] = local_sum;
    }

    __syncthreads();

    if (threadIdx.x < 32) {
        local_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_sum[threadIdx.x] : 0.0f;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }

        if (threadIdx.x == 0) {
            scales[blockIdx.x] = local_sum;
        }
    }

    __syncthreads();

    float scale = size / scales[0];

    // Quantize and pack weights
    for (int i = tid; i < size / 16; i += stride) {
        ternary_t temp[16];
        for (int j = 0; j < 16; ++j) {
            temp[j] = float_to_ternary(weights[i * 16 + j], scale);
        }
        quantized_weights[i].data = pack_ternary(temp);
    }

    // Handle remaining weights
    if (tid == 0 && size % 16 != 0) {
        int remaining = size % 16;
        ternary_t temp[16] = {0};
        for (int j = 0; j < remaining; ++j) {
            temp[j] = float_to_ternary(weights[size - remaining + j], scale);
        }
        quantized_weights[size / 16].data = pack_ternary(temp);
    }
}

__global__ void dequantize_weights_kernel(
    const PackedTernary* __restrict__ quantized_weights,
    float* __restrict__ weights,
    const float* __restrict__ scales,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float scale = scales[0];

    for (int i = tid; i < size / 16; i += stride) {
        uint32_t packed = quantized_weights[i].data;
        for (int j = 0; j < 16; ++j) {
            weights[i * 16 + j] = ternary_to_float(unpack_ternary(packed, j), scale);
        }
    }

    // Handle remaining weights
    if (tid == 0 && size % 16 != 0) {
        int remaining = size % 16;
        uint32_t packed = quantized_weights[size / 16].data;
        for (int j = 0; j < remaining; ++j) {
            weights[size - remaining + j] = ternary_to_float(unpack_ternary(packed, j), scale);
        }
    }
}