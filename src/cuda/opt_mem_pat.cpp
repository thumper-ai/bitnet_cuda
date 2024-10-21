__global__ void optimized_bitnet_fused_quantize_matmul_kernel(
    const float* __restrict__ input,
    const ternary_t* __restrict__ weights,
    half* __restrict__ output,
    const float* __restrict__ input_scale,
    const float* __restrict__ weight_scale,
    int M, int N, int K
) {
    __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
    __shared__ ternary_t weight_tile[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        // Collaborative loading of input and weight tiles
        if (by + ty < M && tile + tx < K)
            input_tile[ty][tx] = input[(by + ty) * K + tile + tx] * input_scale[0];
        if (tile + ty < K && bx + tx < N)
            weight_tile[ty][tx] = weights[(tile + ty) * N + bx + tx];

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += input_tile[ty][k] * ternary_to_float(weight_tile[k][tx], weight_scale[0]);
        }

        __syncthreads();
    }

    // Store the result
    if (by + ty < M && bx + tx < N)
        output[(by + ty) * N + bx + tx] = __float2half(sum);
}