// bitnet_cuda.h
#include <mma.h>
#include <cuda_fp16.h>

// Define tile sizes for Tensor Core operations
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Define overall tile sizes
#define TILE_M 64
#define TILE_N 64
#define TILE_K 64

__global__ void optimized_bitnet_matmul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    float* __restrict__ C_scale,
    int M, int N, int K,
    bool apply_activation
) {
    using namespace nvcuda::wmma;

    // Shared memory for double buffering
    __shared__ half A_shared[2][TILE_M][TILE_K];
    __shared__ half B_shared[2][TILE_K][TILE_N];

    // Fragments for Tensor Core operations
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // Initialize accumulator fragment
    fill_fragment(c_frag, __float2half(0.0f));

    // Persistent threads approach
    for (int block_m = blockIdx.x * TILE_M; block_m < M; block_m += gridDim.x * TILE_M) {
        for (int block_n = blockIdx.y * TILE_N; block_n < N; block_n += gridDim.y * TILE_N) {

            int buffer = 0;
            // Load first tile
            for (int i = threadIdx.x; i < TILE_M * TILE_K / 32; i += blockDim.x) {
                int row = i / TILE_K;
                int col = i % TILE_K;
                A_shared[buffer][row][col] = A[(block_m + row) * K + col];
            }
            for (int i = threadIdx.x; i < TILE_K * TILE_N / 32; i += blockDim.x) {
                int row = i / TILE_N;
                int col = i % TILE_N;
                B_shared[buffer][row][col] = B[row * N + (block_n + col)];
            }
            __syncthreads();

            // Main loop
            for (int k = 0; k < K; k += TILE_K) {
                // Load next tile (double buffering)
                buffer = 1 - buffer;
                if (k + TILE_K < K) {
                    for (int i = threadIdx.x; i < TILE_M * TILE_K / 32; i += blockDim.x) {
                        int row = i / TILE_K;
                        int col = i % TILE_K;
                        A_shared[buffer][row][col] = A[(block_m + row) * K + (k + TILE_K + col)];
                    }
                    for (int i = threadIdx.x; i < TILE_K * TILE_N / 32; i += blockDim.x) {
                        int row = i / TILE_N;
                        int col = i % TILE_N;
                        B_shared[buffer][row][col] = B[(k + TILE_K + row) * N + (block_n + col)];
                    }
                }

                // Compute using Tensor Cores
                #pragma unroll
                for (int m = 0; m < TILE_M; m += WMMA_M) {
                    #pragma unroll
                    for (int n = 0; n < TILE_N; n += WMMA_N) {
                        #pragma unroll
                        for (int k_step = 0; k_step < TILE_K; k_step += WMMA_K) {
                            load_matrix_sync(a_frag, &A_shared[1-buffer][m][k_step], TILE_K);
                            load_matrix_sync(b_frag, &B_shared[1-buffer][k_step][n], TILE_N);
                            mma_sync(c_frag, a_frag, b_frag, c_frag);
                        }
                    }
                }

                __syncthreads();
            }

            // Store results
            #pragma unroll
            for (int m = 0; m < TILE_M; m += WMMA_M) {
                #pragma unroll
                for (int n = 0; n < TILE_N; n += WMMA_N) {
                    half* C_tile = &C[(block_m + m) * N + (block_n + n)];
                    store_matrix_sync(C_tile, c_frag, N, mem_row_major);

                    // Fused activation (e.g., ReLU)
                    if (apply_activation) {
                        #pragma unroll
                        for (int i = 0; i < WMMA_M; ++i) {
                            #pragma unroll
                            for (int j = 0; j < WMMA_N; ++j) {
                                int idx = i * N + j;
                                C_tile[idx] = __hgt(C_tile[idx], __float2half(0.0f)) ? C_tile[idx] : __float2half(0.0f);
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply scaling
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int m = blockIdx.x * TILE_M;
        int n = blockIdx.y * TILE_N;
        if (m < M && n < N) {
            float scale = A_scale[m / TILE_M] * B_scale[n / TILE_N];
            atomicAdd(C_scale, scale);
        }
    }
}