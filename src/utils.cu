│  // src/utils.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include "bitnet_types.h"

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Helper function to convert float to ternary value
__device__ __host__ inline ternary_t float_to_ternary(float x, float scale) {
    float scaled = x * scale;
    if (scaled > 0.5f) return 1;
    if (scaled < -0.5f) return -1;
    return 0;
}

// Helper function to convert ternary value to float
__device__ __host__ inline float ternary_to_float(ternary_t x, float scale) {
    return static_cast<float>(x) * scale;
}

// Helper function to pack 16 ternary values into a single 32-bit integer
__device__ __host__ inline uint32_t pack_ternary(const ternary_t* values) {
    uint32_t packed = 0;
    for (int i = 0; i < 16; ++i) {
        packed |= (uint32_t)(values[i] & 0x3) << (i * 2);
    }
    return packed;
}

// Helper function to unpack a single ternary value from a packed 32-bit integer
__device__ __host__ inline ternary_t unpack_ternary(uint32_t packed, int index) {
    return (packed >> (index * 2)) & 0x3;
}

// Helper function for GELU activation
__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Helper function for RMSNorm
__device__ inline float rms_norm(float x, float epsilon) {
    return x * rsqrtf(__fmaf_rn(x, x, epsilon));
}

// Warp-level reduction sum
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Warp-level reduction max
__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Function to measure memory bandwidth
float measure_memory_bandwidth() {
    const size_t N = 1 << 24;  // 16M elements
    const size_t bytes = N * sizeof(float);

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    CUDA_CHECK(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    float bandwidth = (bytes * 2) / (milliseconds / 1000) / 1e9;  // GB/s

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return bandwidth;
}

// Function to measure compute utilization
float measure_compute_utilization() {
    const int N = 1 << 20;  // 1M elements
    const int ITERATIONS = 100;

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Kernel to perform FMA operations
    auto kernel = [=] __device__ (int idx) {
        float a = d_a[idx];
        float b = d_b[idx];
        float c = d_c[idx];
        #pragma unroll
        for (int i = 0; i < ITERATIONS; ++i) {
            c = __fmaf_rn(a, b, c);
        }
        d_c[idx] = c;
    };

    CUDA_CHECK(cudaEventRecord(start));
    cudaLaunchKernel((void*)kernel, dim3(N/256), dim3(256), nullptr, 0, nullptr);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    float gflops = (2.0f * N * ITERATIONS) / (milliseconds / 1000) / 1e9;  // GFLOPS

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return gflops;
}

// Function to initialize weights with random values
__global__ void initialize_weights_kernel(float* weights, int size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = curand_normal(&state);
    }
}

void initialize_weights(float* weights, int size) {
    unsigned long long seed = 1234ULL;  // Fixed seed for reproducibility
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    initialize_weights_kernel<<<grid_size, block_size>>>(weights, size, seed);
    CUDA_CHECK(cudaGetLastError());
}

// Function to print CUDA device properties
void print_device_properties() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Memory Clock Rate: %.0f MHz\n", prop.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.0f GB/s\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}

// This implementation includes:
// Error checking macro for CUDA calls.
// Helper functions for ternary quantization and dequantization.
// Functions for packing and unpacking ternary values.
// Activation functions (GELU) and normalization (RMSNorm).
// Warp-level reduction operations for sum and max.
// Functions to measure memory bandwidth and compute utilization.
// A kernel for initializing weights with random values.
// A function to print CUDA device properties.
// These utility functions provide a solid foundation for the BitNet CUDA implementation, offering common operations and helper functions that can be used throughout the project. The error checking macro ensures that CUDA errors are caught and reported properly, which is crucial for debugging and maintaining the code.
// Remember to include the corresponding header file (e.g., utils.h) in your project and link this implementation file when compiling. You may need to adjust the include paths and add any necessary CUDA compilation flags when building your project.