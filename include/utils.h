// include/utils.h:
// CUDA_CHECK macro
// pack_ternary, unpack_ternary, float_to_ternary, ternary_to_float functions
// measure_memory_bandwidth, measure_compute_utilization functions

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <cstdint>

// CUDA_CHECK macro for error handling
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error("CUDA error in " + std::string(__FILE__) + " at line " + \
                                     std::to_string(__LINE__) + ": " + \
                                     std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

// Custom data types
typedef int8_t ternary_t;

// Pack 16 ternary values into a single 32-bit integer
__device__ __host__ inline uint32_t pack_ternary(const ternary_t* values) {
    uint32_t packed = 0;
    for (int i = 0; i < 16; ++i) {
        packed |= (uint32_t)(values[i] & 0x3) << (i * 2);
    }
    return packed;
}

// Unpack a single ternary value from a packed 32-bit integer
__device__ __host__ inline ternary_t unpack_ternary(uint32_t packed, int index) {
    return (packed >> (index * 2)) & 0x3;
}

// Convert float to ternary value
__device__ __host__ inline ternary_t float_to_ternary(float x, float scale) {
    float scaled = x * scale;
    if (scaled > 0.5f) return 1;
    if (scaled < -0.5f) return -1;
    return 0;
}

// Convert ternary value to float
__device__ __host__ inline float ternary_to_float(ternary_t x, float scale) {
    return static_cast<float>(x) * scale;
}

// Measure memory bandwidth (in GB/s)
inline float measure_memory_bandwidth() {
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

// Measure compute utilization (in GFLOPS)
inline float measure_compute_utilization() {
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