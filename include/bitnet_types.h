// include/bitnet_types.h

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace bitnet {

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int TILE_SIZE = 16;

// Custom data types
typedef int8_t ternary_t;
typedef int8_t quant_t;

// Structure to hold packed ternary values
struct PackedTernary {
    uint32_t data;
};

// Kernel configuration structure
struct KernelConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
};

// Structure to hold kernel performance metrics
struct KernelMetrics {
    float execution_time;
    float memory_bandwidth;
    float compute_utilization;
};

// Helper functions
__device__ __host__ inline uint32_t pack_ternary(const ternary_t* values) {
    uint32_t packed = 0;
    for (int i = 0; i < 16; ++i) {
        packed |= (uint32_t)(values[i] & 0x3) << (i * 2);
    }
    return packed;
}

__device__ __host__ inline ternary_t unpack_ternary(uint32_t packed, int index) {
    return (packed >> (index * 2)) & 0x3;
}

__device__ __host__ inline ternary_t float_to_ternary(float x, float scale) {
    float scaled = x * scale;
    if (scaled > 0.5f) return 1;
    if (scaled < -0.5f) return -1;
    return 0;
}

__device__ __host__ inline float ternary_to_float(ternary_t x, float scale) {
    return static_cast<float>(x) * scale;
}

__device__ __host__ inline quant_t quantize_activation(float x, float scale) {
    return static_cast<quant_t>(roundf(x * scale * 127.0f));
}

__device__ __host__ inline float dequantize_activation(quant_t x, float scale) {
    return static_cast<float>(x) / (scale * 127.0f);
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

} // namespace bitnet