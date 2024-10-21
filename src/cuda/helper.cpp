#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Custom data type for 1.58-bit (ternary) values
typedef int8_t ternary_t;

// Structure to hold packed ternary values
struct PackedTernary {
    uint32_t data;
};

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