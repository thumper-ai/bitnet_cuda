// tests/test_kernels.cu

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/bitnet_cuda.h"
#include "../include/kernels.h"

// Helper function to initialize test data
void initializeTestData(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Test fixture for BitNet CUDA kernels
class BitNetKernelTest : public ::testing::Test {
protected:
    BitNetCUDA bitnet;
    const int TEST_SIZE = 1024 * 1024;

    void SetUp() override {
        // Initialize CUDA and allocate memory
        cudaSetDevice(0);
    }

    void TearDown() override {
        // Clean up CUDA resources
        cudaDeviceReset();
    }
};

// Test case for quantize_weights_kernel
TEST_F(BitNetKernelTest, QuantizeWeightsKernel) {
    float* h_weights = new float[TEST_SIZE];
    float* d_weights;
    PackedTernary* d_quantized_weights;
    float* d_scales;

    initializeTestData(h_weights, TEST_SIZE);

    CUDA_CHECK(cudaMalloc(&d_weights, TEST_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_quantized_weights, (TEST_SIZE + 15) / 16 * sizeof(PackedTernary)));
    CUDA_CHECK(cudaMalloc(&d_scales, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (TEST_SIZE + block_size - 1) / block_size;

    quantize_weights_kernel<<<grid_size, block_size>>>(d_weights, d_quantized_weights, d_scales, TEST_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify results
    PackedTernary* h_quantized_weights = new PackedTernary[(TEST_SIZE + 15) / 16];
    float h_scale;

    CUDA_CHECK(cudaMemcpy(h_quantized_weights, d_quantized_weights, (TEST_SIZE + 15) / 16 * sizeof(PackedTernary), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_scale, d_scales, sizeof(float), cudaMemcpyDeviceToHost));

    // Check if the scale is reasonable
    EXPECT_GT(h_scale, 0.0f);
    EXPECT_LT(h_scale, 1.0f);

    // Check if the quantized values are within the expected range (-1, 0, 1)
    for (int i = 0; i < (TEST_SIZE + 15) / 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            ternary_t value = unpack_ternary(h_quantized_weights[i].data, j);
            EXPECT_TRUE(value == -1 || value == 0 || value == 1);
        }
    }

    // Clean up
    delete[] h_weights;
    delete[] h_quantized_weights;
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_quantized_weights));
    CUDA_CHECK(cudaFree(d_scales));
}

// Test case for bitnet_matmul_kernel
TEST_F(BitNetKernelTest, BitnetMatmulKernel) {
    const int M = 128, N = 128, K = 128;
    PackedTernary* d_A;
    half* d_B;
    half* d_C;
    float* d_A_scale;
    float* d_B_scale;

    CUDA_CHECK(cudaMalloc(&d_A, (M * K + 15) / 16 * sizeof(PackedTernary)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_A_scale, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_scale, sizeof(float)));

    // Initialize input data
    PackedTernary* h_A = new PackedTernary[(M * K + 15) / 16];
    half* h_B = new half[N * K];
    float h_A_scale = 0.1f, h_B_scale = 0.1f;

    for (int i = 0; i < (M * K + 15) / 16; ++i) {
        h_A[i].data = rand();
    }
    for (int i = 0; i < N * K; ++i) {
        h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A, (M * K + 15) / 16 * sizeof(PackedTernary), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_scale, &h_A_scale, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_scale, &h_B_scale, sizeof(float), cudaMemcpyHostToDevice));

    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);

    bitnet_matmul_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, d_A_scale, d_B_scale, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify results
    half* h_C = new half[M * N];
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    // Perform a simple check (e.g., output is not all zeros)
    bool all_zero = true;
    for (int i = 0; i < M * N; ++i) {
        if (__half2float(h_C[i]) != 0.0f) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero);

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A_scale));
    CUDA_CHECK(cudaFree(d_B_scale));
}

// Test case for bitnet_fused_quantize_layernorm_kernel
TEST_F(BitNetKernelTest, BitnetFusedQuantizeLayernormKernel) {
    const int batch_size = 32, hidden_size = 768;
    float* d_input;
    PackedTernary* d_quantized_output;
    float* d_scales;
    float* d_gamma;
    float* d_beta;

    CUDA_CHECK(cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_quantized_output, (batch_size * hidden_size + 15) / 16 * sizeof(PackedTernary)));
    CUDA_CHECK(cudaMalloc(&d_scales, batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, hidden_size * sizeof(float)));

    // Initialize input data
    float* h_input = new float[batch_size * hidden_size];
    float* h_gamma = new float[hidden_size];
    float* h_beta = new float[hidden_size];

    initializeTestData(h_input, batch_size * hidden_size);
    initializeTestData(h_gamma, hidden_size);
    initializeTestData(h_beta, hidden_size);

    CUDA_CHECK(cudaMemcpy(d_input, h_input, batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = batch_size;

    bitnet_fused_quantize_layernorm_kernel<<<grid_size, block_size>>>(
        d_input, d_quantized_output, d_scales, d_gamma, d_beta, batch_size * hidden_size, hidden_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify results
    PackedTernary* h_quantized_output = new PackedTernary[(batch_size * hidden_size + 15) / 16];
    float* h_scales = new float[batch_size];

    CUDA_CHECK(cudaMemcpy(h_quantized_output, d_quantized_output, (batch_size * hidden_size + 15) / 16 * sizeof(PackedTernary), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scales, d_scales, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Check if the scales are reasonable
    for (int i = 0; i < batch_size; ++i) {
        EXPECT_GT(h_scales[i], 0.0f);
        EXPECT_LT(h_scales[i], 1.0f);
    }

    // Check if the quantized values are within the expected range (-1, 0, 1)
    for (int i = 0; i < (batch_size * hidden_size + 15) / 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            ternary_t value = unpack_ternary(h_quantized_output[i].data, j);
            EXPECT_TRUE(value == -1 || value == 0 || value == 1);
        }
    }

    // Clean up
    delete[] h_input;
    delete[] h_gamma;
    delete[] h_beta;
    delete[] h_quantized_output;
    delete[] h_scales;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_quantized_output));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// This test suite includes:
// A test fixture BitNetKernelTest that sets up and tears down CUDA resources for each test.
// Test cases for the main CUDA kernels:
// quantize_weights_kernel
// bitnet_matmul_kernel
// bitnet_fused_quantize_layernorm_kernel
// Helper functions for initializing test data and unpacking ternary values.
// Comprehensive error checking using the CUDA_CHECK macro.
// Verification of kernel outputs, including checks for reasonable scale values and correct quantization ranges.
// To compile and run these tests:
// Make sure you have the Google Test framework installed.
// Compile the test file with CUDA support, linking against the Google Test library and your BitNet CUDA implementation.
// Run the resulting executable.
// This test suite will help ensure that your CUDA kernels are working correctly and producing reasonable outputs. Remember to add more specific tests as you develop additional functionality for your BitNet implementation.