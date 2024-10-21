#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "../include/bitnet_multi_gpu.h"
#include "../include/utils.h"

class BitNetMultiGPUTest : public ::testing::Test {
protected:
    std::unique_ptr<BitNetMultiGPU> multi_gpu;
    int num_gpus;
    int batch_size;
    int seq_length;
    int hidden_size;
    int num_layers;

    void SetUp() override {
        // Get the number of available GPUs
        CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
        if (num_gpus < 2) {
            GTEST_SKIP() << "Skipping multi-GPU tests. At least 2 GPUs are required.";
        }

        // Use at most 4 GPUs for testing
        num_gpus = std::min(num_gpus, 4);
        
        batch_size = 32 * num_gpus;  // Ensure batch size is divisible by num_gpus
        seq_length = 128;
        hidden_size = 768;
        num_layers = 12;

        multi_gpu = std::make_unique<BitNetMultiGPU>(num_gpus);
    }

    void TearDown() override {
        multi_gpu.reset();
    }
};

TEST_F(BitNetMultiGPUTest, AllocateModelParameters) {
    ASSERT_NO_THROW(multi_gpu->allocate_model_parameters(batch_size, seq_length, hidden_size, num_layers));
}

TEST_F(BitNetMultiGPUTest, ForwardPass) {
    multi_gpu->allocate_model_parameters(batch_size, seq_length, hidden_size, num_layers);

    std::vector<float> input(batch_size * seq_length * hidden_size);
    std::vector<float> output(batch_size * seq_length * hidden_size);

    // Initialize input with some test data
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) / input.size();
    }

    ASSERT_NO_THROW(multi_gpu->forward_pass(input.data(), output.data(), batch_size));

    // Check that output is not all zeros (a very basic check)
    bool all_zeros = true;
    for (float val : output) {
        if (val != 0.0f) {
            all_zeros = false;
            break;
        }
    }
    EXPECT_FALSE(all_zeros);
}

TEST_F(BitNetMultiGPUTest, BackwardPass) {
    multi_gpu->allocate_model_parameters(batch_size, seq_length, hidden_size, num_layers);

    std::vector<float> grad_output(batch_size * seq_length * hidden_size);
    float learning_rate = 0.001f;

    // Initialize grad_output with some test data
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_output[i] = static_cast<float>(i) / grad_output.size();
    }

    ASSERT_NO_THROW(multi_gpu->backward_pass(grad_output.data(), learning_rate));
}

TEST_F(BitNetMultiGPUTest, SynchronizeParameters) {
    multi_gpu->allocate_model_parameters(batch_size, seq_length, hidden_size, num_layers);

    ASSERT_NO_THROW(multi_gpu->synchronize_parameters());
}

TEST_F(BitNetMultiGPUTest, PerformanceComparison) {
    multi_gpu->allocate_model_parameters(batch_size, seq_length, hidden_size, num_layers);

    std::vector<float> input(batch_size * seq_length * hidden_size);
    std::vector<float> output(batch_size * seq_length * hidden_size);

    // Initialize input with some test data
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) / input.size();
    }

    // Measure time for multi-GPU forward pass
    auto start = std::chrono::high_resolution_clock::now();
    multi_gpu->forward_pass(input.data(), output.data(), batch_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto multi_gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Create a single-GPU instance for comparison
    BitNetCUDA single_gpu;
    single_gpu.allocate_model_parameters(batch_size, seq_length, hidden_size, num_layers);

    // Measure time for single-GPU forward pass
    start = std::chrono::high_resolution_clock::now();
    single_gpu.forward_pass(input.data(), output.data(), batch_size);
    end = std::chrono::high_resolution_clock::now();
    auto single_gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Check that multi-GPU is faster than single-GPU
    EXPECT_LT(multi_gpu_duration.count(), single_gpu_duration.count());
    
    std::cout << "Multi-GPU time: " << multi_gpu_duration.count() << " us" << std::endl;
    std::cout << "Single-GPU time: " << single_gpu_duration.count() << " us" << std::endl;
    std::cout << "Speedup: " << static_cast<float>(single_gpu_duration.count()) / multi_gpu_duration.count() << "x" << std::endl;
}

TEST_F(BitNetMultiGPUTest, LargeModelAllocation) {
    // Try to allocate a very large model that wouldn't fit on a single GPU
    int large_hidden_size = 16384;  // 16K hidden size
    int large_num_layers = 24;     // 24 layers

    ASSERT_NO_THROW(multi_gpu->allocate_model_parameters(batch_size, seq_length, large_hidden_size, large_num_layers));
}

TEST_F(BitNetMultiGPUTest, ErrorHandling) {
    // Test with invalid number of GPUs
    EXPECT_THROW(BitNetMultiGPU(0), std::invalid_argument);
    EXPECT_THROW(BitNetMultiGPU(-1), std::invalid_argument);

    // Test with invalid batch size (not divisible by num_gpus)
    EXPECT_THROW(multi_gpu->allocate_model_parameters(batch_size + 1, seq_length, hidden_size, num_layers), std::invalid_argument);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// This test suite includes the following tests:
// AllocateModelParameters: Verifies that model parameters can be allocated across multiple GPUs without errors.
// ForwardPass: Checks that the forward pass can be executed on multiple GPUs and produces non-zero output.
// BackwardPass: Ensures that the backward pass can be performed without errors.
// SynchronizeParameters: Tests the parameter synchronization across GPUs.
// PerformanceComparison: Compares the performance of multi-GPU execution against single-GPU execution, expecting the multi-GPU version to be faster.
// LargeModelAllocation: Attempts to allocate a very large model that wouldn't fit on a single GPU, to test the multi-GPU memory distribution.
// ErrorHandling: Checks that appropriate exceptions are thrown for invalid inputs.
// To compile and run these tests:
// Ensure you have the Google Test framework installed and properly linked in your project.
// Compile this file with CUDA support, linking against the Google Test library and your BitNet CUDA implementation.
// Run the resulting executable.
// Note that this test suite assumes the existence of BitNetMultiGPU and BitNetCUDA classes with the methods used in the tests. You may need to adjust the tests based on your actual implementation details.
// Also, remember to handle cases where fewer than 2 GPUs are available, as multi-GPU tests will be skipped in such scenarios.
// This comprehensive test suite will help ensure that your multi-GPU implementation of BitNet is working correctly and efficiently across different scenarios.