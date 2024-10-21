
// tests/test_memory_management.cu

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/memory_manager.h"
#include "../include/bitnet_types.h"

class BitNetMemoryTest : public ::testing::Test {
protected:
    BitNetMemoryManager* memory_manager;

    void SetUp() override {
        memory_manager = new BitNetMemoryManager();
    }

    void TearDown() override {
        delete memory_manager;
    }
};

TEST_F(BitNetMemoryTest, AllocateAndFreeDeviceMemory) {
    const size_t size = 1024;
    float* d_ptr = memory_manager->allocate<float>("test_float", size);
    ASSERT_NE(d_ptr, nullptr);

    // Verify that we can write to and read from the allocated memory
    float h_data[size];
    for (size_t i = 0; i < size; ++i) {
        h_data[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_ptr, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

    float h_result[size];
    cudaMemcpy(h_result, d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], h_data[i]);
    }

    memory_manager->free("test_float");
}

TEST_F(BitNetMemoryTest, AllocateMultipleTypes) {
    const size_t float_size = 1024;
    const size_t int_size = 512;
    const size_t ternary_size = 2048;

    float* d_float = memory_manager->allocate<float>("test_float", float_size);
    int* d_int = memory_manager->allocate<int>("test_int", int_size);
    ternary_t* d_ternary = memory_manager->allocate<ternary_t>("test_ternary", ternary_size);

    ASSERT_NE(d_float, nullptr);
    ASSERT_NE(d_int, nullptr);
    ASSERT_NE(d_ternary, nullptr);

    memory_manager->free("test_float");
    memory_manager->free("test_int");
    memory_manager->free("test_ternary");
}

TEST_F(BitNetMemoryTest, GetAllocatedMemory) {
    const size_t size = 1024;
    float* d_ptr = memory_manager->allocate<float>("test_float", size);
    ASSERT_NE(d_ptr, nullptr);

    float* retrieved_ptr = memory_manager->get<float>("test_float");
    EXPECT_EQ(d_ptr, retrieved_ptr);

    memory_manager->free("test_float");
}

TEST_F(BitNetMemoryTest, FreeNonexistentMemory) {
    EXPECT_NO_THROW(memory_manager->free("nonexistent"));
}

TEST_F(BitNetMemoryTest, GetNonexistentMemory) {
    EXPECT_EQ(memory_manager->get<float>("nonexistent"), nullptr);
}

TEST_F(BitNetMemoryTest, ReallocateMemory) {
    const size_t initial_size = 1024;
    const size_t new_size = 2048;

    float* d_ptr = memory_manager->allocate<float>("test_float", initial_size);
    ASSERT_NE(d_ptr, nullptr);

    // Fill with initial data
    std::vector<float> h_data(initial_size);
    for (size_t i = 0; i < initial_size; ++i) {
        h_data[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_ptr, h_data.data(), initial_size * sizeof(float), cudaMemcpyHostToDevice);

    // Reallocate
    float* new_d_ptr = memory_manager->allocate<float>("test_float", new_size);
    ASSERT_NE(new_d_ptr, nullptr);

    // Verify that the original data is preserved
    std::vector<float> h_result(new_size);
    cudaMemcpy(h_result.data(), new_d_ptr, new_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < initial_size; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], h_data[i]);
    }

    memory_manager->free("test_float");
}

TEST_F(BitNetMemoryTest, AllocateLargeMemory) {
    const size_t large_size = 1ULL << 30;  // 1 GB
    float* d_ptr = nullptr;
    EXPECT_NO_THROW(d_ptr = memory_manager->allocate<float>("large_allocation", large_size));
    ASSERT_NE(d_ptr, nullptr);
    memory_manager->free("large_allocation");
}

TEST_F(BitNetMemoryTest, StressTest) {
    const int num_allocations = 1000;
    const size_t allocation_size = 1024;

    for (int i = 0; i < num_allocations; ++i) {
        std::string name = "stress_test_" + std::to_string(i);
        float* d_ptr = memory_manager->allocate<float>(name, allocation_size);
        ASSERT_NE(d_ptr, nullptr);
    }

    for (int i = 0; i < num_allocations; ++i) {
        std::string name = "stress_test_" + std::to_string(i);
        memory_manager->free(name);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}



// This test suite covers the following aspects of our memory management:
// Basic allocation and freeing of device memory
// Allocation of multiple types (float, int, ternary_t)
// Retrieving allocated memory
// Handling of non-existent memory (both freeing and getting)
// Reallocation of memory (ensuring data preservation)
// Allocation of large memory blocks
// Stress testing with multiple allocations and deallocations
// To compile and run these tests:
// Ensure you have the Google Test framework installed and properly linked in your project.
// Compile this file with CUDA support, linking against the Google Test library and your BitNet CUDA implementation.
// Run the resulting executable.
// You may need to adjust the include paths based on your project structure. Also, make sure that the BitNetMemoryManager class and ternary_t type are properly defined in the included headers.
// This comprehensive test suite will help ensure that your memory management implementation is robust and behaves correctly under various scenarios. It's particularly important for CUDA programming, where memory management errors can be difficult to debug.