# src/memory_manager.cu
# BitNetMemoryManager implementation

// src/memory_manager.cu

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <memory>
#include <stdexcept>
#include "bitnet_types.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

class BitNetMemoryManager {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool is_unified;
        bool is_pinned;
    };

    std::unordered_map<std::string, MemoryBlock> allocations;
    std::vector<cudaStream_t> streams;

    // Custom memory pool for small allocations
    static constexpr size_t POOL_SIZE = 1024 * 1024 * 1024; // 1GB
    void* memory_pool;
    size_t pool_offset;

public:
    BitNetMemoryManager(int num_streams = 2) : pool_offset(0) {
        streams.resize(num_streams);
        for (auto& stream : streams) {
            CUDA_CHECK(cudaStreamCreate(&stream));
        }

        // Initialize memory pool
        CUDA_CHECK(cudaMalloc(&memory_pool, POOL_SIZE));
    }

    ~BitNetMemoryManager() {
        for (auto& pair : allocations) {
            if (pair.second.is_unified) {
                CUDA_CHECK(cudaFree(pair.second.ptr));
            } else if (pair.second.is_pinned) {
                CUDA_CHECK(cudaFreeHost(pair.second.ptr));
            } else {
                CUDA_CHECK(cudaFree(pair.second.ptr));
            }
        }

        for (auto& stream : streams) {
            CUDA_CHECK(cudaStreamDestroy(stream));
        }

        CUDA_CHECK(cudaFree(memory_pool));
    }

    template<typename T>
    T* allocate(const std::string& name, size_t count, bool use_unified_memory = false, bool use_pinned_memory = false) {
        size_t size = count * sizeof(T);
        void* ptr = nullptr;

        if (allocations.find(name) != allocations.end()) {
            throw std::runtime_error("Memory already allocated for: " + name);
        }

        if (size <= 1024 && !use_unified_memory && !use_pinned_memory) {
            // Use memory pool for small allocations
            if (pool_offset + size > POOL_SIZE) {
                throw std::runtime_error("Memory pool exhausted");
            }
            ptr = static_cast<char*>(memory_pool) + pool_offset;
            pool_offset += size;
        } else if (use_unified_memory) {
            CUDA_CHECK(cudaMallocManaged(&ptr, size));
        } else if (use_pinned_memory) {
            CUDA_CHECK(cudaMallocHost(&ptr, size));
        } else {
            CUDA_CHECK(cudaMalloc(&ptr, size));
        }

        allocations[name] = {ptr, size, use_unified_memory, use_pinned_memory};
        return static_cast<T*>(ptr);
    }

    template<typename T>
    T* get(const std::string& name) {
        auto it = allocations.find(name);
        if (it == allocations.end()) {
            throw std::runtime_error("Memory not allocated for: " + name);
        }
        return static_cast<T*>(it->second.ptr);
    }

    void free(const std::string& name) {
        auto it = allocations.find(name);
        if (it == allocations.end()) {
            throw std::runtime_error("Memory not allocated for: " + name);
        }

        if (it->second.is_unified) {
            CUDA_CHECK(cudaFree(it->second.ptr));
        } else if (it->second.is_pinned) {
            CUDA_CHECK(cudaFreeHost(it->second.ptr));
        } else if (it->second.size > 1024) {
            CUDA_CHECK(cudaFree(it->second.ptr));
        }

        allocations.erase(it);
    }

    template<typename T>
    void copy_to_device(const std::string& name, const T* host_data, size_t count, int stream_id = 0) {
        auto it = allocations.find(name);
        if (it == allocations.end()) {
            throw std::runtime_error("Memory not allocated for: " + name);
        }

        size_t size = count * sizeof(T);
        if (size > it->second.size) {
            throw std::runtime_error("Copy size exceeds allocated size");
        }

        CUDA_CHECK(cudaMemcpyAsync(it->second.ptr, host_data, size, cudaMemcpyHostToDevice, streams[stream_id]));
    }

    template<typename T>
    void copy_to_host(const std::string& name, T* host_data, size_t count, int stream_id = 0) {
        auto it = allocations.find(name);
        if (it == allocations.end()) {
            throw std::runtime_error("Memory not allocated for: " + name);
        }

        size_t size = count * sizeof(T);
        if (size > it->second.size) {
            throw std::runtime_error("Copy size exceeds allocated size");
        }

        CUDA_CHECK(cudaMemcpyAsync(host_data, it->second.ptr, size, cudaMemcpyDeviceToHost, streams[stream_id]));
    }

    void synchronize_stream(int stream_id) {
        CUDA_CHECK(cudaStreamSynchronize(streams[stream_id]));
    }

    void synchronize_all_streams() {
        for (auto& stream : streams) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    cudaStream_t get_stream(int stream_id) {
        if (stream_id >= streams.size()) {
            throw std::runtime_error("Invalid stream ID");
        }
        return streams[stream_id];
    }

    void prefetch_to_gpu(const std::string& name, int device_id, int stream_id = 0) {
        auto it = allocations.find(name);
        if (it == allocations.end()) {
            throw std::runtime_error("Memory not allocated for: " + name);
        }

        if (it->second.is_unified) {
            CUDA_CHECK(cudaMemPrefetchAsync(it->second.ptr, it->second.size, device_id, streams[stream_id]));
        }
    }

    void prefetch_to_cpu(const std::string& name, int stream_id = 0) {
        auto it = allocations.find(name);
        if (it == allocations.end()) {
            throw std::runtime_error("Memory not allocated for: " + name);
        }

        if (it->second.is_unified) {
            CUDA_CHECK(cudaMemPrefetchAsync(it->second.ptr, it->second.size, cudaCpuDeviceId, streams[stream_id]));
        }
    }
};


// This implementation of the BitNetMemoryManager class provides several key features:
// Unified Memory Support: Allows allocation of unified memory for easier management of data that needs to be accessed by both CPU and GPU.
// Pinned Memory Support: Enables allocation of pinned memory for faster host-to-device transfers.
// Custom Memory Pool: Implements a simple memory pool for small allocations to reduce the overhead of frequent cudaMalloc calls.
// Multiple CUDA Streams: Supports multiple CUDA streams for concurrent operations.
// Asynchronous Memory Operations: Provides methods for asynchronous memory copies and prefetching.
// Error Handling: Uses a CUDA_CHECK macro to handle CUDA errors and throw exceptions when errors occur.
// Resource Management: Automatically frees all allocated memory in the destructor.
// Flexible Allocation: Allows allocation of different types of memory (device, unified, or pinned) based on the needs of the application.
// Memory Tracking: Keeps track of all allocations using a name-based system, allowing for easy retrieval and management of allocated memory.
// Prefetching: Includes methods to prefetch unified memory to either GPU or CPU, which can help optimize memory access patterns.
// This implementation provides a robust and flexible memory management system for the BitNet CUDA project. It can be easily integrated into the larger BitNetCUDA class to handle all memory-related operations efficiently.