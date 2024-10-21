// BitNetMemoryManager class
// CustomAllocator class
// include/memory_management.h

#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <memory>
#include <stdexcept>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

class CustomAllocator {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };

    std::vector<MemoryBlock> memory_pool;
    size_t total_allocated_size;
    const size_t min_block_size;

public:
    CustomAllocator(size_t initial_pool_size = 1024 * 1024 * 1024, size_t min_block = 256) 
        : total_allocated_size(0), min_block_size(min_block) {
        void* initial_ptr;
        CUDA_CHECK(cudaMalloc(&initial_ptr, initial_pool_size));
        memory_pool.push_back({initial_ptr, initial_pool_size, false});
    }

    ~CustomAllocator() {
        for (const auto& block : memory_pool) {
            CUDA_CHECK(cudaFree(block.ptr));
        }
    }

    void* allocate(size_t size) {
        size = std::max(size, min_block_size);
        
        // Find a free block that's large enough
        for (auto& block : memory_pool) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                if (block.size > size + min_block_size) {
                    // Split the block
                    size_t remaining_size = block.size - size;
                    void* new_ptr = static_cast<char*>(block.ptr) + size;
                    memory_pool.push_back({new_ptr, remaining_size, false});
                    block.size = size;
                }
                return block.ptr;
            }
        }

        // If no suitable block found, allocate a new one
        void* new_ptr;
        CUDA_CHECK(cudaMalloc(&new_ptr, size));
        memory_pool.push_back({new_ptr, size, true});
        total_allocated_size += size;
        return new_ptr;
    }

    void free(void* ptr) {
        for (auto& block : memory_pool) {
            if (block.ptr == ptr) {
                block.in_use = false;
                // Merge with adjacent free blocks
                merge_free_blocks();
                return;
            }
        }
        throw std::runtime_error("Attempted to free unallocated memory");
    }

private:
    void merge_free_blocks() {
        std::sort(memory_pool.begin(), memory_pool.end(), 
                  [](const MemoryBlock& a, const MemoryBlock& b) { return a.ptr < b.ptr; });

        for (auto it = memory_pool.begin(); it != memory_pool.end(); ++it) {
            if (!it->in_use) {
                auto next = std::next(it);
                while (next != memory_pool.end() && !next->in_use) {
                    it->size += next->size;
                    it = memory_pool.erase(next);
                    next = std::next(it);
                }
            }
        }
    }
};

class BitNetMemoryManager {
private:
    std::unordered_map<std::string, void*> allocations;
    CustomAllocator allocator;

public:
    BitNetMemoryManager() : allocator(1024 * 1024 * 1024) {} // 1GB initial pool

    template<typename T>
    T* allocate(const std::string& name, size_t count) {
        size_t size = count * sizeof(T);
        void* ptr = allocator.allocate(size);
        allocations[name] = ptr;
        return static_cast<T*>(ptr);
    }

    template<typename T>
    T* get(const std::string& name) {
        auto it = allocations.find(name);
        if (it != allocations.end()) {
            return static_cast<T*>(it->second);
        }
        throw std::runtime_error("Allocation not found: " + name);
    }

    void free(const std::string& name) {
        auto it = allocations.find(name);
        if (it != allocations.end()) {
            allocator.free(it->second);
            allocations.erase(it);
        } else {
            throw std::runtime_error("Attempted to free unallocated memory: " + name);
        }
    }

    void* allocate_raw(size_t size) {
        return allocator.allocate(size);
    }

    void free_raw(void* ptr) {
        allocator.free(ptr);
    }

    template<typename T>
    void copy_to_device(const std::string& name, const T* host_data, size_t count) {
        T* device_ptr = get<T>(name);
        CUDA_CHECK(cudaMemcpy(device_ptr, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void copy_to_host(const std::string& name, T* host_data, size_t count) {
        T* device_ptr = get<T>(name);
        CUDA_CHECK(cudaMemcpy(host_data, device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void synchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};