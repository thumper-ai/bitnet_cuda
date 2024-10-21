// include/multi_gpu.h:
// BitNetMultiGPU class
// include/multi_gpu.h

#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <nccl.h>
#include "bitnet_cuda.h"
#include "memory_management.h"

class BitNetMultiGPU {
private:
    std::vector<std::unique_ptr<BitNetCUDA>> gpu_instances;
    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> nccl_comms;
    BitNetMemoryManager memory_manager;
    int num_gpus;

    void init_nccl() {
        ncclUniqueId id;
        CUDA_CHECK(ncclGetUniqueId(&id));
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(ncclCommInitRank(&nccl_comms[i], num_gpus, id, i));
        }
    }

public:
    BitNetMultiGPU(int num_gpus) : num_gpus(num_gpus) {
        gpu_instances.reserve(num_gpus);
        streams.resize(num_gpus);
        nccl_comms.resize(num_gpus);

        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            gpu_instances.emplace_back(std::make_unique<BitNetCUDA>());
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }

        init_nccl();
    }

    ~BitNetMultiGPU() {
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
            CUDA_CHECK(ncclCommDestroy(nccl_comms[i]));
        }
    }

    void allocate_model_parameters(int batch_size, int seq_length, int hidden_size, int num_layers) {
        int batch_per_gpu = batch_size / num_gpus;
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            gpu_instances[i]->allocate_model_parameters(batch_per_gpu, seq_length, hidden_size, num_layers);
        }
    }

    void synchronize_parameters() {
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            auto weights = gpu_instances[i]->get_weights();
            CUDA_CHECK(ncclAllReduce(weights, weights, weights.size(), ncclFloat, ncclSum, nccl_comms[i], streams[i]));
        }
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
    }

    void forward_pass(const float* input, float* output, int batch_size) {
        int batch_per_gpu = batch_size / num_gpus;
        
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            const float* gpu_input = input + i * batch_per_gpu * seq_length * hidden_size;
            float* gpu_output = output + i * batch_per_gpu * seq_length * hidden_size;
            
            gpu_instances[i]->forward_pass(gpu_input, gpu_output, batch_per_gpu);
        }

        // Synchronize outputs across GPUs
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            float* gpu_output = output + i * batch_per_gpu * seq_length * hidden_size;
            CUDA_CHECK(ncclAllReduce(gpu_output, gpu_output, batch_per_gpu * seq_length * hidden_size,
                                     ncclFloat, ncclSum, nccl_comms[i], streams[i]));
        }

        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
    }

    void backward_pass(const float* grad_output, float learning_rate, int batch_size) {
        int batch_per_gpu = batch_size / num_gpus;
        
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            const float* gpu_grad_output = grad_output + i * batch_per_gpu * seq_length * hidden_size;
            
            gpu_instances[i]->backward_pass(gpu_grad_output, learning_rate, batch_per_gpu);
        }

        synchronize_parameters();
    }

    void optimize_memory_usage() {
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            gpu_instances[i]->optimize_memory_usage();
        }
    }

    void create_computation_graphs() {
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            gpu_instances[i]->create_computation_graph();
        }
    }

    void run_computation_graphs() {
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            gpu_instances[i]->run_computation_graph();
        }
    }

    void tune_kernels() {
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            gpu_instances[i]->tune_kernels();
        }
    }
};