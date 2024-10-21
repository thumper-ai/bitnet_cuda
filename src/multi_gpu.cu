// src/multi_gpu.cu

#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include "bitnet_cuda.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

#define NCCL_CHECK(call) \
    do { \
        ncclResult_t error = call; \
        if (error != ncclSuccess) { \
            throw std::runtime_error("NCCL error: " + std::string(ncclGetErrorString(error))); \
        } \
    } while(0)

class BitNetMultiGPU {
private:
    std::vector<std::unique_ptr<BitNetCUDA>> gpu_instances;
    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> nccl_comms;
    int num_gpus;
    int batch_size;
    int seq_length;
    int hidden_size;

    void init_nccl() {
        nccl_comms.resize(num_gpus);
        ncclUniqueId nccl_id;
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            NCCL_CHECK(ncclCommInitRank(&nccl_comms[i], num_gpus, nccl_id, i));
        }
    }

public:
    BitNetMultiGPU(int num_gpus) : num_gpus(num_gpus) {
        if (num_gpus <= 0) {
            throw std::invalid_argument("Number of GPUs must be positive");
        }

        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            gpu_instances.emplace_back(std::make_unique<BitNetCUDA>());
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            streams.push_back(stream);
        }

        init_nccl();
    }

    ~BitNetMultiGPU() {
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
            NCCL_CHECK(ncclCommDestroy(nccl_comms[i]));
        }
    }

    void allocate_model_parameters(int batch_size, int seq_length, int hidden_size, int num_layers) {
        this->batch_size = batch_size;
        this->seq_length = seq_length;
        this->hidden_size = hidden_size;

        int batch_per_gpu = batch_size / num_gpus;
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            gpu_instances[i]->allocate_model_parameters(batch_per_gpu, seq_length, hidden_size, num_layers);
        }
    }

    void forward_pass(const float* input, float* output) {
        int batch_per_gpu = batch_size / num_gpus;
        
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            const float* gpu_input = input + i * batch_per_gpu * seq_length * hidden_size;
            float* gpu_output = output + i * batch_per_gpu * seq_length * hidden_size;
            gpu_instances[i]->forward_pass_async(gpu_input, gpu_output, batch_per_gpu, streams[i]);
        }

        synchronize_all_streams();
    }

    void backward_pass(const float* grad_output, float learning_rate) {
        int batch_per_gpu = batch_size / num_gpus;
        
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            const float* gpu_grad_output = grad_output + i * batch_per_gpu * seq_length * hidden_size;
            gpu_instances[i]->backward_pass_async(gpu_grad_output, learning_rate, batch_per_gpu, streams[i]);
        }

        synchronize_all_streams();
        synchronize_parameters();
    }

    void synchronize_parameters() {
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            auto params = gpu_instances[i]->get_parameters();
            for (auto& param : params) {
                NCCL_CHECK(ncclAllReduce(param.data, param.data, param.size, 
                                         ncclFloat, ncclSum, nccl_comms[i], streams[i]));
            }
        }
        synchronize_all_streams();

        // Normalize parameters
        float scale = 1.0f / num_gpus;
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            gpu_instances[i]->scale_parameters(scale, streams[i]);
        }
        synchronize_all_streams();
    }

    void synchronize_all_streams() {
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
    }

    void train(const float* input, const float* target, int num_epochs, float learning_rate) {
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // Forward pass
            forward_pass(input, nullptr);  // Output is stored internally

            // Compute loss and gradients (assuming this is implemented in BitNetCUDA)
            for (int i = 0; i < num_gpus; ++i) {
                CUDA_CHECK(cudaSetDevice(i));
                gpu_instances[i]->compute_loss_and_gradients(target + i * (batch_size / num_gpus) * seq_length * hidden_size, streams[i]);
            }

            // Backward pass
            backward_pass(nullptr, learning_rate);  // Grad output is stored internally

            if (epoch % 10 == 0) {  // Print loss every 10 epochs
                float total_loss = 0.0f;
                for (int i = 0; i < num_gpus; ++i) {
                    total_loss += gpu_instances[i]->get_loss();
                }
                std::cout << "Epoch " << epoch << ", Loss: " << total_loss / num_gpus << std::endl;
            }
        }
    }

    std::vector<float> get_predictions(const float* input) {
        forward_pass(input, nullptr);  // Output is stored internally

        std::vector<float> predictions(batch_size * seq_length * hidden_size);
        for (int i = 0; i < num_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            int batch_per_gpu = batch_size / num_gpus;
            gpu_instances[i]->copy_output_to_host(predictions.data() + i * batch_per_gpu * seq_length * hidden_size, 
                                                  batch_per_gpu * seq_length * hidden_size, streams[i]);
        }
        synchronize_all_streams();

        return predictions;
    }
};

// This implementation includes the following improvements and additions:
// Error checking macros for CUDA and NCCL calls.
// Initialization of NCCL for efficient inter-GPU communication.
// Proper resource management in the destructor.
// A complete backward_pass method.
// A synchronize_parameters method that uses NCCL for efficient parameter synchronization across GPUs.
// A train method that demonstrates how to use the forward and backward passes in a training loop.
// A get_predictions method for inference.
// Proper synchronization of streams and devices throughout the implementation.
// This implementation provides a robust framework for multi-GPU training and inference with BitNet. It handles data parallelism across GPUs, synchronizes parameters efficiently using NCCL, and provides a high-level interface for training and inference.
// To use this implementation, you would typically:
// Create a BitNetMultiGPU instance with the desired number of GPUs.
// Allocate model parameters.
// Prepare your input data and targets.
// Call the train method for training or get_predictions for inference.
// Remember to compile this code with CUDA and NCCL support, and link against the necessary libraries. Also, ensure that the BitNetCUDA class (which this multi-GPU implementation relies on) has the necessary methods implemented, such as forward_pass_async, backward_pass_async, compute_loss_and_gradients, etc.