// auto_tuning.cu  
// BitNetAutoTuner and AdvancedAutoTuner implementations
│   
// src/auto_tuning.cu

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <nvml.h>
#include <cuda_profiler_api.h>
#include "bitnet_types.h"
// src/auto_tuning.cu


class KernelConfig {
public:
    dim3 grid;
    dim3 block;
    size_t shared_mem;

    KernelConfig(dim3 g, dim3 b, size_t sm) : grid(g), block(b), shared_mem(sm) {}
};

struct KernelMetrics {
    float execution_time;
    float memory_bandwidth;
    float compute_utilization;
    float power_consumption;
};

class BitNetAutoTuner {
private:
    std::unordered_map<std::string, KernelConfig> optimal_configs;

public:
    template<typename KernelFunc>
    void tune_kernel(const std::string& kernel_name, KernelFunc kernel, 
                     const std::vector<KernelConfig>& configs,
                     void* args[], size_t args_size) {
        float best_time = std::numeric_limits<float>::max();
        KernelConfig best_config({0,0,0}, {0,0,0}, 0);

        for (const auto& config : configs) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            kernel<<<config.grid, config.block, config.shared_mem>>>(args);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float time;
            cudaEventElapsedTime(&time, start, stop);

            if (time < best_time) {
                best_time = time;
                best_config = config;
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        optimal_configs[kernel_name] = best_config;
    }

    KernelConfig get_optimal_config(const std::string& kernel_name) {
        auto it = optimal_configs.find(kernel_name);
        if (it != optimal_configs.end()) {
            return it->second;
        }
        // Return a default config if not found
        return KernelConfig({1,1,1}, {256,1,1}, 0);
    }
};


class AdvancedAutoTuner {
private:
    std::unordered_map<std::string, std::vector<std::pair<KernelConfig, KernelMetrics>>> kernel_metrics;
    nvmlDevice_t nvml_device;
    cudaDeviceProp device_prop;

    void init_nvml() {
        NVML_TRY(nvmlInit());
        NVML_TRY(nvmlDeviceGetHandleByIndex(0, &nvml_device));
        CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));
    }

    float measure_memory_bandwidth(cudaEvent_t start, cudaEvent_t stop) {
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        return (2 * device_prop.memoryBusWidth * device_prop.memoryClockRate * 1e-6) / (elapsed_time * 1e-3);
    }

    float measure_compute_utilization() {
        nvmlUtilization_t utilization;
        NVML_TRY(nvmlDeviceGetUtilizationRates(nvml_device, &utilization));
        return static_cast<float>(utilization.gpu) / 100.0f;
    }

    float measure_power_consumption() {
        unsigned int power;
        NVML_TRY(nvmlDeviceGetPowerUsage(nvml_device, &power));
        return static_cast<float>(power) / 1000.0f; // Convert mW to W
    }

public:
    AdvancedAutoTuner() {
        init_nvml();
    }

    ~AdvancedAutoTuner() {
        nvmlShutdown();
    }

    template<typename KernelFunc>
    KernelConfig tune_kernel(const std::string& kernel_name, KernelFunc kernel, 
                             const std::vector<KernelConfig>& configs,
                             void* args[], size_t args_size) {
        std::vector<std::pair<KernelConfig, KernelMetrics>> results;

        for (const auto& config : configs) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            kernel<<<config.grid, config.block, config.shared_mem>>>(args);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float time;
            cudaEventElapsedTime(&time, start, stop);

            float bandwidth = measure_memory_bandwidth();
            float utilization = measure_compute_utilization();
            float power = measure_power_consumption();

            results.push_back({config, {time, bandwidth, utilization, power}});

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        KernelConfig best_config = select_best_config(results);
        kernel_metrics[kernel_name] = results;

        return best_config;
    }

    const std::vector<std::pair<KernelConfig, KernelMetrics>>& get_kernel_metrics(const std::string& kernel_name) const {
        auto it = kernel_metrics.find(kernel_name);
        if (it == kernel_metrics.end()) {
            throw std::runtime_error("No metrics found for kernel: " + kernel_name);
        }
        return it->second;
    }

private:
    KernelConfig select_best_config(const std::vector<std::pair<KernelConfig, KernelMetrics>>& results) {
        float best_score = std::numeric_limits<float>::max();
        KernelConfig best_config({0,0,0}, {0,0,0}, 0);

        for (const auto& [config, metrics] : results) {
            float score = 0.4f * metrics.execution_time + 
                          0.2f * (1.0f / metrics.memory_bandwidth) + 
                          0.2f * (1.0f / metrics.compute_utilization) +
                          0.2f * metrics.power_consumption;
            if (score < best_score) {
                best_score = score;
                best_config = config;
            }
        }

        return best_config;
    }
};

// Helper function to generate a range of kernel configurations
std::vector<KernelConfig> generate_configs(int min_block_size, int max_block_size, int step_size) {
    std::vector<KernelConfig> configs;
    for (int block_size = min_block_size; block_size <= max_block_size; block_size += step_size) {
        configs.emplace_back(dim3(1, 1, 1), dim3(block_size, 1, 1), 0);
    }
    return configs;
}

// Example usage
void example_usage() {
    BitNetAutoTuner simple_tuner;
    AdvancedAutoTuner advanced_tuner;

    // Define your kernel
    auto kernel = [](int* data) __global__ { /* kernel implementation */ };

    // Generate configurations
    auto configs = generate_configs(128, 1024, 128);

    // Tune the kernel
    int* data;
    cudaMalloc(&data, sizeof(int));
    void* args[] = {&data};

    simple_tuner.tune_kernel("example_kernel", kernel, configs, args, sizeof(args));
    auto simple_config = simple_tuner.get_optimal_config("example_kernel");

    auto advanced_config = advanced_tuner.tune_kernel("example_kernel", kernel, configs, args, sizeof(args));

    // Use the optimal configurations
    kernel<<<simple_config.grid, simple_config.block, simple_config.shared_mem>>>(data);
    kernel<<<advanced_config.grid, advanced_config.block, advanced_config.shared_mem>>>(data);

    cudaFree(data);
}

// This implementation provides two autotuning classes:
// BitNetAutoTuner: A simpler autotuner that only considers execution time when selecting the optimal configuration.
// AdvancedAutoTuner: A more sophisticated autotuner that considers multiple performance metrics (execution time, memory bandwidth, compute utilization, and power consumption) when selecting the optimal configuration.
// Key features of this implementation:
// Both tuners support tuning kernels with different configurations (grid size, block size, shared memory).
// The AdvancedAutoTuner uses NVIDIA Management Library (NVML) to measure additional performance metrics.
// The select_best_config method in AdvancedAutoTuner uses a weighted sum approach to balance different performance metrics.
// A helper function generate_configs is provided to easily create a range of kernel configurations for tuning.
// An example usage is provided to demonstrate how to use both tuners.
// To use this in your BitNet implementation:
// Include this file in your main CUDA file.
// Create an instance of either BitNetAutoTuner or AdvancedAutoTuner.
// Define the range of configurations you want to test for each kernel.
// Use the tune_kernel method to find the optimal configuration for each of your kernels.
// Use the optimal configurations when launching your kernels.
// Remember to link against the NVML library when compiling, and to handle any potential errors that might occur during the autotuning process.
