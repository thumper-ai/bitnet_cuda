class BitNetAutoTuner {
private:
    std::unordered_map<std::string, std::pair<dim3, dim3>> optimal_configs;

public:
    template<typename KernelFunc>
    void tune_kernel(const std::string& kernel_name, KernelFunc kernel, 
                     const std::vector<dim3>& block_sizes, 
                     const std::vector<dim3>& grid_sizes,
                     /* other kernel parameters */) {
        float best_time = std::numeric_limits<float>::max();
        dim3 best_block, best_grid;

        for (const auto& block : block_sizes) {
            for (const auto& grid : grid_sizes) {
                cudaEvent_t start, stop;
                CUDA_CHECK(cudaEventCreate(&start));
                CUDA_CHECK(cudaEventCreate(&stop));

                CUDA_CHECK(cudaEventRecord(start));
                kernel<<<grid, block>>>(/* kernel parameters */);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));

                float time;
                CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

                if (time < best_time) {
                    best_time = time;
                    best_block = block;
                    best_grid = grid;
                }

                CUDA_CHECK(cudaEventDestroy(start));
                CUDA_CHECK(cudaEventDestroy(stop));
            }
        }

        optimal_configs[kernel_name] = {best_block, best_grid};
    }

    std::pair<dim3, dim3> get_optimal_config(const std::string& kernel_name) {
        return optimal_configs[kernel_name];
    }
};