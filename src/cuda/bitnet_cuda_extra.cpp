class BitNetCUDA {
private:
    BitNetMemoryManager memory_manager;
    BitNetAutoTuner auto_tuner;
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

public:
    BitNetCUDA() {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    ~BitNetCUDA() {
        CUDA_CHECK(cudaStreamDestroy(stream));
        if (graph_exec) {
            CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
        }
        if (graph) {
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }

    void allocate_model_parameters(int batch_size, int seq_length, int hidden_size, int num_layers) {
        memory_manager.allocate<ternary_t>("weights", hidden_size * hidden_size * num_layers);
        memory_manager.allocate<float>("weight_scales", num_layers);
        memory_manager.allocate<half>("activations", batch_size * seq_length * hidden_size);
        memory_manager.allocate<float>("activation_scales", num_layers + 1);
    }

    void create_computation_graph() {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        // Launch all kernels here
        auto [block, grid] = auto_tuner.get_optimal_config("fused_quantize_matmul");
        bitnet_fused_quantize_matmul_kernel<<<grid, block, 0, stream>>>(
            memory_manager.get<float>("input"),
            memory_manager.get<half>("weights"),
            memory_manager.get<half>("output"),
            memory_manager.get<float>("input_scale"),
            memory_manager.get<float>("weight_scale"),
            /* other parameters */
        );

        // ... other kernel launches

        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    }

    void run_computation_graph() {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void tune_kernels() {
        std::vector<dim3> block_sizes = {{128}, {256}, {512}};
        std::vector<dim3> grid_sizes = {{1}, {2}, {4}};

        auto_tuner.tune_kernel("fused_quantize_matmul", bitnet_fused_quantize_matmul_kernel,
                               block_sizes, grid_sizes,
                               memory_manager.get<float>("input"),
                               memory_manager.get<half>("weights"),
                               memory_manager.get<half>("output"),
                               memory_manager.get<float>("input_scale"),
                               memory_manager.get<float>("weight_scale"),
                               /* other parameters */);

        // ... tune other kernels
    }
};