# BitNet CUDA Implementation

This repository contains a high-performance CUDA implementation of BitNet, a novel neural network architecture that uses extreme quantization (1.58-bit weights) for efficient inference and training.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Building the Project](#building-the-project)
5. [Running Tests](#running-tests)
6. [Profiling](#profiling)
7. [PyTorch Integration](#pytorch-integration)
8. [Usage](#usage)
9. [Multi-GPU Support](#multi-gpu-support)
10; [Configuration Options](#configuration-options)
11. [Contributing](#contributing)
12. [License](#license)


## Prerequisites

- CUDA Toolkit 11.0 or later
- CMake 3.18 or later
- GCC 7.0 or later
- Python 3.6 or later (for PyTorch integration)
- PyTorch 1.7.0 or later (for PyTorch integration)
- NVIDIA GPU with compute capability 7.0 or higher

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bitnet-cuda.git
   cd bitnet-cuda
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Install CUDA Toolkit:
   Follow the instructions on the [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for your operating system.

## Project Structure

```
bitnet-cuda/
├── include/
│   ├── bitnet_cuda.h
│   ├── memory_manager.h
│   ├── auto_tuning.h
│   ├── bitnet_cuda.h
│   ├── bitnet_types.h
│   ├── memory_management.h
│   ├── multi_gpu.h
│   ├── pytorch_extension.h
│   ├── utils.h
│   └── kernels.h
├── src/
│   ├── bitnet_cuda.cu
│   ├── memory_management.cu
│   ├── multi_gpu.cu
│   ├── auto_tuner.cu
│   ├── utils.cu
│   ├── kernels.cu
│   └── kernels/
│       ├── matmul_kernels.cu
│       ├── normalization_kernels.cu
│       ├── linear_kernels.cu
│       ├── persistent_kernels.cu
│       ├── quantization_kernels.cu
│       └── activation_kernels.cu
├── tests/
    ├── CMakeLists.txt
│   ├── test_bitnet_cuda.cu
│   ├── test_memory_manager.cu
│   └── test_kernels.cu
├── pytorch_extension/
│   ├── setup.py
│   └── bitnet_cuda.cpp
├── benchmark/
├── examples/
├── docs/
    ├── TestGuide.md

├── CMakeLists.txt
├── requirements.txt
└── README.md
```

## Building the Project

1. Create a build directory:
   ```
   mkdir build && cd build
   ```

2. Configure the project with CMake:
   ```
   cmake ..
   ```

3. Build the project:
   ```
   make -j$(nproc)
   ```

This will build the main BitNet CUDA library and the test executables.

## Running Tests

After building the project, you can run the tests:

```
ctest --verbose
```

Or run individual test executables:

```
./tests/test_bitnet_cuda
./tests/test_memory_manager
./tests/test_kernels
```

## Profiling

To profile the BitNet CUDA implementation, we use NVIDIA's Nsight Compute and Nsight Systems tools.

1. Profile with Nsight Compute:
   ```
   ncu --set full ./tests/test_bitnet_cuda
   ```

2. Profile with Nsight Systems:
   ```
   nsys profile --stats=true ./tests/test_bitnet_cuda
   ```

Analyze the results using the Nsight Compute and Nsight Systems GUI tools.

## PyTorch Integration

To build the PyTorch extension:

1. Navigate to the `pytorch_extension` directory:
   ```
   cd pytorch_extension
   ```

2. Build the extension:
   ```
   python setup.py install
   ```

3. Test the PyTorch extension:
   ```
   python -m unittest discover tests
   ```

## Usage

Here's a basic example of how to use the BitNet CUDA implementation:

```cpp
#include "bitnet_cuda.h"

int main() {
    BitNetCUDA bitnet(2);  // Create BitNetCUDA instance with 2 GPUs
    
    // Allocate model parameters
    bitnet.allocate_model_parameters(batch_size, seq_length, hidden_size, num_layers);
    
    // Tune kernels (optional)
    bitnet.tune_kernels();
    
    // Create computation graph
    bitnet.create_computation_graph();
    
    // Run forward pass
    bitnet.run_computation_graph();
    
    return 0;
}
```

For PyTorch integration:

```python
import torch
from bitnet_cuda import BitNetCUDAFunction

# Create input tensor
input_tensor = torch.randn(batch_size, seq_length, hidden_size, device='cuda')

# Create BitNet layer
bitnet_layer = BitNetCUDAFunction.apply

# Run forward pass
output = bitnet_layer(input_tensor)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```


## Multi-GPU Support

BitNet CUDA supports multi-GPU execution for large models and datasets. To use multiple GPUs:

1. Specify the number of GPUs when creating the BitNetCUDA instance:
   ```cpp
   BitNetCUDA bitnet(num_gpus);
   ```

2. The implementation will automatically distribute the workload across available GPUs.

3. For PyTorch integration, use `nn.DataParallel` or `nn.DistributedDataParallel` as usual.

## Configuration Options

BitNet CUDA supports various configuration options:

- `--num-gpus`: Number of GPUs to use (default: 1)
- `--batch-size`: Batch size for training/inference (default: 32)
- `--hidden-size`: Hidden size of the model (default: 768)
- `--num-layers`: Number of layers in the model (default: 12)
- `--quantization-bits`: Number of bits for quantization (default: 1.58)

For a complete list of options, run:
```
./bitnet_benchmark --help
```
