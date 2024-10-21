# **Guide for Running All Tests in `bitnet_cuda`**

This guide provides detailed instructions on how to compile and run all tests for the `bitnet_cuda` project, specifically focusing on the test files `test_kernels.cu` and `test_memory_management.cu` located in the `tests/` directory. We will cover the required tools, setup procedures, compilation steps, command-line arguments, options, and the overall flow of the testing process.

---

## **Table of Contents**

1. [Prerequisites](#1-prerequisites)
2. [Understanding the Test Files](#2-understanding-the-test-files)
   - [2.1. `test_kernels.cu`](#21-test_kernelscu)
   - [2.2. `test_memory_management.cu`](#22-test_memory_managementcu)
3. [Setting Up the Environment](#3-setting-up-the-environment)
   - [3.1. Installing CUDA Toolkit](#31-installing-cuda-toolkit)
   - [3.2. Installing Google Test Framework](#32-installing-google-test-framework)
4. [Compiling the Tests](#4-compiling-the-tests)
   - [4.1. Compile Options](#41-compile-options)
   - [4.2. Compilation Steps](#42-compilation-steps)
5. [Running the Tests](#5-running-the-tests)
   - [5.1. Test Executables](#51-test-executables)
   - [5.2. Command-Line Arguments and Options](#52-command-line-arguments-and-options)
6. [Understanding Test Flow](#6-understanding-test-flow)
   - [6.1. Flow of `test_kernels.cu`](#61-flow-of-test_kernelscu)
   - [6.2. Flow of `test_memory_management.cu`](#62-flow-of-test_memory_managementcu)
7. [Analyzing Test Results](#7-analyzing-test-results)
8. [Automating the Test Process](#8-automating-the-test-process)
9. [Troubleshooting](#9-troubleshooting)
10. [Additional Resources](#10-additional-resources)

---

## **1. Prerequisites**

Before running the tests, ensure that your system meets the following requirements:

- **Operating System**: Linux or macOS (Windows users may need to adjust commands accordingly)
- **CUDA Toolkit**: Version compatible with your NVIDIA GPU
- **NVIDIA GPU Drivers**: Latest version recommended
- **Compiler**: Supports C++11 (e.g., GCC 5.0 or newer)
- **CMake**: Version 3.12 or higher
- **Google Test Framework**: Installed on your system
- **Git**: For cloning repositories if needed

---

## **2. Understanding the Test Files**

### **2.1. `test_kernels.cu`**

**Purpose**: Tests the main CUDA kernels implemented in the `bitnet_cuda` library to ensure they function correctly.

**Key Components**:

- **Test Fixture**: `BitNetKernelTest`
- **Test Cases**:
  - `QuantizeWeightsKernel`
  - `BitnetMatmulKernel`
  - `BitnetFusedQuantizeLayernormKernel`
- **Helper Functions**: `initializeTestData`

**Location**:

```cpp:tests/test_kernels.cu
// tests/test_kernels.cu
```

### **2.2. `test_memory_management.cu`**

**Purpose**: Validates the memory management functionalities provided by the `BitNetMemoryManager` class.

**Key Components**:

- **Test Fixture**: `BitNetMemoryTest`
- **Test Cases**:
  - `AllocateAndFreeDeviceMemory`
  - `AllocateMultipleTypes`
  - `GetAllocatedMemory`
  - `FreeNonexistentMemory`
  - `GetNonexistentMemory`
  - `ReallocateMemory`
  - `AllocateLargeMemory`
  - `StressTest`

**Location**:

```cpp:tests/test_memory_management.cu
// tests/test_memory_management.cu
```

---

## **3. Setting Up the Environment**

### **3.1. Installing CUDA Toolkit**

1. **Download CUDA Toolkit**:

   - Visit the [NVIDIA CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads) page.
   - Select your operating system, architecture, distribution, and version.
   - Follow the provided instructions for downloading.

2. **Install CUDA Toolkit**:

   - Follow the installation instructions specific to your operating system.
   - Ensure that you also install the NVIDIA driver if prompted.

3. **Set Environment Variables**:

   - Add CUDA paths to your environment variables.

     ```bash
     export CUDA_HOME=/usr/local/cuda
     export PATH=$CUDA_HOME/bin:$PATH
     export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
     ```

   - Add these lines to your `~/.bashrc` or `~/.bash_profile` to make them persistent.

### **3.2. Installing Google Test Framework**

**Option 1: Install via Package Manager (Ubuntu/Debian)**

```bash
sudo apt-get update
sudo apt-get install libgtest-dev
sudo apt-get install cmake
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp *.a /usr/lib
```

**Option 2: Build from Source**

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/google/googletest.git
   cd googletest
   ```

2. **Build and Install**:

   ```bash
   mkdir build && cd build
   cmake ..
   make
   sudo make install
   ```

---

## **4. Compiling the Tests**

### **4.1. Compile Options**

When compiling the CUDA tests, we need to specify:

- **Include Paths**:
  - `-I../include`: Includes the project's header files.
  - `-I /usr/local/include`: Includes standard and third-party headers (adjust if necessary).
- **Library Paths**:
  - `-L /usr/local/lib`: Path to the libraries (adjust if necessary).
- **Libraries to Link**:
  - `-lcudart`: CUDA runtime library.
  - `-lgtest`: Google Test library.
  - `-lgtest_main`: Google Test main function.
- **Compiler Flags**:
  - `-std=c++11`: Use the C++11 standard.

### **4.2. Compilation Steps**

**Step 1: Navigate to the Project Directory**

```bash
cd path/to/bitnet_cuda
```

**Step 2: Create a Build Directory**

```bash
mkdir build && cd build
```

**Step 3: Compile `test_kernels.cu`**

```bash
nvcc -std=c++11 -I ../include -I /usr/local/include \
     -L /usr/local/lib \
     ../tests/test_kernels.cu -o test_kernels \
     -lcudart -lgtest -lgtest_main
```

**Step 4: Compile `test_memory_management.cu`**

```bash
nvcc -std=c++11 -I ../include -I /usr/local/include \
     -L /usr/local/lib \
     ../tests/test_memory_management.cu -o test_memory_management \
     -lcudart -lgtest -lgtest_main
```

**Notes**:

- **Adjust Paths**: Ensure that the paths to include directories and libraries reflect your system's configuration.
- **CUDA Version Compatibility**: If you encounter errors, check that your CUDA version is compatible with your GPU and the code.
- **Dependencies**: Make sure all dependencies are installed and accessible.

---

## **5. Running the Tests**

### **5.1. Test Executables**

After compilation, you should have the following executables in your `build/` directory:

- `test_kernels`
- `test_memory_management`

### **5.2. Command-Line Arguments and Options**

Google Test provides several command-line options to control test execution.

**Common Options**:

- **`--gtest_list_tests`**: List all available tests.

  ```bash
  ./test_kernels --gtest_list_tests
  ```

- **`--gtest_filter`**: Run specific tests.

  ```bash
  ./test_kernels --gtest_filter=BitNetKernelTest.QuantizeWeightsKernel
  ```

- **`--gtest_repeat`**: Repeat tests multiple times.

  ```bash
  ./test_memory_management --gtest_repeat=5
  ```

- **`--gtest_break_on_failure`**: Stop execution upon first failure.

  ```bash
  ./test_kernels --gtest_break_on_failure
  ```

- **`--gtest_output`**: Output test results to a file.

  ```bash
  ./test_kernels --gtest_output=xml:test_results.xml
  ```

**Running All Tests**

To run all tests in `test_kernels`:

```bash
./test_kernels
```

To run all tests in `test_memory_management`:

```bash
./test_memory_management
```

**Running Specific Test Cases**

Example: Run only the `AllocateAndFreeDeviceMemory` test in `test_memory_management`:

```bash
./test_memory_management --gtest_filter=BitNetMemoryTest.AllocateAndFreeDeviceMemory
```

---

## **6. Understanding Test Flow**

### **6.1. Flow of `test_kernels.cu`

The `test_kernels.cu` file tests the core CUDA kernels. Here's how the flow works:

1. **Test Fixture Setup (`BitNetKernelTest`)**:

   - **`SetUp`**: Initializes CUDA resources and sets the device.
   - **`TearDown`**: Resets the CUDA device to clean up resources.

2. **Test Cases**:

   - **`QuantizeWeightsKernel`**:
     - Initializes host data with random values.
     - Allocates device memory and copies data to the device.
     - Launches the `quantize_weights_kernel`.
     - Synchronizes the device and checks for errors.
     - Copies results back to the host.
     - Verifies that the scales are within a reasonable range and quantized values are correctly in `{-1, 0, 1}`.
     - Frees memory.

   - **`BitnetMatmulKernel`**:
     - Sets matrix dimensions.
     - Allocates and initializes device memory for inputs and outputs.
     - Launches the `bitnet_matmul_kernel`.
     - Synchronizes and checks for errors.
     - Copies and verifies results on the host.
     - Checks that the output is not all zeros.
     - Frees memory.

   - **`BitnetFusedQuantizeLayernormKernel`**:
     - Sets batch size and hidden size.
     - Allocates and initializes device memory for inputs and parameters.
     - Launches the `bitnet_fused_quantize_layernorm_kernel`.
     - Synchronizes and checks for errors.
     - Copies results back to the host.
     - Verifies scales and quantized output values.
     - Frees memory.

3. **Test Execution**:

   - The `main` function initializes the Google Test framework and runs all tests.

### **6.2. Flow of `test_memory_management.cu`**

The `test_memory_management.cu` file tests the `BitNetMemoryManager` class.

1. **Test Fixture Setup (`BitNetMemoryTest`)**:

   - **`SetUp`**: Creates an instance of `BitNetMemoryManager`.
   - **`TearDown`**: Deletes the memory manager instance.

2. **Test Cases**:

   - **`AllocateAndFreeDeviceMemory`**:
     - Allocates device memory for floats.
     - Copies data to and from the device.
     - Verifies data integrity.
     - Frees memory.

   - **`AllocateMultipleTypes`**:
     - Allocates memory for different data types (`float`, `int`, `ternary_t`).
     - Verifies allocations.
     - Frees memory.

   - **`GetAllocatedMemory`**:
     - Allocates memory and retrieves it by name.
     - Verifies that the pointers match.
     - Frees memory.

   - **`FreeNonexistentMemory`**:
     - Attempts to free a non-existent memory block.
     - Expects no exceptions or errors.

   - **`GetNonexistentMemory`**:
     - Attempts to get a non-existent memory block.
     - Expects a `nullptr` return.

   - **`ReallocateMemory`**:
     - Allocates memory with an initial size.
     - Reallocates with a new size.
     - Verifies data preservation.
     - Frees memory.

   - **`AllocateLargeMemory`**:
     - Attempts to allocate a large memory block (e.g., 1 GB).
     - Verifies allocation success.
     - Frees memory.

   - **`StressTest`**:
     - Performs multiple allocations and deallocations in a loop.
     - Verifies stability under stress.

3. **Test Execution**:

   - The `main` function initializes the Google Test framework and runs all tests.

---

## **7. Analyzing Test Results**

When you run the test executables, you will see output indicating the progress and results of each test case.

**Example Output**:

```
[==========] Running 3 tests from 1 test case.
[----------] Global test environment set-up.
[----------] 3 tests from BitNetKernelTest
[ RUN      ] BitNetKernelTest.QuantizeWeightsKernel
[       OK ] BitNetKernelTest.QuantizeWeightsKernel (123 ms)
[ RUN      ] BitNetKernelTest.BitnetMatmulKernel
[       OK ] BitNetKernelTest.BitnetMatmulKernel (456 ms)
[ RUN      ] BitNetKernelTest.BitnetFusedQuantizeLayernormKernel
[       OK ] BitNetKernelTest.BitnetFusedQuantizeLayernormKernel (789 ms)
[----------] 3 tests from BitNetKernelTest (1368 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test case ran. (1368 ms total)
[  PASSED  ] 3 tests.
```

- **`[ RUN      ]`**: Indicates the start of a test case.
- **`[       OK ]`**: Test passed successfully.
- **`[  FAILED  ]`**: Test failed (details will follow).
- **`[==========]`**: Summary of tests run.

**Analyzing Failures**:

If a test fails, Google Test will provide detailed information about the failure, including:

- File name and line number where the failure occurred.
- Expected versus actual values.
- Failure messages from assertions (e.g., `EXPECT_EQ`, `EXPECT_TRUE`).

**Example Failure**:

```
[ RUN      ] BitNetKernelTest.QuantizeWeightsKernel
test_kernels.cu:97: Failure
Value of: value == -1 || value == 0 || value == 1
  Actual: false
Expected: true
[  FAILED  ] BitNetKernelTest.QuantizeWeightsKernel (123 ms)
```

---

## **8. Automating the Test Process**

To streamline the compilation and execution of tests, you can create a shell script or use a makefile.

### **Creating a Shell Script**

**File Name**: `run_all_tests.sh`

```bash
#!/bin/bash

set -e

echo "Compiling test_kernels.cu..."
nvcc -std=c++11 -I ../include -I /usr/local/include \
     -L /usr/local/lib \
     ../tests/test_kernels.cu -o test_kernels \
     -lcudart -lgtest -lgtest_main

echo "Compiling test_memory_management.cu..."
nvcc -std=c++11 -I ../include -I /usr/local/include \
     -L /usr/local/lib \
     ../tests/test_memory_management.cu -o test_memory_management \
     -lcudart -lgtest -lgtest_main

echo "Running test_kernels..."
./test_kernels

echo "Running test_memory_management..."
./test_memory_management
```

**Make the Script Executable and Run It**

```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```

### **Using a Makefile**

**File Name**: `Makefile`

```makefile
NVCC = nvcc
CXXFLAGS = -std=c++11 -I ../include -I /usr/local/include
LDFLAGS = -L /usr/local/lib -lcudart -lgtest -lgtest_main
TESTS = test_kernels test_memory_management

all: $(TESTS)

test_kernels: ../tests/test_kernels.cu
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

test_memory_management: ../tests/test_memory_management.cu
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run: $(TESTS)
	@echo "Running test_kernels..."
	./test_kernels
	@echo "Running test_memory_management..."
	./test_memory_management

clean:
	rm -f $(TESTS)
```

**Usage**

- Compile Tests:

  ```bash
  make
  ```

- Run Tests:

  ```bash
  make run
  ```

- Clean Up:

  ```bash
  make clean
  ```

---

## **9. Troubleshooting**

### **Common Issues and Solutions**

1. **Missing Libraries or Headers**

   - **Error Message**: `fatal error: gtest/gtest.h: No such file or directory`

     **Solution**: Ensure that the include path `-I /usr/local/include` (or wherever Google Test is installed) is correct.

2. **CUDA Version Mismatch**

   - **Error Message**: CUDA version mismatch errors during compilation or runtime.

     **Solution**: Verify that your NVIDIA drivers and CUDA Toolkit versions are compatible and up to date.

3. **Linking Errors**

   - **Error Message**: Undefined references to Google Test functions.

     **Solution**: Ensure you are linking against `-lgtest` and `-lgtest_main`.

4. **Permission Denied**

   - **Error Message**: `permission denied` when running scripts or executables.

     **Solution**: Check file permissions and make executables runnable using `chmod +x filename`.

5. **Insufficient GPU Memory**

   - **Error Message**: CUDA out-of-memory errors during test execution.

     **Solution**: Reduce the test data sizes (e.g., `TEST_SIZE`, `M`, `N`, `K`) in the test files to fit within your GPU's memory limits.

### **Debugging Tips**

- **Use Debug Build**: Compile with debugging symbols by adding `-G` to the NVCC flags.

  ```bash
  nvcc -G -std=c++11 ...
  ```

- **Run Under Debugger**: Use `cuda-gdb` to debug CUDA applications.

  ```bash
  cuda-gdb ./test_kernels
  ```

- **Check Device Status**: Use `nvidia-smi` to monitor GPU usage and ensure the device is not in an unstable state.

  ```bash
  nvidia-smi
  ```

- **Add Verbose Output**: Modify test cases to include additional `std::cout` statements for debugging purposes.

---

## **10. Additional Resources**

- **Google Test Documentation**: [Google Test Primer](https://google.github.io/googletest/primer.html)
- **CUDA C++ Programming Guide**: [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- **NVIDIA Developer Forums**: [CUDA Setup and Installation](https://forums.developer.nvidia.com/c/gpu-accelerated-libraries/cuda-setup-and-installation)
- **CUDA-GDB Debugger**: [CUDA-GDB Documentation](https://docs.nvidia.com/cuda/cuda-gdb/index.html)
- **BitNet CUDA Project Documentation**: Refer to any provided documentation within the `docs/` directory of the project.

---

**Note**: The exact paths and commands may vary depending on your system configuration and where you've installed dependencies. Always replace placeholder paths with the actual paths on your system.

By following this guide, you should be able to compile and run all tests for the `bitnet_cuda` project with a detailed understanding of each step involved. If you encounter any issues or have further questions, feel free to reach out for assistance.

---

**Happy Testing!**