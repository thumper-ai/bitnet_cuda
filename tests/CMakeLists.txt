# Minimum CMake version
cmake_minimum_required(VERSION 3.18)

# Find required packages
find_package(CUDA REQUIRED)
find_package(GTest REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Set CUDA architecture flags (adjust based on your target GPUs)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)

# Define test sources
set(TEST_SOURCES
    test_memory_management.cu
    test_auto_tuning.cu
    test_kernels.cu
    test_multi_gpu.cu
    test_pytorch_extension.cpp
)

# Create test executable
add_executable(bitnet_tests ${TEST_SOURCES})

# Link libraries
target_link_libraries(bitnet_tests
    PRIVATE
    bitnet_cuda
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

# Set CUDA specific properties
set_target_properties(bitnet_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add CUDA compiler flags
target_compile_options(bitnet_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -Xptxas -O3
    -Xcompiler -O3
>)

# Enable testing
enable_testing()

# Add test
add_test(NAME BitNetTests COMMAND bitnet_tests)

# Custom target for running tests with verbose output
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS bitnet_tests
)

# If using NVTX for profiling
find_library(NVTX_LIBRARY nvToolsExt PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
if(NVTX_LIBRARY)
    target_link_libraries(bitnet_tests PRIVATE ${NVTX_LIBRARY})
    target_compile_definitions(bitnet_tests PRIVATE USE_NVTX)
endif()

# If using PyTorch extension
find_package(Torch QUIET)
if(TORCH_FOUND)
    target_link_libraries(bitnet_tests PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(bitnet_tests PRIVATE WITH_PYTORCH)
endif()

# Copy any necessary test data
file(COPY ${CMAKE_SOURCE_DIR}/tests/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# This CMakeLists.txt file does the following:
# Sets up the required CMake version and finds necessary packages (CUDA and GTest).
# Sets C++ and CUDA standards.
# Configures CUDA architecture flags.
# Sets up include directories.
# Defines test sources.
# Creates the test executable and links necessary libraries.
# Sets CUDA-specific properties and compiler flags.
# Enables testing and adds a custom target for running tests with verbose output.
# Adds optional support for NVTX profiling.
# Adds optional support for PyTorch extensions.
# Copies any necessary test data to the build directory.
# To use this CMakeLists.txt:
# Place it in your tests directory.
# Make sure you have a main CMakeLists.txt file in your project root that includes the tests directory.
# Create the test files mentioned in TEST_SOURCES.
# Build your project using CMake.
# Run the tests using ctest or the run_tests target.
# Remember to adjust the CUDA architecture flags and any other settings as necessary for your specific project setup and target hardware.