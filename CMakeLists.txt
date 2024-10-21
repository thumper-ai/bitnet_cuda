cmake_minimum_required(VERSION 3.18)
project(BitNetCUDA CUDA CXX)

# Version checking
find_package(CUDA 11.0 REQUIRED)
if(CUDA_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA version must be at least 11.0")
endif()

find_package(CUDAToolkit REQUIRED)

# Options
option(BITNET_ENABLE_PYTORCH "Enable PyTorch integration" ON)
option(BITNET_ENABLE_MULTI_GPU "Enable multi-GPU support" ON)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g")

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)

# Dependencies
include(cmake/CPM.cmake)
CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 7.1.3
)

if(BITNET_ENABLE_PYTORCH)
    find_package(Torch REQUIRED)
    add_definitions(-DBITNET_ENABLE_PYTORCH)
endif()

if(BITNET_ENABLE_MULTI_GPU)
    add_definitions(-DBITNET_ENABLE_MULTI_GPU)
endif()

# Source files
set(BITNET_SOURCES
    src/bitnet_cuda.cu
    src/memory_manager.cu
    src/auto_tuner.cu
    src/kernels/quantization_kernels.cu
    src/kernels/matmul_kernels.cu
    src/kernels/normalization_kernels.cu
)

# Main library
add_library(bitnet_cuda STATIC ${BITNET_SOURCES})
target_include_directories(bitnet_cuda PUBLIC include)
target_link_libraries(bitnet_cuda PUBLIC CUDA::cudart CUDA::cublas fmt::fmt)

if(BITNET_ENABLE_PYTORCH)
    target_link_libraries(bitnet_cuda PUBLIC ${TORCH_LIBRARIES})
endif()

# Installation
install(TARGETS bitnet_cuda
        EXPORT BitNetCUDATargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(DIRECTORY include/
        DESTINATION include/bitnet_cuda
        FILES_MATCHING PATTERN "*.h")

install(EXPORT BitNetCUDATargets
        FILE BitNetCUDATargets.cmake
        NAMESPACE BitNetCUDA::
        DESTINATION lib/cmake/BitNetCUDA)