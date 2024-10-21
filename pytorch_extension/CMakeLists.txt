# pytorch_extension/CMakeLists.txt

find_package(Torch REQUIRED)

add_library(bitnet_pytorch_ext SHARED
    src/pytorch_extension.cu
)

target_link_libraries(bitnet_pytorch_ext
    PRIVATE
    bitnet_cuda
    ${TORCH_LIBRARIES}
)

set_target_properties(bitnet_pytorch_ext PROPERTIES PREFIX "")