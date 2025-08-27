cmake_minimum_required(VERSION 3.18)
project(Conv2D LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Dynamically find PyTorch
execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
    find_package(Torch REQUIRED)
else()
    message(WARNING "PyTorch CMake path not found. Compiling without PyTorch support.")
    set(TORCH_LIBRARIES "")
    add_definitions(-DNO_TORCH)
endif()

# Source files
set(SRC_FILES
    src/naive_conv.cu
    src/tiled_conv.cu
    src/coal_tiled_conv.cu
    benchmark/benchmark.cu
)

add_executable(conv_benchmark ${SRC_FILES})

set_target_properties(conv_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75  # or whatever matches your GPU
)

target_link_libraries(conv_benchmark PRIVATE cublas ${TORCH_LIBRARIES})

add_custom_target(run_benchmark
    COMMAND conv_benchmark
    DEPENDS conv_benchmark
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
