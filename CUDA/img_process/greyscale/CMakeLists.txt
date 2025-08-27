cmake_minimum_required(VERSION 3.10)
project(CUDA_Greyscale_Benchmark CUDA CXX)

# Set CUDA architectures to compile for
set(CMAKE_CUDA_ARCHITECTURES 52 60 70 75 80 86)  # Adjust based on your target GPUs

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14)

# Add kernel implementation files
add_library(kernel_implementations STATIC
    src/naive_greyscale.cu
    src/coalesced_greyscale.cu
    src/mutlipixel_greysc.cu
    src/fma_greyscale.cu
    src/shared_greysc.cu
    src/vectorized_greysc.cu
)
set_target_properties(kernel_implementations PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Benchmark executable
cuda_add_executable(greyscale_benchmark 
    src/benchmark.cu
)

target_link_libraries(greyscale_benchmark
    kernel_implementations
    ${CUDA_LIBRARIES}
)




# Installation rules
install(TARGETS greyscale_benchmark DESTINATION bin)