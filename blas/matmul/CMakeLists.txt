cmake_minimum_required(VERSION 3.18)
project(MatmulBenchmark LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CUDA_ARCHITECTURES 75)

set(SRC_FILES
    src/naive_gemm.cu
    src/optim_tiled_gemm.cu
    src/regblock_gemm.cu
    src/tiled_gemm.cu
    src/warp_tiling_gemm.cu
    src/wmma_gemm.cu
)

add_executable(benchmark src/benchmark.cu ${SRC_FILES})

set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
)

target_link_libraries(benchmark PRIVATE cublas)

add_custom_target(run_benchmark
    COMMAND benchmark
    DEPENDS benchmark
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)