cmake_minimum_required(VERSION 3.18)
project(RELU LANGUAGES CUDA CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find PyTorch via Python
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
    add_definitions(-DNO_TORCH)  # A define to conditionally exclude PyTorch code if desired
endif()

# List your source files
set(SRC_FILES
    src/relu_naive.cu
    src/relu_vectorized.cu
    src/relu_optim.cu
    benchmark/benchmark.cu
)

# Create the executable
add_executable(benchmark ${SRC_FILES})

# Set CUDA properties (e.g., separable compilation, GPU arch)
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75  # Adjust as needed
)

# Link to CUDA's cuBLAS and PyTorch (if found)
target_link_libraries(benchmark PRIVATE cublas ${TORCH_LIBRARIES})

# Optional: custom target to run the benchmark
add_custom_target(run_benchmark
    COMMAND benchmark
    DEPENDS benchmark
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
