cmake_minimum_required(VERSION 3.18)
project(RELU LANGUAGES CUDA CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch via Python
execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_PYTHON_RESULT
)

# Debug output
message(STATUS "Python result: ${TORCH_PYTHON_RESULT}")
message(STATUS "Torch cmake prefix path: ${TORCH_CMAKE_PREFIX_PATH}")

# Find the Torch package.
if(TORCH_CMAKE_PREFIX_PATH AND TORCH_PYTHON_RESULT EQUAL 0)
    set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
    message(STATUS "CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
    
    find_package(Torch REQUIRED)
    
    if(TORCH_FOUND)
        message(STATUS "Torch found successfully")
        message(STATUS "Torch version: ${TORCH_VERSION}")
        message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
    else()
        message(FATAL_ERROR "Torch package found but TORCH_FOUND is false")
    endif()
else()
    message(FATAL_ERROR "PyTorch CMake path not found or Python command failed. Cannot compile.")
endif()

# List your source files
set(SRC_FILES
    src/relu_naive.cu
    src/relu_vectorized.cu
    src/relu_optim.cu
    src/relu_fp16x8.cu
    benchmark/benchmark.cu
)

# Create the executable
add_executable(benchmark ${SRC_FILES})

# Set CUDA properties
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75  # Adjust as needed for your GPU
)

# Try different linking approaches
if(TARGET torch)
    message(STATUS "Using torch target")
    target_link_libraries(benchmark PRIVATE torch)
elseif(TARGET Torch::torch)
    message(STATUS "Using Torch::torch target")
    target_link_libraries(benchmark PRIVATE Torch::torch)
elseif(TORCH_LIBRARIES)
    message(STATUS "Using TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
    target_link_libraries(benchmark PRIVATE ${TORCH_LIBRARIES})
else()
    message(FATAL_ERROR "No suitable Torch target found")
endif()

# Include directories
if(TORCH_INCLUDE_DIRS)
    target_include_directories(benchmark PRIVATE ${TORCH_INCLUDE_DIRS})
endif()

# Optional: custom target to run the benchmark
add_custom_target(run_benchmark
    COMMAND benchmark
    DEPENDS benchmark
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)