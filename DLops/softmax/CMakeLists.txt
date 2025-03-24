cmake_minimum_required(VERSION 3.10)
project(cuda_softmax LANGUAGES CXX CUDA)

# Set C++ standard to C++17 (required for PyTorch)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA configuration
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# PyTorch configuration
# Option 1: Use find_package with a hint to the libtorch path
# Uncomment and modify the path if you know where libtorch is installed
# set(CMAKE_PREFIX_PATH "/path/to/libtorch")
# find_package(Torch REQUIRED)

# Option 2: Use direct configuration for PyTorch
# For systems where PyTorch is installed through pip
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
    # Define empty variables to allow compilation without PyTorch
    set(TORCH_LIBRARIES "")
    set(TORCH_INCLUDE_DIRS "")
    
    # Add a compile definition to conditionally exclude PyTorch code
    add_definitions(-DNO_TORCH)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if(TORCH_INCLUDE_DIRS)
    include_directories(${TORCH_INCLUDE_DIRS})
endif()

# Add source files
set(SOURCES
    src/softmax_naive.cu
    src/softmax_shared.cu
    src/softmax_warp.cu
    src/softmax_block.cu
    src/softmax_sota.cu
    benchmark/main.cu
)

# Create the executable
add_executable(softmax_benchmark ${SOURCES})

# Link against CUDA and PyTorch libraries (if available)
target_link_libraries(softmax_benchmark ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})

# Set CUDA flags
set_target_properties(softmax_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80"  # Adjust based on your GPU architecture
)

# Install the executable
install(TARGETS softmax_benchmark DESTINATION bin)