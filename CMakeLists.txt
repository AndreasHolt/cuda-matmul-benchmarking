cmake_minimum_required(VERSION 3.18)
project(matmul CUDA CXX)

# Enable CUDA language support
enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

# Set the source files
set(SOURCES
        ./src/main.cu
        src/naive_matmul.cu
        src/naive_matmul.cuh
        src/matmul_helpers.cu
        src/matmul_helpers.cuh
        src/cpu_matmul.cu
        src/cpu_matmul.cuh)

# Create the executable
add_executable(matmul ${SOURCES})

# Include CUDA directories
target_include_directories(matmul PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Set CUDA specific properties
set_target_properties(matmul PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA runtime and other necessary libraries
target_link_libraries(matmul PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
)