cmake_minimum_required(VERSION 3.18)
project(KNN CUDA CXX)

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
set(SOURCES main.cu)

# Create the executable
add_executable(KNN ${SOURCES})

# Include CUDA directories
target_include_directories(KNN PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Set CUDA specific properties
set_target_properties(KNN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA runtime and other necessary libraries
target_link_libraries(KNN PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
)