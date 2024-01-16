cmake_minimum_required(VERSION 3.18)
project(cuda_shittery CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_package(CUDAToolkit)

add_executable(cuda_shittery main.cu)

set_target_properties(cuda_shittery PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75)

target_link_libraries(cuda_shittery PRIVATE CUDA::curand)
