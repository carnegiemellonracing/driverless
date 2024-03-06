cmake_minimum_required(VERSION 3.18)
project(cuda_shittery C CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)

find_package(CUDAToolkit)
find_package(sdl2)
find_package(OpenGL)

include_directories(extern/include)
include_directories(./)

add_executable(cuda_shittery extern/src/glad.c gl_experiments/headless_render.cpp gl_experiments/swap_rg.cu)

set_target_properties(cuda_shittery PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75)

target_link_libraries(cuda_shittery PRIVATE
        ${CMAKE_DL_LIBS}
        ${SDL2_LIBRARIES}
        OpenGL::GL OpenGL::GLU)

target_compile_options(cuda_shittery PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        >)
