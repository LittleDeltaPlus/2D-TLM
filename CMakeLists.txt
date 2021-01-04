cmake_minimum_required(VERSION 3.15)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.2/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

project(2D_TLM LANGUAGES CUDA CXX)


add_executable(2D_TLM 2DTLM.cu)

set_target_properties(
        2D_TLM
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)