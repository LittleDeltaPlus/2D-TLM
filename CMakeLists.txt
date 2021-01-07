cmake_minimum_required(VERSION 3.15)
project(2D_TLM)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2")

add_executable(2D_TLM cpu_TLM.cpp)