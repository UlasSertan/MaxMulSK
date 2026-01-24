cmake_minimum_required(VERSION 3.30)
project(MatrixLibrary)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

add_executable(MatrixLibrary main.cpp
        Matrix.hpp
        GEMMKernels.cpp
        GEMMKernels.hpp)
