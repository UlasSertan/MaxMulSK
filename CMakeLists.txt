cmake_minimum_required(VERSION 3.30)
project(MatrixLibrary LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES arm64)

set(CMAKE_C_COMPILER   "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

add_executable(MatrixLibrary
        main.cpp
        neon/GEMMKernels.cpp
        neon/test_neon.cpp
        sme/SME-GEMMKernels.cpp
        sme/test_sme.cpp
)

target_compile_options(MatrixLibrary PRIVATE
        -O3 -Wall -Wextra
        -mcpu=apple-m4
)

# NEON: ARMv8.4 baseline
set_source_files_properties(neon/GEMMKernels.cpp neon/test_neon.cpp PROPERTIES
        COMPILE_OPTIONS "-march=armv8.4-a"
)

# SME: ARMv8.7 + SME/SME2 extensions
set_source_files_properties(sme/SME-GEMMKernels.cpp sme/test_sme.cpp PROPERTIES
        COMPILE_OPTIONS "-march=armv8.7-a+sme+sme2"
)

target_include_directories(MatrixLibrary PRIVATE
        /opt/homebrew/opt/libomp/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(MatrixLibrary PRIVATE
        /opt/homebrew/opt/libomp/lib/libomp.dylib
)
