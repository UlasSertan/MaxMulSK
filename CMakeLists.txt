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
        -fopenmp
)

# NEON: ARMv8.4 baseline
set_source_files_properties(neon/GEMMKernels.cpp neon/test_neon.cpp PROPERTIES
        COMPILE_OPTIONS "-march=armv8.4-a"
)

# SME: ARMv8.7 + SME/SME2 extensions (re-enable with the sources above)
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

# =============================================================================
# bench_compare — single-thread: our NEON vs Apple Accelerate
# =============================================================================
add_executable(bench_compare
        bench/bench_compare.cpp
        neon/GEMMKernels.cpp
)

target_compile_options(bench_compare PRIVATE
        -O3
        -mcpu=apple-m4
        -march=armv8.4-a
        -fopenmp
)

target_include_directories(bench_compare PRIVATE
        /opt/homebrew/opt/libomp/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(bench_compare PRIVATE
        /opt/homebrew/opt/libomp/lib/libomp.dylib
        "-framework Accelerate"
        # OpenBLAS loaded at runtime via dlopen — no link-time dependency needed
)
