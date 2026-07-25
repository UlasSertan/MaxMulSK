cmake_minimum_required(VERSION 3.30)

# =============================================================================
# Toolchain & dependency discovery
#
# Apple's bundled Clang does not support the SME/SME2 intrinsics this project
# needs, so an LLVM/Clang from Homebrew is required. Everything below is
# discovered rather than hardcoded; override any of it explicitly:
#
#   cmake -B build -DCMAKE_CXX_COMPILER=/path/to/clang++
#   HOMEBREW_PREFIX=/custom/prefix cmake -B build
#
# Compiler selection must happen before project(), hence the placement.
# =============================================================================

if(DEFINED ENV{HOMEBREW_PREFIX})
    set(HOMEBREW_PREFIX "$ENV{HOMEBREW_PREFIX}")
else()
    find_program(BREW_EXECUTABLE brew)
    if(BREW_EXECUTABLE)
        execute_process(COMMAND "${BREW_EXECUTABLE}" --prefix
                        OUTPUT_VARIABLE HOMEBREW_PREFIX
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
    endif()
endif()
if(NOT HOMEBREW_PREFIX)
    set(HOMEBREW_PREFIX "/opt/homebrew")  # Apple Silicon default
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(LLVM_C_COMPILER   clang   HINTS "${HOMEBREW_PREFIX}/opt/llvm/bin" NO_DEFAULT_PATH)
    find_program(LLVM_CXX_COMPILER clang++ HINTS "${HOMEBREW_PREFIX}/opt/llvm/bin" NO_DEFAULT_PATH)
    if(LLVM_C_COMPILER AND LLVM_CXX_COMPILER)
        set(CMAKE_C_COMPILER   "${LLVM_C_COMPILER}")
        set(CMAKE_CXX_COMPILER "${LLVM_CXX_COMPILER}")
    else()
        message(FATAL_ERROR
            "LLVM/Clang not found under ${HOMEBREW_PREFIX}/opt/llvm/bin.\n"
            "Install it with 'brew install llvm', or point CMake at your own:\n"
            "  cmake -B build -DCMAKE_C_COMPILER=... -DCMAKE_CXX_COMPILER=...")
    endif()
endif()

project(MaxMulSK LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES arm64)

# OpenMP runtime (Homebrew libomp — Apple's toolchain ships no libomp)
find_path(LIBOMP_INCLUDE_DIR omp.h
          HINTS "${HOMEBREW_PREFIX}/opt/libomp/include")
find_library(LIBOMP_LIBRARY omp
             HINTS "${HOMEBREW_PREFIX}/opt/libomp/lib")
if(NOT LIBOMP_INCLUDE_DIR OR NOT LIBOMP_LIBRARY)
    message(FATAL_ERROR
        "libomp not found under ${HOMEBREW_PREFIX}/opt/libomp.\n"
        "Install it with 'brew install libomp', or set -DLIBOMP_INCLUDE_DIR=... "
        "and -DLIBOMP_LIBRARY=...")
endif()

add_executable(MaxMulSK
        main.cpp
        neon/neon-8x12.cpp
        neon/test_neon.cpp
        sme/sme-4x1.cpp
        sme/sme-2x2.cpp
        sme/sme-1x4.cpp
        sme/sme-1x4-sym.cpp
        sme/sme-1x4-sym-zainout.cpp
        sme/sme-4x1-zapack.cpp
        sme/test_sme.cpp
)

target_compile_options(MaxMulSK PRIVATE
        -O3 -Wall -Wextra
        -mcpu=apple-m4
        -fopenmp
)

# NEON: ARMv8.4 baseline
set_source_files_properties(neon/neon-8x12.cpp neon/test_neon.cpp PROPERTIES
        COMPILE_OPTIONS "-march=armv8.4-a"
)

# SME: ARMv8.7 + SME/SME2 extensions (re-enable with the sources above)
set_source_files_properties(sme/sme-4x1.cpp sme/sme-2x2.cpp sme/sme-1x4.cpp sme/sme-1x4-sym.cpp sme/sme-1x4-sym-zainout.cpp sme/sme-4x1-zapack.cpp sme/test_sme.cpp PROPERTIES
        COMPILE_OPTIONS "-march=armv8.7-a+sme+sme2"
)

target_include_directories(MaxMulSK PRIVATE
        ${LIBOMP_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(MaxMulSK PRIVATE
        ${LIBOMP_LIBRARY}
)

# =============================================================================
# bench_compare — single-thread: our NEON vs Apple Accelerate
# =============================================================================
add_executable(bench_compare
        bench/bench_compare.cpp
        neon/neon-8x12.cpp
        sme/sme-4x1.cpp
        sme/sme-2x2.cpp
        sme/sme-4x1-zapack.cpp
)

target_compile_options(bench_compare PRIVATE
        -O3
        -mcpu=apple-m4
        -march=armv8.4-a
        -fopenmp
)

target_include_directories(bench_compare PRIVATE
        ${LIBOMP_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(bench_compare PRIVATE
        ${LIBOMP_LIBRARY}
        "-framework Accelerate"
        # OpenBLAS loaded at runtime via dlopen — no link-time dependency needed
)

# =============================================================================
# bench_profile — single-library, single-size driver for scripts/profile.sh /
# scripts/profile_power.sh runs against Accelerate or OpenBLAS.
# =============================================================================
add_executable(bench_profile
        bench/bench_profile.cpp
)

target_compile_options(bench_profile PRIVATE
        -O3
        -mcpu=apple-m4
        -march=armv8.4-a
        -fopenmp
)

target_include_directories(bench_profile PRIVATE
        ${LIBOMP_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(bench_profile PRIVATE
        ${LIBOMP_LIBRARY}
        "-framework Accelerate"
)

# =============================================================================
# matmul — head-to-head N=2048 single-thread benchmark: our SME 1x4SymZAInOut
# kernel vs Apple Accelerate. Shares A/B buffers and best-of-7 timing. Built
# and run via scripts/run_matmul.sh.
# =============================================================================
add_executable(matmul
        experiments/matmul.cpp
        sme/sme-1x4-sym-zainout.cpp
)

target_compile_options(matmul PRIVATE
        -O3 -Wall -Wextra
        -mcpu=apple-m4
        -march=armv8.7-a+sme+sme2
        -fopenmp
)

target_include_directories(matmul PRIVATE
        ${LIBOMP_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(matmul PRIVATE
        ${LIBOMP_LIBRARY}
        "-framework Accelerate"
)
