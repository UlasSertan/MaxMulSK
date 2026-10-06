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
        sme/v1/sme-4x1.cpp
        sme/v1/sme-2x2.cpp
        sme/v1/sme-1x4.cpp
        sme/v1/sme-1x4-sym.cpp
        sme/experimental/v1/sme-1x4-sym-zainout.cpp
        sme/experimental/v1/sme-4x1-zapack.cpp
        sme/support/test_sme.cpp
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

# SME: ARMv8.7 + SME/SME2 extensions, applied to every SME translation unit in
# the tree so whichever target compiles one gets the same flag.
set_source_files_properties(sme/experimental/v1/sme-1x4-sym-zainout.cpp sme/experimental/v1/sme-4x1-zapack.cpp sme/experimental/v2/sme-1x4-acc-fast.cpp sme/experimental/v3/sme-1x4-acc-fast-kcout.cpp sme/experimental/v3/sme-1x4-acc-kcout-bdirect.cpp sme/experimental/v4/sme-1x4-acc-kcout-small.cpp sme/experimental/v4/sme-1x4-bstream-ilv1.cpp sme/experimental/v4/sme-1x4-bstream-ilv2.cpp sme/experimental/v4/sme-1x4-bstream-ilv4.cpp sme/experimental/v4/sme-1x4-bstream-reg1.cpp sme/experimental/v4/sme-1x4-bstream-reg3.cpp sme/experimental/v4/sme-1x4-bstream-v3sched.cpp sme/support/gemm_tuning.cpp sme/support/test_sme.cpp sme/v1/sme-1x4-sym.cpp sme/v1/sme-1x4.cpp sme/v1/sme-2x2.cpp sme/v1/sme-4x1.cpp sme/v2/sme-1x4-acc-kc.cpp sme/v2/sme-1x4-acc.cpp sme/v3/sme-1x4-acc-kcout.cpp sme/v3/sme-2x2-acc-kcout.cpp sme/v3/sme-4x1-acc-kcout.cpp sme/v4/sme-1x4-general.cpp sme/v4/sme-1x4-kcout-mc16-apack4za-bonce-gen.cpp sme/v4/sme-1x4-kcout-mc16-apack4za-bonce-zapres.cpp sme/v4/sme-1x4-kcout-mc16-apack4za-bonce.cpp sme/v4/sme-1x4-kcout-ncblock-apack4za.cpp sme/v4/sme-1x4-panelfuse.cpp sme/v4/sme-1x4-v4c-dispatch.cpp sme/v4/sme-1x4-kcout-ncblock-fusepack.cpp sme/v5/sme-1x4-fusedpack.cpp sme/experimental/v5/sme-1x4-splitunit.cpp sme/experimental/v5/sme-1x4-splitunit-onestream.cpp PROPERTIES
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
# C API (capi/maxmulsk.h): v3 and v4c behind two extern "C" functions, for
# linking from other local codebases. Only capi/ is exposed to consumers.
# The SME flags come from the set_source_files_properties list above.
# =============================================================================

add_library(maxmulsk_capi STATIC
        capi/maxmulsk.cpp
        sme/v3/sme-1x4-acc-kcout.cpp
        sme/support/gemm_tuning.cpp
        sme/v4/sme-1x4-kcout-ncblock-apack4za.cpp
        sme/v4/sme-1x4-v4c-dispatch.cpp
)

target_compile_options(maxmulsk_capi PRIVATE
        -O3 -Wall -Wextra
        -mcpu=apple-m4
)

target_include_directories(maxmulsk_capi
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/capi
)

# =============================================================================
# MaxMulSK Trio (sme/trio): one SME compute thread fed by two NEON packer
# threads. trio::Engine is the interface (sme/trio/trio.hpp).
# =============================================================================

add_library(maxmulsk_trio STATIC
        sme/trio/trio_sme.cpp
        sme/trio/trio_packers.cpp
)

target_compile_options(maxmulsk_trio PRIVATE
        -O3 -Wall -Wextra
        -mcpu=apple-m4 -march=armv8.7-a+sme+sme2
)

target_include_directories(maxmulsk_trio
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/sme/trio
)

# =============================================================================

# =============================================================================
# Benchmarks, comparisons against other libraries, sweeps and profiling drivers
# live in the private MaxMulSK-Lab repo, which builds this tree by path.
# Nothing here depends on the Lab.
# =============================================================================
