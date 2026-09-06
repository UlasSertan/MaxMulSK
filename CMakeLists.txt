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

# =============================================================================
# Arm KleidiAI (benchmark comparison only)
#
# KleidiAI ships fp32 SME2/NEON matmul micro-kernels that make a useful
# reference point for our own kernels. It is fetched and built from source at a
# pinned tag rather than vendored, so updating it is a one-line change here.
#
# Offline or don't want the dependency:
#   cmake -B build -DMAXMULSK_WITH_KLEIDIAI=OFF     # skip it; bench_compare
#                                                   # prints a notice and omits
#                                                   # the KleidiAI columns
# Already have a checkout:
#   cmake -B build -DFETCHCONTENT_SOURCE_DIR_KLEIDIAI=/path/to/kleidiai
# =============================================================================
option(MAXMULSK_WITH_KLEIDIAI "Fetch and build Arm KleidiAI for bench_compare" ON)

if(MAXMULSK_WITH_KLEIDIAI)
    include(FetchContent)

    # KleidiAI's own tests pull in googletest; we only need the library.
    set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

    set(KLEIDIAI_PINNED_TAG "v1.30.0")
    FetchContent_Declare(kleidiai
            GIT_REPOSITORY https://github.com/ARM-software/kleidiai.git
            GIT_TAG        ${KLEIDIAI_PINNED_TAG}
            GIT_SHALLOW    TRUE
            SYSTEM
            EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(kleidiai)
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
set_source_files_properties(sme/sme-4x1.cpp sme/sme-2x2.cpp sme/sme-1x4.cpp sme/sme-1x4-sym.cpp sme/sme-1x4-sym-zainout.cpp sme/sme-4x1-zapack.cpp sme/sme-1x4-acc.cpp sme/sme-1x4-acc-kc.cpp sme/test_sme.cpp PROPERTIES
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
        bench/kleidiai_gemm.cpp
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

# KleidiAI's PUBLIC include directories come with the link. Without it,
# bench/kleidiai_gemm.cpp compiles to a stub and the columns are skipped.
if(MAXMULSK_WITH_KLEIDIAI)
    target_link_libraries(bench_compare PRIVATE kleidiai)
    target_compile_definitions(bench_compare PRIVATE MAXMULSK_HAVE_KLEIDIAI)
endif()

# =============================================================================
# benchmark_maxmul_vs_kleidiai — the fair head-to-head: MaxMulSK's SME kernels
# vs KleidiAI's fp32 SME2 micro-kernels, single-thread, end-to-end and
# kernel-only. Isolated from the production kernels: it links them, it does not
# modify them.
#
#   cmake --build <dir> --target benchmark_maxmul_vs_kleidiai
#   ./<dir>/benchmark_maxmul_vs_kleidiai [out.csv]
# =============================================================================
if(MAXMULSK_WITH_KLEIDIAI)
    # Recorded in the benchmark's own environment banner.
    execute_process(COMMAND git rev-parse --short HEAD
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE MAXMULSK_GIT_REV
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT MAXMULSK_GIT_REV)
        set(MAXMULSK_GIT_REV "unknown")
    endif()
    set(BENCH_KAI_FLAGS "-O3 -Wall -Wextra -mcpu=apple-m4 -fopenmp")

    # llama.cpp / ggml CPU baseline. Built separately (CPU-only: Metal,
    # Accelerate, BLAS and ggml's KleidiAI integration all OFF) and pointed at
    # with -DLLAMA_CPP_DIR / -DLLAMA_CPP_BUILD_DIR. Without them the benchmark
    # still builds; the ggml columns are simply absent.
    set(LLAMA_CPP_DIR       "" CACHE PATH "llama.cpp source checkout")
    set(LLAMA_CPP_BUILD_DIR "" CACHE PATH "llama.cpp build directory (CPU-only)")

    add_executable(benchmark_maxmul_vs_kleidiai
            bench/benchmark_maxmul_vs_kleidiai.cpp
            bench/maxmulsk_sme_adapter.cpp
            bench/ggml_gemm_baseline.cpp
            bench/kleidiai_gemm.cpp
            sme/sme-4x1.cpp
            sme/sme-2x2.cpp
            sme/sme-1x4.cpp
            sme/sme-1x4-sym.cpp
            sme/sme-1x4-sym-zainout.cpp
            sme/sme-4x1-zapack.cpp
            sme/sme-1x4-acc.cpp
            sme/sme-1x4-acc-kc.cpp
    )

    target_compile_options(benchmark_maxmul_vs_kleidiai PRIVATE
            -O3 -Wall -Wextra
            -mcpu=apple-m4
            -fopenmp
    )

    # The adapter calls the kernels' __arm_streaming entry points, so it needs
    # the SME extensions too. sme/*.cpp already carry this flag from the
    # directory-scope property set further up; the driver TU does not need it.
    set_source_files_properties(bench/maxmulsk_sme_adapter.cpp PROPERTIES
            COMPILE_OPTIONS "-march=armv8.7-a+sme+sme2"
    )

    target_compile_definitions(benchmark_maxmul_vs_kleidiai PRIVATE
            MAXMULSK_HAVE_KLEIDIAI
            BENCH_KLEIDIAI_VERSION="${KLEIDIAI_PINNED_TAG}"
            BENCH_GIT_REV="${MAXMULSK_GIT_REV}"
            BENCH_CXX_FLAGS="${BENCH_KAI_FLAGS}"
    )

    target_include_directories(benchmark_maxmul_vs_kleidiai PRIVATE
            ${LIBOMP_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(benchmark_maxmul_vs_kleidiai PRIVATE
            ${LIBOMP_LIBRARY}
            kleidiai
            "-framework Accelerate"   # reference SGEMM + optional third baseline
    )

    if(LLAMA_CPP_DIR AND LLAMA_CPP_BUILD_DIR)
        execute_process(COMMAND git rev-parse --short HEAD
                        WORKING_DIRECTORY ${LLAMA_CPP_DIR}
                        OUTPUT_VARIABLE LLAMA_CPP_REV
                        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        target_include_directories(benchmark_maxmul_vs_kleidiai PRIVATE
                ${LLAMA_CPP_DIR}/ggml/include)
        target_link_libraries(benchmark_maxmul_vs_kleidiai PRIVATE
                ${LLAMA_CPP_BUILD_DIR}/ggml/src/libggml.a
                ${LLAMA_CPP_BUILD_DIR}/ggml/src/libggml-cpu.a
                ${LLAMA_CPP_BUILD_DIR}/ggml/src/libggml-base.a)
        target_compile_definitions(benchmark_maxmul_vs_kleidiai PRIVATE
                BENCH_LLAMA_REV="${LLAMA_CPP_REV}")

        # A llama.cpp build with GGML_CPU_KLEIDIAI=ON additionally exposes the
        # "CPU_KLEIDIAI" extra buffer type, which is the only route to ggml's
        # fp32 SME2 KleidiAI kernel. Point LLAMA_CPP_KLEIDIAI_LIB at that build's
        # libkleidiai.a to enable the llama-kleidiai-f32 row; without it the
        # benchmark still builds and simply omits that row.
        set(LLAMA_CPP_KLEIDIAI_LIB "" CACHE FILEPATH
            "libkleidiai.a from a GGML_CPU_KLEIDIAI=ON llama.cpp build")
        set(LLAMA_CPP_KLEIDIAI_VER "" CACHE STRING
            "KleidiAI version that llama.cpp fetched (ggml pins its own)")
        if(LLAMA_CPP_KLEIDIAI_LIB)
            target_link_libraries(benchmark_maxmul_vs_kleidiai PRIVATE
                    ${LLAMA_CPP_KLEIDIAI_LIB})
            target_compile_definitions(benchmark_maxmul_vs_kleidiai PRIVATE
                    BENCH_GGML_KLEIDIAI
                    BENCH_GGML_KAI_VER="${LLAMA_CPP_KLEIDIAI_VER}"
                    BENCH_GGML_FLAGS="GGML_METAL=OFF GGML_ACCELERATE=OFF GGML_BLAS=OFF GGML_CPU_KLEIDIAI=ON GGML_OPENMP=OFF GGML_NATIVE=ON")
        else()
            target_compile_definitions(benchmark_maxmul_vs_kleidiai PRIVATE
                    BENCH_GGML_FLAGS="GGML_METAL=OFF GGML_ACCELERATE=OFF GGML_BLAS=OFF GGML_CPU_KLEIDIAI=OFF GGML_OPENMP=OFF GGML_NATIVE=ON")
        endif()
    endif()
endif()

# =============================================================================
# energy_bench — one implementation, one shape, held busy for a fixed wall time
# so bench/energy_bench.sh can wrap it in powermetrics and derive J and J/GFLOP.
# Deliberately has no ggml dependency: it only needs our kernels, KleidiAI and
# Accelerate.
# =============================================================================
if(MAXMULSK_WITH_KLEIDIAI)
    add_executable(energy_bench
            bench/energy_bench.cpp
            bench/maxmulsk_sme_adapter.cpp
            bench/kleidiai_gemm.cpp
            sme/sme-4x1.cpp
            sme/sme-2x2.cpp
            sme/sme-1x4.cpp
            sme/sme-1x4-sym.cpp
            sme/sme-1x4-sym-zainout.cpp
            sme/sme-4x1-zapack.cpp
            sme/sme-1x4-acc.cpp
            sme/sme-1x4-acc-kc.cpp
    )
    target_compile_options(energy_bench PRIVATE
            -O3 -Wall -Wextra -mcpu=apple-m4 -fopenmp)
    target_compile_definitions(energy_bench PRIVATE MAXMULSK_HAVE_KLEIDIAI)
    target_include_directories(energy_bench PRIVATE
            ${LIBOMP_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(energy_bench PRIVATE
            ${LIBOMP_LIBRARY} kleidiai "-framework Accelerate")
endif()

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
