//
// Created on 24.01.2026.
//

#include "GEMMKernels.hpp"

// 1. Intel/AMD (x86_64) - AVX2
#if defined(__AVX2__)
    #include <immintrin.h>
    #define USE_AVX2 1

// 2. Apple Silicon / ARM64 - NEON
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
    #define USE_NEON 1

// 3. Last resort (Fallback)
#else
    #define USE_SCALAR 1
#endif


namespace GEMM {

#ifdef  USE_AVX2

    // Kernel size = 4x3
    static void multiply_kernel_avx2(const float* A, const float* B, float* C,
        size_t M, size_t N, size_t K) {}

#endif

#ifdef  USE_NEON

    // Kernel size = 8x3
    static void multiply_kernel_neon(const float* A, const float* B, float* C,
        size_t M, size_t N, size_t K) {}

#endif

#if defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define RESTRICT __restrict
#else
    #define RESTRICT

    static void multiply_kernel_scalar(const float* RESTRICT A, const float* RESTRICT B, float* RESTRICT C,
        size_t M, size_t N, size_t K) {}

#endif

    // Dispatcher
    void multiply(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {


#if defined(USE_AVX2)
        // Intel
        multiply_kernel_avx2(A, B, C, M, N, K);

#elif defined(USE_NEON)
        // Apple Silicon
        multiply_kernel_neon(A, B, C, M, N, K);

#else
        // Neither
        multiply_kernel_scalar(A, B, C, M, N, K);
#endif

    }
} // namespace GEMM