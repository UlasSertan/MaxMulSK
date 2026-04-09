#pragma once

#include <cstddef>

namespace SMEKernels {

    // =========================================================================
    // PACKING (Streaming Mode)
    // =========================================================================

    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming;

    void pack_B_streaming(const float* B, float* packed_B,
                          size_t N_curr, size_t K_curr,
                          size_t curr_row, size_t N, size_t curr_col) __arm_streaming;

    // =========================================================================
    // DISPATCH WRAPPERS
    // =========================================================================

    void pack_A_dispatch(const float* A, float* packed_A,
                         size_t M_curr, size_t K_curr,
                         size_t curr_row, size_t curr_col, size_t K);

    void pack_B_dispatch(const float* B, float* packed_B,
                         size_t N_curr, size_t K_curr,
                         size_t curr_row, size_t N, size_t curr_col);

    // =========================================================================
    // MICRO KERNEL
    // =========================================================================

    void micro_kernel_4x1(float* packed_A, float* packed_B, float* C,
                          size_t K_curr, size_t wide_of_C) __arm_out("za") __arm_streaming;

    // =========================================================================
    // MAIN DRIVER
    // =========================================================================

    __arm_new("za")
    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N) __arm_streaming;

} // namespace SMEKernels
