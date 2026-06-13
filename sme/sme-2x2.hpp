#pragma once

#include <cstddef>

namespace SMEKernels2x2 {

    // =========================================================================
    // PACKING (Streaming Mode)
    // =========================================================================

    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming __arm_out("za");

    void pack_B_streaming(const float* B, float* packed_B,
                          size_t N_curr, size_t K_curr,
                          size_t curr_row, size_t N, size_t curr_col) __arm_streaming;

    // =========================================================================
    // MICRO KERNEL — 2x2 tile layout (2*SVL rows × 2*SVL cols)
    // =========================================================================

    void micro_kernel_2x2(float* packed_A, float* packed_B, float* C,
                          size_t K_curr, size_t wide_of_C) __arm_out("za") __arm_streaming;

    // =========================================================================
    // MAIN DRIVER
    // =========================================================================

    __arm_locally_streaming __arm_new("za")
    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N);

} // namespace SMEKernels2x2
