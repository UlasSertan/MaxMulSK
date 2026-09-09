#pragma once

#include <cstddef>

namespace SMEKernels1x4 {

    // =========================================================================
    // PACKING (Streaming Mode) — B-inner-packing variant
    // pack_A: single SVL-wide panel per m-step (GS = SVL)
    // pack_B: 4-panel interleaved per k-step (GS = 4*SVL)
    // =========================================================================

    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming;

    void pack_B_streaming(const float* B, float* packed_B,
                          size_t N_curr, size_t K_curr,
                          size_t curr_row, size_t N, size_t curr_col) __arm_streaming;

    // =========================================================================
    // MICRO KERNEL — 1x4 tile layout (SVL rows × 4*SVL cols)
    // ZA_i = A ⊗ B_i  (same A panel, 4 different B panels)
    // =========================================================================

    void micro_kernel_1x4(float* packed_A, float* packed_B, float* C,
                          size_t K_curr, size_t wide_of_C) __arm_out("za") __arm_streaming;

    // =========================================================================
    // MAIN DRIVER — B-inner-packing + 1x4 kernel
    // Loop order: M → K → pack_A → N → pack_B → (jr, ir) → micro_kernel_1x4
    // =========================================================================

    __arm_locally_streaming __arm_new("za")
    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N);

} // namespace SMEKernels1x4
