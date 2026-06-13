#pragma once

#include <cstddef>

// =============================================================================
// SME 1x4-sym: B-inner-packing, x4-grouped B loads — true mirror of 4x1.
// Schedule per 4 k-steps: 1 A x4 load + 4 B x4 loads + 16 svmopa.
// (Current SMEKernels1x4 is the x1-interleaved variant; this is the retired
// ~1148 GFLOPS x4-grouped baseline brought back for A-inner vs B-inner study.)
// =============================================================================

namespace SMEKernels1x4Sym {

    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming;

    void pack_B_streaming(const float* B, float* packed_B,
                          size_t N_curr, size_t K_curr,
                          size_t curr_row, size_t N, size_t curr_col) __arm_streaming;

    void micro_kernel_1x4(float* packed_A, float* packed_B, float* C,
                          size_t K_curr, size_t wide_of_C) __arm_out("za") __arm_streaming;

    __arm_locally_streaming __arm_new("za")
    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N);

} // namespace SMEKernels1x4Sym
