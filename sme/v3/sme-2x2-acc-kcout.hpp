#pragma once

#include <cstddef>

#include "../support/gemm_tuning.hpp"

// =============================================================================
// SME 2x2-Acc-KcOut: the 1x4-Acc treatment applied to the 2x2 geometry.
//
//   - ZA lifetime lifted out of the micro-kernel; K is innermost, so ZA holds
//     one C tile's partial sum across a whole K panel and drains once.
//   - The ZA->C store drops its read-modify-write and OVERWRITES.
//   - The store is row-major: svst1_f32_x2 per row, 64 narrow stores -> 32.
//     (1x4 reached 16 because its tile is 4*SVL wide; a 2x2 grid is only 2.)
//   - pack_A already used the ZA transpose in the shipped 2x2, so that step of
//     the chain needed no work here.
//   - An OUTER Kc loop above the tile nest keeps the packed panels Kc deep
//     rather than K deep. Each C tile is visited K/Kc times: first panel
//     overwrites, the rest accumulate.
//
// SEMANTICS: C = A*B (overwrite), NOT C += A*B. The caller must NOT pre-zero C.
// The shipped SMEKernels2x2 accumulates; this one does not.
// =============================================================================

namespace SMEKernels2x2AccKcOut {

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

    // Accumulates into ZA only; the driver owns zeroing and writeback.
    void micro_kernel_2x2(float* packed_A, float* packed_B,
                          size_t K_curr) __arm_inout("za") __arm_streaming;

    // =========================================================================
    // MAIN DRIVER
    // =========================================================================

    // Streaming implementation with the blocking passed in, for sweeps.
    __arm_locally_streaming __arm_new("za")
    void run_multiplication_blocked(const float* A, const float* B, float* C,
                                    size_t M, size_t K, size_t N,
                                    MaxMulSK::tuning::Blocking b);

    // Public entry point. NOT streaming: it chooses the blocking in normal mode
    // and then enters streaming once.
    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N);

} // namespace SMEKernels2x2AccKcOut
