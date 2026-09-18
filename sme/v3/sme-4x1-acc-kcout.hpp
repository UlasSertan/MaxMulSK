#pragma once

#include <cstddef>

#include "../support/gemm_tuning.hpp"

// =============================================================================
// SME 4x1-Acc-KcOut: the 1x4-Acc treatment applied to the 4x1 geometry,
// completing the migration across all three micro-kernel shapes.
//
//   - ZA lifetime lifted out of the micro-kernel; K innermost, one drain per
//     C tile per K panel.
//   - The ZA->C store drops its read-modify-write and OVERWRITES.
//   - An OUTER Kc loop keeps the packed panels Kc deep rather than K deep.
//
// What does NOT transfer: the row-major store rework. 4x1's four ZA tiles stack
// VERTICALLY and its output tile is only 1*SVL wide, so every row is a single
// vector and there is no wider store to reach for. 1x4 collapsed 64 narrow
// stores to 16 and 2x2 to 32; here it stays 64. Expect a correspondingly
// smaller win.
//
// SEMANTICS: C = A*B (overwrite), NOT C += A*B. Do NOT pre-zero C. The shipped
// SMEKernels4x1 accumulates; this one does not.
// =============================================================================

namespace SMEKernels4x1AccKcOut {

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
    // MICRO KERNEL
    // =========================================================================

    // Accumulates into ZA only; the driver owns zeroing and writeback.
    void micro_kernel_4x1(float* packed_A, float* packed_B,
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

} // namespace SMEKernels4x1AccKcOut
