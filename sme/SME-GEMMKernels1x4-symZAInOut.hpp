#pragma once

#include <cstddef>

// =============================================================================
// SME 1x4-symZAInOut: ZA in_out experiment branch of 1x4-sym (TODO §5).
//
// Splits the original micro_kernel_1x4 into three streaming SME functions
// that share ZA via the ABI attributes:
//
//   kernel_zero    __arm_out("za")    : svzero_za once per output tile
//   kernel_compute __arm_inout("za")  : pure FMOPA loop, ZA persists across calls
//   kernel_store   __arm_in("za")     : ZA -> C writeback once per tile
//
// run_multiplication keeps __arm_new("za") so it owns the ZA lifetime; the
// three callees participate in that scope. Driver loop is reordered so that
// for each (m, n) strip we pack the FULL K of A and B once, then for each
// output tile (jr, ir) we pay the zero/store overhead only once and call
// kernel_compute many times across K_inner_tile chunks.
// =============================================================================

namespace SMEKernels1x4SymZAInOut {

    // pack_A uses ZA tile 0 as transpose scratch. __arm_out("za") declares
    // that it clobbers the entire ZA scope; safe because the driver only
    // calls it when no live accumulator state exists in ZA.
    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_out("za") __arm_streaming;

    void pack_B_streaming(const float* B, float* packed_B,
                          size_t N_curr, size_t K_curr,
                          size_t curr_row, size_t N, size_t curr_col) __arm_streaming;

    void kernel_zero() __arm_out("za") __arm_streaming;

    void kernel_compute(const float* packed_A, const float* packed_B,
                        size_t K_curr) __arm_inout("za") __arm_streaming;

    void kernel_store(float* C, size_t wide_of_C) __arm_in("za") __arm_streaming;

    __arm_locally_streaming __arm_new("za")
    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N);

} // namespace SMEKernels1x4SymZAInOut
