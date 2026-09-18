#pragma once

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>

// =============================================================================
// EXPERIMENTAL, INCOMPLETE. Do not use outside the v4 experiments.
//
// SME 1x4, B streamed from memory, prefetch depth 1  (CONTROL)
//
// DEPTH 1: each B quad is loaded immediately before the four svmopa that use
// it. No look-ahead. This is the baseline the deeper window is measured
// against; it is the "nobuffer" variant that measured 1461.6 GFLOP/s.
//
// SEMANTICS: C = A*B (overwrite), NOT C += A*B. Each C tile is written exactly
// once with the finished K sum, so the ZA->C writeback stores directly. The
// caller must NOT pre-zero C and must not rely on C's prior contents.
//
// CONSEQUENCE: correct only when M is a multiple of SVL and N is a multiple of
// 4*SVL. Anything else writes partial tiles at full width and is silently
// wrong. There is no edge-tile path yet.
// =============================================================================

namespace SMEKernels1x4BStreamReg1 {

    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming __arm_out("za");

    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N);

} // namespace SMEKernels1x4BStreamReg1
