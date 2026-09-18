#pragma once

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>

// =============================================================================
// EXPERIMENTAL, INCOMPLETE. Do not use outside the v4 experiments.
//
// SME 1x4, B streamed from memory, prefetch depth 3  (EXPERIMENT)
//
// DEPTH 3: three B quads are kept in flight, so each load is issued 12 svmopa
// before its use instead of immediately before it. Costs 4 more live Z
// registers (20 of 32) and adds no memory instructions at all -- the whole
// point, after the staging-buffer attempt lost on exactly that axis.
//
// SEMANTICS: C = A*B (overwrite), NOT C += A*B. Each C tile is written exactly
// once with the finished K sum, so the ZA->C writeback stores directly. The
// caller must NOT pre-zero C and must not rely on C's prior contents.
//
// CONSEQUENCE: correct only when M is a multiple of SVL and N is a multiple of
// 4*SVL. Anything else writes partial tiles at full width and is silently
// wrong. There is no edge-tile path yet.
// =============================================================================

namespace SMEKernels1x4BStreamReg3 {

    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming __arm_out("za");

    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N);

} // namespace SMEKernels1x4BStreamReg3
