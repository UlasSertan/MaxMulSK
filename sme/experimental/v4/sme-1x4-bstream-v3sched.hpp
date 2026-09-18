#pragma once

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>

// =============================================================================
// EXPERIMENTAL, INCOMPLETE. Do not use outside the v4 experiments.
//
// SME 1x4 with B read straight from memory at stride N, using the schedule
// shape of the hand-written v3 kernels: per k-step ZA0, ZA1, [one x4 load],
// ZA2, ZA3, plus a fifth load at end-of-iteration.
//
// SEMANTICS: C = A*B (overwrite). The caller must NOT pre-zero C.
// CONSEQUENCE: correct only when M is a multiple of SVL and N a multiple of
// 4*SVL. There is no edge-tile path yet.
// =============================================================================

namespace SMEKernels1x4BStreamV3sched {

    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming __arm_out("za");

    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N);

} // namespace SMEKernels1x4BStreamV3sched
