#pragma once

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>

// =============================================================================
// EXPERIMENT F2 -- 1x4-kcout-mc16-apack4za-bonce.
//
// Mc=16 / Kc=2048 / Nc=64, loop order Kc -> Mc -> Nc, A packed inline across all
// four ZA tiles, B packed once per (Kc, N panel). Compute and C writeback are
// the v3 acc-kcout bodies, unchanged.
//
// SEMANTICS: C = A*B (overwrite). The caller need not pre-zero C, and C's prior
// contents do not enter the result.
//
// TARGET SHAPE ONLY: M=11008, K=4096, N=256, and 16 fp32 streaming lanes.
// run_multiplication_f2 returns false and leaves C untouched for anything else.
// There is no tail path and no dispatch.
// =============================================================================

namespace SMEKernels1x4KcOutMc16Apack4ZaBonce {

    // TEST ONLY: fills packed_A (16*2048 floats) for one (kk, m) panel using the
    // driver's own packing macro, so the layout can be compared elementwise.
    bool probe_pack_A(const float* A, size_t M, size_t K,
                      size_t kk, size_t m, float* packed_A);

    bool run_multiplication_f2(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N);

} // namespace SMEKernels1x4KcOutMc16Apack4ZaBonce
