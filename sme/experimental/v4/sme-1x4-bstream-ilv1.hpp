#pragma once

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>

// =============================================================================
// EXPERIMENTAL, INCOMPLETE. Do not use outside the v4 experiments.
//
// SME 1x4, B streamed, one memory op per svmopa (svld1_f32)
// Here: never two in a row -- an svld1_f32 sits after every one.
//
// SEMANTICS: C = A*B (overwrite), NOT C += A*B. The caller must NOT pre-zero C.
//
// CONSEQUENCE: correct only when M is a multiple of SVL and N is a multiple of
// 4*SVL. There is no edge-tile path yet.
// =============================================================================

namespace SMEKernels1x4BStreamIlv1 {

    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming __arm_out("za");

    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N);

} // namespace SMEKernels1x4BStreamIlv1
