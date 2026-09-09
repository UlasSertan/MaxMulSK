#pragma once

#include <cstddef>

// =============================================================================
// SME 1x4-Acc-KcOut-BDirect: acc-kcout with pack_B REMOVED.
//
// pack_B was a pure copy -- it read 4*SVL consecutive floats of B's row k and
// wrote them to 4*SVL consecutive floats of the panel, no reordering. So the
// micro-kernel can read B in place and step by N between k-steps instead of by
// 4*SVL. That drops a full read+write+read pass over B. Only the ragged N tail
// still needs packing, because it needs zero padding.
//
// Expected to favour SMALL N (the LLM shapes, where the k-step stride is a few
// KB) and to risk hurting LARGE N, where the stride grows and the prefetcher
// has less to work with.
//
// Below: 1x4-Acc with an OUTER Kc loop above the M loop.
//
// The whole M x N sweep runs once per Kc slice of K, so the packed buffers
// scale with Kc rather than K and stay inside L2 on large-K shapes. Each C
// tile is therefore visited K/Kc times: the first panel overwrites, the rest
// accumulate. The C = A*B contract below is unchanged -- do NOT pre-zero C.
//
// Same packing layout and same FMOPA schedule as 1x4-sym. The only difference
// is where the ZA lifetime boundaries sit: the micro-kernel no longer zeroes ZA
// on entry nor writes ZA back to C on exit. It only accumulates into whatever
// ZA already holds. The driver owns the ZA lifetime: loop order is M -> N -> K
// with K innermost, so for each C tile fixed by (M, N) it zeroes ZA once,
// accumulates over every KC chunk, and stores once after the K loop.
// A and B are packed over the full K (A first), so the packing buffers scale
// with K rather than with K_tile.
//
// Goal: keep ZA state live for as long as possible.
//
// SEMANTICS: C = A*B (overwrite), NOT C += A*B. Because K is innermost, each C
// tile is written exactly once with the finished sum, so the ZA->C writeback
// stores directly instead of doing a read-modify-write. The caller must NOT
// pre-zero C, and must not rely on C's prior contents. Every other MaxMulSK SME
// kernel accumulates; this one does not.
// =============================================================================

namespace SMEKernels1x4AccKcOutBDirect {

    // Uses ZA tile 0 as a transpose buffer (horizontal write, vertical read),
    // hence __arm_out("za"): it clobbers ZA. The driver only calls this while no
    // accumulation is live.
    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming __arm_out("za");

    void pack_B_streaming(const float* B, float* packed_B,
                          size_t N_curr, size_t K_curr,
                          size_t curr_row, size_t N, size_t curr_col) __arm_streaming;

    // Accumulates into ZA only. Does not zero ZA and does not touch C; both are
    // the driver's responsibility now, hence __arm_inout("za") rather than
    // __arm_out("za") and the dropped C / wide_of_C parameters.
    // B_stride: floats between consecutive k-steps of B. 4*SVL for a packed
    // panel, N for B read in its native row-major layout.
    void micro_kernel_1x4(float* packed_A, const float* packed_B,
                          size_t K_curr, size_t B_stride) __arm_inout("za") __arm_streaming;

    __arm_locally_streaming __arm_new("za")
    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N);

} // namespace SMEKernels1x4AccKcOutBDirect
