#pragma once
// ONE-STREAM copy of sme-1x4-acc-kcout (2026-09-22): same kernel, no streaming-mode
// exits inside a call (buffers kept across calls), plus a batch entry.

#include <cstddef>

#include "../support/gemm_tuning.hpp"

// =============================================================================
// SME 1x4-Acc-KcOut: 1x4-Acc with an OUTER Kc loop above the M loop.
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

namespace SMEKernels1x4AccKcOutOneStream {

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
    void micro_kernel_1x4(float* packed_A, float* packed_B,
                          size_t K_curr) __arm_inout("za") __arm_streaming;


    // Public entry point. Deliberately NOT streaming: choosing the blocking is a
    // table walk plus possibly an environment read, and neither belongs inside a
    // streaming region. It selects in normal mode, then enters streaming once.
    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N);

    // Several GEMMs in ONE streaming region: one smstart/smstop for all of them.
    struct Gemm { const float* A; const float* B; float* C; size_t M, K, N; };
    void run_batch(const Gemm* gemms, size_t n);

} // namespace SMEKernels1x4AccKcOutOneStream
