#pragma once
// ONE-STREAM copy of sme-1x4-kcout-ncblock-apack4za + the v4c rule (2026-09-22):
// buffers kept across calls, streaming body, one smstart/smstop per call or per batch.

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>

// =============================================================================
// EXPERIMENT NB -- 1x4-kcout-ncblock-apack4za.
//
// F2's four-ZA inline A packing and compute body, with an Nc block added so the
// packed working set is (Nc + Mc) * Kc floats instead of growing with N.
// Loop order Kc -> Nc -> Mc -> n(64) -> m(16) -> compute(Kc).
//
// SEMANTICS: C = A*B (overwrite). The caller need not pre-zero C.
//
// WITH TAIL SUPPORT. M, N and K may be anything; short final microtiles go
// through a scratch path. Still requires 16 fp32 streaming lanes and Mc/Nc/Kc
// that are multiples of 16/64/64, since the panel slot arithmetic depends on it.
//
// The full-tile path is deliberately byte-identical to the tail-free kernel this
// replaced (measured 2026-09-13, TU hash 546284d4..., archived under
// bench/results/2026-09-13/source):
// same packing macro, same store_za, no extra branch inside the micro-kernel
// loop. Edge work happens in a 16x64 staging block and a 16x64 C scratch, both
// allocated once outside every loop.
// =============================================================================

namespace SMEKernels1x4V4COneStream {

    struct Blocking { size_t Mc = 16, Nc = 256, Kc = 2048; };

    enum class Support {
        Native, UnsupportedMTail, UnsupportedNTail, UnsupportedKTail,
        UnsupportedVectorLength, BadBlocking, AllocationFailed, Unsupported,
    };

    struct Counts {
        size_t kc_slices, nc_blocks, mc_blocks;
        size_t a_panel_packs, b_panel_packs, microkernel_calls;
    };

    Support classify(size_t M, size_t K, size_t N, const Blocking& b);
    void capacity(size_t M, size_t K, size_t N, const Blocking& b,
                  size_t* packed_A_bytes, size_t* packed_B_bytes);
    void counts(size_t M, size_t K, size_t N, const Blocking& b, Counts* out);

    // TEST ONLY: fills one 16-row A panel slot using the driver's own macro.
    bool probe_pack_A(const float* A, size_t M, size_t K,
                      size_t kk, size_t m, size_t kcl, float* pA);
    // TEST ONLY: packs one 64-column B panel (kcl x 64) exactly as the driver does.
    bool probe_pack_B(const float* B, size_t K, size_t N,
                      size_t kk, size_t n, size_t kcl, float* pB);

    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N, const Blocking& b);

    // v4c: blocking from the shape (same rule as sme-1x4-v4c-dispatch).
    Blocking choose(size_t M, size_t K, size_t N);
    Support run_multiplication(const float* A, const float* B, float* C, size_t M, size_t K, size_t N);
    // Several GEMMs in ONE streaming region (one smstart/smstop for all).
    struct Gemm { const float* A; const float* B; float* C; size_t M, K, N; };
    Support run_batch(const Gemm* gemms, size_t n);

} // namespace SMEKernels1x4V4COneStream
