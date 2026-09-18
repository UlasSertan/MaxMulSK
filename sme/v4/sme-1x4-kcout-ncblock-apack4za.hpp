#pragma once

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
// NO TAIL PATH: native only when M%16==0, N%64==0, K%64==0, with 16 fp32
// streaming lanes, and with Mc/Nc/Kc multiples of 16/64/64. A short FINAL
// macroblock is supported. Anything else returns a reason and leaves C
// untouched; this kernel never substitutes another one.
// =============================================================================

namespace SMEKernels1x4KcOutNcBlockApack4Za {

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

    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N, const Blocking& b);

} // namespace SMEKernels1x4KcOutNcBlockApack4Za
