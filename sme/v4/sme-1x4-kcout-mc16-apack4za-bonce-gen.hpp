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
// NO TAIL PATH. Native only when M%16==0, N%64==0 and K%64==0, with 16 fp32
// streaming lanes. Anything else returns an Unsupported reason and leaves C
// untouched. Kc is min(2048, K). packed_B holds every N panel of one Kc
// slice, so it grows linearly with N -- call capacity() before running a
// wide-N shape.
// =============================================================================

namespace SMEKernels1x4KcOutMc16Apack4ZaBonceGen {

    // Why a call did or did not run natively. Never silently substitutes another
    // kernel: a caller that wants a fallback must pick one itself and label the
    // result as such.
    enum class Support {
        Native,                    // ran here
        UnsupportedMTail,          // M % 16 != 0
        UnsupportedNTail,          // N % 64 != 0
        UnsupportedKTail,          // K % 64 != 0
        UnsupportedVectorLength,   // streaming fp32 lanes != 16
        AllocationFailed,
        Unsupported,
    };

    Support classify(size_t M, size_t K, size_t N);

    // Packed A and B bytes this kernel would allocate for the shape.
    void capacity(size_t K, size_t N, size_t* packed_A_bytes, size_t* packed_B_bytes);

    // TEST ONLY: fills packed_A (16*2048 floats) for one (kk, m) panel using the
    // driver's own packing macro, so the layout can be compared elementwise.
    bool probe_pack_A(const float* A, size_t M, size_t K,
                      size_t kk, size_t m, float* packed_A);

    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N);

} // namespace SMEKernels1x4KcOutMc16Apack4ZaBonceGen
