#pragma once

#include <cstddef>

// =============================================================================
// 1x4-general -- one entry point that routes between two existing kernels.
//
//   M == K == N   ->  sme/v3/sme-1x4-acc-kcout   (v3, with its tuning table)
//   otherwise     ->  sme/v4 v4c                 (Nc-blocked, two-profile rule)
//
// The square case goes to v3 because that is where v3 still leads: in
// final5.csv (2026-09-13) the Nc-blocked kernel lost to v3 on 256^3 and 512^3
// and was level on the rest, while off the diagonal it was ahead.
//
// This is a routing layer and nothing else. It owns no packing, no blocking and
// no compute; it does not run both paths and it does not measure anything at
// runtime. It lives in its own translation unit so both kernels it calls stay
// byte-identical to the ones already measured.
//
// SEMANTICS: C = A*B (overwrite). The caller need not pre-zero C.
//
// SUPPORT: the square path is v3's, which handles any shape. The non-square
// path is v4c's, which is native only for M%16==0, N%64==0, K%64==0 with 16 fp32
// streaming lanes. A shape v4c refuses is reported as Unsupported and C is left
// untouched -- this layer does NOT silently fall back to v3 for it, because a
// fallback result is not the routed kernel's performance.
// =============================================================================

namespace SMEKernels1x4General {

    enum class Path { V3Square, V4C, Unsupported };

    // Pure function of the shape; exposed so a benchmark can record the route
    // without running anything.
    Path route(size_t M, size_t K, size_t N);

    Path run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N);

} // namespace SMEKernels1x4General
