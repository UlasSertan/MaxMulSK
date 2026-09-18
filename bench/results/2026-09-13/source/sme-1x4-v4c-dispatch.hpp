#pragma once

#include "sme-1x4-kcout-ncblock-apack4za.hpp"
#include <cstddef>

// =============================================================================
// v4c -- shape-driven choice between two FIXED blocking profiles for the
// Nc-blocked v4 kernel. Nothing else differs from calling that kernel directly.
//
//   v4a = Mc16 / Nc512  / Kc2048
//   v4b = Mc32 / Nc1024 / Kc2048
//
// This lives in its own translation unit so the measured kernel TU stays
// byte-identical to the one in final5.csv.
//
// It is a RULE, not a lookup and not an autotuner: no shape-ID table, no exact
// dimension matching, no running both profiles and keeping the faster one.
// Every input runs exactly one path, chosen once before anything is allocated.
// =============================================================================

namespace SMEKernels1x4V4C {

    using Blocking = SMEKernels1x4KcOutNcBlockApack4Za::Blocking;
    using Support  = SMEKernels1x4KcOutNcBlockApack4Za::Support;

    // Pure function of the shape. Exposed so a benchmark can record which
    // profile was taken without running anything.
    Blocking choose(size_t M, size_t K, size_t N);

    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N);

} // namespace SMEKernels1x4V4C
