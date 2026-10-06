#pragma once

#include "sme-1x4-kcout-ncblock-apack4za.hpp"

#include <cstddef>

// =============================================================================
// MILESTONE (2026-09-27): development continues in sme/v5/sme-1x4-fusedpack;
// this file stays as measured in Lab EXP-FUSEPACK, with its experiment switches.
//
// EXPERIMENT -- 1x4-kcout-ncblock-fusepack: a copy of the Nc-blocked v4 kernel
// (sme-1x4-kcout-ncblock-apack4za, left untouched) with the packing hoisted out
// of the compute loops and the first A block of every Nc block packed together
// with that Nc block's B in one fused routine.
//
//   for kk:
//     for nn:
//       fuse_pack(B block nn, A block mm = 0)   -- B rows fill the A chunk slots
//       compute(mm = 0)
//       for mm = Mc, 2*Mc, ...:
//         pack_A(A block mm)                     -- same macro as the original
//         compute(mm)
//
// Micro-kernel, packed layouts, ZA->C writeback and the order in which C tiles
// are computed are unchanged, so C is bit-identical to the original kernel run
// with the same Blocking.
//
// MAXMULSK_FUSEPACK_FILL (default 0) also places B rows in the A chunk's FILL
// phase. Homebrew clang 22.1.8 miscompiles that form (ZA slice base emitted as
// an immediate); build with FILL = 1 only on a compiler whose output has been
// checked. The default DRAIN-only form compiles correctly on 22.1.8.
//
// MAXMULSK_FUSEPACK_NS lets the Lab link several builds of this TU into one
// binary under different namespaces.
// =============================================================================

#ifndef MAXMULSK_FUSEPACK_NS
#define MAXMULSK_FUSEPACK_NS SMEKernels1x4KcOutNcBlockFusePack
#endif

namespace MAXMULSK_FUSEPACK_NS {

    using Blocking = SMEKernels1x4KcOutNcBlockApack4Za::Blocking;
    using Support  = SMEKernels1x4KcOutNcBlockApack4Za::Support;

    // Whether this build places B rows in the FILL slots too (1) or only in the
    // DRAIN slots (0). Exposed so a benchmark can record what it linked.
    int fill_slots_enabled();

    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N, const Blocking& b);

    // Caller-owned workspace variant: no allocation inside the call. The buffer
    // must be 64-byte aligned and at least workspace_bytes(M, K, N, b) long; its
    // contents need no initialisation and may be reused across calls.
    size_t workspace_bytes(size_t M, size_t K, size_t N, const Blocking& b);
    Support run_multiplication_ws(const float* A, const float* B, float* C,
                                  size_t M, size_t K, size_t N, const Blocking& b,
                                  void* ws, size_t ws_bytes);

} // namespace MAXMULSK_FUSEPACK_NS
