#pragma once

#include "sme/v4/sme-1x4-kcout-ncblock-apack4za.hpp"

#include <cstddef>

// =============================================================================
// v5 (prefetch copy, v5.1 candidate; 2026-09-28) -- 1x4 fused-pack GEMM (from 2026-09-27; evidence: Lab EXP-FUSEPACK).
//
// Same 1x4 micro-kernel, packed layouts and ZA->C writeback as the Nc-blocked v4
// kernel, so C is bit-identical to v4 run with the same Blocking. What changes
// is where and how packing happens:
//
//   for kk:
//     for nn:
//       fuse_pack(B block nn, A block mm = 0)   -- B read row by row, its row
//       compute(mm = 0)                            copies placed in the A
//       for mm = Mc, 2*Mc, ...:                    transpose's DRAIN slots
//         pack_A(A block mm)
//         compute(mm)
//
// Packing is hoisted out of the compute loops; the mm = 0 round is peeled so no
// branch decides whether a block needs packing. One allocation per call, or none
// with run_multiplication_ws.
//
// Per tile: ZA is zeroed, the micro-kernel accumulates one K panel, and C is
// written with ST1W straight from ZA. On later K panels the old C is first added
// into ZA with the SME2 multi-vector FADD (no MOVA), the next tile's C rows being
// prefetched while this one computes. Summation order is v4's: C + (p0 + p1 + ...).
//
// Build-time parameters (all optional):
//   MAXMULSK_V5PF_FUSE  1 (default) fuse B into the A pack; 0 = hoist only, the
//                     A block and the B block packed back to back.
//   MAXMULSK_V5PF_PFA   prfm distance on A in chunks (default 1); 0 = off.
//   MAXMULSK_V5PF_PFB   prfm distance on B in rows (default 0 = off).
//   MAXMULSK_V5PF_CPA   1 = prefetch the next A pack's source during compute (default 0).
//   MAXMULSK_V5PF_CPB   1 = prefetch the next B pack's source during compute (default 0).
//   MAXMULSK_V5PF_FILL  0 (default) B rows only in DRAIN slots. 1 also uses the
//                     FILL slots; Homebrew clang 22.1.8 miscompiles that form
//                     (ZA slice base emitted as an immediate) -- only build it
//                     with a compiler whose output has been checked.
//   MAXMULSK_V5PF_NS    namespace override, so the Lab can link several builds.
// Toolchain: developed and measured with Homebrew LLVM 21.1.8.
// =============================================================================

#ifndef MAXMULSK_V5PF_NS
#define MAXMULSK_V5PF_NS SMEKernels1x4FusedPackPf
#endif

namespace MAXMULSK_V5PF_NS {

    using Blocking = SMEKernels1x4KcOutNcBlockApack4Za::Blocking;
    using Support  = SMEKernels1x4KcOutNcBlockApack4Za::Support;

    // Whether this build places B rows in the FILL slots too (1) or only in the
    // DRAIN slots (0). Exposed so a benchmark can record what it linked.
    int fill_slots_enabled();

    // Plan for a shape. Rule R3 (2026-09-27, Lab v5_grid): one fixed plan,
    // Mc 64 / Nc 1024 / Kc 1024, which the driver clamps to the shape (Mc to M,
    // Nc to N, Kc to K). On the v5 grid it averaged 0.978 of each shape's best
    // measured plan on headline35 and 0.994 on six held-out shapes. Pure;
    // exposed so a benchmark can record the plan it ran.
    Blocking choose(size_t M, size_t K, size_t N);

    // Plans with choose(); one allocation per call, freed on return.
    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N);

    // Explicit plan; one allocation per call, freed on return.
    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N, const Blocking& b);

    // Caller-owned workspace variant: no allocation inside the call. The buffer
    // must be 64-byte aligned and at least workspace_bytes(M, K, N, b) long; its
    // contents need no initialisation and may be reused across calls.
    size_t workspace_bytes(size_t M, size_t K, size_t N, const Blocking& b);
    Support run_multiplication_ws(const float* A, const float* B, float* C,
                                  size_t M, size_t K, size_t N, const Blocking& b,
                                  void* ws, size_t ws_bytes);

} // namespace MAXMULSK_V5PF_NS
