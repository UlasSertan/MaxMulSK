#pragma once

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <cstdint>

// =============================================================================
// ARCH-001 PanelFuse, experiment 2: new macroflow, existing packers and compute.
// See the .cpp header. FULL TILES ONLY: M%16, N%64, K%64 must be zero.
//
// SEMANTICS: C = A*B (overwrite). The caller need not pre-zero C.
// =============================================================================

namespace SMEKernels1x4PanelFuse {

    // What to do when neither all of A nor all of B fits the budget.
    enum class BothLarge {
        RetainB,   // keep a B rectangle, traverse M, re-prepare A per B rectangle
        RetainA,   // keep an A rectangle, traverse N, re-prepare B per A rectangle
    };

    struct Params {
        size_t    Kc           = 2048;
        size_t    budget_bytes = 7864320;        // 7.5 MiB, the ledger's upper bound
        BothLarge both_large   = BothLarge::RetainB;
    };

    enum class Support {
        Native, UnsupportedMTail, UnsupportedNTail, UnsupportedKTail,
        UnsupportedVectorLength, BadParams, AllocationFailed, Unsupported,
    };
    enum class Retained { A, B };

    struct Plan {
        Retained retained;
        bool     forced_by_policy;     // neither operand fit; both_large decided
        size_t   t_budget, t_outer, t_inner;
        size_t   packed_A_bytes, packed_B_bytes;
    };

    // Counted during the run. pack/compute cycles are only filled in when
    // timed is set (via set_phase_timing), and that costs two counter reads per
    // phase -- do not use a timed run as the headline number.
    struct Stats {
        bool     timed = false;
        size_t   a_panel_packs = 0, b_panel_packs = 0;
        size_t   phases = 0, microkernel_calls = 0;
        uint64_t pack_cycles = 0, compute_cycles = 0, counter_hz = 0;
    };

    extern bool st_timed_default;
    void set_phase_timing(bool on);

    Support classify(size_t M, size_t K, size_t N, const Params& p);
    Plan    plan(size_t M, size_t K, size_t N, const Params& p);
    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N,
                               const Params& p, Stats* st = nullptr);

} // namespace SMEKernels1x4PanelFuse
