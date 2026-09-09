#pragma once

#include <cstddef>

// =============================================================================
// Shape-dependent cache-blocking parameters.
//
// The kernels ship with one set of tile sizes that is good everywhere and best
// nowhere in particular. This table overrides them for the shapes where a
// measured configuration reliably beat the default.
//
// Rules this table follows, so it stays honest:
//
//   1. An entry exists ONLY where a configuration beat the default across
//      THREE separate runs, not one. Every single-run "win" this project has
//      seen so far (+16.5% at 2048^3, +8.3% at 256^3, +13.8% for the ZA pack)
//      evaporated on repetition.
//   2. Square shapes only (M == K == N). Non-square and LLM-style shapes fall
//      through to the default; the sweep found nothing for them, and
//      N_tile = 512 actively costs 0.6-7.6% on the large-K shapes.
//   3. Each entry names the date and the measured margin, so a later run that
//      disagrees can be compared against what was actually claimed.
//
// M_step and N_step are NOT here. They are the micro-kernel's output tile
// geometry, not a blocking choice.
// =============================================================================

namespace MaxMulSK::tuning {

struct Blocking {
    size_t M_tile   = 1024;
    size_t N_tile   = 64;     // must be a multiple of N_step = 4*SVL
    size_t Kc       = 2048;   // outer K panel
    bool   ir_outer = false;  // swap the jr/ir nest; only meaningful when
                              // N_tile > N_step
};

// Picks the blocking for one GEMM. Cheap, pure, and callable from normal mode:
// the kernels call it BEFORE entering streaming mode so neither the table walk
// nor the environment read happens inside a streaming region.
//
// MAXMULSK_BLOCKING="M_tile,N_tile,Kc,ir_outer" overrides everything, so a
// sweep can be driven without recompiling. Malformed values are ignored.
Blocking select(size_t M, size_t K, size_t N);

// What select() would return with the environment override ignored. Exposed so
// a benchmark can report the table's own choice separately from an override.
Blocking select_from_table(size_t M, size_t K, size_t N);

}  // namespace MaxMulSK::tuning
