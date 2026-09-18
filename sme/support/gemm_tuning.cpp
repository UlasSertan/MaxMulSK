#include "gemm_tuning.hpp"

#include <cstdlib>
#include <cstdio>

namespace MaxMulSK::tuning {

namespace {

struct Entry {
    size_t   min_side;   // applies to square shapes with M == K == N >= this
    Blocking blocking;
};

// Shipped defaults, one per geometry. These are the tile sizes each kernel was
// written with; the tables below only override them where a measurement earned
// it.
constexpr Blocking kDefault1x4{1024,   64, 2048, false};
constexpr Blocking kDefault2x2{  64,  512, 2048, false};
constexpr Blocking kDefault4x1{  64, 1024, 2048, false};

// -----------------------------------------------------------------------
// 1x4, measured 2026-09-09. Median of three runs with 90 s cooldowns.
//
// A caveat worth carrying: at 2048^3 the DEFAULT is what is unstable, not the
// entry. Its three runs were 1278 / 1297 / 1505 GFLOP/s while the entry gave
// 1492 / 1506 / 1542. So the entry is not reliably 16% faster; it reliably
// reaches what the default only sometimes reaches. Why the default is bimodal
// there is not established.
//
// Nothing above 4096^3 was measured, so the 4096 row is an extrapolation for
// larger squares. If that band starts to matter, measure it.
// -----------------------------------------------------------------------
constexpr Entry kTable1x4[] = {
    { 4096, { 2048, 512, 4096, true } },   // +6.0% median of 3 (1541 -> 1634)
    { 2048, { 1024, 512, 2048, true } },   // +16.1% median of 3 (1297 -> 1506)
};

// -----------------------------------------------------------------------
// 2x2, measured 2026-09-09. Swept 28 to 264 configurations per shape, then the
// shortlist re-measured over three runs with cooldowns.
//
// One axis carried the whole result: N_tile 512 -> 1024. The sweep's own picks
// added M_tile and the jr/ir swap on top, but re-running showed those worth at
// most ~1%, inside the noise, and the plain N_tile change was the more stable
// of the two at 2048^3 (run spread 2.6% against 5.5%). So the entry changes one
// parameter, which is also the one that can be explained.
//
// Below 2048^3 the default already wins or ties: at 1024^3 the best
// configuration out of 132 was +0.6%.
// -----------------------------------------------------------------------
constexpr Entry kTable2x2[] = {
    { 2048, { 64, 1024, 2048, false } },   // +10.6% at 2048^3, +13.9% at 4096^3
};

// 4x1: swept the same way on 2026-09-09 and left empty on purpose. Its shipped
// 64/1024/2048 came first out of 21 and 36 configurations at 256^3 and 512^3,
// and never lost by more than 0.3% anywhere else. There is nothing to override.
//
// That sweep did find something else: pack_A addressed only one 4-panel group,
// so every M_tile above 64 returned silently wrong results. Fixed in
// sme/v3/sme-4x1-acc-kcout.cpp; the same latent bug still sits in the v1 kernel,
// where M_tile is fixed at 64 and never triggers it.
constexpr Entry kTable4x1[] = {};

bool parse_env(Blocking& out) {
    const char* s = std::getenv("MAXMULSK_BLOCKING");
    if (!s || !*s) return false;
    unsigned long long mt = 0, nt = 0, kc = 0, iro = 0;
    if (std::sscanf(s, "%llu,%llu,%llu,%llu", &mt, &nt, &kc, &iro) != 4) return false;
    if (!mt || !nt || !kc) return false;
    out.M_tile   = static_cast<size_t>(mt);
    out.N_tile   = static_cast<size_t>(nt);
    out.Kc       = static_cast<size_t>(kc);
    out.ir_outer = iro != 0;
    return true;
}

}  // namespace

Blocking select_from_table(Kernel k, size_t M, size_t K, size_t N) {
    const Entry* table = nullptr;
    size_t count = 0;
    Blocking b{};
    switch (k) {
        case Kernel::Sme1x4KcOut: b = kDefault1x4; table = kTable1x4; count = 2; break;
        case Kernel::Sme2x2KcOut: b = kDefault2x2; table = kTable2x2; count = 1; break;
        case Kernel::Sme4x1KcOut: b = kDefault4x1; table = kTable4x1; count = 0; break;
    }
    if (M != K || K != N) return b;     // square shapes only
    for (size_t i = 0; i < count; i++)
        if (M >= table[i].min_side) return table[i].blocking;
    return b;
}

Blocking select(Kernel k, size_t M, size_t K, size_t N) {
    Blocking b = select_from_table(k, M, K, N);
    parse_env(b);                       // override wins if present and well-formed
    return b;
}

}  // namespace MaxMulSK::tuning
