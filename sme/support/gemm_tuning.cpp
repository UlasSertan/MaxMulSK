#include "gemm_tuning.hpp"

#include <cstdlib>
#include <cstdio>

namespace MaxMulSK::tuning {

namespace {

struct Entry {
    size_t   min_side;   // applies to square shapes with M == K == N >= this
    Blocking blocking;
    const char* provenance;
};

// Ordered largest-first; the first match wins.
//
// Measured 2026-09-09 on Apple M4, single thread, 1x4-Acc-KcOut, median of
// three runs with 90 s cooldowns between them. The margin quoted is against
// the default on the same three runs.
//
// A caveat worth carrying: at 2048^3 the DEFAULT is what is unstable, not the
// entry. Its three runs were 1278 / 1297 / 1505 GFLOP/s while the entry gave
// 1492 / 1506 / 1542. So the entry is not reliably 16% faster; it reliably
// reaches what the default only sometimes reaches. Why the default is bimodal
// there is not established.
constexpr Entry kTable[] = {
    { 4096, { 2048, 512, 4096, true }, "2026-09-09, +6.0% median of 3 (1541 -> 1634)" },
    { 2048, { 1024, 512, 2048, true }, "2026-09-09, +16.1% median of 3 (1297 -> 1506)" },
};

// >= 4096 uses the 4096 entry, which is an EXTRAPOLATION: nothing above 4096^3
// was measured. If that band starts to matter, measure it rather than trusting
// this row.

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

Blocking select_from_table(size_t M, size_t K, size_t N) {
    Blocking b{};                       // the default
    if (M != K || K != N) return b;     // square shapes only
    for (const Entry& e : kTable)
        if (M >= e.min_side) return e.blocking;
    return b;
}

Blocking select(size_t M, size_t K, size_t N) {
    Blocking b = select_from_table(M, K, N);
    parse_env(b);                       // override wins if present and well-formed
    return b;
}

}  // namespace MaxMulSK::tuning
