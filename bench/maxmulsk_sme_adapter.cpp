// bench/maxmulsk_sme_adapter.cpp — see maxmulsk_sme_adapter.hpp.
//
// Compiled with -march=armv8.7-a+sme+sme2 (the same flags as sme/*.cpp),
// because the shipped pack_* and micro_kernel_* entry points are
// __arm_streaming and can only be called from streaming-mode code.

#include "maxmulsk_sme_adapter.hpp"

#include <algorithm>
#include <cstdlib>
#include <vector>

#include <arm_sme.h>

#include "../sme/sme-4x1.hpp"
#include "../sme/sme-2x2.hpp"
#include "../sme/sme-1x4.hpp"
#include "../sme/sme-1x4-sym.hpp"
#include "../sme/sme-1x4-sym-zainout.hpp"
#include "../sme/sme-4x1-zapack.hpp"
#include "../sme/sme-1x4-acc.hpp"
#include "../sme/sme-1x4-acc-kc.hpp"
#include "../sme/sme-1x4-acc-kcout.hpp"
#include "../sme/sme-1x4-acc-fast-kcout.hpp"

namespace MaxMulSK {
namespace {

// Kernel ids usable as template arguments.
enum : int { K4x1 = 0, K2x2, K1x4, K1x4SYM, K4x1ZAP };

// Geometry transcribed from each shipped driver. m_outer selects the loop
// order; b_block_stride selects the packed-B offset formula; zero_partial_A
// mirrors the edge-strip zero-fill that 4x1 and ZAPack perform before pack_A.
template <int KID> struct Geom;

template <> struct Geom<K4x1> {
    static constexpr size_t M_tile = 64, K_tile = 2048, N_tile = 1024;
    static constexpr size_t m_mul = 4, n_mul = 1;      // steps in units of SVL
    static constexpr bool m_outer = false;             // n -> k -> m
    static constexpr bool b_block_stride = false;      // packed_B + jr*kc
    static constexpr bool zero_partial_A = true;
};
template <> struct Geom<K2x2> {
    static constexpr size_t M_tile = 64, K_tile = 1024, N_tile = 512;
    static constexpr size_t m_mul = 2, n_mul = 2;
    static constexpr bool m_outer = false;
    static constexpr bool b_block_stride = true;       // packed_B + (jr/N_step)*kc*N_step
    static constexpr bool zero_partial_A = false;
};
template <> struct Geom<K1x4> {
    static constexpr size_t M_tile = 1024, K_tile = 2048, N_tile = 64;
    static constexpr size_t m_mul = 1, n_mul = 4;
    static constexpr bool m_outer = true;              // m -> k -> n
    static constexpr bool b_block_stride = false;
    static constexpr bool zero_partial_A = false;
};
template <> struct Geom<K1x4SYM> : Geom<K1x4> {};
template <> struct Geom<K4x1ZAP> {
    static constexpr size_t M_tile = 128, K_tile = 1024, N_tile = 512;
    static constexpr size_t m_mul = 4, n_mul = 1;
    static constexpr bool m_outer = false;
    static constexpr bool b_block_stride = false;
    static constexpr bool zero_partial_A = true;
};

// -- forwarders to the shipped entry points --------------------------------
// pack_A is declared __arm_out("za") for every variant even though only 2x2 and
// ZAPack actually use ZA as transpose scratch: over-declaring a clobber is
// always sound, and it keeps one signature for the template.

template <int KID>
static void pack_A(const float* A, float* p, size_t mc, size_t kc,
                   size_t m, size_t k, size_t K) __arm_streaming __arm_out("za") {
    if constexpr (KID == K4x1)         SMEKernels4x1::pack_A_streaming(A, p, mc, kc, m, k, K);
    else if constexpr (KID == K2x2)    SMEKernels2x2::pack_A_streaming(A, p, mc, kc, m, k, K);
    else if constexpr (KID == K1x4)    SMEKernels1x4::pack_A_streaming(A, p, mc, kc, m, k, K);
    else if constexpr (KID == K1x4SYM) SMEKernels1x4Sym::pack_A_streaming(A, p, mc, kc, m, k, K);
    else                               SMEKernels4x1ZAPack::pack_A_streaming(A, p, mc, kc, m, k, K);
}

template <int KID>
static void pack_B(const float* B, float* p, size_t nc, size_t kc,
                   size_t k, size_t N, size_t n) __arm_streaming {
    if constexpr (KID == K4x1)         SMEKernels4x1::pack_B_streaming(B, p, nc, kc, k, N, n);
    else if constexpr (KID == K2x2)    SMEKernels2x2::pack_B_streaming(B, p, nc, kc, k, N, n);
    else if constexpr (KID == K1x4)    SMEKernels1x4::pack_B_streaming(B, p, nc, kc, k, N, n);
    else if constexpr (KID == K1x4SYM) SMEKernels1x4Sym::pack_B_streaming(B, p, nc, kc, k, N, n);
    else                               SMEKernels4x1ZAPack::pack_B_streaming(B, p, nc, kc, k, N, n);
}

template <int KID>
static void micro(float* pa, float* pb, float* C,
                  size_t kc, size_t ldc) __arm_streaming __arm_out("za") {
    if constexpr (KID == K4x1)         SMEKernels4x1::micro_kernel_4x1(pa, pb, C, kc, ldc);
    else if constexpr (KID == K2x2)    SMEKernels2x2::micro_kernel_2x2(pa, pb, C, kc, ldc);
    else if constexpr (KID == K1x4)    SMEKernels1x4::micro_kernel_1x4(pa, pb, C, kc, ldc);
    else if constexpr (KID == K1x4SYM) SMEKernels1x4Sym::micro_kernel_1x4(pa, pb, C, kc, ldc);
    else                               SMEKernels4x1ZAPack::micro_kernel_4x1(pa, pb, C, kc, ldc);
}

// SVE-store zero fill; a scalar loop here would be turned into __arm_sc_memset,
// which has no implementation (see TODO.md BUG-5).
static void zero_streaming(float* p, size_t n_floats, size_t SVL) __arm_streaming {
    const svbool_t pg = svptrue_b32();
    const svfloat32_t z = svdup_f32(0.0f);
    for (size_t i = 0; i < n_floats; i += SVL) svst1_f32(pg, p + i, z);
}

// Block tables: one entry per (k,m) A block and per (k,n) B block. Offsets are
// exact, so no space is wasted on the edge blocks of skinny shapes.
struct Blocks {
    size_t SVL = 0, M_step = 0, N_step = 0;
    size_t M = 0, K = 0, N = 0;
    size_t nm = 0, nk = 0, nn = 0;               // block counts
    std::vector<size_t> offA, offB;              // in floats
    size_t floatsA = 0, floatsB = 0;
};

template <int KID>
static Blocks plan(size_t M, size_t K, size_t N, size_t SVL) {
    using G = Geom<KID>;
    Blocks b;
    b.SVL = SVL;
    b.M_step = G::m_mul * SVL;
    b.N_step = G::n_mul * SVL;
    b.M = M; b.K = K; b.N = N;
    b.nm = (M + G::M_tile - 1) / G::M_tile;
    b.nk = (K + G::K_tile - 1) / G::K_tile;
    b.nn = (N + G::N_tile - 1) / G::N_tile;

    auto roundup = [](size_t a, size_t q) { return ((a + q - 1) / q) * q; };

    b.offA.resize(b.nk * b.nm);
    for (size_t ki = 0; ki < b.nk; ++ki) {
        const size_t kc = std::min(G::K_tile, K - ki * G::K_tile);
        for (size_t mi = 0; mi < b.nm; ++mi) {
            const size_t mc = std::min(G::M_tile, M - mi * G::M_tile);
            b.offA[ki * b.nm + mi] = b.floatsA;
            b.floatsA += roundup(mc, b.M_step) * kc;
        }
    }
    b.offB.resize(b.nk * b.nn);
    for (size_t ki = 0; ki < b.nk; ++ki) {
        const size_t kc = std::min(G::K_tile, K - ki * G::K_tile);
        for (size_t ni = 0; ni < b.nn; ++ni) {
            const size_t nc = std::min(G::N_tile, N - ni * G::N_tile);
            b.offB[ki * b.nn + ni] = b.floatsB;
            b.floatsB += roundup(nc, b.N_step) * kc;
        }
    }
    return b;
}

// Pass 1: packing only, into the big buffers.
template <int KID>
__arm_locally_streaming __arm_new("za") __attribute__((noinline))
static void prepack(const float* A, const float* B, const Blocks& b,
                    float* bufA, float* bufB) {
    using G = Geom<KID>;
    const size_t SVL = b.SVL, M_step = b.M_step;

    for (size_t ki = 0; ki < b.nk; ++ki) {
        const size_t k = ki * G::K_tile;
        const size_t kc = std::min(G::K_tile, b.K - k);

        for (size_t ni = 0; ni < b.nn; ++ni) {
            const size_t n = ni * G::N_tile;
            const size_t nc = std::min(G::N_tile, b.N - n);
            pack_B<KID>(B, bufB + b.offB[ki * b.nn + ni], nc, kc, k, b.N, n);
        }
        for (size_t mi = 0; mi < b.nm; ++mi) {
            const size_t m = mi * G::M_tile;
            const size_t mc = std::min(G::M_tile, b.M - m);
            float* pa = bufA + b.offA[ki * b.nm + mi];
            if constexpr (G::zero_partial_A) {
                if (mc % M_step != 0)
                    zero_streaming(pa + (mc / M_step) * M_step * kc, M_step * kc, SVL);
            }
            pack_A<KID>(A, pa, mc, kc, m, k, b.K);
        }
    }
}

// Pass 2: the driver's loop nest with the packing calls removed.
template <int KID>
__arm_locally_streaming __arm_new("za") __attribute__((noinline))
static void compute(float* C, const Blocks& b,
                    const float* bufA, const float* bufB, float* C_scratch) {
    using G = Geom<KID>;
    const size_t SVL = b.SVL, M_step = b.M_step, N_step = b.N_step, N = b.N;

    // One (n,k,m) or (m,k,n) triple; body is identical to the shipped driver's
    // inner (jr, ir) nest.
    auto tile = [&](size_t m, size_t mc, size_t mi,
                    size_t n, size_t nc, size_t ni,
                    size_t kc, size_t ki) __arm_streaming __arm_out("za") {
        const float* pA_blk = bufA + b.offA[ki * b.nm + mi];
        const float* pB_blk = bufB + b.offB[ki * b.nn + ni];

        for (size_t jr = 0; jr < nc; jr += N_step) {
            const size_t n_rem = nc - jr;
            const float* pB = G::b_block_stride ? pB_blk + (jr / N_step) * kc * N_step
                                                : pB_blk + jr * kc;
            for (size_t ir = 0; ir < mc; ir += M_step) {
                const size_t m_rem = mc - ir;
                const bool m_tail = m_rem < M_step;
                const bool n_tail = n_rem < N_step;
                float* pA = const_cast<float*>(pA_blk) + ir * kc;

                if (!m_tail && !n_tail) {
                    micro<KID>(pA, const_cast<float*>(pB),
                               C + (m + ir) * N + (n + jr), kc, N);
                } else {
                    const size_t rows = std::min(m_rem, M_step);
                    const size_t cols = std::min(n_rem, N_step);
                    zero_streaming(C_scratch, M_step * N_step, SVL);
                    micro<KID>(pA, const_cast<float*>(pB), C_scratch, kc, N_step);
                    float* C_dst = C + (m + ir) * N + (n + jr);
                    for (size_t row = 0; row < rows; ++row)
                        for (size_t col = 0; col < cols; ++col)
                            C_dst[row * N + col] += C_scratch[row * N_step + col];
                }
            }
        }
    };

    if constexpr (G::m_outer) {              // m -> k -> n
        for (size_t mi = 0; mi < b.nm; ++mi) {
            const size_t m = mi * G::M_tile, mc = std::min(G::M_tile, b.M - m);
            for (size_t ki = 0; ki < b.nk; ++ki) {
                const size_t k = ki * G::K_tile, kc = std::min(G::K_tile, b.K - k);
                for (size_t ni = 0; ni < b.nn; ++ni) {
                    const size_t n = ni * G::N_tile, nc = std::min(G::N_tile, b.N - n);
                    tile(m, mc, mi, n, nc, ni, kc, ki);
                }
            }
        }
    } else {                                  // n -> k -> m
        for (size_t ni = 0; ni < b.nn; ++ni) {
            const size_t n = ni * G::N_tile, nc = std::min(G::N_tile, b.N - n);
            for (size_t ki = 0; ki < b.nk; ++ki) {
                const size_t k = ki * G::K_tile, kc = std::min(G::K_tile, b.K - k);
                for (size_t mi = 0; mi < b.nm; ++mi) {
                    const size_t m = mi * G::M_tile, mc = std::min(G::M_tile, b.M - m);
                    tile(m, mc, mi, n, nc, ni, kc, ki);
                }
            }
        }
    }
}

// ---- Experiment 1: hot micro-kernel -------------------------------------

// One batch of micro-kernel invocations with streaming mode entered once
// around the whole loop, matching the shipped driver's structure.
template <int KID>
__arm_locally_streaming __arm_new("za") __attribute__((noinline))
static void hot_batch(float* pa, float* pb, float* tile,
                      size_t Kc, size_t ldc, size_t reps) {
    for (size_t r = 0; r < reps; ++r) micro<KID>(pa, pb, tile, Kc, ldc);
}

// A single invocation that enters and leaves streaming mode itself, so a
// non-streaming caller loop pays smstart/smstop per call. This mirrors
// KleidiAI's API granularity.
template <int KID>
__arm_locally_streaming __arm_new("za") __attribute__((noinline))
static void hot_one(float* pa, float* pb, float* tile, size_t Kc, size_t ldc) {
    micro<KID>(pa, pb, tile, Kc, ldc);
}

template <int KID>
__arm_locally_streaming __arm_new("za") __attribute__((noinline))
static void hot_pack(const float* A_small, const float* B_small,
                     float* pa, float* pb,
                     size_t m_step, size_t n_step, size_t Kc) {
    pack_A<KID>(A_small, pa, m_step, Kc, /*curr_row=*/0, /*curr_col=*/0, /*K=*/Kc);
    pack_B<KID>(B_small, pb, n_step, Kc, /*curr_row=*/0, /*N=*/n_step, /*curr_col=*/0);
}

// =========================================================================
// 1x4-Acc / 1x4-Acc-Kc
//
// These do not fit the Geom-templated path above: their micro-kernel takes no C
// and owns no ZA lifetime, their loop order is m->n->K with K innermost, and
// their operands are packed over the full K rather than per K tile. They get
// their own plan/prepack/compute below.
//
// The ZA->C writeback is re-implemented here rather than exported from the
// kernel: it is byte-identical to the kernel's own file-local store_za, and
// keeping it on the benchmark side avoids widening the kernel's public header
// just to be measurable. Like the kernel's, it OVERWRITES C - these variants
// compute C = A*B, not C += A*B.
// =========================================================================

struct AccGeom {
    static constexpr size_t M_tile = 1024, K_tile = 2048, N_tile = 64;
    static constexpr size_t m_mul = 1, n_mul = 4;
};
// Sub-chunk of a K tile. 0 = one call per K tile (Acc); the AccKc value must
// match `constexpr size_t Kc` in sme/sme-1x4-acc-kc.cpp, since this replays
// that driver's loop.
template <bool KC> struct AccChunk { static constexpr size_t value = 0; };
template <> struct AccChunk<true> { static constexpr size_t value = 2048; };

template <bool KC>
static void acc_micro(float* pa, float* pb, size_t k) __arm_streaming __arm_inout("za") {
    if constexpr (KC) SMEKernels1x4AccKc::micro_kernel_1x4(pa, pb, k);
    else              SMEKernels1x4Acc::micro_kernel_1x4(pa, pb, k);
}

// __arm_out("za"): the Acc kernels' pack_A uses ZA tile 0 as a transpose buffer.
template <bool KC>
static void acc_pack_A(const float* A, float* p, size_t mc, size_t K,
                       size_t m) __arm_streaming __arm_out("za") {
    if constexpr (KC) SMEKernels1x4AccKc::pack_A_streaming(A, p, mc, K, m, 0, K);
    else              SMEKernels1x4Acc::pack_A_streaming(A, p, mc, K, m, 0, K);
}

template <bool KC>
static void acc_pack_B(const float* B, float* p, size_t nc, size_t K,
                       size_t N, size_t n) __arm_streaming {
    if constexpr (KC) SMEKernels1x4AccKc::pack_B_streaming(B, p, nc, K, 0, N, n);
    else              SMEKernels1x4Acc::pack_B_streaming(B, p, nc, K, 0, N, n);
}

static void acc_store_za(float* C, size_t wide_of_C) __arm_in("za") __arm_streaming {
    const size_t SVL = static_cast<size_t>(svcntsw());
    const svbool_t pg = svptrue_b32();
    svfloat32_t inactive = svundef_f32();
    // Row-major, mirroring the kernel's store_za: slice i of ZA0..ZA3 is one
    // contiguous 4*SVL row of the output tile, so it goes out in a single
    // svst1_f32_x4 instead of four strided narrow stores.
    const svcount_t pn = svptrue_c32();
    for (size_t i = 0; i < SVL; i++) {
        svfloat32x4_t row = svcreate4_f32(
            svread_hor_za32_f32_m(inactive, pg, 0, (uint32_t)i),
            svread_hor_za32_f32_m(inactive, pg, 1, (uint32_t)i),
            svread_hor_za32_f32_m(inactive, pg, 2, (uint32_t)i),
            svread_hor_za32_f32_m(inactive, pg, 3, (uint32_t)i));
        svst1_f32_x4(pn, C + i * wide_of_C, row);
    }
}

// One C tile: accumulate over all of K, honouring the variant's granularity.
template <bool KC>
static void acc_accumulate_K(const float* pA_panel, const float* pB_panel,
                             size_t K, size_t SVL, size_t N_step) __arm_streaming __arm_inout("za") {
    constexpr size_t chunk = AccChunk<KC>::value;
    for (size_t k = 0; k < K; k += AccGeom::K_tile) {
        const size_t kc = std::min(AccGeom::K_tile, K - k);
        if constexpr (chunk == 0) {
            acc_micro<KC>(const_cast<float*>(pA_panel) + k * SVL,
                          const_cast<float*>(pB_panel) + k * N_step, kc);
        } else {
            for (size_t kk = 0; kk < kc; kk += chunk) {
                const size_t c = std::min(chunk, kc - kk);
                acc_micro<KC>(const_cast<float*>(pA_panel) + (k + kk) * SVL,
                              const_cast<float*>(pB_panel) + (k + kk) * N_step, c);
            }
        }
    }
}

// Block tables: A indexed by m only, B by n only - both hold the full K.
struct AccBlocks {
    size_t SVL = 0, M_step = 0, N_step = 0;
    size_t M = 0, K = 0, N = 0, nm = 0, nn = 0;
    std::vector<size_t> offA, offB;
    size_t floatsA = 0, floatsB = 0;
};

static AccBlocks acc_plan(size_t M, size_t K, size_t N, size_t SVL) {
    AccBlocks b;
    b.SVL = SVL;
    b.M_step = AccGeom::m_mul * SVL;
    b.N_step = AccGeom::n_mul * SVL;
    b.M = M; b.K = K; b.N = N;
    b.nm = (M + AccGeom::M_tile - 1) / AccGeom::M_tile;
    b.nn = (N + AccGeom::N_tile - 1) / AccGeom::N_tile;
    auto roundup = [](size_t a, size_t q) { return ((a + q - 1) / q) * q; };
    b.offA.resize(b.nm);
    for (size_t mi = 0; mi < b.nm; ++mi) {
        const size_t mc = std::min(AccGeom::M_tile, M - mi * AccGeom::M_tile);
        b.offA[mi] = b.floatsA;
        b.floatsA += roundup(mc, b.M_step) * K;
    }
    b.offB.resize(b.nn);
    for (size_t ni = 0; ni < b.nn; ++ni) {
        const size_t nc = std::min(AccGeom::N_tile, N - ni * AccGeom::N_tile);
        b.offB[ni] = b.floatsB;
        b.floatsB += roundup(nc, b.N_step) * K;
    }
    return b;
}

template <bool KC>
__arm_locally_streaming __arm_new("za") __attribute__((noinline))
static void acc_prepack(const float* A, const float* B, const AccBlocks& b,
                        float* bufA, float* bufB) {
    for (size_t mi = 0; mi < b.nm; ++mi) {
        const size_t m = mi * AccGeom::M_tile;
        const size_t mc = std::min(AccGeom::M_tile, b.M - m);
        acc_pack_A<KC>(A, bufA + b.offA[mi], mc, b.K, m);
    }
    for (size_t ni = 0; ni < b.nn; ++ni) {
        const size_t n = ni * AccGeom::N_tile;
        const size_t nc = std::min(AccGeom::N_tile, b.N - n);
        acc_pack_B<KC>(B, bufB + b.offB[ni], nc, b.K, b.N, n);
    }
}

template <bool KC>
__arm_locally_streaming __arm_new("za") __attribute__((noinline))
static void acc_compute(float* C, const AccBlocks& b,
                        const float* bufA, const float* bufB, float* C_scratch) {
    const size_t SVL = b.SVL, M_step = b.M_step, N_step = b.N_step, N = b.N, K = b.K;
    for (size_t mi = 0; mi < b.nm; ++mi) {
        const size_t m = mi * AccGeom::M_tile;
        const size_t mc = std::min(AccGeom::M_tile, b.M - m);
        for (size_t ni = 0; ni < b.nn; ++ni) {
            const size_t n = ni * AccGeom::N_tile;
            const size_t nc = std::min(AccGeom::N_tile, b.N - n);
            for (size_t jr = 0; jr < nc; jr += N_step) {
                const size_t n_rem = nc - jr;
                for (size_t ir = 0; ir < mc; ir += M_step) {
                    const size_t m_rem = mc - ir;
                    const bool m_tail = m_rem < M_step;
                    const bool n_tail = n_rem < N_step;

                    svzero_za();
                    acc_accumulate_K<KC>(bufA + b.offA[mi] + ir * K,
                                         bufB + b.offB[ni] + jr * K,
                                         K, SVL, N_step);
                    if (!m_tail && !n_tail) {
                        acc_store_za(C + (m + ir) * N + (n + jr), N);
                    } else {
                        const size_t rows = std::min(m_rem, M_step);
                        const size_t cols = std::min(n_rem, N_step);
                        // Overwriting store: no pre-zeroing, and the scatter
                        // assigns. Predicated SVE rather than a scalar loop,
                        // which would lower to __arm_sc_memcpy (TODO.md BUG-5).
                        acc_store_za(C_scratch, N_step);
                        float* C_dst = C + (m + ir) * N + (n + jr);
                        for (size_t row = 0; row < rows; ++row) {
                            const float* src = C_scratch + row * N_step;
                            float* dst_row = C_dst + row * N;
                            for (size_t col = 0; col < cols; col += SVL) {
                                svbool_t pg_c = svwhilelt_b32_u64(col, cols);
                                svst1_f32(pg_c, dst_row + col, svld1_f32(pg_c, src + col));
                            }
                        }
                    }
                }
            }
        }
    }
}

// Hot micro-kernel batches for the Acc variants.
template <bool KC>
__arm_locally_streaming __arm_new("za") __attribute__((noinline))
static void acc_hot_batch(float* pa, float* pb, float* tile,
                          size_t Kc, size_t ldc, size_t reps) {
    constexpr size_t chunk = AccChunk<KC>::value;
    for (size_t r = 0; r < reps; ++r) {
        svzero_za();
        if constexpr (chunk == 0) {
            acc_micro<KC>(pa, pb, Kc);
        } else {
            const size_t SVL = static_cast<size_t>(svcntsw());
            for (size_t kk = 0; kk < Kc; kk += chunk) {
                const size_t c = std::min(chunk, Kc - kk);
                acc_micro<KC>(pa + kk * SVL, pb + kk * (4 * SVL), c);
            }
        }
        acc_store_za(tile, ldc);
    }
}

template <bool KC>
__arm_locally_streaming __arm_new("za") __attribute__((noinline))
static void acc_hot_batch_amortized(float* pa, float* pb, float* tile,
                                    size_t Kc, size_t ldc, size_t reps) {
    constexpr size_t chunk = AccChunk<KC>::value;
    const size_t SVL = static_cast<size_t>(svcntsw());
    svzero_za();
    for (size_t r = 0; r < reps; ++r) {
        if constexpr (chunk == 0) {
            acc_micro<KC>(pa, pb, Kc);
        } else {
            for (size_t kk = 0; kk < Kc; kk += chunk) {
                const size_t c = std::min(chunk, Kc - kk);
                acc_micro<KC>(pa + kk * SVL, pb + kk * (4 * SVL), c);
            }
        }
    }
    acc_store_za(tile, ldc);
}

template <bool KC>
__arm_locally_streaming __arm_new("za") __attribute__((noinline))
static void acc_hot_pack(const float* A_small, const float* B_small,
                         float* pa, float* pb,
                         size_t m_step, size_t n_step, size_t Kc) {
    if constexpr (KC) {
        SMEKernels1x4AccKc::pack_A_streaming(A_small, pa, m_step, Kc, 0, 0, Kc);
        SMEKernels1x4AccKc::pack_B_streaming(B_small, pb, n_step, Kc, 0, n_step, 0);
    } else {
        SMEKernels1x4Acc::pack_A_streaming(A_small, pa, m_step, Kc, 0, 0, Kc);
        SMEKernels1x4Acc::pack_B_streaming(B_small, pb, n_step, Kc, 0, n_step, 0);
    }
}

struct FreeDeleter { void operator()(void* p) const { std::free(p); } };
using Buf = std::unique_ptr<float[], FreeDeleter>;

static Buf alloc64(size_t floats) {
    const size_t bytes = ((floats * sizeof(float)) + 63) & ~size_t(63);
    return Buf(static_cast<float*>(std::aligned_alloc(64, bytes ? bytes : 64)));
}

}  // namespace

// ---------------------------------------------------------------------------

const char* name(Kernel k) {
    switch (k) {
        case Kernel::Sme4x1:        return "SMEKernels4x1::run_multiplication";
        case Kernel::Sme2x2:        return "SMEKernels2x2::run_multiplication";
        case Kernel::Sme1x4:        return "SMEKernels1x4::run_multiplication";
        case Kernel::Sme1x4Sym:     return "SMEKernels1x4Sym::run_multiplication";
        case Kernel::Sme4x1ZAPack:  return "SMEKernels4x1ZAPack::run_multiplication";
        case Kernel::Sme1x4SymZAIO: return "SMEKernels1x4SymZAInOut::run_multiplication";
        case Kernel::Sme1x4Acc:     return "SMEKernels1x4Acc::run_multiplication";
        case Kernel::Sme1x4AccKc:   return "SMEKernels1x4AccKc::run_multiplication";
        case Kernel::Sme1x4AccKcOut:     return "SMEKernels1x4AccKcOut::run_multiplication";
        case Kernel::Sme1x4AccFastKcOut: return "SMEKernels1x4AccFastKcOut::run_multiplication";
    }
    return "?";
}

const char* label(Kernel k) {
    switch (k) {
        case Kernel::Sme4x1:        return "SME 4x1";
        case Kernel::Sme2x2:        return "SME 2x2";
        case Kernel::Sme1x4:        return "SME 1x4";
        case Kernel::Sme1x4Sym:     return "SME 1x4sym";
        case Kernel::Sme4x1ZAPack:  return "SME 4x1ZP";
        case Kernel::Sme1x4SymZAIO: return "SME 1x4ZAIO";
        case Kernel::Sme1x4Acc:     return "SME 1x4Acc";
        case Kernel::Sme1x4AccKc:   return "SME 1x4AccKc";
        case Kernel::Sme1x4AccKcOut:     return "SME 1x4AccKcOut";
        case Kernel::Sme1x4AccFastKcOut: return "SME 1x4AccFastKcOut";
    }
    return "?";
}

bool has_kernel_only(Kernel k) {
    return k != Kernel::Sme1x4SymZAIO &&
           k != Kernel::Sme1x4AccKcOut &&
           k != Kernel::Sme1x4AccFastKcOut;
}

bool overwrites_C(Kernel k) {
    // The outer-Kc variants touch each C tile K/Kc times, but the first panel
    // still overwrites, so the caller must not pre-zero them either.
    return k == Kernel::Sme1x4Acc || k == Kernel::Sme1x4AccKc ||
           k == Kernel::Sme1x4AccKcOut || k == Kernel::Sme1x4AccFastKcOut;
}

// Only the two kernels the Prepacked / HotMicro replay understands. The
// outer-Kc variants overwrite C too, but they never reach those paths
// (has_kernel_only() is false for them), so they are deliberately excluded here.
static bool is_acc(Kernel k) {
    return k == Kernel::Sme1x4Acc || k == Kernel::Sme1x4AccKc;
}

__arm_locally_streaming __attribute__((noinline))
static size_t svl_words_streaming() { return static_cast<size_t>(svcntsw()); }

size_t svl_words() { return svl_words_streaming(); }

void run(Kernel k, const float* A, const float* B, float* C,
         size_t M, size_t K, size_t N) {
    switch (k) {
        case Kernel::Sme4x1:        SMEKernels4x1::run_multiplication(A, B, C, M, K, N); break;
        case Kernel::Sme2x2:        SMEKernels2x2::run_multiplication(A, B, C, M, K, N); break;
        case Kernel::Sme1x4:        SMEKernels1x4::run_multiplication(A, B, C, M, K, N); break;
        case Kernel::Sme1x4Sym:     SMEKernels1x4Sym::run_multiplication(A, B, C, M, K, N); break;
        case Kernel::Sme4x1ZAPack:  SMEKernels4x1ZAPack::run_multiplication(A, B, C, M, K, N); break;
        case Kernel::Sme1x4SymZAIO: SMEKernels1x4SymZAInOut::run_multiplication(A, B, C, M, K, N); break;
        case Kernel::Sme1x4Acc:     SMEKernels1x4Acc::run_multiplication(A, B, C, M, K, N); break;
        case Kernel::Sme1x4AccKc:   SMEKernels1x4AccKc::run_multiplication(A, B, C, M, K, N); break;
        case Kernel::Sme1x4AccKcOut:     SMEKernels1x4AccKcOut::run_multiplication(A, B, C, M, K, N); break;
        case Kernel::Sme1x4AccFastKcOut: SMEKernels1x4AccFastKcOut::run_multiplication(A, B, C, M, K, N); break;
    }
}

struct Prepacked::Impl {
    Kernel kind = Kernel::Sme4x1;
    Blocks blocks;
    AccBlocks acc_blocks;
    Buf bufA, bufB, scratch;
};

Prepacked::Prepacked() : impl_(std::make_unique<Impl>()) {}
Prepacked::~Prepacked() = default;

void Prepacked::prepare(Kernel k, const float* A, const float* B,
                        size_t M, size_t K, size_t N) {
    Impl& s = *impl_;
    s.kind = k;
    const size_t SVL = svl_words();

    auto build = [&](auto kid_tag) {
        constexpr int KID = decltype(kid_tag)::value;
        s.blocks = plan<KID>(M, K, N, SVL);
        s.bufA = alloc64(s.blocks.floatsA);
        s.bufB = alloc64(s.blocks.floatsB);
        s.scratch = alloc64(s.blocks.M_step * s.blocks.N_step);
        // Touch every page before packing so the pack pass, and not some later
        // timed region, pays the faults.
        std::fill_n(s.bufA.get(), s.blocks.floatsA, 0.0f);
        std::fill_n(s.bufB.get(), s.blocks.floatsB, 0.0f);
        std::fill_n(s.scratch.get(), s.blocks.M_step * s.blocks.N_step, 0.0f);
        prepack<KID>(A, B, s.blocks, s.bufA.get(), s.bufB.get());
    };

    switch (k) {
        case Kernel::Sme4x1:       build(std::integral_constant<int, K4x1>{});    break;
        case Kernel::Sme2x2:       build(std::integral_constant<int, K2x2>{});    break;
        case Kernel::Sme1x4:       build(std::integral_constant<int, K1x4>{});    break;
        case Kernel::Sme1x4Sym:    build(std::integral_constant<int, K1x4SYM>{}); break;
        case Kernel::Sme4x1ZAPack: build(std::integral_constant<int, K4x1ZAP>{}); break;
        case Kernel::Sme1x4Acc:
        case Kernel::Sme1x4AccKc: {
            s.acc_blocks = acc_plan(M, K, N, SVL);
            s.bufA = alloc64(s.acc_blocks.floatsA);
            s.bufB = alloc64(s.acc_blocks.floatsB);
            s.scratch = alloc64(s.acc_blocks.M_step * s.acc_blocks.N_step);
            std::fill_n(s.bufA.get(), s.acc_blocks.floatsA, 0.0f);
            std::fill_n(s.bufB.get(), s.acc_blocks.floatsB, 0.0f);
            std::fill_n(s.scratch.get(), s.acc_blocks.M_step * s.acc_blocks.N_step, 0.0f);
            if (k == Kernel::Sme1x4Acc)
                acc_prepack<false>(A, B, s.acc_blocks, s.bufA.get(), s.bufB.get());
            else
                acc_prepack<true>(A, B, s.acc_blocks, s.bufA.get(), s.bufB.get());
            break;
        }
        case Kernel::Sme1x4SymZAIO:
        case Kernel::Sme1x4AccKcOut:
        case Kernel::Sme1x4AccFastKcOut: break;  // unsupported; see has_kernel_only()
    }
}

void Prepacked::compute(float* C) const {
    const Impl& s = *impl_;
    switch (s.kind) {
        case Kernel::Sme4x1:
            compute<K4x1>(C, s.blocks, s.bufA.get(), s.bufB.get(), s.scratch.get()); break;
        case Kernel::Sme2x2:
            compute<K2x2>(C, s.blocks, s.bufA.get(), s.bufB.get(), s.scratch.get()); break;
        case Kernel::Sme1x4:
            compute<K1x4>(C, s.blocks, s.bufA.get(), s.bufB.get(), s.scratch.get()); break;
        case Kernel::Sme1x4Sym:
            compute<K1x4SYM>(C, s.blocks, s.bufA.get(), s.bufB.get(), s.scratch.get()); break;
        case Kernel::Sme4x1ZAPack:
            compute<K4x1ZAP>(C, s.blocks, s.bufA.get(), s.bufB.get(), s.scratch.get()); break;
        case Kernel::Sme1x4Acc:
            acc_compute<false>(C, s.acc_blocks, s.bufA.get(), s.bufB.get(), s.scratch.get()); break;
        case Kernel::Sme1x4AccKc:
            acc_compute<true>(C, s.acc_blocks, s.bufA.get(), s.bufB.get(), s.scratch.get()); break;
        case Kernel::Sme1x4SymZAIO:
        case Kernel::Sme1x4AccKcOut:
        case Kernel::Sme1x4AccFastKcOut: break;  // see has_kernel_only()
    }
}

struct HotMicro::Impl {
    Kernel kind = Kernel::Sme4x1;
    size_t Kc = 0, m_step = 0, n_step = 0;
    Buf bufA, bufB, tile;
};

HotMicro::HotMicro() : impl_(std::make_unique<Impl>()) {}
HotMicro::~HotMicro() = default;

size_t HotMicro::m_step() const { return impl_->m_step; }
size_t HotMicro::n_step() const { return impl_->n_step; }

size_t HotMicro::panel_bytes() const {
    const Impl& s = *impl_;
    return (s.m_step * s.Kc + s.n_step * s.Kc + s.m_step * s.n_step) * sizeof(float);
}

void HotMicro::zero_tile() {
    std::fill_n(impl_->tile.get(), impl_->m_step * impl_->n_step, 0.0f);
}

const float* HotMicro::tile() const { return impl_->tile.get(); }

void HotMicro::prepare(Kernel k, size_t Kc, const float* A_small, const float* B_small) {
    Impl& s = *impl_;
    s.kind = k;
    s.Kc = Kc;
    const size_t SVL = svl_words();

    auto build = [&](auto kid_tag) {
        constexpr int KID = decltype(kid_tag)::value;
        using G = Geom<KID>;
        s.m_step = G::m_mul * SVL;
        s.n_step = G::n_mul * SVL;
        s.bufA = alloc64(s.m_step * Kc);
        s.bufB = alloc64(s.n_step * Kc);
        s.tile = alloc64(s.m_step * s.n_step);
        std::fill_n(s.bufA.get(), s.m_step * Kc, 0.0f);
        std::fill_n(s.bufB.get(), s.n_step * Kc, 0.0f);
        std::fill_n(s.tile.get(), s.m_step * s.n_step, 0.0f);
        hot_pack<KID>(A_small, B_small, s.bufA.get(), s.bufB.get(),
                      s.m_step, s.n_step, Kc);
    };

    switch (k) {
        case Kernel::Sme4x1:       build(std::integral_constant<int, K4x1>{});    break;
        case Kernel::Sme2x2:       build(std::integral_constant<int, K2x2>{});    break;
        case Kernel::Sme1x4:       build(std::integral_constant<int, K1x4>{});    break;
        case Kernel::Sme1x4Sym:    build(std::integral_constant<int, K1x4SYM>{}); break;
        case Kernel::Sme4x1ZAPack: build(std::integral_constant<int, K4x1ZAP>{}); break;
        case Kernel::Sme1x4Acc:
        case Kernel::Sme1x4AccKc: {
            s.m_step = AccGeom::m_mul * SVL;
            s.n_step = AccGeom::n_mul * SVL;
            s.bufA = alloc64(s.m_step * Kc);
            s.bufB = alloc64(s.n_step * Kc);
            s.tile = alloc64(s.m_step * s.n_step);
            std::fill_n(s.bufA.get(), s.m_step * Kc, 0.0f);
            std::fill_n(s.bufB.get(), s.n_step * Kc, 0.0f);
            std::fill_n(s.tile.get(), s.m_step * s.n_step, 0.0f);
            if (k == Kernel::Sme1x4Acc)
                acc_hot_pack<false>(A_small, B_small, s.bufA.get(), s.bufB.get(),
                                    s.m_step, s.n_step, Kc);
            else
                acc_hot_pack<true>(A_small, B_small, s.bufA.get(), s.bufB.get(),
                                   s.m_step, s.n_step, Kc);
            break;
        }
        case Kernel::Sme1x4SymZAIO:
        case Kernel::Sme1x4AccKcOut:
        case Kernel::Sme1x4AccFastKcOut: break;  // see has_kernel_only()
    }
}

bool HotMicro::supports_amortized() const { return is_acc(impl_->kind); }

void HotMicro::run_batch_amortized(size_t reps) const {
    const Impl& s = *impl_;
    if (s.kind == Kernel::Sme1x4Acc)
        acc_hot_batch_amortized<false>(s.bufA.get(), s.bufB.get(), s.tile.get(),
                                       s.Kc, s.n_step, reps);
    else if (s.kind == Kernel::Sme1x4AccKc)
        acc_hot_batch_amortized<true>(s.bufA.get(), s.bufB.get(), s.tile.get(),
                                      s.Kc, s.n_step, reps);
}

void HotMicro::run_batch(size_t reps) const {
    const Impl& s = *impl_;
    switch (s.kind) {
        case Kernel::Sme4x1:
            hot_batch<K4x1>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step, reps); break;
        case Kernel::Sme2x2:
            hot_batch<K2x2>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step, reps); break;
        case Kernel::Sme1x4:
            hot_batch<K1x4>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step, reps); break;
        case Kernel::Sme1x4Sym:
            hot_batch<K1x4SYM>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step, reps); break;
        case Kernel::Sme4x1ZAPack:
            hot_batch<K4x1ZAP>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step, reps); break;
        case Kernel::Sme1x4Acc:
            acc_hot_batch<false>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step, reps); break;
        case Kernel::Sme1x4AccKc:
            acc_hot_batch<true>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step, reps); break;
        case Kernel::Sme1x4SymZAIO:
        case Kernel::Sme1x4AccKcOut:
        case Kernel::Sme1x4AccFastKcOut: break;  // see has_kernel_only()
    }
}

void HotMicro::run_batch_isolated_streaming(size_t reps) const {
    const Impl& s = *impl_;
    for (size_t r = 0; r < reps; ++r) {
        switch (s.kind) {
            case Kernel::Sme4x1:
                hot_one<K4x1>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step); break;
            case Kernel::Sme2x2:
                hot_one<K2x2>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step); break;
            case Kernel::Sme1x4:
                hot_one<K1x4>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step); break;
            case Kernel::Sme1x4Sym:
                hot_one<K1x4SYM>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step); break;
            case Kernel::Sme4x1ZAPack:
                hot_one<K4x1ZAP>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step); break;
            case Kernel::Sme1x4Acc:
                acc_hot_batch<false>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step, 1); break;
            case Kernel::Sme1x4AccKc:
                acc_hot_batch<true>(s.bufA.get(), s.bufB.get(), s.tile.get(), s.Kc, s.n_step, 1); break;
            case Kernel::Sme1x4SymZAIO:
            case Kernel::Sme1x4AccKcOut:
            case Kernel::Sme1x4AccFastKcOut: break;  // see has_kernel_only()
        }
    }
}

size_t Prepacked::packed_bytes() const {
    if (is_acc(impl_->kind))
        return (impl_->acc_blocks.floatsA + impl_->acc_blocks.floatsB) * sizeof(float);
    return (impl_->blocks.floatsA + impl_->blocks.floatsB) * sizeof(float);
}

}  // namespace MaxMulSK
