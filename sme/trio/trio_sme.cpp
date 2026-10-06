#include "trio_internal.hpp"

#include <arm_acle.h>
#include <arm_sme.h>
#include <arm_sve.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>

#define RESTRICT __restrict__
#define PF_L2KEEP(p) __pldx(0, 1, 0, (const void*)(p))

// SME: A pack macros (ZA transpose), used only by the warm-up, start
#define FK_T(g) ((g) / 4)
#define FK_C(g) (4 * ((g) % 4))
#define FK_GROUP(g)                                                            \
    svcreate4_f32(svread_ver_za32_f32_m(zu, pg, FK_T(g), FK_C(g) + 0),         \
                  svread_ver_za32_f32_m(zu, pg, FK_T(g), FK_C(g) + 1),         \
                  svread_ver_za32_f32_m(zu, pg, FK_T(g), FK_C(g) + 2),         \
                  svread_ver_za32_f32_m(zu, pg, FK_T(g), FK_C(g) + 3))
#define FK_DRAIN(g)                                                            \
    svst1_f32_x4(p4, out + (g) * 64, s0);                                      \
    s0 = FK_GROUP((g) + 2);                                                    \
    svst1_f32_x4(p4, out + ((g) + 1) * 64, s1);                                \
    s1 = FK_GROUP((g) + 3);
#define FK_PACK_A_KP_CHUNK(SRC, STRIDE)                                        \
    do {                                                                       \
        svfloat32x4_t q0 = svld1_f32_x4(p4, (SRC) + 0 * (STRIDE));             \
        svfloat32x4_t q1 = svld1_f32_x4(p4, (SRC) + 1 * (STRIDE));             \
        for (uint32_t r = 0; r < 14; r += 2) {                                 \
            svwrite_hor_za32_f32_m(0, r, pg, svget4_f32(q0, 0));               \
            svwrite_hor_za32_f32_m(1, r, pg, svget4_f32(q0, 1));               \
            svwrite_hor_za32_f32_m(2, r, pg, svget4_f32(q0, 2));               \
            svwrite_hor_za32_f32_m(3, r, pg, svget4_f32(q0, 3));               \
            q0 = svld1_f32_x4(p4, (SRC) + (r + 2) * (STRIDE));                 \
            svwrite_hor_za32_f32_m(0, r + 1, pg, svget4_f32(q1, 0));           \
            svwrite_hor_za32_f32_m(1, r + 1, pg, svget4_f32(q1, 1));           \
            svwrite_hor_za32_f32_m(2, r + 1, pg, svget4_f32(q1, 2));           \
            svwrite_hor_za32_f32_m(3, r + 1, pg, svget4_f32(q1, 3));           \
            q1 = svld1_f32_x4(p4, (SRC) + (r + 3) * (STRIDE));                 \
        }                                                                      \
        svwrite_hor_za32_f32_m(0, 14, pg, svget4_f32(q0, 0));                  \
        svwrite_hor_za32_f32_m(1, 14, pg, svget4_f32(q0, 1));                  \
        svwrite_hor_za32_f32_m(2, 14, pg, svget4_f32(q0, 2));                  \
        svwrite_hor_za32_f32_m(3, 14, pg, svget4_f32(q0, 3));                  \
        svwrite_hor_za32_f32_m(0, 15, pg, svget4_f32(q1, 0));                  \
        svwrite_hor_za32_f32_m(1, 15, pg, svget4_f32(q1, 1));                  \
        svwrite_hor_za32_f32_m(2, 15, pg, svget4_f32(q1, 2));                  \
        svwrite_hor_za32_f32_m(3, 15, pg, svget4_f32(q1, 3));                  \
        float* const out = pA + kp * 16;                                       \
        svfloat32x4_t s0 = FK_GROUP(0);                                        \
        svfloat32x4_t s1 = FK_GROUP(1);                                        \
        FK_DRAIN(0)  FK_DRAIN(2)  FK_DRAIN(4)  FK_DRAIN(6)                     \
        FK_DRAIN(8)  FK_DRAIN(10) FK_DRAIN(12)                                 \
        svst1_f32_x4(p4, out + 14 * 64, s0);                                   \
        svst1_f32_x4(p4, out + 15 * 64, s1);                                   \
    } while (0)
#define FK_PACK_A_EDGE_CHUNK()                                                 \
    do {                                                                       \
        for (size_t r = 0; r < 16; r++) {                                      \
            const size_t have = (r < rows) ? kr : 0;                           \
            const float* srow = arow_base + (r < rows ? r : 0) * K + kp;       \
            float* drow = stage + r * 64;                                      \
            for (size_t j = 0; j < 64; j += SVL) {                             \
                const svbool_t pv = svwhilelt_b32_u64(j, have);                \
                svst1_f32(pg, drow + j, svld1_f32(pv, srow + j));              \
            }                                                                  \
        }                                                                      \
        FK_PACK_A_KP_CHUNK(stage, 64);                                         \
    } while (0)

// SME: A pack macros end

namespace trio::detail {
namespace {

    __attribute__((noinline))
    // SME: compute (micro-kernel) starts
    static void micro_kernel_1x4(float* RESTRICT packed_A, float* RESTRICT packed_B,
                          size_t K_curr) __arm_inout("za") __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t pg = svptrue_b32();
        const svcount_t pg4 = svptrue_c32();
        const size_t GS_B = 4 * SVL;

        const float* pA = packed_A;
        const float* pB = packed_B;

        const size_t K_main = (K_curr / 4) * 4;

        if (K_main >= 4) {
            svfloat32x4_t a_x4 = svld1_f32_x4(pg4, pA); pA += 4 * SVL;
            svfloat32_t a0 = svget4_f32(a_x4, 0);
            svfloat32_t a1 = svget4_f32(a_x4, 1);
            svfloat32_t a2 = svget4_f32(a_x4, 2);
            svfloat32_t a3 = svget4_f32(a_x4, 3);

            svfloat32x4_t b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;

            for (size_t k = 0; k < K_main - 4; k += 4) {
                svfloat32_t b0 = svget4_f32(b_x4, 0);
                svfloat32_t b1 = svget4_f32(b_x4, 1);
                svfloat32_t b2 = svget4_f32(b_x4, 2);
                svfloat32_t b3 = svget4_f32(b_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
                svmopa_za32_f32_m(2, pg, pg, a0, b2);
                svmopa_za32_f32_m(3, pg, pg, a0, b3);

                b0 = svget4_f32(b_x4, 0);
                b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2);
                b3 = svget4_f32(b_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a1, b0);
                svmopa_za32_f32_m(1, pg, pg, a1, b1);
                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
                svmopa_za32_f32_m(2, pg, pg, a1, b2);
                svmopa_za32_f32_m(3, pg, pg, a1, b3);

                b0 = svget4_f32(b_x4, 0);
                b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2);
                b3 = svget4_f32(b_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a2, b0);
                svmopa_za32_f32_m(1, pg, pg, a2, b1);
                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
                svmopa_za32_f32_m(2, pg, pg, a2, b2);
                svmopa_za32_f32_m(3, pg, pg, a2, b3);

                b0 = svget4_f32(b_x4, 0);
                b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2);
                b3 = svget4_f32(b_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a3, b0);
                svmopa_za32_f32_m(1, pg, pg, a3, b1);
                a_x4 = svld1_f32_x4(pg4, pA); pA += 4 * SVL;
                svmopa_za32_f32_m(2, pg, pg, a3, b2);
                svmopa_za32_f32_m(3, pg, pg, a3, b3);

                a0 = svget4_f32(a_x4, 0);
                a1 = svget4_f32(a_x4, 1);
                a2 = svget4_f32(a_x4, 2);
                a3 = svget4_f32(a_x4, 3);
                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
            }

            {
                svfloat32_t b0 = svget4_f32(b_x4, 0);
                svfloat32_t b1 = svget4_f32(b_x4, 1);
                svfloat32_t b2 = svget4_f32(b_x4, 2);
                svfloat32_t b3 = svget4_f32(b_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                svmopa_za32_f32_m(2, pg, pg, a0, b2);
                svmopa_za32_f32_m(3, pg, pg, a0, b3);

                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
                b0 = svget4_f32(b_x4, 0); b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2); b3 = svget4_f32(b_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a1, b0);
                svmopa_za32_f32_m(1, pg, pg, a1, b1);
                svmopa_za32_f32_m(2, pg, pg, a1, b2);
                svmopa_za32_f32_m(3, pg, pg, a1, b3);

                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
                b0 = svget4_f32(b_x4, 0); b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2); b3 = svget4_f32(b_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a2, b0);
                svmopa_za32_f32_m(1, pg, pg, a2, b1);
                svmopa_za32_f32_m(2, pg, pg, a2, b2);
                svmopa_za32_f32_m(3, pg, pg, a2, b3);

                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
                b0 = svget4_f32(b_x4, 0); b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2); b3 = svget4_f32(b_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a3, b0);
                svmopa_za32_f32_m(1, pg, pg, a3, b1);
                svmopa_za32_f32_m(2, pg, pg, a3, b2);
                svmopa_za32_f32_m(3, pg, pg, a3, b3);
            }
        }

        for (size_t k = K_main; k < K_curr; k++) {
            svfloat32_t a = svld1_f32(pg, pA); pA += SVL;
            svfloat32x4_t b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
            svmopa_za32_f32_m(0, pg, pg, a, svget4_f32(b_x4, 0));
            svmopa_za32_f32_m(1, pg, pg, a, svget4_f32(b_x4, 1));
            svmopa_za32_f32_m(2, pg, pg, a, svget4_f32(b_x4, 2));
            svmopa_za32_f32_m(3, pg, pg, a, svget4_f32(b_x4, 3));
        }

    }

    __attribute__((noinline))
    // SME: compute ends

    // SME: C writeback (ZA -> C) starts
    static void store_za(float* RESTRICT C, size_t wide_of_C) __arm_in("za") __arm_streaming {
        const svbool_t pg = svptrue_b32();
        for (uint32_t i = 0; i < 16; i++) {
            float* const row = C + i * wide_of_C;
            svst1_hor_za32(0, i, pg, row + 0);
            svst1_hor_za32(1, i, pg, row + 16);
            svst1_hor_za32(2, i, pg, row + 32);
            svst1_hor_za32(3, i, pg, row + 48);
        }
    }

    __attribute__((noinline))
    static void add_c_into_za(const float* RESTRICT C, size_t wide_of_C) __arm_inout("za") __arm_streaming {
        const svbool_t pg = svptrue_b32();
        for (uint32_t t = 0; t < 4; t++)
            for (uint32_t i = 0; i < 4; i++) {
                const float* const c = C + t * 16;
                const svfloat32x4_t z = svcreate4_f32(
                    svld1_f32(pg, c + (i + 0) * wide_of_C), svld1_f32(pg, c + (i + 4) * wide_of_C),
                    svld1_f32(pg, c + (i + 8) * wide_of_C), svld1_f32(pg, c + (i + 12) * wide_of_C));
                svadd_za32_f32_vg1x4(4 * i + t, z);
            }
    }

    // SME: C writeback ends

    // SME: C prefetch starts
    __attribute__((always_inline)) static inline void prefetch_c_tile(const float* C, size_t wide_of_C) __arm_streaming_compatible {
        for (size_t i = 0; i < 16; i++) {
            const float* row = C + i * wide_of_C;
            PF_L2KEEP(row); PF_L2KEEP(reinterpret_cast<const char*>(row) + 128);
        }
    }

    // SME: C prefetch ends

    // SME: warm-up A pack starts
    __attribute__((always_inline)) static inline
    void pack_A_block(const float* A, size_t K, size_t mm, size_t mcl, size_t kk, size_t kcl, size_t Kc_pad,
                      float* packed_A, float* stage) __arm_streaming __arm_inout("za") {
        const svbool_t  pg = svptrue_b32();
        const svcount_t p4 = svptrue_c32();
        const svfloat32_t zu = svundef_f32();
        const size_t SVL = static_cast<size_t>(svcntsw());
        for (size_t ir = 0; ir < mcl; ir += 16) {
            float* const pA = packed_A + (ir / 16) * (Kc_pad * 16);
            const size_t rows = std::min<size_t>(16, mcl - ir);
            const float* const arow_base = A + (mm + ir) * K + kk;
            for (size_t kp = 0; kp < kcl; kp += 64) {
                const size_t kr = std::min<size_t>(64, kcl - kp);
                if (rows == 16 && kr == 64) { const float* const arow = arow_base + kp; FK_PACK_A_KP_CHUNK(arow, K); }
                else { FK_PACK_A_EDGE_CHUNK(); }
            }
        }
    }

    // SME: warm-up A pack ends

    // SME: C prefetch, next tile
    __attribute__((always_inline)) static inline
    void pf_c_next(float* C, size_t N, size_t mm, size_t mcl, size_t nn, size_t ncl, size_t nr, size_t ir, bool first_k) __arm_streaming_compatible {
        if (first_k) return;
        const size_t nir = ir + 16 < mcl ? ir + 16 : 0, nnr = ir + 16 < mcl ? nr : nr + 64;
        if (nnr < ncl && nir + 16 <= mcl && nnr + 64 <= ncl) prefetch_c_tile(C + (mm + nir) * N + (nn + nnr), N);
    }

    // SME: streaming region for one call starts
    __arm_locally_streaming __arm_new("za") __attribute__((noinline))
    static bool run_streaming(Shared& sh, float* C, float* stage) {
        if (static_cast<size_t>(svcntsw()) != 16) return false;
        const Call cl = sh.call;
        const float* const A = cl.A;
        const size_t M = cl.M, K = cl.K, N = cl.N, Mc = cl.Mc, Nc = cl.Nc, Kc = cl.Kc, Kc_pad = cl.Kc_pad;
        const size_t depth = cl.depth, P = kP;
        const float* const ring = cl.ring;
        float* const aring = cl.aring;
        // SME: warm-up (first A and B unit) starts
        {
            pack_A_block(A, K, 0, std::min(Mc, M), 0, std::min(Kc, K), Kc_pad, aring, stage);
            sh.a_ready[0].v.store(1, std::memory_order_release);
            const svcount_t wpn = svptrue_c32();
            const Unit& un = cl.units[0];
            float* const dst = cl.ring;
            for (size_t r = 0; r < un.rows; r++)
                for (size_t q = 0; q < un.npan; q++)
                    svst1_f32_x4(wpn, dst + q * (Kc * 64) + r * 64, svld1_f32_x4(wpn, un.src + r * N + q * 64));
            sh.ready[0].v.store(1, std::memory_order_release);
        }
        // SME: warm-up ends
        // SME: compute loop starts
        size_t ub = 0, akey = 0;
        for (size_t kk = 0; kk < K; kk += Kc) {
            const size_t kcl = std::min(Kc, K - kk);
            const bool first_k = (kk == 0);
            for (size_t nn = 0; nn < N; nn += Nc) {
                const size_t ncl = std::min(Nc, N - nn), nunits = (ncl + 64 * P - 1) / (64 * P);
                for (size_t mm = 0; mm < M; mm += Mc) {
                    const size_t mcl = std::min(Mc, M - mm);
                    const bool last_mm = mm + Mc >= M;
                    const size_t s = akey % 2;
                    while (sh.a_ready[s].v.load(std::memory_order_acquire) != akey + 1) {}
                    float* const aslot = aring + s * (Mc * Kc_pad);
                    for (size_t nr = 0; nr < ncl; nr += 64) {
                        const size_t u = ub + nr / (64 * P);
                        const size_t slot = u % depth;
                        if (mm == 0 && nr % (64 * P) == 0)
                            while (sh.ready[slot].v.load(std::memory_order_acquire) != (uint64_t)u + 1) {}
                        float* const pB = const_cast<float*>(ring) + slot * (Kc * 64 * P) + ((nr / 64) % P) * (Kc * 64);
                        for (size_t ir = 0; ir < mcl; ir += 16) {
                            float* const dstC = C + (mm + ir) * N + (nn + nr);
                            svzero_za();
                            pf_c_next(C, N, mm, mcl, nn, ncl, nr, ir, first_k);
                            micro_kernel_1x4(aslot + (ir / 16) * (Kc_pad * 16), pB, kcl);
                            if (!first_k) add_c_into_za(dstC, N);
                            store_za(dstC, N);
                        }
                        if (last_mm && (nr + 64 >= ncl || (nr + 64) % (64 * P) == 0))
                            sh.cons.b.store((uint64_t)u + 1, std::memory_order_release);
                    }
                    sh.cons.a.store(akey + 1, std::memory_order_release);
                    akey++;
                }
                ub += nunits;
            }
        }
        // SME: compute loop ends
        return true;
    }
    // SME: streaming region ends

    __arm_locally_streaming __attribute__((noinline))
    static size_t lanes_streaming() { return static_cast<size_t>(svcntsw()); }

}

    bool run_sme(Shared& sh, float* C, float* stage) { return run_streaming(sh, C, stage); }
    size_t streaming_lanes() { return lanes_streaming(); }

}
