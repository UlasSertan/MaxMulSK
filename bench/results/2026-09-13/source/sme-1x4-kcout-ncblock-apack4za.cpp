#include "sme-1x4-kcout-ncblock-apack4za.hpp"

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <algorithm>

#define RESTRICT __restrict__

// =============================================================================
// EXPERIMENT NB -- 1x4-kcout-ncblock-apack4za
//
// F2 with an Nc block added. F2 packed every N panel of a Kc slice up front, so
// its packed B grew with N; measured 2026-09-13, its collapse tracked packed-B
// size rather than M or N directly. Here B is packed one (Kc, Nc) block at a
// time, which caps the packed working set at (Nc + Mc) * Kc floats whatever N is.
//
// Loop order:  Kc -> Nc -> Mc -> n(64 cols) -> m(16 rows) -> compute(Kc)
//
//   * B is packed once per (Kc, Nc) block, on the first Mc round, and held for
//     every later Mc round of that block.
//   * A is packed once per (Kc, Nc, Mc) block, on that block's first 64-column
//     panel, and reused by the remaining panels of the same Nc block.
//   * A is re-packed for the NEXT Nc block. That is the price paid on purpose to
//     bound B's working set: the A-pack count carries a factor of ceil(N/Nc).
//   * Mc is NOT packed up front. On the first n panel each 16-row chunk is packed
//     into its own slot and computed immediately, keeping preparation and
//     consumption adjacent.
//
// The inline four-ZA A packing, the compute micro-kernel with its load/MOPA
// schedule, the packed B layout and both C writeback bodies are carried over
// unchanged from the measured F2.
// =============================================================================

// Output group g of the vertical drain is four consecutive vertical slices of
// one ZA tile: tile g/4, first column 4*(g%4). Both must be compile-time
// constants -- the tile operand of svread_ver_za32 is an immediate.
#define NB_T(g) ((g) / 4)
#define NB_C(g) (4 * ((g) % 4))
#define NB_GROUP(g)                                                            \
    svcreate4_f32(svread_ver_za32_f32_m(zu, pg, NB_T(g), NB_C(g) + 0),         \
                  svread_ver_za32_f32_m(zu, pg, NB_T(g), NB_C(g) + 1),         \
                  svread_ver_za32_f32_m(zu, pg, NB_T(g), NB_C(g) + 2),         \
                  svread_ver_za32_f32_m(zu, pg, NB_T(g), NB_C(g) + 3))

// One steady step: store the group a Z set already holds, then refill that same
// Z set with the group two ahead. Store operand is consumed before the refill,
// so the two x4 store streams never wait on each other.
#define NB_DRAIN(g)                                                            \
    svst1_f32_x4(p4, out + (g) * 64, s0);                                      \
    s0 = NB_GROUP((g) + 2);                                                    \
    svst1_f32_x4(p4, out + ((g) + 1) * 64, s1);                                \
    s1 = NB_GROUP((g) + 3);

#define NB_PACK_A_KP_CHUNK()                                           \
    do {                                                               \
        /* ---- HORIZONTAL FILL: memory -> Z -> ZA ------------- */    \
        /* No svzero_za here: every ZA element that is read back */    \
        /* below is written first -- all 16 slices of all four */      \
        /* tiles. Merging predication cannot expose stale ZA. */       \
        /* One x4 load brings the four consecutive 16-float */         \
        /* vectors of ONE source row, which land in tiles 0..3 at */   \
        /* the same slice. q0 and q1 are two independent four-Z */     \
        /* groups so a load is always in flight while ZA is */         \
        /* written. */                                                 \
        svfloat32x4_t q0 = svld1_f32_x4(p4, arow + 0 * K);             \
        svfloat32x4_t q1 = svld1_f32_x4(p4, arow + 1 * K);             \
        for (uint32_t r = 0; r < 14; r += 2) {                         \
            svwrite_hor_za32_f32_m(0, r, pg, svget4_f32(q0, 0));       \
            svwrite_hor_za32_f32_m(1, r, pg, svget4_f32(q0, 1));       \
            svwrite_hor_za32_f32_m(2, r, pg, svget4_f32(q0, 2));       \
            svwrite_hor_za32_f32_m(3, r, pg, svget4_f32(q0, 3));       \
            /* every component of q0 is consumed; refill it */         \
            q0 = svld1_f32_x4(p4, arow + (r + 2) * K);                 \
            svwrite_hor_za32_f32_m(0, r + 1, pg, svget4_f32(q1, 0));   \
            svwrite_hor_za32_f32_m(1, r + 1, pg, svget4_f32(q1, 1));   \
            svwrite_hor_za32_f32_m(2, r + 1, pg, svget4_f32(q1, 2));   \
            svwrite_hor_za32_f32_m(3, r + 1, pg, svget4_f32(q1, 3));   \
            q1 = svld1_f32_x4(p4, arow + (r + 3) * K);                 \
        }                                                              \
        /* DRAIN: rows 14 and 15 are already in q0/q1. No load of */   \
        /* rows 16/17 -- they do not exist in this 16-row panel. */    \
        svwrite_hor_za32_f32_m(0, 14, pg, svget4_f32(q0, 0));          \
        svwrite_hor_za32_f32_m(1, 14, pg, svget4_f32(q0, 1));          \
        svwrite_hor_za32_f32_m(2, 14, pg, svget4_f32(q0, 2));          \
        svwrite_hor_za32_f32_m(3, 14, pg, svget4_f32(q0, 3));          \
        svwrite_hor_za32_f32_m(0, 15, pg, svget4_f32(q1, 0));          \
        svwrite_hor_za32_f32_m(1, 15, pg, svget4_f32(q1, 1));          \
        svwrite_hor_za32_f32_m(2, 15, pg, svget4_f32(q1, 2));          \
        svwrite_hor_za32_f32_m(3, 15, pg, svget4_f32(q1, 3));          \
        /* ---- VERTICAL DRAIN: ZA -> Z -> packed_A ------------ */    \
        /* Reading tile t vertically is the transpose for free: */     \
        /* vertical slice c of tile t is A[m+0..15][kk+kp+16t+c], */   \
        /* which is exactly packed_A[(kp+16t+c)*16 + 0..15]. */        \
        /* Output group g = four vertical slices = 64 floats. */       \
        /* Groups 0-3 come from ZA0, 4-7 ZA1, 8-11 ZA2, 12-15 ZA3; */  \
        /* the pipeline is NOT restarted at a tile boundary. */        \
        float* const out = pA + kp * 16;                         \
        svfloat32x4_t s0 = NB_GROUP(0);                                \
        svfloat32x4_t s1 = NB_GROUP(1);                                \
        /* Static unroll: the tile operand of svread_ver_za32 must */  \
        /* be an immediate, so g cannot be a runtime variable. */      \
        NB_DRAIN(0)  NB_DRAIN(2)  NB_DRAIN(4)  NB_DRAIN(6)             \
        NB_DRAIN(8)  NB_DRAIN(10) NB_DRAIN(12)                         \
        /* DRAIN: s0 holds group 14 (ZA3 vertical 8..11), */           \
        /* s1 holds group 15 (ZA3 vertical 12..15). No further */      \
        /* extraction -- the two stores are back to back. */           \
        svst1_f32_x4(p4, out + 14 * 64, s0);                           \
        svst1_f32_x4(p4, out + 15 * 64, s1);                           \
    } while (0)

namespace SMEKernels1x4KcOutNcBlockApack4Za {

    // PACK B  (K x N) -> 4-panel interleaved layout
    // Each jr-group (4*SVL cols) gets its own contiguous K_curr * 4*SVL block.
    // Layout per k-step: [panel0_SVL | panel1_SVL | panel2_SVL | panel3_SVL]
    // Micro-kernel reads all 4 panels with a single svld1_f32_x4 per k-step.
    // =========================================================================
    __attribute__((noinline))
    void pack_B_streaming(const float* B, float* packed_B,
                          size_t N_curr, size_t K_curr,
                          size_t curr_row, size_t N, size_t curr_col) __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svcount_t full_mask4 = svptrue_c32();
        const svbool_t  pg         = svptrue_b32();
        const size_t GS = 4 * SVL;
        const size_t group_stride = K_curr * GS;
        const size_t N_full = (N_curr / GS) * GS;

        size_t n = 0;

        for (; n < N_full; n += GS) {
            float* block_base = packed_B + (n / GS) * group_stride;

            for (size_t k = 0; k < K_curr; k++) {
                svfloat32x4_t vec = svld1_f32_x4(full_mask4,
                                                  B + (k + curr_row) * N + curr_col + n);
                float* out = block_base + k * GS;
                svst1_f32(pg, out + 0 * SVL, svget4_f32(vec, 0));
                svst1_f32(pg, out + 1 * SVL, svget4_f32(vec, 1));
                svst1_f32(pg, out + 2 * SVL, svget4_f32(vec, 2));
                svst1_f32(pg, out + 3 * SVL, svget4_f32(vec, 3));
            }
        }

        // N tail: predicated lanes zero invalid columns. Kernel still reads
        // 4*SVL per k-step from this block; zero lanes contribute nothing.
        if (n < N_curr) {
            float* block_base = packed_B + (n / GS) * group_stride;
            svbool_t mask0 = svwhilelt_b32_u64(n + 0 * SVL, N_curr);
            svbool_t mask1 = svwhilelt_b32_u64(n + 1 * SVL, N_curr);
            svbool_t mask2 = svwhilelt_b32_u64(n + 2 * SVL, N_curr);
            svbool_t mask3 = svwhilelt_b32_u64(n + 3 * SVL, N_curr);

            for (size_t k = 0; k < K_curr; k++) {
                const float* src = B + (k + curr_row) * N + curr_col + n;
                svfloat32_t v0 = svld1_f32(mask0, src + 0 * SVL);
                svfloat32_t v1 = svld1_f32(mask1, src + 1 * SVL);
                svfloat32_t v2 = svld1_f32(mask2, src + 2 * SVL);
                svfloat32_t v3 = svld1_f32(mask3, src + 3 * SVL);
                float* out = block_base + k * GS;
                svst1_f32(pg, out + 0 * SVL, v0);
                svst1_f32(pg, out + 1 * SVL, v1);
                svst1_f32(pg, out + 2 * SVL, v2);
                svst1_f32(pg, out + 3 * SVL, v3);
            }
        }
    }

    // MICRO KERNEL 1x4-sym: SVL rows x 4*SVL cols using ZA accumulator.
    // ZA_i = A x B_i  -- single A vec, 4 different B panels per k-step.
    // K-unrolled by 4, software-pipelined as the A<->B mirror of 4x1.
    //
    // Per group of 4 k-steps:
    //   1 A x4 load   (covers 4 k-steps' SVL-wide A vecs)
    //   4 B x4 loads  (one per k-step, each = 4 B panels)
    //   16 svmopa
    //
    // Schedule per 4 k-steps:
    //   PROLOGUE: load A x4 (k+0..k+3), load B x4 (k+0)
    //   group k+0: ZA0(a0,b0), ZA1(a0,b1), [load B(k+1)], ZA2(a0,b2), ZA3(a0,b3)
    //   group k+1: ZA0(a1,b0), ZA1(a1,b1), [load B(k+2)], ZA2(a1,b2), ZA3(a1,b3)
    //   group k+2: ZA0(a2,b0), ZA1(a2,b1), [load B(k+3)], ZA2(a2,b2), ZA3(a2,b3)
    //   group k+3: ZA0(a3,b0), ZA1(a3,b1), [load next A x4], ZA2(a3,b2), ZA3(a3,b3)
    //   end-of-iter: load next iter's first B x4
    // Same-tile write gap: 4 svmopa slots + 1 load (matches 4x1).
    // =========================================================================
    __attribute__((noinline))
    void micro_kernel_1x4(float* RESTRICT packed_A, float* RESTRICT packed_B,
                          size_t K_curr) __arm_inout("za") __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t pg = svptrue_b32();
        const svcount_t pg4 = svptrue_c32();
        const size_t GS_B = 4 * SVL; // B group stride per k-step

        const float* pA = packed_A;
        const float* pB = packed_B;

        const size_t K_main = (K_curr / 4) * 4;

        if (K_main >= 4) {
            // PROLOGUE: load first A x4 (covers k+0..k+3) and first B x4 (k+0)
            svfloat32x4_t a_x4 = svld1_f32_x4(pg4, pA); pA += 4 * SVL;
            svfloat32_t a0 = svget4_f32(a_x4, 0);
            svfloat32_t a1 = svget4_f32(a_x4, 1);
            svfloat32_t a2 = svget4_f32(a_x4, 2);
            svfloat32_t a3 = svget4_f32(a_x4, 3);

            svfloat32x4_t b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;

            for (size_t k = 0; k < K_main - 4; k += 4) {
                // Extract current B panels (k-step k+0)
                svfloat32_t b0 = svget4_f32(b_x4, 0);
                svfloat32_t b1 = svget4_f32(b_x4, 1);
                svfloat32_t b2 = svget4_f32(b_x4, 2);
                svfloat32_t b3 = svget4_f32(b_x4, 3);

                // group k+0: ZA0, ZA1, [load B(k+1)], ZA2, ZA3
                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
                svmopa_za32_f32_m(2, pg, pg, a0, b2);
                svmopa_za32_f32_m(3, pg, pg, a0, b3);

                b0 = svget4_f32(b_x4, 0);
                b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2);
                b3 = svget4_f32(b_x4, 3);

                // group k+1: ZA0, ZA1, [load B(k+2)], ZA2, ZA3
                svmopa_za32_f32_m(0, pg, pg, a1, b0);
                svmopa_za32_f32_m(1, pg, pg, a1, b1);
                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
                svmopa_za32_f32_m(2, pg, pg, a1, b2);
                svmopa_za32_f32_m(3, pg, pg, a1, b3);

                b0 = svget4_f32(b_x4, 0);
                b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2);
                b3 = svget4_f32(b_x4, 3);

                // group k+2: ZA0, ZA1, [load B(k+3)], ZA2, ZA3
                svmopa_za32_f32_m(0, pg, pg, a2, b0);
                svmopa_za32_f32_m(1, pg, pg, a2, b1);
                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
                svmopa_za32_f32_m(2, pg, pg, a2, b2);
                svmopa_za32_f32_m(3, pg, pg, a2, b3);

                b0 = svget4_f32(b_x4, 0);
                b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2);
                b3 = svget4_f32(b_x4, 3);

                // group k+3: ZA0, ZA1, [load next A x4], ZA2, ZA3
                svmopa_za32_f32_m(0, pg, pg, a3, b0);
                svmopa_za32_f32_m(1, pg, pg, a3, b1);
                a_x4 = svld1_f32_x4(pg4, pA); pA += 4 * SVL;
                svmopa_za32_f32_m(2, pg, pg, a3, b2);
                svmopa_za32_f32_m(3, pg, pg, a3, b3);

                a0 = svget4_f32(a_x4, 0);
                a1 = svget4_f32(a_x4, 1);
                a2 = svget4_f32(a_x4, 2);
                a3 = svget4_f32(a_x4, 3);
                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B; // prefetch next iter's first B
            }

            // EPILOGUE: last group of 4 (no next prefetch)
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

        // K TAIL: remaining 0-3 iterations, one at a time.
        // Mirror of 4x1 K-tail: 1 A x1 load + 1 B x4 load + 4 svmopa.
        for (size_t k = K_main; k < K_curr; k++) {
            svfloat32_t a = svld1_f32(pg, pA); pA += SVL;
            svfloat32x4_t b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
            svmopa_za32_f32_m(0, pg, pg, a, svget4_f32(b_x4, 0));
            svmopa_za32_f32_m(1, pg, pg, a, svget4_f32(b_x4, 1));
            svmopa_za32_f32_m(2, pg, pg, a, svget4_f32(b_x4, 2));
            svmopa_za32_f32_m(3, pg, pg, a, svget4_f32(b_x4, 3));
        }

    }

    // ZA -> C writeback, lifted out of the micro-kernel.
    //
    // This OVERWRITES C rather than accumulating into it. That is sound here and
    // not in 1x4-sym: with K innermost, each C tile is stored exactly once with
    // the complete K sum, so there is nothing in C to preserve. Dropping the
    // read-modify-write removes 64 loads and 64 adds per tile (4 KiB of reads).
    //
    // CONSEQUENCE: run_multiplication computes C = A*B, not C += A*B. The caller
    // does NOT need to zero C first. This differs from the other MaxMulSK SME
    // kernels, which all accumulate.
    // Tiles stack horizontally: ZA_i -> C[0..SVL, i*SVL..(i+1)*SVL]
    // =========================================================================
    __attribute__((noinline))
    static void store_za(float* RESTRICT C, size_t wide_of_C) __arm_in("za") __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t pg = svptrue_b32();
        svfloat32_t inactive = svundef_f32();

        // Row-major traversal instead of tile-major. The four ZA tiles stack
        // horizontally, so slice i of ZA0..ZA3 lands on ONE contiguous 4*SVL
        // row of the output tile -- one svst1_f32_x4 per row. Tile-major
        // traversal (ZA0's 16 rows, then ZA1's...) cannot do this: consecutive
        // rows of one tile are wide_of_C apart, and SME2 has no strided
        // multi-vector store. 64 narrow stores become 16 wide ones.
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

    __attribute__((noinline))
    static void store_za_add(float* RESTRICT C, size_t wide_of_C) __arm_in("za") __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t pg = svptrue_b32();
        const svcount_t pn = svptrue_c32();
        svfloat32_t inactive = svundef_f32();

        for (size_t i = 0; i < SVL; i++) {
            svfloat32x4_t za = svcreate4_f32(
                svread_hor_za32_f32_m(inactive, pg, 0, (uint32_t)i),
                svread_hor_za32_f32_m(inactive, pg, 1, (uint32_t)i),
                svread_hor_za32_f32_m(inactive, pg, 2, (uint32_t)i),
                svread_hor_za32_f32_m(inactive, pg, 3, (uint32_t)i));
            svfloat32x4_t old = svld1_f32_x4(pn, C + i * wide_of_C);
            svfloat32x4_t sum = svcreate4_f32(
                svadd_f32_x(pg, svget4_f32(za, 0), svget4_f32(old, 0)),
                svadd_f32_x(pg, svget4_f32(za, 1), svget4_f32(old, 1)),
                svadd_f32_x(pg, svget4_f32(za, 2), svget4_f32(old, 2)),
                svadd_f32_x(pg, svget4_f32(za, 3), svget4_f32(old, 3)));
            svst1_f32_x4(pn, C + i * wide_of_C, sum);
        }
    }

    struct FreeDeleter { void operator()(void* p) { std::free(p); } };
    using AlignedBuffer = std::unique_ptr<float[], FreeDeleter>;

    Support classify(size_t M, size_t K, size_t N, const Blocking& b) {
        if (M == 0 || K == 0 || N == 0)  return Support::Unsupported;
        if (M % 16 != 0)                 return Support::UnsupportedMTail;
        if (N % 64 != 0)                 return Support::UnsupportedNTail;
        if (K % 64 != 0)                 return Support::UnsupportedKTail;
        // A macroblock must be a whole number of microtiles and packing chunks.
        // A SHORT FINAL macroblock is fine: Mc=128 against M=64 just makes the
        // single block 64 rows. What is rejected is a macroblock size that is
        // not itself a multiple of the microtile.
        if (b.Mc == 0 || b.Mc % 16 != 0) return Support::BadBlocking;
        if (b.Nc == 0 || b.Nc % 64 != 0) return Support::BadBlocking;
        if (b.Kc == 0 || b.Kc % 64 != 0) return Support::BadBlocking;
        return Support::Native;
    }

    // Allocation capacity. Panel slot strides are built from the ALLOCATED Kc
    // depth, not from the current block's real length, so a short final Kc / Nc /
    // Mc block leaves a gap rather than shifting every later slot. Packing and
    // compute address slots the same way.
    void capacity(size_t M, size_t K, size_t N, const Blocking& b,
                  size_t* packed_A_bytes, size_t* packed_B_bytes) {
        const size_t Kc = std::min(b.Kc, K);
        const size_t Nc = std::min(b.Nc, N);
        const size_t Mc = std::min(b.Mc, M);
        *packed_A_bytes = Mc * Kc * sizeof(float);
        *packed_B_bytes = Nc * Kc * sizeof(float);
    }

    // Preparation and compute counts, derived from the loop structure alone. No
    // instrumentation is added to the hot path.
    void counts(size_t M, size_t K, size_t N, const Blocking& b, Counts* out) {
        const size_t Kc = std::min(b.Kc, K);
        const size_t Nc = std::min(b.Nc, N);
        const size_t Mc = std::min(b.Mc, M);
        out->kc_slices = (K + Kc - 1) / Kc;
        out->nc_blocks = (N + Nc - 1) / Nc;
        out->mc_blocks = (M + Mc - 1) / Mc;
        // Every 64-column B panel, once per Kc slice.
        out->b_panel_packs = out->kc_slices * (N / 64);
        // Every 16-row A panel, once per (Kc slice, Nc block). This is the term
        // that grows as Nc shrinks.
        out->a_panel_packs = out->kc_slices * out->nc_blocks * (M / 16);
        // Compute does not depend on the macroblock sizes when microtile aligned.
        out->microkernel_calls = out->kc_slices * (N / 64) * (M / 16);
    }

    __arm_locally_streaming __arm_new("za")
    __attribute__((noinline))
    static bool run_streaming(const float* A, const float* B, float* C,
                              size_t M, size_t K, size_t N,
                              float* packed_A, float* packed_B,
                              size_t Kc, size_t Nc, size_t Mc) {
        if (static_cast<size_t>(svcntsw()) != 16) return false;

        const svbool_t  pg = svptrue_b32();
        const svcount_t p4 = svptrue_c32();
        const svfloat32_t zu = svundef_f32();

        for (size_t kk = 0; kk < K; kk += Kc) {
            const size_t kcl = std::min(Kc, K - kk);
            const bool first_k = (kk == 0);

            for (size_t nn = 0; nn < N; nn += Nc) {
                const size_t ncl = std::min(Nc, N - nn);

                for (size_t mm = 0; mm < M; mm += Mc) {
                    const size_t mcl = std::min(Mc, M - mm);

                    for (size_t nr = 0; nr < ncl; nr += 64) {
                        float* const pB = packed_B + (nr / 64) * (Kc * 64);

                        // Prepared once for the whole Nc block, on its first Mc
                        // round, and held for every later Mc round.
                        if (mm == 0) {
                            pack_B_streaming(B, pB, 64, kcl, kk, N, nn + nr);
                        }

                        for (size_t ir = 0; ir < mcl; ir += 16) {
                            float* const pA = packed_A + (ir / 16) * (Kc * 16);

                            // Packed on this Nc block's first panel and reused by
                            // the rest of it; re-packed for the next Nc block.
                            if (nr == 0) {
                                const float* const arow_base = A + (mm + ir) * K + kk;
                                for (size_t kp = 0; kp < kcl; kp += 64) {
                                    const float* const arow = arow_base + kp;
                                    NB_PACK_A_KP_CHUNK();
                                }
                            }

                            svzero_za();
                            micro_kernel_1x4(pA, pB, kcl);

                            float* const dstC = C + (mm + ir) * N + (nn + nr);
                            if (first_k) store_za(dstC, N);
                            else         store_za_add(dstC, N);
                        }
                    }
                }
            }
        }
        return true;
    }

    // TEST ONLY. Expands the SAME NB_PACK_A_KP_CHUNK macro as the driver, so the
    // layout check verifies the code that actually runs.
    __arm_locally_streaming __arm_new("za")
    bool probe_pack_A(const float* A, size_t M, size_t K,
                      size_t kk, size_t m, size_t kcl, float* pA) {
        if (static_cast<size_t>(svcntsw()) != 16) return false;
        if (m + 16 > M || kk + kcl > K || kcl == 0 || kcl % 64 != 0) return false;
        const svbool_t  pg = svptrue_b32();
        const svcount_t p4 = svptrue_c32();
        const svfloat32_t zu = svundef_f32();
        const float* const arow_base = A + m * K + kk;
        for (size_t kp = 0; kp < kcl; kp += 64) {
            const float* const arow = arow_base + kp;
            NB_PACK_A_KP_CHUNK();
        }
        return true;
    }

    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N, const Blocking& b) {
        const Support s = classify(M, K, N, b);
        if (s != Support::Native) return s;

        // Clamp to the shape so a macroblock larger than the matrix does not
        // over-allocate; the loops then see a single short block.
        const size_t Kc = std::min(b.Kc, K);
        const size_t Nc = std::min(b.Nc, N);
        const size_t Mc = std::min(b.Mc, M);

        AlignedBuffer packed_A(static_cast<float*>(
            std::aligned_alloc(64, Mc * Kc * sizeof(float))));
        AlignedBuffer packed_B(static_cast<float*>(
            std::aligned_alloc(64, Nc * Kc * sizeof(float))));
        if (!packed_A || !packed_B) return Support::AllocationFailed;

        return run_streaming(A, B, C, M, K, N, packed_A.get(), packed_B.get(), Kc, Nc, Mc)
                   ? Support::Native : Support::UnsupportedVectorLength;
    }

} // namespace SMEKernels1x4KcOutNcBlockApack4Za
