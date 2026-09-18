#include "sme-1x4-kcout-mc16-apack4za-bonce-zapres.hpp"

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <algorithm>

#define RESTRICT __restrict__

// =============================================================================
// EXPERIMENT F2-ZAPRES -- F2 with pack_B_streaming declared __arm_preserves("za")
//
// Derived from sme/v3/sme-1x4-acc-kcout.cpp at commit 8f1ec86 (with that file's
// uncommitted working-tree edits; see bench/results/2026-09-10/environment.txt).
// pack_B_streaming, micro_kernel_1x4, store_za and store_za_add are carried over
// VERBATIM -- the compute load/MOPA schedule and the C writeback are not part of
// this experiment.
//
// What changes versus the reference driver:
//   * Mc goes from 1024 to 16, so one A panel is a single 16-row micro-tile and
//     it is consumed by the four N panels immediately after it is built.
//   * A is packed INLINE in the driver, not through pack_A_streaming, and it
//     uses all four ZA tiles instead of tile 0 alone: 64 k-steps per round
//     instead of 16, with x4 loads feeding horizontal writes and x4 stores
//     draining vertical reads.
//   * B is packed once per (Kc, N panel) on the first M round and reused by
//     every later M round, instead of being repacked per M block.
//
// Loop order is Kc -> Mc -> Nc. Target shape only; see run_multiplication_f2.
// These are hypotheses about locality, not measured results.
// =============================================================================

// Output group g of the vertical drain is four consecutive vertical slices of
// one ZA tile: tile g/4, first column 4*(g%4). Both must be compile-time
// constants -- the tile operand of svread_ver_za32 is an immediate.
#define ZP_T(g) ((g) / 4)
#define ZP_C(g) (4 * ((g) % 4))
#define ZP_GROUP(g)                                                            \
    svcreate4_f32(svread_ver_za32_f32_m(zu, pg, ZP_T(g), ZP_C(g) + 0),         \
                  svread_ver_za32_f32_m(zu, pg, ZP_T(g), ZP_C(g) + 1),         \
                  svread_ver_za32_f32_m(zu, pg, ZP_T(g), ZP_C(g) + 2),         \
                  svread_ver_za32_f32_m(zu, pg, ZP_T(g), ZP_C(g) + 3))

// One steady step: store the group a Z set already holds, then refill that same
// Z set with the group two ahead. Store operand is consumed before the refill,
// so the two x4 store streams never wait on each other.
#define ZP_DRAIN(g)                                                            \
    svst1_f32_x4(p4, out + (g) * 64, s0);                                      \
    s0 = ZP_GROUP((g) + 2);                                                    \
    svst1_f32_x4(p4, out + ((g) + 1) * 64, s1);                                \
    s1 = ZP_GROUP((g) + 3);

#define ZP_PACK_A_KP_CHUNK()                                           \
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
        float* const out = packed_A + kp * 16;                         \
        svfloat32x4_t s0 = ZP_GROUP(0);                                \
        svfloat32x4_t s1 = ZP_GROUP(1);                                \
        /* Static unroll: the tile operand of svread_ver_za32 must */  \
        /* be an immediate, so g cannot be a runtime variable. */      \
        ZP_DRAIN(0)  ZP_DRAIN(2)  ZP_DRAIN(4)  ZP_DRAIN(6)             \
        ZP_DRAIN(8)  ZP_DRAIN(10) ZP_DRAIN(12)                         \
        /* DRAIN: s0 holds group 14 (ZA3 vertical 8..11), */           \
        /* s1 holds group 15 (ZA3 vertical 12..15). No further */      \
        /* extraction -- the two stores are back to back. */           \
        svst1_f32_x4(p4, out + 14 * 64, s0);                           \
        svst1_f32_x4(p4, out + 15 * 64, s1);                           \
    } while (0)                                                        \

namespace SMEKernels1x4KcOutMc16Apack4ZaBonceZaPres {

    // PACK B  (K x N) -> 4-panel interleaved layout
    // Each jr-group (4*SVL cols) gets its own contiguous K_curr * 4*SVL block.
    // Layout per k-step: [panel0_SVL | panel1_SVL | panel2_SVL | panel3_SVL]
    // Micro-kernel reads all 4 panels with a single svld1_f32_x4 per k-step.
    // =========================================================================
    // ONLY change versus the measured F2: this body neither reads nor writes ZA
    // -- verified in the generated assembly, 104 instructions, zero ZA or
    // streaming-state opcodes and no calls -- so it can be declared as
    // preserving ZA. Without that, the ABI forces the caller to arm a TPIDR2
    // lazy save around every call. noinline is kept, as is the body itself.
    __attribute__((noinline))
    void pack_B_streaming(const float* B, float* packed_B,
                          size_t N_curr, size_t K_curr,
                          size_t curr_row, size_t N, size_t curr_col)
        __arm_streaming __arm_preserves("za") {
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

    // =========================================================================
    // F2 STREAMING BODY
    //
    // ZA lifetime, which the attributes have to match:
    //   * inline A packing WRITES all four tiles (no zero needed: every element
    //     read back is written first) and leaves nothing live;
    //   * svzero_za() then starts a fresh C accumulator per micro-tile;
    //   * micro_kernel_1x4 accumulates into that live ZA (__arm_inout);
    //   * store_za / store_za_add drain it, after which nothing is live.
    // Packing and accumulation never overlap, so no save/restore is required.
    // =========================================================================
    __arm_locally_streaming __arm_new("za")
    __attribute__((noinline))
    static bool run_f2_streaming(const float* A, const float* B, float* C,
                                 size_t M, size_t K, size_t N,
                                 float* packed_A, float* packed_B) {
        // The layouts and unrolls below are written for 16 fp32 lanes. Anything
        // else would silently mis-address, so refuse rather than assume.
        if (static_cast<size_t>(svcntsw()) != 16) return false;

        constexpr size_t Kc = 2048;
        constexpr size_t Mc = 16;
        constexpr size_t Nc = 64;

        const svbool_t  pg = svptrue_b32();
        const svcount_t p4 = svptrue_c32();
        const svfloat32_t zu = svundef_f32();

        for (size_t kk = 0; kk < K; kk += Kc) {
            const bool first_k = (kk == 0);

            for (size_t m = 0; m < M; m += Mc) {
                for (size_t n = 0; n < N; n += Nc) {
                    const size_t panel_id = n / Nc;
                    float* const pB = packed_B + panel_id * (Kc * Nc);

                    // B PACK ONCE: only on the first M round of this Kc slice.
                    // The four panels are NOT prepared together up front; each
                    // is built the first time its own n is reached.
                    if (m == 0) {
                        pack_B_streaming(B, pB, Nc, Kc, kk, N, n);
                    }

                    if (n == 0) {
                        // A PACK, INLINE. Built once per (kk, m) and then reused
                        // by n = 64, 128, 192 without repacking.
                        for (size_t kp = 0; kp < Kc; kp += 64) {
                            const float* const arow = A + m * K + kk + kp;

                        ZP_PACK_A_KP_CHUNK();
                        }
                    }

                    // ZA becomes the C accumulator now; full zero.
                    svzero_za();

                    // Compute body carried over unchanged. Both pointers start
                    // at the base of their panel on every call.
                    micro_kernel_1x4(packed_A, pB, Kc);

                    float* const dstC = C + m * N + n;
                    if (first_k) {
                        store_za(dstC, N);        // C is overwritten, not read
                    } else {
                        store_za_add(dstC, N);    // old C + this Kc partial sum
                    }
                }
            }
        }
        return true;
    }

    // TEST ONLY. Expands the SAME ZP_PACK_A_KP_CHUNK macro as the driver, so
    // what is verified here is the code that actually runs; it is a second
    // expansion, not a refactor of the hot path into a callable helper.
    __arm_locally_streaming __arm_new("za")
    bool probe_pack_A(const float* A, size_t M, size_t K,
                      size_t kk, size_t m, float* packed_A) {
        if (static_cast<size_t>(svcntsw()) != 16) return false;
        if (m + 16 > M || kk + 2048 > K) return false;
        const svbool_t  pg = svptrue_b32();
        const svcount_t p4 = svptrue_c32();
        const svfloat32_t zu = svundef_f32();
        for (size_t kp = 0; kp < 2048; kp += 64) {
            const float* const arow = A + m * K + kk + kp;
            ZP_PACK_A_KP_CHUNK();
        }
        return true;
    }

    // Public entry. Deliberately NOT streaming: allocation and the shape check
    // belong outside the streaming region, which is entered exactly once.
    //
    // Returns false and leaves C untouched for anything but the target shape.
    // There is no dispatch and no general tail path in this experiment; running
    // another shape here would silently assume full tiles.
    bool run_multiplication_f2(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N) {
        if (!(M == 11008 && K == 4096 && N == 256)) return false;

        constexpr size_t Kc = 2048, Mc = 16, Nc = 64;
        AlignedBuffer packed_A(static_cast<float*>(
            std::aligned_alloc(64, Mc * Kc * sizeof(float))));          // 128 KiB
        AlignedBuffer packed_B(static_cast<float*>(
            std::aligned_alloc(64, 4 * Nc * Kc * sizeof(float))));      // 2 MiB
        if (!packed_A || !packed_B) return false;

        return run_f2_streaming(A, B, C, M, K, N, packed_A.get(), packed_B.get());
    }

} // namespace SMEKernels1x4KcOutMc16Apack4ZaBonceZaPres
