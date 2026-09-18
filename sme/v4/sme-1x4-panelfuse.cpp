#include "sme-1x4-panelfuse.hpp"

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <algorithm>

#define RESTRICT __restrict__

// =============================================================================
// ARCH-001 PanelFuse -- experiment 2 of the ledger's separation plan:
//   the new macroflow, with the EXISTING packers and the EXISTING compute body.
//
// Nothing about how a panel is packed or multiplied changes here. What changes
// is which panels are prepared together, how long they are kept, and the rule
// that stops preparation:
//
//   * A packed working set of A and B panels is prepared under a byte budget,
//     with equal NUMBERS of 16 x kcl A panels and kcl x 64 B panels. In bytes
//     that is the ledger's 1:4 rule (one pair costs 320 * kcl bytes).
//   * Then a compute phase consumes every (A panel, B panel) pair of the
//     prepared rectangle with NO packing inside it.
//   * One operand is retained and only the other's next region is prepared:
//     if all of A fits the budget, A is kept for the whole of N; if all of B
//     fits, B is kept for the whole of M; if neither fits, a policy decides.
//
// The rectangle rule, per phase:  t = min(remaining panels of the operand being
// advanced, panels of the retained region, budget / (320 * kcl)).
//
// FULL TILES ONLY in this version: M % 16, N % 64, K % 64 must all be zero;
// anything else is refused with C untouched. Tail support lives in the
// ncblock-apack4za-tail kernel and can be carried over once this macroflow has
// been measured.
//
// Pack counts are counted, not assumed. Whether either operand is packed once
// per GEMM is a property of the shape and the policy, reported per run.
// =============================================================================

// Output group g of the vertical drain is four consecutive vertical slices of
// one ZA tile: tile g/4, first column 4*(g%4). Both must be compile-time
// constants -- the tile operand of svread_ver_za32 is an immediate.
#define PF_T(g) ((g) / 4)
#define PF_C(g) (4 * ((g) % 4))
#define PF_GROUP(g)                                                            \
    svcreate4_f32(svread_ver_za32_f32_m(zu, pg, PF_T(g), PF_C(g) + 0),         \
                  svread_ver_za32_f32_m(zu, pg, PF_T(g), PF_C(g) + 1),         \
                  svread_ver_za32_f32_m(zu, pg, PF_T(g), PF_C(g) + 2),         \
                  svread_ver_za32_f32_m(zu, pg, PF_T(g), PF_C(g) + 3))

// One steady step: store the group a Z set already holds, then refill that same
// Z set with the group two ahead. Store operand is consumed before the refill,
// so the two x4 store streams never wait on each other.
#define PF_DRAIN(g)                                                            \
    svst1_f32_x4(p4, out + (g) * 64, s0);                                      \
    s0 = PF_GROUP((g) + 2);                                                    \
    svst1_f32_x4(p4, out + ((g) + 1) * 64, s1);                                \
    s1 = PF_GROUP((g) + 3);

#define PF_PACK_A_KP_CHUNK()                                           \
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
        svfloat32x4_t s0 = PF_GROUP(0);                                \
        svfloat32x4_t s1 = PF_GROUP(1);                                \
        /* Static unroll: the tile operand of svread_ver_za32 must */  \
        /* be an immediate, so g cannot be a runtime variable. */      \
        PF_DRAIN(0)  PF_DRAIN(2)  PF_DRAIN(4)  PF_DRAIN(6)             \
        PF_DRAIN(8)  PF_DRAIN(10) PF_DRAIN(12)                         \
        /* DRAIN: s0 holds group 14 (ZA3 vertical 8..11), */           \
        /* s1 holds group 15 (ZA3 vertical 12..15). No further */      \
        /* extraction -- the two stores are back to back. */           \
        svst1_f32_x4(p4, out + 14 * 64, s0);                           \
        svst1_f32_x4(p4, out + 15 * 64, s1);                           \
    } while (0)

namespace SMEKernels1x4PanelFuse {

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

    // Cycle counter, readable in streaming mode with one instruction. Used only
    // when Stats::timed is set, and only at phase boundaries, never per tile.
    static inline uint64_t rd_cycles() { uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v; }
    static inline uint64_t rd_freq()   { uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v; }

    Support classify(size_t M, size_t K, size_t N, const Params& p) {
        if (M == 0 || K == 0 || N == 0)          return Support::Unsupported;
        if (M % 16 != 0)                          return Support::UnsupportedMTail;
        if (N % 64 != 0)                          return Support::UnsupportedNTail;
        if (K % 64 != 0)                          return Support::UnsupportedKTail;
        if (p.Kc == 0 || p.Kc % 64 != 0)          return Support::BadParams;
        if (p.budget_bytes < 320 * 64)            return Support::BadParams;   // one pair at kcl=64
        return Support::Native;
    }

    // What the macroflow will do for a shape, without running it.
    Plan plan(size_t M, size_t K, size_t N, const Params& p) {
        Plan pl{};
        const size_t Kc  = std::min(p.Kc, K);
        const size_t kcl = Kc;                                   // first slice
        const size_t tb  = p.budget_bytes / (320 * kcl);
        const size_t nA = M / 16, nB = N / 64;
        pl.t_budget = tb;
        if      (nA <= tb) pl.retained = Retained::A;
        else if (nB <= tb) pl.retained = Retained::B;
        else               pl.retained = (p.both_large == BothLarge::RetainB) ? Retained::B : Retained::A;
        pl.forced_by_policy = !(nA <= tb) && !(nB <= tb);
        pl.t_outer = std::min(tb, pl.retained == Retained::A ? nA : nB);
        pl.t_inner = std::min(pl.t_outer, pl.retained == Retained::A ? nB : nA);
        pl.packed_A_bytes = p.budget_bytes / 5;
        pl.packed_B_bytes = p.budget_bytes - pl.packed_A_bytes;
        return pl;
    }

    __arm_locally_streaming __arm_new("za")
    __attribute__((noinline))
    static bool run_streaming(const float* A, const float* B, float* C,
                              size_t M, size_t K, size_t N,
                              float* packed_A, float* packed_B,
                              const Params& p, Stats* st) {
        if (static_cast<size_t>(svcntsw()) != 16) return false;

        const svbool_t  pg = svptrue_b32();
        const svcount_t p4 = svptrue_c32();
        const svfloat32_t zu = svundef_f32();
        const size_t Kc = std::min(p.Kc, K);
        const size_t nA_total = M / 16, nB_total = N / 64;
        const bool timed = st && st->timed;
        uint64_t t_pack = 0, t_comp = 0;

        // ---- panel preparation, existing packers, unchanged --------------------
        // A panel a (rows 16a..16a+15) of slice (kk, kcl) -> slot s of packed_A.
        auto prep_A = [&](size_t a, size_t s, size_t kk, size_t kcl) __arm_streaming __arm_inout("za") {
            float* const pA = packed_A + s * (kcl * 16);
            const float* const arow_base = A + (16 * a) * K + kk;
            for (size_t kp = 0; kp < kcl; kp += 64) {
                const float* const arow = arow_base + kp;
                PF_PACK_A_KP_CHUNK();
            }
            if (st) st->a_panel_packs++;
        };
        // B panel b (cols 64b..64b+63) -> slot s of packed_B.
        auto prep_B = [&](size_t b, size_t s, size_t kk, size_t kcl) __arm_streaming {
            pack_B_streaming(B, packed_B + s * (kcl * 64), 64, kcl, kk, N, 64 * b);
            if (st) st->b_panel_packs++;
        };

        for (size_t kk = 0; kk < K; kk += Kc) {
            const size_t kcl = std::min(Kc, K - kk);
            const bool first_k = (kk == 0);
            const size_t tb = p.budget_bytes / (320 * kcl);

            // ---- orientation: which operand is retained for this slice -------
            Retained ret;
            if      (nA_total <= tb) ret = Retained::A;
            else if (nB_total <= tb) ret = Retained::B;
            else ret = (p.both_large == BothLarge::RetainB) ? Retained::B : Retained::A;

            const size_t outer_total = (ret == Retained::A) ? nA_total : nB_total;
            const size_t inner_total = (ret == Retained::A) ? nB_total : nA_total;

            for (size_t o0 = 0; o0 < outer_total; ) {
                const size_t t_o = std::min(tb, outer_total - o0);

                // Prepare the retained region once for the whole inner traversal.
                uint64_t c0 = timed ? rd_cycles() : 0;
                for (size_t s = 0; s < t_o; s++)
                    (ret == Retained::A) ? prep_A(o0 + s, s, kk, kcl) : prep_B(o0 + s, s, kk, kcl);
                if (timed) t_pack += rd_cycles() - c0;

                for (size_t i0 = 0; i0 < inner_total; ) {
                    // Equal panel counts: the 1:4 rule. Never more of the inner
                    // operand than the retained region can pair with.
                    const size_t t_i = std::min(t_o, inner_total - i0);

                    c0 = timed ? rd_cycles() : 0;
                    for (size_t s = 0; s < t_i; s++)
                        (ret == Retained::A) ? prep_B(i0 + s, s, kk, kcl) : prep_A(i0 + s, s, kk, kcl);
                    if (timed) t_pack += rd_cycles() - c0;
                    if (st) st->phases++;

                    // ---- COMPUTE PHASE: every pair, no packing in here ---------
                    // Order is B panel outer, A panel inner, as the ledger states.
                    const size_t nb = (ret == Retained::A) ? t_i : t_o;
                    const size_t na = (ret == Retained::A) ? t_o : t_i;
                    const size_t b_base = (ret == Retained::A) ? i0 : o0;
                    const size_t a_base = (ret == Retained::A) ? o0 : i0;

                    c0 = timed ? rd_cycles() : 0;
                    for (size_t bi = 0; bi < nb; bi++) {
                        float* const pB = packed_B + bi * (kcl * 64);
                        for (size_t ai = 0; ai < na; ai++) {
                            float* const pA = packed_A + ai * (kcl * 16);
                            svzero_za();
                            micro_kernel_1x4(pA, pB, kcl);
                            float* const dstC = C + (16 * (a_base + ai)) * N + 64 * (b_base + bi);
                            if (first_k) store_za(dstC, N);
                            else         store_za_add(dstC, N);
                        }
                    }
                    if (timed) t_comp += rd_cycles() - c0;
                    if (st) st->microkernel_calls += nb * na;

                    i0 += t_i;
                }
                o0 += t_o;
            }
        }
        if (st) {
            st->pack_cycles = t_pack; st->compute_cycles = t_comp;
            st->counter_hz = rd_freq();
        }
        return true;
    }

    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N,
                               const Params& p, Stats* st) {
        const Support s = classify(M, K, N, p);
        if (s != Support::Native) return s;
        if (st) { *st = Stats{}; st->timed = st_timed_default; }

        // One A panel is a quarter of one B panel in bytes, and panels are
        // prepared in equal numbers, so the budget splits 1:4 by construction.
        const size_t a_bytes = p.budget_bytes / 5;
        const size_t b_bytes = p.budget_bytes - a_bytes;
        AlignedBuffer packed_A(static_cast<float*>(std::aligned_alloc(64, ((a_bytes + 63) / 64) * 64)));
        AlignedBuffer packed_B(static_cast<float*>(std::aligned_alloc(64, ((b_bytes + 63) / 64) * 64)));
        if (!packed_A || !packed_B) return Support::AllocationFailed;

        return run_streaming(A, B, C, M, K, N, packed_A.get(), packed_B.get(), p, st)
                   ? Support::Native : Support::UnsupportedVectorLength;
    }

    bool st_timed_default = false;
    void set_phase_timing(bool on) { st_timed_default = on; }

} // namespace SMEKernels1x4PanelFuse
