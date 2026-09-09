#include "sme-1x4-acc-kcout-bdirect.hpp"

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <memory>
#include <algorithm>

#define RESTRICT __restrict__

// =============================================================================
// SME 1x4-Acc  (experimental; copied from 1x4-sym)
//
// Identical to 1x4-sym except that ZA zeroing and the ZA->C writeback have been
// lifted out of the micro-kernel and into the driver, so ZA stays live across
// the whole accumulation instead of being zeroed and drained per call.
//
// Same packing layout as SMEKernels1x4 (pack_A: single SVL-wide panel per
// m-step, GS=SVL ;  pack_B: 4-panel interleaved per k-step, GS=4*SVL), but the
// micro-kernel mirrors 4x1's load schedule exactly under the A<->B swap:
//
//   4x1     per 4 k-steps : 4 A-x4 loads  + 1 B-x4 load  + 16 svmopa
//   1x4-sym per 4 k-steps : 1 A-x4 load   + 4 B-x4 loads + 16 svmopa
//
// Inside one k-step all 4 svmopa write different ZA tiles (ZA_i = a*b_i) so
// the same-tile gap stays 4 slots wide, matching 4x1.
// =============================================================================

namespace SMEKernels1x4AccKcOutBDirect {

    // =========================================================================
    // PACK A  (M x K) -> single-panel layout (GS = SVL)
    // Each m-step (SVL rows) gets its own contiguous K_curr * SVL block.
    // Within a block per k-step: SVL contiguous floats = one column of A.
    // Micro-kernel can then read 4 consecutive k-steps with one svld1_f32_x4.
    // =========================================================================
    __attribute__((noinline))
    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming __arm_out("za") {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const size_t K_full = (K_curr / SVL) * SVL;
        const size_t GS = SVL; // group stride: single panel per k-step

        const svbool_t pg = svptrue_b32();
        const svbool_t pfalse = svpfalse_b();

        for (size_t m = 0; m < M_curr; m += SVL) {
            const float* row_base = A + (m + curr_row) * K + curr_col;
            size_t m_idx = m / SVL;
            float* block_base = packed_A + m_idx * SVL * K_curr;
            size_t k = 0;
            size_t rows_here = std::min(SVL, M_curr - m);

            svbool_t p0 = (0 < rows_here) ? pg : pfalse;
            svbool_t p1 = (1 < rows_here) ? pg : pfalse;
            svbool_t p2 = (2 < rows_here) ? pg : pfalse;
            svbool_t p3 = (3 < rows_here) ? pg : pfalse;
            svbool_t p4 = (4 < rows_here) ? pg : pfalse;
            svbool_t p5 = (5 < rows_here) ? pg : pfalse;
            svbool_t p6 = (6 < rows_here) ? pg : pfalse;
            svbool_t p7 = (7 < rows_here) ? pg : pfalse;
            svbool_t p8 = (8 < rows_here) ? pg : pfalse;
            svbool_t p9 = (9 < rows_here) ? pg : pfalse;
            svbool_t p10 = (10 < rows_here) ? pg : pfalse;
            svbool_t p11 = (11 < rows_here) ? pg : pfalse;
            svbool_t p12 = (12 < rows_here) ? pg : pfalse;
            svbool_t p13 = (13 < rows_here) ? pg : pfalse;
            svbool_t p14 = (14 < rows_here) ? pg : pfalse;
            svbool_t p15 = (15 < rows_here) ? pg : pfalse;

            // A packed via ZA as a transpose buffer instead of an SVE butterfly:
            // load 16 rows of A as ZA horizontal slices, read them back as
            // vertical slices, which is the transpose for free. Same output
            // layout as the butterfly version it replaces.
            //
            // Safe here because ZA holds nothing live at this point: the driver
            // packs A and B for a block before it zeroes ZA and starts
            // accumulating, and never packs while an accumulation is in flight.
            //
            // Predicated-off horizontal loads MERGE, so a short m-tail would
            // otherwise read whatever the previous block left in ZA. Zero once.
            if (rows_here < SVL) {
                svzero_za();
            }

            for (; k < K_full; k += SVL) {
                svld1_hor_za32(0, 0,  p0,  row_base + k +  0*K);
                svld1_hor_za32(0, 1,  p1,  row_base + k +  1*K);
                svld1_hor_za32(0, 2,  p2,  row_base + k +  2*K);
                svld1_hor_za32(0, 3,  p3,  row_base + k +  3*K);
                svld1_hor_za32(0, 4,  p4,  row_base + k +  4*K);
                svld1_hor_za32(0, 5,  p5,  row_base + k +  5*K);
                svld1_hor_za32(0, 6,  p6,  row_base + k +  6*K);
                svld1_hor_za32(0, 7,  p7,  row_base + k +  7*K);
                svld1_hor_za32(0, 8,  p8,  row_base + k +  8*K);
                svld1_hor_za32(0, 9,  p9,  row_base + k +  9*K);
                svld1_hor_za32(0, 10, p10, row_base + k + 10*K);
                svld1_hor_za32(0, 11, p11, row_base + k + 11*K);
                svld1_hor_za32(0, 12, p12, row_base + k + 12*K);
                svld1_hor_za32(0, 13, p13, row_base + k + 13*K);
                svld1_hor_za32(0, 14, p14, row_base + k + 14*K);
                svld1_hor_za32(0, 15, p15, row_base + k + 15*K);

                float* out = block_base + k * GS;
                svst1_ver_za32(0, 0,  pg, out +  0*GS);
                svst1_ver_za32(0, 1,  pg, out +  1*GS);
                svst1_ver_za32(0, 2,  pg, out +  2*GS);
                svst1_ver_za32(0, 3,  pg, out +  3*GS);
                svst1_ver_za32(0, 4,  pg, out +  4*GS);
                svst1_ver_za32(0, 5,  pg, out +  5*GS);
                svst1_ver_za32(0, 6,  pg, out +  6*GS);
                svst1_ver_za32(0, 7,  pg, out +  7*GS);
                svst1_ver_za32(0, 8,  pg, out +  8*GS);
                svst1_ver_za32(0, 9,  pg, out +  9*GS);
                svst1_ver_za32(0, 10, pg, out + 10*GS);
                svst1_ver_za32(0, 11, pg, out + 11*GS);
                svst1_ver_za32(0, 12, pg, out + 12*GS);
                svst1_ver_za32(0, 13, pg, out + 13*GS);
                svst1_ver_za32(0, 14, pg, out + 14*GS);
                svst1_ver_za32(0, 15, pg, out + 15*GS);
            }

            // K tail: one column at a time via tmp buffer
            float tmp[16];
            svst1_f32(pg, tmp, svdup_f32(0.0f));
            for (; k < K_curr; k++) {
                for (size_t row = 0; row < rows_here; row++)
                    tmp[row] = row_base[k + row * K];
                svfloat32_t col_vec = svld1_f32(pg, tmp);
                svst1_f32(pg, block_base + k * GS, col_vec);
                svst1_f32(pg, tmp, svdup_f32(0.0f));
            }
        }
    }

    // =========================================================================
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

    // =========================================================================
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
    // B_stride is the distance in floats between consecutive k-steps of the B
    // operand. It used to be hard-wired to 4*SVL because B always arrived
    // pre-packed. It is a parameter now so the same code can walk B in its
    // NATIVE row-major layout, where consecutive k-steps are N apart.
    //
    // That substitution is legal because pack_B never reordered anything: its
    // inner loop read 4*SVL consecutive floats of B's row k and wrote them to
    // 4*SVL consecutive floats of the panel. The bytes the micro-kernel wants
    // for one k-step are already contiguous in B; only the stride between
    // k-steps differs.
    void micro_kernel_1x4(float* RESTRICT packed_A, const float* RESTRICT packed_B,
                          size_t K_curr, size_t B_stride) __arm_inout("za") __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t pg = svptrue_b32();
        const svcount_t pg4 = svptrue_c32();
        const size_t GS_B = B_stride; // floats between consecutive k-steps of B

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

    // =========================================================================
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

    // =========================================================================
    // Accumulating ZA -> C writeback, for every K panel after the first.
    //
    // The outer Kc loop makes each C tile the destination of K/Kc separate
    // partial sums, so only the FIRST panel may overwrite; the rest must fold
    // into what is already there. This is the read-modify-write that §0.10
    // removed, deliberately reintroduced -- but paid K/Kc-1 times per tile
    // instead of once per K_tile, which is the trade the outer loop is making.
    // =========================================================================
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

    // =========================================================================
    // MAIN DRIVER -- B-inner-packing + 1x4-sym kernel
    // Loop order: M -> K -> pack_A -> N -> pack_B -> (jr, ir) -> micro_kernel
    // pack_A once per (m, k); pack_B once per (m, k, n).
    // Edge tiles (m_rem < M_step || n_rem < N_step) use scratch buffer.
    // =========================================================================
    struct FreeDeleter { void operator()(void* p) { std::free(p); } };
    using AlignedBuffer = std::unique_ptr<float[], FreeDeleter>;

    __arm_locally_streaming __arm_new("za")
    __attribute__((noinline))
    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N) {
        const size_t SVL = static_cast<size_t>(svcntsw());

        constexpr size_t M_tile = 1024;
        constexpr size_t K_tile = 2048;   // KC: ZA accumulation chunk, innermost
        constexpr size_t N_tile = 64;

        const size_t M_step = 1 * SVL;
        const size_t N_step = 4 * SVL;

        // OUTER K PANEL -- the change that defines this variant.
        //
        // The whole M x N sweep now runs once per Kc slice of K, so the packed
        // buffers scale with Kc instead of with K. That is the entire point:
        // keep the packed working set inside L2 when K is large. The price is
        // that every C tile is revisited K/Kc times instead of once.
        //
        // Kc is a tuning knob, not a correctness one. 2048 matches K_tile, so at
        // this setting the inner k loop degenerates to one call per panel.
        constexpr size_t Kc = 2048;

        AlignedBuffer packed_A(static_cast<float*>(
            std::aligned_alloc(64, M_tile * Kc * sizeof(float))));
        // No packed_B. B is consumed in place; only the ragged N tail is packed,
        // into a single N_step-wide panel, because a partial column block needs
        // its dead lanes zeroed and B in place cannot provide that.
        AlignedBuffer packed_B_tail(static_cast<float*>(
            std::aligned_alloc(64, N_step * Kc * sizeof(float))));

        AlignedBuffer C_scratch(static_cast<float*>(
            std::aligned_alloc(64, M_step * N_step * sizeof(float))));

        for (size_t kk = 0; kk < K; kk += Kc) {
            const size_t kcl = std::min(Kc, K - kk);

            // Only the first K panel may overwrite C; later panels fold into
            // what the earlier ones left. The kernel as a whole still computes
            // C = A*B, so the caller must NOT pre-zero C -- same contract as
            // 1x4-Acc, even though C is now touched more than once.
            const bool first_k = (kk == 0);

            for (size_t m = 0; m < M; m += M_tile) {
                size_t mc = std::min(M_tile, M - m);
                pack_A_streaming(A, packed_A.get(), mc, kcl, m, kk, K);

                for (size_t n = 0; n < N; n += N_tile) {
                    size_t nc = std::min(N_tile, N - n);

                    for (size_t jr = 0; jr < nc; jr += N_step) {
                        size_t n_rem = nc - jr;

                        // B's source for this column block. A full block reads B
                        // directly at row kk, column n+jr, stepping N per
                        // k-step. A ragged one goes through the shipped packer
                        // into the tail panel, stepping N_step.
                        const float* b_src;
                        size_t b_stride;
                        if (n_rem >= N_step) {
                            b_src = B + kk * N + (n + jr);
                            b_stride = N;
                        } else {
                            pack_B_streaming(B, packed_B_tail.get(), n_rem, kcl,
                                             kk, N, n + jr);
                            b_src = packed_B_tail.get();
                            b_stride = N_step;
                        }

                        for (size_t ir = 0; ir < mc; ir += M_step) {
                            size_t m_rem = mc - ir;

                            bool m_tail = m_rem < M_step;
                            bool n_tail = n_rem < N_step;

                            // Packed panels are kcl deep now, so the per-tile
                            // stride is ir * kcl / jr * kcl, not ir * K.
                            svzero_za();
                            for (size_t k = 0; k < kcl; k += K_tile) {
                                size_t kc = std::min(K_tile, kcl - k);
                                micro_kernel_1x4(
                                    packed_A.get() + ir * kcl + k * SVL,
                                    b_src + k * b_stride,
                                    kc, b_stride);
                            }

                            if (!m_tail && !n_tail) {
                                float* dst = C + (m + ir) * N + (n + jr);
                                if (first_k) store_za(dst, N);
                                else         store_za_add(dst, N);
                            } else {
                                size_t rows = std::min(m_rem, M_step);
                                size_t cols = std::min(n_rem, N_step);

                                store_za(C_scratch.get(), N_step);

                                // Predicated SVE, never a scalar copy: a scalar
                                // loop here gets recognised as memcpy and
                                // lowered to __arm_sc_memcpy, which has no
                                // implementation in streaming mode (BUG-5).
                                float* C_dst = C + (m + ir) * N + (n + jr);
                                const float* src = C_scratch.get();
                                for (size_t row = 0; row < rows; row++) {
                                    const float* sp = src + row * N_step;
                                    float* dst_row = C_dst + row * N;
                                    for (size_t col = 0; col < cols; col += SVL) {
                                        svbool_t pg_c = svwhilelt_b32_u64(col, cols);
                                        svfloat32_t v = svld1_f32(pg_c, sp + col);
                                        if (!first_k)
                                            v = svadd_f32_x(pg_c, v,
                                                            svld1_f32(pg_c, dst_row + col));
                                        svst1_f32(pg_c, dst_row + col, v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

} // namespace SMEKernels1x4AccKcOutBDirect
