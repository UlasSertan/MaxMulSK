#include "sme-1x4-sym.hpp"

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <memory>
#include <algorithm>

#define RESTRICT __restrict__

// =============================================================================
// SME 1x4-sym
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

namespace SMEKernels1x4Sym {

    // =========================================================================
    // PACK A  (M x K) -> single-panel layout (GS = SVL)
    // Each m-step (SVL rows) gets its own contiguous K_curr * SVL block.
    // Within a block per k-step: SVL contiguous floats = one column of A.
    // Micro-kernel can then read 4 consecutive k-steps with one svld1_f32_x4.
    // =========================================================================
    __attribute__((noinline))
    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming {
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

            // Main loop: butterfly-transpose SVLxSVL block into SVL column vecs
            for (; k < K_full; k += SVL) {
                svfloat32_t r0  = svld1_f32(p0, row_base + k);
                svfloat32_t r1  = svld1_f32(p1, row_base + k +  K);
                svfloat32_t r2  = svld1_f32(p2, row_base + k +  2*K);
                svfloat32_t r3  = svld1_f32(p3, row_base + k +  3*K);
                svfloat32_t r4  = svld1_f32(p4, row_base + k +  4*K);
                svfloat32_t r5  = svld1_f32(p5, row_base + k +  5*K);
                svfloat32_t r6  = svld1_f32(p6, row_base + k +  6*K);
                svfloat32_t r7  = svld1_f32(p7, row_base + k +  7*K);
                svfloat32_t r8  = svld1_f32(p8, row_base + k +  8*K);
                svfloat32_t r9  = svld1_f32(p9, row_base + k +  9*K);
                svfloat32_t r10 = svld1_f32(p10, row_base + k + 10*K);
                svfloat32_t r11 = svld1_f32(p11, row_base + k + 11*K);
                svfloat32_t r12 = svld1_f32(p12, row_base + k + 12*K);
                svfloat32_t r13 = svld1_f32(p13, row_base + k + 13*K);
                svfloat32_t r14 = svld1_f32(p14, row_base + k + 14*K);
                svfloat32_t r15 = svld1_f32(p15, row_base + k + 15*K);

                svfloat32_t s1_0L = svzip1_f32(r0,  r8);
                svfloat32_t s1_0H = svzip2_f32(r0,  r8);
                svfloat32_t s1_1L = svzip1_f32(r1,  r9);
                svfloat32_t s1_1H = svzip2_f32(r1,  r9);
                svfloat32_t s1_2L = svzip1_f32(r2,  r10);
                svfloat32_t s1_2H = svzip2_f32(r2,  r10);
                svfloat32_t s1_3L = svzip1_f32(r3,  r11);
                svfloat32_t s1_3H = svzip2_f32(r3,  r11);
                svfloat32_t s1_4L = svzip1_f32(r4,  r12);
                svfloat32_t s1_4H = svzip2_f32(r4,  r12);
                svfloat32_t s1_5L = svzip1_f32(r5, r13);
                svfloat32_t s1_5H = svzip2_f32(r5, r13);
                svfloat32_t s1_6L = svzip1_f32(r6, r14);
                svfloat32_t s1_6H = svzip2_f32(r6, r14);
                svfloat32_t s1_7L = svzip1_f32(r7, r15);
                svfloat32_t s1_7H = svzip2_f32(r7, r15);

                svfloat32_t s2_0L = svzip1_f32(s1_0L, s1_4L);
                svfloat32_t s2_0H = svzip2_f32(s1_0L, s1_4L);
                svfloat32_t s2_1L = svzip1_f32(s1_0H, s1_4H);
                svfloat32_t s2_1H = svzip2_f32(s1_0H, s1_4H);
                svfloat32_t s2_2L = svzip1_f32(s1_1L, s1_5L);
                svfloat32_t s2_2H = svzip2_f32(s1_1L, s1_5L);
                svfloat32_t s2_3L = svzip1_f32(s1_1H, s1_5H);
                svfloat32_t s2_3H = svzip2_f32(s1_1H, s1_5H);
                svfloat32_t s2_4L = svzip1_f32(s1_2L, s1_6L);
                svfloat32_t s2_4H = svzip2_f32(s1_2L, s1_6L);
                svfloat32_t s2_5L = svzip1_f32(s1_2H, s1_6H);
                svfloat32_t s2_5H = svzip2_f32(s1_2H, s1_6H);
                svfloat32_t s2_6L = svzip1_f32(s1_3L, s1_7L);
                svfloat32_t s2_6H = svzip2_f32(s1_3L, s1_7L);
                svfloat32_t s2_7L = svzip1_f32(s1_3H, s1_7H);
                svfloat32_t s2_7H = svzip2_f32(s1_3H, s1_7H);

                svfloat32_t s3_0L = svzip1_f32(s2_0L, s2_4L);
                svfloat32_t s3_0H = svzip2_f32(s2_0L, s2_4L);
                svfloat32_t s3_1L = svzip1_f32(s2_0H, s2_4H);
                svfloat32_t s3_1H = svzip2_f32(s2_0H, s2_4H);
                svfloat32_t s3_2L = svzip1_f32(s2_1L, s2_5L);
                svfloat32_t s3_2H = svzip2_f32(s2_1L, s2_5L);
                svfloat32_t s3_3L = svzip1_f32(s2_1H, s2_5H);
                svfloat32_t s3_3H = svzip2_f32(s2_1H, s2_5H);
                svfloat32_t s3_4L = svzip1_f32(s2_2L, s2_6L);
                svfloat32_t s3_4H = svzip2_f32(s2_2L, s2_6L);
                svfloat32_t s3_5L = svzip1_f32(s2_2H, s2_6H);
                svfloat32_t s3_5H = svzip2_f32(s2_2H, s2_6H);
                svfloat32_t s3_6L = svzip1_f32(s2_3L, s2_7L);
                svfloat32_t s3_6H = svzip2_f32(s2_3L, s2_7L);
                svfloat32_t s3_7L = svzip1_f32(s2_3H, s2_7H);
                svfloat32_t s3_7H = svzip2_f32(s2_3H, s2_7H);

                svfloat32_t col0  = svzip1_f32(s3_0L, s3_4L);
                svfloat32_t col1  = svzip2_f32(s3_0L, s3_4L);
                svfloat32_t col2  = svzip1_f32(s3_0H, s3_4H);
                svfloat32_t col3  = svzip2_f32(s3_0H, s3_4H);
                svfloat32_t col4  = svzip1_f32(s3_1L, s3_5L);
                svfloat32_t col5  = svzip2_f32(s3_1L, s3_5L);
                svfloat32_t col6  = svzip1_f32(s3_1H, s3_5H);
                svfloat32_t col7  = svzip2_f32(s3_1H, s3_5H);
                svfloat32_t col8  = svzip1_f32(s3_2L, s3_6L);
                svfloat32_t col9  = svzip2_f32(s3_2L, s3_6L);
                svfloat32_t col10 = svzip1_f32(s3_2H, s3_6H);
                svfloat32_t col11 = svzip2_f32(s3_2H, s3_6H);
                svfloat32_t col12 = svzip1_f32(s3_3L, s3_7L);
                svfloat32_t col13 = svzip2_f32(s3_3L, s3_7L);
                svfloat32_t col14 = svzip1_f32(s3_3H, s3_7H);
                svfloat32_t col15 = svzip2_f32(s3_3H, s3_7H);

                float* out = block_base + k * GS;
                svst1_f32(pg, out +  0*GS, col0);
                svst1_f32(pg, out +  1*GS, col1);
                svst1_f32(pg, out +  2*GS, col2);
                svst1_f32(pg, out +  3*GS, col3);
                svst1_f32(pg, out +  4*GS, col4);
                svst1_f32(pg, out +  5*GS, col5);
                svst1_f32(pg, out +  6*GS, col6);
                svst1_f32(pg, out +  7*GS, col7);
                svst1_f32(pg, out +  8*GS, col8);
                svst1_f32(pg, out +  9*GS, col9);
                svst1_f32(pg, out + 10*GS, col10);
                svst1_f32(pg, out + 11*GS, col11);
                svst1_f32(pg, out + 12*GS, col12);
                svst1_f32(pg, out + 13*GS, col13);
                svst1_f32(pg, out + 14*GS, col14);
                svst1_f32(pg, out + 15*GS, col15);
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
    void micro_kernel_1x4(float* RESTRICT packed_A, float* RESTRICT packed_B, float* RESTRICT C,
                          size_t K_curr, size_t wide_of_C) __arm_out("za") __arm_streaming {
        svzero_za();
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

        svfloat32_t inactive = svundef_f32();

        // Tiles stack horizontally: ZA_i -> C[0..SVL, i*SVL..(i+1)*SVL]
        #define STORE_ZA_TILE(TILE)                                              \
        do {                                                                     \
            float* C_tile = C + (TILE) * SVL;                                    \
            for (int i = 0; i < (int)SVL; i++) {                                 \
                svfloat32_t result = svread_hor_za32_f32_m(inactive, pg, TILE, i);\
                float* ptr = C_tile + i * wide_of_C;                             \
                svfloat32_t existing = svld1_f32(pg, ptr);                       \
                svst1_f32(pg, ptr, svadd_f32_x(pg, existing, result));           \
            }                                                                    \
        } while (0)

        STORE_ZA_TILE(0);
        STORE_ZA_TILE(1);
        STORE_ZA_TILE(2);
        STORE_ZA_TILE(3);

        #undef STORE_ZA_TILE
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
        constexpr size_t K_tile = 2048;
        constexpr size_t N_tile = 64;

        const size_t M_step = 1 * SVL;
        const size_t N_step = 4 * SVL;

        AlignedBuffer packed_A(static_cast<float*>(
            std::aligned_alloc(64, M_tile * K_tile * sizeof(float))));
        AlignedBuffer packed_B(static_cast<float*>(
            std::aligned_alloc(64, K_tile * N_tile * sizeof(float))));

        AlignedBuffer C_scratch(static_cast<float*>(
            std::aligned_alloc(64, M_step * N_step * sizeof(float))));

        for (size_t m = 0; m < M; m += M_tile) {
            size_t mc = std::min(M_tile, M - m);
            for (size_t k = 0; k < K; k += K_tile) {
                size_t kc = std::min(K_tile, K - k);
                pack_A_streaming(A, packed_A.get(), mc, kc, m, k, K);

                for (size_t n = 0; n < N; n += N_tile) {
                    size_t nc = std::min(N_tile, N - n);
                    pack_B_streaming(B, packed_B.get(), nc, kc, k, N, n);

                    for (size_t jr = 0; jr < nc; jr += N_step) {
                        size_t n_rem = nc - jr;
                        for (size_t ir = 0; ir < mc; ir += M_step) {
                            size_t m_rem = mc - ir;

                            bool m_tail = m_rem < M_step;
                            bool n_tail = n_rem < N_step;

                            if (!m_tail && !n_tail) {
                                micro_kernel_1x4(
                                    packed_A.get() + ir * kc,
                                    packed_B.get() + jr * kc,
                                    C + (m + ir) * N + (n + jr),
                                    kc, N);
                            } else {
                                size_t rows = std::min(m_rem, M_step);
                                size_t cols = std::min(n_rem, N_step);

                                {
                                    svbool_t pg_z = svptrue_b32();
                                    svfloat32_t zero = svdup_f32(0.0f);
                                    float* zp = C_scratch.get();
                                    for (size_t i = 0; i < M_step * N_step; i += SVL)
                                        svst1_f32(pg_z, zp + i, zero);
                                }

                                micro_kernel_1x4(
                                    packed_A.get() + ir * kc,
                                    packed_B.get() + jr * kc,
                                    C_scratch.get(),
                                    kc, N_step);

                                float* C_dst = C + (m + ir) * N + (n + jr);
                                const float* src = C_scratch.get();
                                for (size_t row = 0; row < rows; row++) {
                                    for (size_t col = 0; col < cols; col++)
                                        C_dst[row * N + col] += src[row * N_step + col];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

} // namespace SMEKernels1x4Sym
