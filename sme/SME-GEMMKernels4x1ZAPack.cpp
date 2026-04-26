#include "SME-GEMMKernels4x1ZAPack.hpp"

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <memory>
#include <algorithm>

#define RESTRICT __restrict__

namespace SMEKernels4x1ZAPack {

    // =========================================================================
    // PACK A  (M x K) → 4-panel interleaved layout
    // Layout per k-step: [panel0_SVL | panel1_SVL | panel2_SVL | panel3_SVL]
    // Micro-kernel reads all 4 panels with a single svld1_f32_x4 per k-step
    // =========================================================================
    __attribute__((noinline))
    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming __arm_out("za") {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const size_t K_full = (K_curr / SVL) * SVL;
        const size_t GS = 4 * SVL; // group stride: 4 panels interleaved per k-step

        const svbool_t pg = svptrue_b32();
        const svbool_t pfalse = svpfalse_b();

        for (size_t m = 0; m < M_curr; m += SVL) {
            const float* row_base = A + (m + curr_row) * K + curr_col;
            size_t p = m / SVL; // panel index within group (0-3)
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

            // M-tail: predicated-off horizontal loads are merging (keep prior ZA contents).
            // Zero tile 0 once per m-block so unwritten slices contribute zeros on read-out.
            if (rows_here < SVL) {
                svzero_za();
            }

            for (; k < K_full; k += SVL) {
                // Load 16 rows of A as horizontal slices of ZA tile 0
                svld1_hor_za32(0, 0,  p0,  row_base + k + 0*K);
                svld1_hor_za32(0, 1,  p1,  row_base + k + 1*K);
                svld1_hor_za32(0, 2,  p2,  row_base + k + 2*K);
                svld1_hor_za32(0, 3,  p3,  row_base + k + 3*K);
                svld1_hor_za32(0, 4,  p4,  row_base + k + 4*K);
                svld1_hor_za32(0, 5,  p5,  row_base + k + 5*K);
                svld1_hor_za32(0, 6,  p6,  row_base + k + 6*K);
                svld1_hor_za32(0, 7,  p7,  row_base + k + 7*K);
                svld1_hor_za32(0, 8,  p8,  row_base + k + 8*K);
                svld1_hor_za32(0, 9,  p9,  row_base + k + 9*K);
                svld1_hor_za32(0, 10, p10, row_base + k + 10*K);
                svld1_hor_za32(0, 11, p11, row_base + k + 11*K);
                svld1_hor_za32(0, 12, p12, row_base + k + 12*K);
                svld1_hor_za32(0, 13, p13, row_base + k + 13*K);
                svld1_hor_za32(0, 14, p14, row_base + k + 14*K);
                svld1_hor_za32(0, 15, p15, row_base + k + 15*K);

                // Read 16 columns as vertical slices → transposed tile in packed_A layout
                float* out = packed_A + k * GS + p * SVL;
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

            // K tail: one column at a time via tmp buffer (ZA trick doesn't help for 1-wide strips)
            float tmp[16];
            svst1_f32(pg, tmp, svdup_f32(0.0f));
            for (; k < K_curr; k++) {
                for (size_t row = 0; row < rows_here; row++)
                    tmp[row] = row_base[k + row * K];
                svfloat32_t col_vec = svld1_f32(pg, tmp);
                svst1_f32(pg, packed_A + k * GS + p * SVL, col_vec);
                svst1_f32(pg, tmp, svdup_f32(0.0f));
            }
        }
    }

    // =========================================================================
    // PACK B  (K x N) → panel-major, SVL-wide panels
    // =========================================================================
    __attribute__((noinline))
    void pack_B_streaming(const float* B, float* packed_B,
                          size_t N_curr, size_t K_curr,
                          size_t curr_row, size_t N, size_t curr_col) __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svcount_t full_mask4 = svptrue_c32();
        const svbool_t  pg         = svptrue_b32();
        const size_t vector_size = SVL * 4;
        const size_t N_full = (N_curr / vector_size) * vector_size;
        const size_t panel_stride = K_curr * SVL;

        size_t n = 0;

        // Main loop: process 4*SVL columns at a time using x4 load
        for (; n < N_full; n += vector_size) {
            float* dst0 = packed_B + n * K_curr;
            float* dst1 = dst0 + panel_stride;
            float* dst2 = dst1 + panel_stride;
            float* dst3 = dst2 + panel_stride;

            for (size_t k = 0; k < K_curr; k++) {
                svfloat32x4_t vec = svld1_f32_x4(full_mask4,
                                                  B + (k + curr_row) * N + curr_col + n);
                svst1_f32(pg, dst0, svget4_f32(vec, 0));
                svst1_f32(pg, dst1, svget4_f32(vec, 1));
                svst1_f32(pg, dst2, svget4_f32(vec, 2));
                svst1_f32(pg, dst3, svget4_f32(vec, 3));
                dst0 += SVL;
                dst1 += SVL;
                dst2 += SVL;
                dst3 += SVL;
            }
        }

        // Tail: remaining columns (< 4*SVL), one SVL panel at a time
        for (; n < N_curr; n += SVL) {
            float* dst = packed_B + n * K_curr;
            svbool_t ld_mask = svwhilelt_b32_u64(n, N_curr);

            for (size_t k = 0; k < K_curr; k++) {
                svfloat32_t vec = svld1_f32(ld_mask,
                                            B + (k + curr_row) * N + curr_col + n);
                svst1_f32(pg, dst, vec);
                dst += SVL;
            }
        }
    }

    // =========================================================================
    // MICRO KERNEL: (4*SVL) x SVL output tile using ZA accumulator
    // K-unrolled by 4, software-pipelined: loads interleaved between svmopa
    // to hide ZA accumulator write-after-write latency.
    //
    // Schedule per 4 k-steps (16 svmopa + 5 x4 loads):
    //   PROLOGUE: load B x4 (4 b's), load A x4 (k+0)
    //   group k+0: ZA0, ZA1, [load A(k+1)], ZA2, ZA3
    //   group k+1: ZA0, ZA1, [load A(k+2)], ZA2, ZA3
    //   group k+2: ZA0, ZA1, [load A(k+3)], ZA2, ZA3
    //   group k+3: ZA0, ZA1, [load next B],  ZA2, ZA3
    // Gap between same-tile writes: 4 slots (was 3 without interleaving)
    // =========================================================================
    __attribute__((noinline))
    void micro_kernel_4x1(float* RESTRICT packed_A, float* RESTRICT packed_B, float* RESTRICT C,
                          size_t K_curr, size_t wide_of_C) __arm_out("za") __arm_streaming {
        svzero_za();
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t pg = svptrue_b32();
        const svcount_t pg4 = svptrue_c32();
        const size_t GS = 4 * SVL; // group stride per k-step

        const float* pA = packed_A;
        const float* pB = packed_B;

        const size_t K_main = (K_curr / 4) * 4;

        if (K_main >= 4) {
            // PROLOGUE: load first B x4 and first A x4
            svfloat32x4_t b_x4 = svld1_f32_x4(pg4, pB); pB += 4 * SVL;
            svfloat32_t b0 = svget4_f32(b_x4, 0);
            svfloat32_t b1 = svget4_f32(b_x4, 1);
            svfloat32_t b2 = svget4_f32(b_x4, 2);
            svfloat32_t b3 = svget4_f32(b_x4, 3);

            svfloat32x4_t a_x4 = svld1_f32_x4(pg4, pA); pA += GS;

            for (size_t k = 0; k < K_main - 4; k += 4) {
                // Extract current A into named regs before overwriting a_x4
                svfloat32_t a0 = svget4_f32(a_x4, 0);
                svfloat32_t a1 = svget4_f32(a_x4, 1);
                svfloat32_t a2 = svget4_f32(a_x4, 2);
                svfloat32_t a3 = svget4_f32(a_x4, 3);

                // group k+0: ZA0, ZA1, [load A(k+1)], ZA2, ZA3
                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a1, b0);
                a_x4 = svld1_f32_x4(pg4, pA); pA += GS;
                svmopa_za32_f32_m(2, pg, pg, a2, b0);
                svmopa_za32_f32_m(3, pg, pg, a3, b0);

                a0 = svget4_f32(a_x4, 0);
                a1 = svget4_f32(a_x4, 1);
                a2 = svget4_f32(a_x4, 2);
                a3 = svget4_f32(a_x4, 3);

                // group k+1: ZA0, ZA1, [load A(k+2)], ZA2, ZA3
                svmopa_za32_f32_m(0, pg, pg, a0, b1);
                svmopa_za32_f32_m(1, pg, pg, a1, b1);
                a_x4 = svld1_f32_x4(pg4, pA); pA += GS;
                svmopa_za32_f32_m(2, pg, pg, a2, b1);
                svmopa_za32_f32_m(3, pg, pg, a3, b1);

                a0 = svget4_f32(a_x4, 0);
                a1 = svget4_f32(a_x4, 1);
                a2 = svget4_f32(a_x4, 2);
                a3 = svget4_f32(a_x4, 3);

                // group k+2: ZA0, ZA1, [load A(k+3)], ZA2, ZA3
                svmopa_za32_f32_m(0, pg, pg, a0, b2);
                svmopa_za32_f32_m(1, pg, pg, a1, b2);
                a_x4 = svld1_f32_x4(pg4, pA); pA += GS;
                svmopa_za32_f32_m(2, pg, pg, a2, b2);
                svmopa_za32_f32_m(3, pg, pg, a3, b2);

                a0 = svget4_f32(a_x4, 0);
                a1 = svget4_f32(a_x4, 1);
                a2 = svget4_f32(a_x4, 2);
                a3 = svget4_f32(a_x4, 3);

                // group k+3: ZA0, ZA1, [load next B x4], ZA2, ZA3
                svmopa_za32_f32_m(0, pg, pg, a0, b3);
                svmopa_za32_f32_m(1, pg, pg, a1, b3);
                b_x4 = svld1_f32_x4(pg4, pB); pB += 4 * SVL;
                svmopa_za32_f32_m(2, pg, pg, a2, b3);
                svmopa_za32_f32_m(3, pg, pg, a3, b3);

                b0 = svget4_f32(b_x4, 0);
                b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2);
                b3 = svget4_f32(b_x4, 3);
                a_x4 = svld1_f32_x4(pg4, pA); pA += GS;
            }

            // EPILOGUE: last group of 4 (no next B/A to prefetch)
            {
                svfloat32_t a0 = svget4_f32(a_x4, 0);
                svfloat32_t a1 = svget4_f32(a_x4, 1);
                svfloat32_t a2 = svget4_f32(a_x4, 2);
                svfloat32_t a3 = svget4_f32(a_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a1, b0);
                svmopa_za32_f32_m(2, pg, pg, a2, b0);
                svmopa_za32_f32_m(3, pg, pg, a3, b0);

                a_x4 = svld1_f32_x4(pg4, pA); pA += GS;
                a0 = svget4_f32(a_x4, 0); a1 = svget4_f32(a_x4, 1);
                a2 = svget4_f32(a_x4, 2); a3 = svget4_f32(a_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a0, b1);
                svmopa_za32_f32_m(1, pg, pg, a1, b1);
                svmopa_za32_f32_m(2, pg, pg, a2, b1);
                svmopa_za32_f32_m(3, pg, pg, a3, b1);

                a_x4 = svld1_f32_x4(pg4, pA); pA += GS;
                a0 = svget4_f32(a_x4, 0); a1 = svget4_f32(a_x4, 1);
                a2 = svget4_f32(a_x4, 2); a3 = svget4_f32(a_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a0, b2);
                svmopa_za32_f32_m(1, pg, pg, a1, b2);
                svmopa_za32_f32_m(2, pg, pg, a2, b2);
                svmopa_za32_f32_m(3, pg, pg, a3, b2);

                a_x4 = svld1_f32_x4(pg4, pA); pA += GS;
                a0 = svget4_f32(a_x4, 0); a1 = svget4_f32(a_x4, 1);
                a2 = svget4_f32(a_x4, 2); a3 = svget4_f32(a_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a0, b3);
                svmopa_za32_f32_m(1, pg, pg, a1, b3);
                svmopa_za32_f32_m(2, pg, pg, a2, b3);
                svmopa_za32_f32_m(3, pg, pg, a3, b3);
            }
        }

        // K TAIL: remaining 0-3 iterations, one at a time
        for (size_t k = K_main; k < K_curr; k++) {
            svfloat32x4_t a_x4 = svld1_f32_x4(pg4, pA); pA += GS;
            svfloat32_t b0 = svld1_f32(pg, pB); pB += SVL;
            svmopa_za32_f32_m(0, pg, pg, svget4_f32(a_x4, 0), b0);
            svmopa_za32_f32_m(1, pg, pg, svget4_f32(a_x4, 1), b0);
            svmopa_za32_f32_m(2, pg, pg, svget4_f32(a_x4, 2), b0);
            svmopa_za32_f32_m(3, pg, pg, svget4_f32(a_x4, 3), b0);
        }

        svfloat32_t inactive = svundef_f32();

        // Store each ZA tile back into C, accumulating with existing values.
        // Tile index must be a compile-time constant — unroll manually.
        #define STORE_ZA_TILE(TILE)                                              \
        do {                                                                     \
            float* C_tile = C + (TILE) * SVL * wide_of_C;                        \
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
    // MAIN DRIVER
    // =========================================================================
    struct FreeDeleter { void operator()(void* p) { std::free(p); } };
    using AlignedBuffer = std::unique_ptr<float[], FreeDeleter>;

    __arm_locally_streaming __arm_new("za")
    __attribute__((noinline))
    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N) {
        const size_t SVL = static_cast<size_t>(svcntsw());

        constexpr size_t M_tile = 128;
        constexpr size_t K_tile = 1024;
        constexpr size_t N_tile = 512;

        AlignedBuffer packed_A(static_cast<float*>(
            std::aligned_alloc(64, M_tile * K_tile * sizeof(float))));
        AlignedBuffer packed_B(static_cast<float*>(
            std::aligned_alloc(64, K_tile * N_tile * sizeof(float))));

        const size_t M_step = 4 * SVL;
        const size_t N_step = 1 * SVL;

        for (size_t n = 0; n < N; n += N_tile) {
            size_t nc = std::min(N_tile, N - n);
            for (size_t k = 0; k < K; k += K_tile) {
                size_t kc = std::min(K_tile, K - k);
                pack_B_streaming(B, packed_B.get(), nc, kc, k, N, n);

                for (size_t m = 0; m < M; m += M_tile) {
                    size_t mc = std::min(M_tile, M - m);
                    pack_A_streaming(A, packed_A.get(), mc, kc, m, k, K);

                    for (size_t jr = 0; jr < nc; jr += N_step) {
                        for (size_t ir = 0; ir < mc; ir += M_step) {
                            micro_kernel_4x1(
                                packed_A.get() + ir * kc,
                                packed_B.get() + jr * kc,
                                C + (m + ir) * N + (n + jr),
                                kc, N);
                        }
                    }
                }
            }
        }
    }

} // namespace SMEKernels4x1ZAPack
