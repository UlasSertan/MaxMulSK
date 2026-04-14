#include "SME-GEMMKernels2x2.hpp"

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <memory>
#include <algorithm>

#define RESTRICT __restrict__

namespace SMEKernels2x2 {

    // =========================================================================
    // PACK A  (M x K) → panel-major, transposed, SVL-wide columns
    // =========================================================================
    __attribute__((noinline))
    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const size_t K_full = (K_curr / SVL) * SVL;
        size_t m = 0;

        // Predicates to handle tail cases
        const svbool_t pg = svptrue_b32();
        const svbool_t pfalse = svpfalse_b();

        for (m; m < M_curr; m += SVL) {
            float* panel_base = packed_A + (m / SVL) * (K_curr * SVL);
            const float* row_base = A + (m + curr_row) * K + curr_col;
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

            // Main loop: load SVL-wide rows and transpose into column panels
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

                // 4-stage butterfly transpose (zip pairs → groups of 4 → 8 → 16)
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

                float* out = panel_base + k * SVL;
                svst1_f32(pg, out +  0*SVL, col0);
                svst1_f32(pg, out +  1*SVL, col1);
                svst1_f32(pg, out +  2*SVL, col2);
                svst1_f32(pg, out +  3*SVL, col3);
                svst1_f32(pg, out +  4*SVL, col4);
                svst1_f32(pg, out +  5*SVL, col5);
                svst1_f32(pg, out +  6*SVL, col6);
                svst1_f32(pg, out +  7*SVL, col7);
                svst1_f32(pg, out +  8*SVL, col8);
                svst1_f32(pg, out +  9*SVL, col9);
                svst1_f32(pg, out + 10*SVL, col10);
                svst1_f32(pg, out + 11*SVL, col11);
                svst1_f32(pg, out + 12*SVL, col12);
                svst1_f32(pg, out + 13*SVL, col13);
                svst1_f32(pg, out + 14*SVL, col14);
                svst1_f32(pg, out + 15*SVL, col15);
            }

            // K tail: gather one column at a time via tmp buffer
            // Zero tmp using SVE (avoids __arm_sc_memset from scalar loops)
            float tmp[16];
            svst1_f32(pg, tmp, svdup_f32(0.0f));
            for (; k < K_curr; k++) {
                for (size_t row = 0; row < rows_here; row++)
                    tmp[row] = row_base[k + row * K];
                svfloat32_t col_vec = svld1_f32(pg, tmp);
                svst1_f32(pg, panel_base + k * SVL, col_vec);
                // Re-zero only the lanes we wrote (avoid memset)
                svst1_f32(pg, tmp, svdup_f32(0.0f));
            }
        }
    }

    // =========================================================================
    // PACK B  (K x N) → 2*SVL-wide panels, interleaved per k-step
    // Layout per panel: [b0_k0 | b1_k0 | b0_k1 | b1_k1 | ...]
    // Each k-step stores 2*SVL contiguous floats (two SVL vectors side by side)
    // =========================================================================
    __attribute__((noinline))
    void pack_B_streaming(const float* B, float* packed_B,
                          size_t N_curr, size_t K_curr,
                          size_t curr_row, size_t N, size_t curr_col) __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t pg = svptrue_b32();
        const size_t panel_width = 2 * SVL;
        const size_t panel_stride = K_curr * panel_width; // total floats per 2*SVL panel

        size_t n = 0;

        // Main loop: process 2*SVL columns at a time
        for (; n + panel_width <= N_curr; n += panel_width) {
            float* dst = packed_B + (n / panel_width) * panel_stride;

            for (size_t k = 0; k < K_curr; k++) {
                const float* src = B + (k + curr_row) * N + curr_col + n;
                svfloat32_t v0 = svld1_f32(pg, src);
                svfloat32_t v1 = svld1_f32(pg, src + SVL);
                svst1_f32(pg, dst,       v0);
                svst1_f32(pg, dst + SVL, v1);
                dst += panel_width;
            }
        }

        // Tail: remaining columns (< 2*SVL)
        if (n < N_curr) {
            float* dst = packed_B + (n / panel_width) * panel_stride;
            svbool_t mask0 = svwhilelt_b32_u64(n, N_curr);
            svbool_t mask1 = svwhilelt_b32_u64(n + SVL, N_curr);

            for (size_t k = 0; k < K_curr; k++) {
                const float* src = B + (k + curr_row) * N + curr_col + n;
                svfloat32_t v0 = svld1_f32(mask0, src);
                svfloat32_t v1 = svld1_f32(mask1, src + SVL);
                svst1_f32(pg, dst,       v0);
                svst1_f32(pg, dst + SVL, v1);
                dst += panel_width;
            }
        }
    }

    // =========================================================================
    // MICRO KERNEL 2x2: (2*SVL) rows × (2*SVL) cols using ZA accumulator
    // Uses 4 ZA tiles:
    //   ZA0 = A_panel0 × B_panel0  (top-left)
    //   ZA1 = A_panel0 × B_panel1  (top-right)
    //   ZA2 = A_panel1 × B_panel0  (bottom-left)
    //   ZA3 = A_panel1 × B_panel1  (bottom-right)
    // =========================================================================
    __attribute__((noinline))
    void micro_kernel_2x2(float* RESTRICT packed_A, float* RESTRICT packed_B, float* RESTRICT C,
                          size_t K_curr, size_t wide_of_C) __arm_out("za") __arm_streaming {
        svzero_za();
        const size_t SVL = static_cast<size_t>(svcntsw());
        svbool_t pg = svptrue_b32();
        const size_t ps = SVL * K_curr;

        // A has 2 panels (separate, stride ps apart)
        // B has interleaved layout: [b0 | b1] per k-step, stride 2*SVL
        const float* pA0 = packed_A + 0*ps;
        const float* pA1 = packed_A + 1*ps;
        const float* pB  = packed_B;

        // PROLOGUE: load k=0 data
        svfloat32_t a0 = svld1_f32(pg, pA0); pA0 += SVL;
        svfloat32_t a1 = svld1_f32(pg, pA1); pA1 += SVL;
        svfloat32_t b0 = svld1_f32(pg, pB);
        svfloat32_t b1 = svld1_f32(pg, pB + SVL); pB += 2*SVL;

        // MAIN LOOP
        for (size_t k = 0; k < K_curr - 1; k++) {
            svmopa_za32_f32_m(0, pg, pg, a0, b0);  // top-left
            svmopa_za32_f32_m(1, pg, pg, a0, b1);  // top-right
            svfloat32_t next_a0 = svld1_f32(pg, pA0); pA0 += SVL;

            svmopa_za32_f32_m(2, pg, pg, a1, b0);  // bottom-left
            svfloat32_t next_b0 = svld1_f32(pg, pB);

            svmopa_za32_f32_m(3, pg, pg, a1, b1);  // bottom-right
            svfloat32_t next_a1 = svld1_f32(pg, pA1); pA1 += SVL;
            svfloat32_t next_b1 = svld1_f32(pg, pB + SVL); pB += 2*SVL;

            a0 = next_a0;
            a1 = next_a1;
            b0 = next_b0;
            b1 = next_b1;
        }

        // EPILOGUE: last k iteration
        svmopa_za32_f32_m(0, pg, pg, a0, b0);
        svmopa_za32_f32_m(1, pg, pg, a0, b1);
        svmopa_za32_f32_m(2, pg, pg, a1, b0);
        svmopa_za32_f32_m(3, pg, pg, a1, b1);

        svfloat32_t inactive = svundef_f32();

        // Store ZA tiles back into C, accumulating with existing values.
        // ZA0 → C[0..SVL, 0..SVL]        (top-left)
        // ZA1 → C[0..SVL, SVL..2*SVL]    (top-right)
        // ZA2 → C[SVL..2*SVL, 0..SVL]    (bottom-left)
        // ZA3 → C[SVL..2*SVL, SVL..2*SVL](bottom-right)

        #define STORE_ZA_TILE_2x2(TILE, ROW_OFF, COL_OFF)                        \
        do {                                                                      \
            float* C_tile = C + (ROW_OFF) * wide_of_C + (COL_OFF);               \
            for (int i = 0; i < (int)SVL; i++) {                                  \
                svfloat32_t result = svread_hor_za32_f32_m(inactive, pg, TILE, i); \
                float* ptr = C_tile + i * wide_of_C;                              \
                svfloat32_t existing = svld1_f32(pg, ptr);                        \
                svst1_f32(pg, ptr, svadd_f32_x(pg, existing, result));            \
            }                                                                     \
        } while (0)

        STORE_ZA_TILE_2x2(0, 0,   0);     // top-left
        STORE_ZA_TILE_2x2(1, 0,   SVL);   // top-right
        STORE_ZA_TILE_2x2(2, SVL, 0);     // bottom-left
        STORE_ZA_TILE_2x2(3, SVL, SVL);   // bottom-right

        #undef STORE_ZA_TILE_2x2
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

        constexpr size_t M_tile = 256;
        constexpr size_t K_tile = 2048;
        constexpr size_t N_tile = 512;

        const size_t M_step = 2 * SVL;
        const size_t N_step = 2 * SVL;

        AlignedBuffer packed_A(static_cast<float*>(
            std::aligned_alloc(64, M_tile * K_tile * sizeof(float))));
        AlignedBuffer packed_B(static_cast<float*>(
            std::aligned_alloc(64, K_tile * N_tile * sizeof(float))));

        // Scratch buffer for edge tiles: full 2*SVL × 2*SVL
        // Used when remaining rows or cols < 2*SVL
        AlignedBuffer C_scratch(static_cast<float*>(
            std::aligned_alloc(64, M_step * N_step * sizeof(float))));

        for (size_t n = 0; n < N; n += N_tile) {
            size_t nc = std::min(N_tile, N - n);
            for (size_t k = 0; k < K; k += K_tile) {
                size_t kc = std::min(K_tile, K - k);
                pack_B_streaming(B, packed_B.get(), nc, kc, k, N, n);

                for (size_t m = 0; m < M; m += M_tile) {
                    size_t mc = std::min(M_tile, M - m);
                    pack_A_streaming(A, packed_A.get(), mc, kc, m, k, K);

                    for (size_t jr = 0; jr < nc; jr += N_step) {
                        size_t n_rem = nc - jr;
                        for (size_t ir = 0; ir < mc; ir += M_step) {
                            size_t m_rem = mc - ir;

                            bool m_tail = m_rem < M_step;
                            bool n_tail = n_rem < N_step;

                            if (!m_tail && !n_tail) {
                                // Full tile — write directly into C
                                micro_kernel_2x2(
                                    packed_A.get() + ir * kc,
                                    packed_B.get() + (jr / N_step) * kc * N_step,
                                    C + (m + ir) * N + (n + jr),
                                    kc, N);
                            } else {
                                // Edge tile — compute into scratch, scatter valid part to C
                                size_t rows = std::min(m_rem, M_step);
                                size_t cols = std::min(n_rem, N_step);

                                // Zero scratch with SVE stores (avoids __arm_sc_memset)
                                {
                                    svbool_t pg_z = svptrue_b32();
                                    svfloat32_t zero = svdup_f32(0.0f);
                                    float* zp = C_scratch.get();
                                    for (size_t i = 0; i < M_step * N_step; i += SVL)
                                        svst1_f32(pg_z, zp + i, zero);
                                }

                                micro_kernel_2x2(
                                    packed_A.get() + ir * kc,
                                    packed_B.get() + (jr / N_step) * kc * N_step,
                                    C_scratch.get(),
                                    kc, N_step);

                                // Scatter valid rows × cols back into C
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

} // namespace SMEKernels2x2
