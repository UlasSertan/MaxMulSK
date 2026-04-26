#include "SME-GEMMKernels2x2.hpp"

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <memory>
#include <algorithm>

#define RESTRICT __restrict__

namespace SMEKernels2x2 {

    // =========================================================================
    // PACK A  (M x K) → 2-panel interleaved layout (ZA-based transpose)
    // Layout per k-step: [panel0_SVL | panel1_SVL]
    // Per 16×16 tile: 16 hor-load into ZA0 rows → 16 ver-store from ZA0 cols.
    // The load/store sequence performs the transpose via ZA tile geometry.
    // =========================================================================
    __attribute__((noinline))
    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming __arm_out("za") {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const size_t K_full = (K_curr / SVL) * SVL;
        const size_t GS = 2 * SVL; // group stride: 2 panels interleaved per k-step

        const svbool_t pg = svptrue_b32();
        const svbool_t pfalse = svpfalse_b();

        for (size_t m = 0; m < M_curr; m += SVL) {
            const float* row_base = A + (m + curr_row) * K + curr_col;
            size_t p = (m / SVL) % 2;  // local panel index within M_step group (0 or 1)
            size_t group = (m / SVL) / 2; // which M_step group
            size_t group_offset = group * GS * K_curr; // base offset for this group
            size_t k = 0;
            size_t rows_here = std::min(SVL, M_curr - m);

            svbool_t p0  = (0  < rows_here) ? pg : pfalse;
            svbool_t p1  = (1  < rows_here) ? pg : pfalse;
            svbool_t p2  = (2  < rows_here) ? pg : pfalse;
            svbool_t p3  = (3  < rows_here) ? pg : pfalse;
            svbool_t p4  = (4  < rows_here) ? pg : pfalse;
            svbool_t p5  = (5  < rows_here) ? pg : pfalse;
            svbool_t p6  = (6  < rows_here) ? pg : pfalse;
            svbool_t p7  = (7  < rows_here) ? pg : pfalse;
            svbool_t p8  = (8  < rows_here) ? pg : pfalse;
            svbool_t p9  = (9  < rows_here) ? pg : pfalse;
            svbool_t p10 = (10 < rows_here) ? pg : pfalse;
            svbool_t p11 = (11 < rows_here) ? pg : pfalse;
            svbool_t p12 = (12 < rows_here) ? pg : pfalse;
            svbool_t p13 = (13 < rows_here) ? pg : pfalse;
            svbool_t p14 = (14 < rows_here) ? pg : pfalse;
            svbool_t p15 = (15 < rows_here) ? pg : pfalse;

            // M-tail: predicated-off hor-loads are merging — unwritten slices keep
            // prior ZA contents. Zero tile 0 once so they read back as zero.
            if (rows_here < SVL) {
                svzero_za();
            }

            // Main loop: write 16 rows as ZA hor-slices, read 16 cols as ver-slices → transpose
            for (; k < K_full; k += SVL) {
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

                float* out = packed_A + group_offset + k * GS + p * SVL;
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

            // K tail: one column at a time via tmp buffer (ZA trick is 16x16-only)
            float tmp[16];
            svst1_f32(pg, tmp, svdup_f32(0.0f));
            for (; k < K_curr; k++) {
                for (size_t row = 0; row < rows_here; row++)
                    tmp[row] = row_base[k + row * K];
                svfloat32_t col_vec = svld1_f32(pg, tmp);
                svst1_f32(pg, packed_A + group_offset + k * GS + p * SVL, col_vec);
                svst1_f32(pg, tmp, svdup_f32(0.0f));
            }
        }
    }

    // =========================================================================
    // PACK B  (K x N) → 2*SVL-wide panels, interleaved per k-step
    // Layout per panel: [b0_k0 | b1_k0 | b0_k1 | b1_k1 | ...]
    // Each k-step stores 2*SVL contiguous floats (two SVL vectors side by side).
    // Main loop uses paired (x2) load/store — both sides are contiguous in memory,
    // so we get one-instruction-per-side instead of two.
    // =========================================================================
    __attribute__((noinline))
    void pack_B_streaming(const float* B, float* packed_B,
                          size_t N_curr, size_t K_curr,
                          size_t curr_row, size_t N, size_t curr_col) __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t  pg  = svptrue_b32();
        const svcount_t pg2 = svptrue_c32();       // governs both halves of an x2 pair
        const size_t panel_width = 2 * SVL;
        const size_t panel_stride = K_curr * panel_width; // total floats per 2*SVL panel

        size_t n = 0;

        // Main loop: process 2*SVL columns at a time via paired load/store
        for (; n + panel_width <= N_curr; n += panel_width) {
            float* dst = packed_B + (n / panel_width) * panel_stride;

            for (size_t k = 0; k < K_curr; k++) {
                const float* src = B + (k + curr_row) * N + curr_col + n;
                svfloat32x2_t vec = svld1_f32_x2(pg2, src);
                svst1_f32_x2(pg2, dst, vec);
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
    // K-unrolled by 4, software-pipelined: loads interleaved between svmopa.
    //
    // A interleaved: [a0|a1] per k-step, GS=2*SVL. x4 load covers 2 k-steps.
    // B interleaved: [b0|b1] per k-step, 2*SVL.   x4 load covers 2 k-steps.
    //
    // Per 4 k-steps: 4 x4 loads (2 A + 2 B) → 16 svmopa
    // Schedule (gap between same-tile writes = 4 slots):
    //   group k+0: ZA0, ZA1, [load A(k+2,k+3)], ZA2, ZA3
    //   group k+1: ZA0, ZA1, [load B(k+2,k+3)], ZA2, ZA3
    //   group k+2: ZA0, ZA1, [load next A(k+0,k+1)], ZA2, ZA3
    //   group k+3: ZA0, ZA1, [load next B(k+0,k+1)], ZA2, ZA3
    // =========================================================================
    __attribute__((noinline))
    void micro_kernel_2x2(float* RESTRICT packed_A, float* RESTRICT packed_B, float* RESTRICT C,
                          size_t K_curr, size_t wide_of_C) __arm_out("za") __arm_streaming {
        svzero_za();
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t pg = svptrue_b32();
        const svcount_t pg4 = svptrue_c32();

        const float* pA = packed_A;
        const float* pB = packed_B;

        const size_t K_main = (K_curr / 4) * 4;

        if (K_main >= 4) {
            // PROLOGUE: load A x4 (k+0,k+1) and B x4 (k+0,k+1)
            // A x4 = [a0_k0, a1_k0, a0_k1, a1_k1]
            // B x4 = [b0_k0, b1_k0, b0_k1, b1_k1]
            svfloat32x4_t a_x4_01 = svld1_f32_x4(pg4, pA);       pA += 4 * SVL;
            svfloat32x4_t b_x4_01 = svld1_f32_x4(pg4, pB);       pB += 4 * SVL;
            svfloat32x4_t a_x4_23 = svld1_f32_x4(pg4, pA);       pA += 4 * SVL;
            svfloat32x4_t b_x4_23 = svld1_f32_x4(pg4, pB);       pB += 4 * SVL;

            for (size_t k = 0; k < K_main - 4; k += 4) {
                // k+0: a from a_x4_01[0,1], b from b_x4_01[0,1]
                svfloat32_t a0 = svget4_f32(a_x4_01, 0);
                svfloat32_t a1 = svget4_f32(a_x4_01, 1);
                svfloat32_t b0 = svget4_f32(b_x4_01, 0);
                svfloat32_t b1 = svget4_f32(b_x4_01, 1);

                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                svfloat32x4_t next_a_01 = svld1_f32_x4(pg4, pA); pA += 4 * SVL;
                svmopa_za32_f32_m(2, pg, pg, a1, b0);
                svmopa_za32_f32_m(3, pg, pg, a1, b1);

                // k+1: a from a_x4_01[2,3], b from b_x4_01[2,3]
                a0 = svget4_f32(a_x4_01, 2);
                a1 = svget4_f32(a_x4_01, 3);
                b0 = svget4_f32(b_x4_01, 2);
                b1 = svget4_f32(b_x4_01, 3);

                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                svfloat32x4_t next_b_01 = svld1_f32_x4(pg4, pB); pB += 4 * SVL;
                svmopa_za32_f32_m(2, pg, pg, a1, b0);
                svmopa_za32_f32_m(3, pg, pg, a1, b1);

                // k+2: a from a_x4_23[0,1], b from b_x4_23[0,1]
                a0 = svget4_f32(a_x4_23, 0);
                a1 = svget4_f32(a_x4_23, 1);
                b0 = svget4_f32(b_x4_23, 0);
                b1 = svget4_f32(b_x4_23, 1);

                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                svfloat32x4_t next_a_23 = svld1_f32_x4(pg4, pA); pA += 4 * SVL;
                svmopa_za32_f32_m(2, pg, pg, a1, b0);
                svmopa_za32_f32_m(3, pg, pg, a1, b1);

                // k+3: a from a_x4_23[2,3], b from b_x4_23[2,3]
                a0 = svget4_f32(a_x4_23, 2);
                a1 = svget4_f32(a_x4_23, 3);
                b0 = svget4_f32(b_x4_23, 2);
                b1 = svget4_f32(b_x4_23, 3);

                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                svfloat32x4_t next_b_23 = svld1_f32_x4(pg4, pB); pB += 4 * SVL;
                svmopa_za32_f32_m(2, pg, pg, a1, b0);
                svmopa_za32_f32_m(3, pg, pg, a1, b1);

                a_x4_01 = next_a_01;
                b_x4_01 = next_b_01;
                a_x4_23 = next_a_23;
                b_x4_23 = next_b_23;
            }

            // EPILOGUE: last group of 4 (no prefetch)
            {
                svfloat32_t a0 = svget4_f32(a_x4_01, 0);
                svfloat32_t a1 = svget4_f32(a_x4_01, 1);
                svfloat32_t b0 = svget4_f32(b_x4_01, 0);
                svfloat32_t b1 = svget4_f32(b_x4_01, 1);
                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                svmopa_za32_f32_m(2, pg, pg, a1, b0);
                svmopa_za32_f32_m(3, pg, pg, a1, b1);

                a0 = svget4_f32(a_x4_01, 2); a1 = svget4_f32(a_x4_01, 3);
                b0 = svget4_f32(b_x4_01, 2); b1 = svget4_f32(b_x4_01, 3);
                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                svmopa_za32_f32_m(2, pg, pg, a1, b0);
                svmopa_za32_f32_m(3, pg, pg, a1, b1);

                a0 = svget4_f32(a_x4_23, 0); a1 = svget4_f32(a_x4_23, 1);
                b0 = svget4_f32(b_x4_23, 0); b1 = svget4_f32(b_x4_23, 1);
                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                svmopa_za32_f32_m(2, pg, pg, a1, b0);
                svmopa_za32_f32_m(3, pg, pg, a1, b1);

                a0 = svget4_f32(a_x4_23, 2); a1 = svget4_f32(a_x4_23, 3);
                b0 = svget4_f32(b_x4_23, 2); b1 = svget4_f32(b_x4_23, 3);
                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                svmopa_za32_f32_m(2, pg, pg, a1, b0);
                svmopa_za32_f32_m(3, pg, pg, a1, b1);
            }
        }

        // K TAIL: remaining 0-3 iterations, one at a time
        // A interleaved: [a0|a1] per k-step at pA, stride 2*SVL
        for (size_t k = K_main; k < K_curr; k++) {
            svfloat32_t a0 = svld1_f32(pg, pA);
            svfloat32_t a1 = svld1_f32(pg, pA + SVL); pA += 2 * SVL;
            svfloat32_t b0 = svld1_f32(pg, pB);
            svfloat32_t b1 = svld1_f32(pg, pB + SVL); pB += 2 * SVL;
            svmopa_za32_f32_m(0, pg, pg, a0, b0);
            svmopa_za32_f32_m(1, pg, pg, a0, b1);
            svmopa_za32_f32_m(2, pg, pg, a1, b0);
            svmopa_za32_f32_m(3, pg, pg, a1, b1);
        }

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

        constexpr size_t M_tile = 64;
        constexpr size_t K_tile = 1024;
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
