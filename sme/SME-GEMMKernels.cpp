#include "SME-GEMMKernels.hpp"

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <memory>
#include <algorithm>

#define RESTRICT __restrict__

namespace SMEKernels {

    // =========================================================================
    // PACK A  (M x K) → panel-major, transposed, SVL-wide columns
    // =========================================================================
    __attribute__((noinline))
    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t pg = svptrue_b32();
        const size_t K_full = (K_curr / SVL) * SVL;

        for (size_t m = 0; m < M_curr; m += SVL) {
            float* panel_base = packed_A + (m / SVL) * (K_curr * SVL);
            const float* row_base = A + (m + curr_row) * K + curr_col;
            size_t k = 0;

            // Main loop: load SVL-wide rows and transpose into column panels
            for (; k < K_full; k += SVL) {
                svfloat32_t r0  = svld1_f32(pg, row_base + k);
                svfloat32_t r1  = svld1_f32(pg, row_base + k +  K);
                svfloat32_t r2  = svld1_f32(pg, row_base + k +  2*K);
                svfloat32_t r3  = svld1_f32(pg, row_base + k +  3*K);
                svfloat32_t r4  = svld1_f32(pg, row_base + k +  4*K);
                svfloat32_t r5  = svld1_f32(pg, row_base + k +  5*K);
                svfloat32_t r6  = svld1_f32(pg, row_base + k +  6*K);
                svfloat32_t r7  = svld1_f32(pg, row_base + k +  7*K);
                svfloat32_t r8  = svld1_f32(pg, row_base + k +  8*K);
                svfloat32_t r9  = svld1_f32(pg, row_base + k +  9*K);
                svfloat32_t r10 = svld1_f32(pg, row_base + k + 10*K);
                svfloat32_t r11 = svld1_f32(pg, row_base + k + 11*K);
                svfloat32_t r12 = svld1_f32(pg, row_base + k + 12*K);
                svfloat32_t r13 = svld1_f32(pg, row_base + k + 13*K);
                svfloat32_t r14 = svld1_f32(pg, row_base + k + 14*K);
                svfloat32_t r15 = svld1_f32(pg, row_base + k + 15*K);

                // 4-stage butterfly transpose (zip pairs → groups of 4 → 8 → 16)
                svfloat32_t s1_0L = svzip1_f32(r0,  r1);
                svfloat32_t s1_0H = svzip2_f32(r0,  r1);
                svfloat32_t s1_1L = svzip1_f32(r2,  r3);
                svfloat32_t s1_1H = svzip2_f32(r2,  r3);
                svfloat32_t s1_2L = svzip1_f32(r4,  r5);
                svfloat32_t s1_2H = svzip2_f32(r4,  r5);
                svfloat32_t s1_3L = svzip1_f32(r6,  r7);
                svfloat32_t s1_3H = svzip2_f32(r6,  r7);
                svfloat32_t s1_4L = svzip1_f32(r8,  r9);
                svfloat32_t s1_4H = svzip2_f32(r8,  r9);
                svfloat32_t s1_5L = svzip1_f32(r10, r11);
                svfloat32_t s1_5H = svzip2_f32(r10, r11);
                svfloat32_t s1_6L = svzip1_f32(r12, r13);
                svfloat32_t s1_6H = svzip2_f32(r12, r13);
                svfloat32_t s1_7L = svzip1_f32(r14, r15);
                svfloat32_t s1_7H = svzip2_f32(r14, r15);

                svfloat32_t s2_0L = svzip1_f32(s1_0L, s1_2L);
                svfloat32_t s2_0H = svzip2_f32(s1_0L, s1_2L);
                svfloat32_t s2_1L = svzip1_f32(s1_0H, s1_2H);
                svfloat32_t s2_1H = svzip2_f32(s1_0H, s1_2H);
                svfloat32_t s2_2L = svzip1_f32(s1_1L, s1_3L);
                svfloat32_t s2_2H = svzip2_f32(s1_1L, s1_3L);
                svfloat32_t s2_3L = svzip1_f32(s1_1H, s1_3H);
                svfloat32_t s2_3H = svzip2_f32(s1_1H, s1_3H);
                svfloat32_t s2_4L = svzip1_f32(s1_4L, s1_6L);
                svfloat32_t s2_4H = svzip2_f32(s1_4L, s1_6L);
                svfloat32_t s2_5L = svzip1_f32(s1_4H, s1_6H);
                svfloat32_t s2_5H = svzip2_f32(s1_4H, s1_6H);
                svfloat32_t s2_6L = svzip1_f32(s1_5L, s1_7L);
                svfloat32_t s2_6H = svzip2_f32(s1_5L, s1_7L);
                svfloat32_t s2_7L = svzip1_f32(s1_5H, s1_7H);
                svfloat32_t s2_7H = svzip2_f32(s1_5H, s1_7H);

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

            // Scalar tail for K not divisible by SVL
            float tmp[16];
            for (; k < K_curr; k++) {
                for (size_t row = 0; row < SVL; row++)
                    tmp[row] = row_base[k + row * K];
                svfloat32_t col_vec = svld1_f32(pg, tmp);
                svst1_f32(pg, panel_base + k * SVL, col_vec);
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
    // =========================================================================
    __attribute__((noinline))
    void micro_kernel_4x1(float* RESTRICT packed_A, float* RESTRICT packed_B, float* RESTRICT C,
                          size_t K_curr, size_t wide_of_C) __arm_out("za") __arm_streaming {
        svzero_za();
        const size_t SVL = static_cast<size_t>(svcntsw());
        svbool_t pg = svptrue_b32();
        const size_t ps = SVL * K_curr;

        for (size_t k = 0; k < K_curr; k++) {
            svfloat32_t a0 = svld1(pg, packed_A + 0*ps + k*SVL);
            svfloat32_t a1 = svld1(pg, packed_A + 1*ps + k*SVL);
            svfloat32_t a2 = svld1(pg, packed_A + 2*ps + k*SVL);
            svfloat32_t a3 = svld1(pg, packed_A + 3*ps + k*SVL);
            svfloat32_t b0 = svld1(pg, packed_B + k*SVL);

            svmopa_za32_f32_m(0, pg, pg, a0, b0);
            svmopa_za32_f32_m(1, pg, pg, a1, b0);
            svmopa_za32_f32_m(2, pg, pg, a2, b0);
            svmopa_za32_f32_m(3, pg, pg, a3, b0);
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

        constexpr size_t M_tile = 64;
        constexpr size_t K_tile = 256;
        constexpr size_t N_tile = 64;

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

} // namespace SMEKernels
