#include "GEMMKernels.hpp"

#include <arm_neon.h>
#include <cstddef>
#include <memory>
#include <algorithm>
#include <omp.h>

#define RESTRICT __restrict__

namespace GEMM {

    // =========================================================================
    // MICRO KERNEL: 8 rows x 12 columns (8x3 NEON vectors)
    // A is packed column-major per 8-row panel (transposed)
    // B is packed row-major per 12-column panel
    // =========================================================================
    static void multiply_kernel_neon_8x3(
            const float* RESTRICT A,
            const float* RESTRICT B,
            float* RESTRICT C,
            size_t Kc, size_t N) {

        float32x4_t c[8][3];
        const float32x4_t zero = vdupq_n_f32(0.0f);

        for (size_t i = 0; i < 8; i++) {
            c[i][0] = zero;
            c[i][1] = zero;
            c[i][2] = zero;
        }

        const float* a_ptr = A;
        const float* b_ptr = B;
        size_t k = 0;

        // Unrolled 4x along K
        for (; k < (Kc & ~3UL); k += 4) {

            float32x4_t a0_3 = vld1q_f32(a_ptr);
            float32x4_t a4_7 = vld1q_f32(a_ptr + 4);
            float32x4_t b0   = vld1q_f32(b_ptr);
            float32x4_t b1   = vld1q_f32(b_ptr + 4);
            float32x4_t b2   = vld1q_f32(b_ptr + 8);

            // K+0
            float32x4_t a0_3_next = vld1q_f32(a_ptr + 8);
            float32x4_t a4_7_next = vld1q_f32(a_ptr + 12);

            c[0][0] = vfmaq_laneq_f32(c[0][0], b0, a0_3, 0);
            c[0][1] = vfmaq_laneq_f32(c[0][1], b1, a0_3, 0);
            c[0][2] = vfmaq_laneq_f32(c[0][2], b2, a0_3, 0);
            c[1][0] = vfmaq_laneq_f32(c[1][0], b0, a0_3, 1);
            c[1][1] = vfmaq_laneq_f32(c[1][1], b1, a0_3, 1);
            c[1][2] = vfmaq_laneq_f32(c[1][2], b2, a0_3, 1);

            float32x4_t b0_next = vld1q_f32(b_ptr + 12);

            c[2][0] = vfmaq_laneq_f32(c[2][0], b0, a0_3, 2);
            c[2][1] = vfmaq_laneq_f32(c[2][1], b1, a0_3, 2);
            c[2][2] = vfmaq_laneq_f32(c[2][2], b2, a0_3, 2);
            c[3][0] = vfmaq_laneq_f32(c[3][0], b0, a0_3, 3);
            c[3][1] = vfmaq_laneq_f32(c[3][1], b1, a0_3, 3);
            c[3][2] = vfmaq_laneq_f32(c[3][2], b2, a0_3, 3);
            c[4][0] = vfmaq_laneq_f32(c[4][0], b0, a4_7, 0);
            c[4][1] = vfmaq_laneq_f32(c[4][1], b1, a4_7, 0);
            c[4][2] = vfmaq_laneq_f32(c[4][2], b2, a4_7, 0);
            c[5][0] = vfmaq_laneq_f32(c[5][0], b0, a4_7, 1);
            c[5][1] = vfmaq_laneq_f32(c[5][1], b1, a4_7, 1);
            c[5][2] = vfmaq_laneq_f32(c[5][2], b2, a4_7, 1);
            c[6][0] = vfmaq_laneq_f32(c[6][0], b0, a4_7, 2);
            c[6][1] = vfmaq_laneq_f32(c[6][1], b1, a4_7, 2);
            c[6][2] = vfmaq_laneq_f32(c[6][2], b2, a4_7, 2);
            c[7][0] = vfmaq_laneq_f32(c[7][0], b0, a4_7, 3);
            c[7][1] = vfmaq_laneq_f32(c[7][1], b1, a4_7, 3);
            c[7][2] = vfmaq_laneq_f32(c[7][2], b2, a4_7, 3);

            // K+1
            a0_3 = a0_3_next;
            a4_7 = a4_7_next;
            b0   = b0_next;
            b1   = vld1q_f32(b_ptr + 16);
            b2   = vld1q_f32(b_ptr + 20);

            a0_3_next = vld1q_f32(a_ptr + 16);
            a4_7_next = vld1q_f32(a_ptr + 20);

            c[0][0] = vfmaq_laneq_f32(c[0][0], b0, a0_3, 0);
            c[0][1] = vfmaq_laneq_f32(c[0][1], b1, a0_3, 0);
            c[0][2] = vfmaq_laneq_f32(c[0][2], b2, a0_3, 0);
            c[1][0] = vfmaq_laneq_f32(c[1][0], b0, a0_3, 1);
            c[1][1] = vfmaq_laneq_f32(c[1][1], b1, a0_3, 1);
            c[1][2] = vfmaq_laneq_f32(c[1][2], b2, a0_3, 1);

            b0_next = vld1q_f32(b_ptr + 24);

            c[2][0] = vfmaq_laneq_f32(c[2][0], b0, a0_3, 2);
            c[2][1] = vfmaq_laneq_f32(c[2][1], b1, a0_3, 2);
            c[2][2] = vfmaq_laneq_f32(c[2][2], b2, a0_3, 2);
            c[3][0] = vfmaq_laneq_f32(c[3][0], b0, a0_3, 3);
            c[3][1] = vfmaq_laneq_f32(c[3][1], b1, a0_3, 3);
            c[3][2] = vfmaq_laneq_f32(c[3][2], b2, a0_3, 3);
            c[4][0] = vfmaq_laneq_f32(c[4][0], b0, a4_7, 0);
            c[4][1] = vfmaq_laneq_f32(c[4][1], b1, a4_7, 0);
            c[4][2] = vfmaq_laneq_f32(c[4][2], b2, a4_7, 0);
            c[5][0] = vfmaq_laneq_f32(c[5][0], b0, a4_7, 1);
            c[5][1] = vfmaq_laneq_f32(c[5][1], b1, a4_7, 1);
            c[5][2] = vfmaq_laneq_f32(c[5][2], b2, a4_7, 1);
            c[6][0] = vfmaq_laneq_f32(c[6][0], b0, a4_7, 2);
            c[6][1] = vfmaq_laneq_f32(c[6][1], b1, a4_7, 2);
            c[6][2] = vfmaq_laneq_f32(c[6][2], b2, a4_7, 2);
            c[7][0] = vfmaq_laneq_f32(c[7][0], b0, a4_7, 3);
            c[7][1] = vfmaq_laneq_f32(c[7][1], b1, a4_7, 3);
            c[7][2] = vfmaq_laneq_f32(c[7][2], b2, a4_7, 3);

            // K+2
            a0_3 = a0_3_next;
            a4_7 = a4_7_next;
            b0   = b0_next;
            b1   = vld1q_f32(b_ptr + 28);
            b2   = vld1q_f32(b_ptr + 32);

            a0_3_next = vld1q_f32(a_ptr + 24);
            a4_7_next = vld1q_f32(a_ptr + 28);

            c[0][0] = vfmaq_laneq_f32(c[0][0], b0, a0_3, 0);
            c[0][1] = vfmaq_laneq_f32(c[0][1], b1, a0_3, 0);
            c[0][2] = vfmaq_laneq_f32(c[0][2], b2, a0_3, 0);
            c[1][0] = vfmaq_laneq_f32(c[1][0], b0, a0_3, 1);
            c[1][1] = vfmaq_laneq_f32(c[1][1], b1, a0_3, 1);
            c[1][2] = vfmaq_laneq_f32(c[1][2], b2, a0_3, 1);

            b0_next = vld1q_f32(b_ptr + 36);

            c[2][0] = vfmaq_laneq_f32(c[2][0], b0, a0_3, 2);
            c[2][1] = vfmaq_laneq_f32(c[2][1], b1, a0_3, 2);
            c[2][2] = vfmaq_laneq_f32(c[2][2], b2, a0_3, 2);
            c[3][0] = vfmaq_laneq_f32(c[3][0], b0, a0_3, 3);
            c[3][1] = vfmaq_laneq_f32(c[3][1], b1, a0_3, 3);
            c[3][2] = vfmaq_laneq_f32(c[3][2], b2, a0_3, 3);
            c[4][0] = vfmaq_laneq_f32(c[4][0], b0, a4_7, 0);
            c[4][1] = vfmaq_laneq_f32(c[4][1], b1, a4_7, 0);
            c[4][2] = vfmaq_laneq_f32(c[4][2], b2, a4_7, 0);
            c[5][0] = vfmaq_laneq_f32(c[5][0], b0, a4_7, 1);
            c[5][1] = vfmaq_laneq_f32(c[5][1], b1, a4_7, 1);
            c[5][2] = vfmaq_laneq_f32(c[5][2], b2, a4_7, 1);
            c[6][0] = vfmaq_laneq_f32(c[6][0], b0, a4_7, 2);
            c[6][1] = vfmaq_laneq_f32(c[6][1], b1, a4_7, 2);
            c[6][2] = vfmaq_laneq_f32(c[6][2], b2, a4_7, 2);
            c[7][0] = vfmaq_laneq_f32(c[7][0], b0, a4_7, 3);
            c[7][1] = vfmaq_laneq_f32(c[7][1], b1, a4_7, 3);
            c[7][2] = vfmaq_laneq_f32(c[7][2], b2, a4_7, 3);

            // K+3
            a0_3 = a0_3_next;
            a4_7 = a4_7_next;
            b0   = b0_next;
            b1   = vld1q_f32(b_ptr + 40);
            b2   = vld1q_f32(b_ptr + 44);

            c[0][0] = vfmaq_laneq_f32(c[0][0], b0, a0_3, 0);
            c[0][1] = vfmaq_laneq_f32(c[0][1], b1, a0_3, 0);
            c[0][2] = vfmaq_laneq_f32(c[0][2], b2, a0_3, 0);
            c[1][0] = vfmaq_laneq_f32(c[1][0], b0, a0_3, 1);
            c[1][1] = vfmaq_laneq_f32(c[1][1], b1, a0_3, 1);
            c[1][2] = vfmaq_laneq_f32(c[1][2], b2, a0_3, 1);
            c[2][0] = vfmaq_laneq_f32(c[2][0], b0, a0_3, 2);
            c[2][1] = vfmaq_laneq_f32(c[2][1], b1, a0_3, 2);
            c[2][2] = vfmaq_laneq_f32(c[2][2], b2, a0_3, 2);
            c[3][0] = vfmaq_laneq_f32(c[3][0], b0, a0_3, 3);
            c[3][1] = vfmaq_laneq_f32(c[3][1], b1, a0_3, 3);
            c[3][2] = vfmaq_laneq_f32(c[3][2], b2, a0_3, 3);
            c[4][0] = vfmaq_laneq_f32(c[4][0], b0, a4_7, 0);
            c[4][1] = vfmaq_laneq_f32(c[4][1], b1, a4_7, 0);
            c[4][2] = vfmaq_laneq_f32(c[4][2], b2, a4_7, 0);
            c[5][0] = vfmaq_laneq_f32(c[5][0], b0, a4_7, 1);
            c[5][1] = vfmaq_laneq_f32(c[5][1], b1, a4_7, 1);
            c[5][2] = vfmaq_laneq_f32(c[5][2], b2, a4_7, 1);
            c[6][0] = vfmaq_laneq_f32(c[6][0], b0, a4_7, 2);
            c[6][1] = vfmaq_laneq_f32(c[6][1], b1, a4_7, 2);
            c[6][2] = vfmaq_laneq_f32(c[6][2], b2, a4_7, 2);
            c[7][0] = vfmaq_laneq_f32(c[7][0], b0, a4_7, 3);
            c[7][1] = vfmaq_laneq_f32(c[7][1], b1, a4_7, 3);
            c[7][2] = vfmaq_laneq_f32(c[7][2], b2, a4_7, 3);

            a_ptr += 32;
            b_ptr += 48;
        }

        // Scalar tail for remaining K
        for (; k < Kc; k++) {
            float32x4_t a0_3 = vld1q_f32(a_ptr);
            float32x4_t a4_7 = vld1q_f32(a_ptr + 4);
            float32x4_t b0   = vld1q_f32(b_ptr);
            float32x4_t b1   = vld1q_f32(b_ptr + 4);
            float32x4_t b2   = vld1q_f32(b_ptr + 8);

            c[0][0] = vfmaq_laneq_f32(c[0][0], b0, a0_3, 0);
            c[0][1] = vfmaq_laneq_f32(c[0][1], b1, a0_3, 0);
            c[0][2] = vfmaq_laneq_f32(c[0][2], b2, a0_3, 0);
            c[1][0] = vfmaq_laneq_f32(c[1][0], b0, a0_3, 1);
            c[1][1] = vfmaq_laneq_f32(c[1][1], b1, a0_3, 1);
            c[1][2] = vfmaq_laneq_f32(c[1][2], b2, a0_3, 1);
            c[2][0] = vfmaq_laneq_f32(c[2][0], b0, a0_3, 2);
            c[2][1] = vfmaq_laneq_f32(c[2][1], b1, a0_3, 2);
            c[2][2] = vfmaq_laneq_f32(c[2][2], b2, a0_3, 2);
            c[3][0] = vfmaq_laneq_f32(c[3][0], b0, a0_3, 3);
            c[3][1] = vfmaq_laneq_f32(c[3][1], b1, a0_3, 3);
            c[3][2] = vfmaq_laneq_f32(c[3][2], b2, a0_3, 3);
            c[4][0] = vfmaq_laneq_f32(c[4][0], b0, a4_7, 0);
            c[4][1] = vfmaq_laneq_f32(c[4][1], b1, a4_7, 0);
            c[4][2] = vfmaq_laneq_f32(c[4][2], b2, a4_7, 0);
            c[5][0] = vfmaq_laneq_f32(c[5][0], b0, a4_7, 1);
            c[5][1] = vfmaq_laneq_f32(c[5][1], b1, a4_7, 1);
            c[5][2] = vfmaq_laneq_f32(c[5][2], b2, a4_7, 1);
            c[6][0] = vfmaq_laneq_f32(c[6][0], b0, a4_7, 2);
            c[6][1] = vfmaq_laneq_f32(c[6][1], b1, a4_7, 2);
            c[6][2] = vfmaq_laneq_f32(c[6][2], b2, a4_7, 2);
            c[7][0] = vfmaq_laneq_f32(c[7][0], b0, a4_7, 3);
            c[7][1] = vfmaq_laneq_f32(c[7][1], b1, a4_7, 3);
            c[7][2] = vfmaq_laneq_f32(c[7][2], b2, a4_7, 3);

            a_ptr += 8;
            b_ptr += 12;
        }

        // Store: accumulate into C (row-major, stride N)
        for (size_t i = 0; i < 8; i++) {
            float* c_ptr = C + i * N;
            float32x4_t c0 = vld1q_f32(c_ptr);
            vst1q_f32(c_ptr,     vaddq_f32(c0, c[i][0]));
            c0 = vld1q_f32(c_ptr + 4);
            vst1q_f32(c_ptr + 4, vaddq_f32(c0, c[i][1]));
            c0 = vld1q_f32(c_ptr + 8);
            vst1q_f32(c_ptr + 8, vaddq_f32(c0, c[i][2]));
        }
    }

    // =========================================================================
    // PACKING
    // =========================================================================

    void transpose_8x4(const float* src, float* dest, size_t K) {
        float32x4_t v0 = vld1q_f32(src);
        float32x4_t v1 = vld1q_f32(src + K);
        float32x4_t v2 = vld1q_f32(src + 2*K);
        float32x4_t v3 = vld1q_f32(src + 3*K);
        float32x4_t v4 = vld1q_f32(src + 4*K);
        float32x4_t v5 = vld1q_f32(src + 5*K);
        float32x4_t v6 = vld1q_f32(src + 6*K);
        float32x4_t v7 = vld1q_f32(src + 7*K);

        float32x4x2_t v01 = vzipq_f32(v0, v1);
        float32x4x2_t v23 = vzipq_f32(v2, v3);
        float32x4x2_t v45 = vzipq_f32(v4, v5);
        float32x4x2_t v67 = vzipq_f32(v6, v7);

        float32x4_t r0 = vcombine_f32(vget_low_f32(v01.val[0]),  vget_low_f32(v23.val[0]));
        float32x4_t r1 = vcombine_f32(vget_high_f32(v01.val[0]), vget_high_f32(v23.val[0]));
        float32x4_t r2 = vcombine_f32(vget_low_f32(v01.val[1]),  vget_low_f32(v23.val[1]));
        float32x4_t r3 = vcombine_f32(vget_high_f32(v01.val[1]), vget_high_f32(v23.val[1]));
        float32x4_t r4 = vcombine_f32(vget_low_f32(v45.val[0]),  vget_low_f32(v67.val[0]));
        float32x4_t r5 = vcombine_f32(vget_high_f32(v45.val[0]), vget_high_f32(v67.val[0]));
        float32x4_t r6 = vcombine_f32(vget_low_f32(v45.val[1]),  vget_low_f32(v67.val[1]));
        float32x4_t r7 = vcombine_f32(vget_high_f32(v45.val[1]), vget_high_f32(v67.val[1]));

        // Layout: [k0_r0..r7, k1_r0..r7, k2_r0..r7, k3_r0..r7]
        vst1q_f32(dest + 0,  r0);
        vst1q_f32(dest + 4,  r4);
        vst1q_f32(dest + 8,  r1);
        vst1q_f32(dest + 12, r5);
        vst1q_f32(dest + 16, r2);
        vst1q_f32(dest + 20, r6);
        vst1q_f32(dest + 24, r3);
        vst1q_f32(dest + 28, r7);
    }

    void pack_A_block(const float* src, float* pack_A, size_t Mc, size_t Kc, size_t K) {
        float* write_ptr = pack_A;
        for (size_t i = 0; i < Mc; i += 8) {
            for (size_t k = 0; k < Kc; k += 4) {
                transpose_8x4(src + i * K + k, write_ptr, K);
                write_ptr += 32;
            }
        }
    }

    void pack_B_block(const float* B, float* pack_B,
                      size_t j_outer, size_t Nc, size_t Kc, size_t N) {
        float* dst = pack_B;
        for (size_t j = 0; j < Nc; j += 12) {
            for (size_t k = 0; k < Kc; k++) {
                const float* src = B + k * N + j + j_outer;
                vst1q_f32(dst,     vld1q_f32(src));
                vst1q_f32(dst + 4, vld1q_f32(src + 4));
                vst1q_f32(dst + 8, vld1q_f32(src + 8));
                dst += 12;
            }
        }
    }

    // =========================================================================
    // DISPATCHER
    // =========================================================================

    void multiply(const float* A, const float* B, float* C,
                  size_t M, size_t Nc, size_t Kc, size_t N) {
        for (size_t i = 0; i < M; i += 8) {
            for (size_t j = 0; j < Nc; j += 12) {
                multiply_kernel_neon_8x3(&A[i * Kc], &B[j * Kc], &C[i * N + j], Kc, N);
            }
        }
    }

    // =========================================================================
    // TOP-LEVEL ENTRY POINT
    // =========================================================================

    struct FreeDeleter { void operator()(void* p) { std::free(p); } };
    using AlignedBuffer = std::unique_ptr<float[], FreeDeleter>;

    void package(const float* A, const float* B, float* C,
                 size_t M, size_t N, size_t K) {
        constexpr size_t Kc = 256;
        constexpr size_t Mc = 64;
        constexpr size_t Nc = 1020; // must be multiple of 12
        constexpr size_t Nc_aligned = ((Nc + 11) / 12) * 12;

        #pragma omp parallel for
        for (size_t i = 0; i < M * N; i++) C[i] = 0.0f;

        void* ptr = std::aligned_alloc(64, Kc * Nc_aligned * sizeof(float));
        AlignedBuffer packed_B(static_cast<float*>(ptr));

        for (size_t k_out = 0; k_out < K; k_out += Kc) {
            size_t current_Kc = std::min(Kc, K - k_out);
            for (size_t j = 0; j < N; j += Nc) {
                pack_B_block(B + k_out * N, packed_B.get(), j, Nc, current_Kc, N);
                #pragma omp parallel
                {
                    alignas(64) float packed_A[Mc * Kc];
                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < M; i += Mc) {
                        pack_A_block(A + i * K + k_out, packed_A, Mc, current_Kc, K);
                        multiply(packed_A, packed_B.get(), C + i * N + j, Mc, Nc, current_Kc, N);
                    }
                }
            }
        }
    }

} // namespace GEMM
