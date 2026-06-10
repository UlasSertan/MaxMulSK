#include "SME-GEMMKernels1x4-symZAInOut.hpp"

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <memory>
#include <algorithm>

#define RESTRICT __restrict__

// =============================================================================
// SME 1x4-symZAInOut -- ZA in_out experiment (TODO §5)
//
// Goal: stop zeroing ZA and storing ZA->C on every micro-kernel call. With
// __arm_inout("za") the FMOPA loop can be called many times for the same
// output tile while ZA stays live; we pay svzero_za and the store-back loop
// only once per output tile instead of once per K-block.
//
// Structure:
//   kernel_zero    __arm_out("za")    -- svzero_za only
//   kernel_compute __arm_inout("za")  -- pure FMOPA loop (no zero, no store)
//   kernel_store   __arm_in("za")     -- ZA -> C with += semantics
//
// Driver:
//   for each (m, n) strip
//     pack_A for the FULL K  (M_tile * K floats)
//     pack_B for the FULL K  (K * N_tile floats)
//     for each output tile (jr, ir)
//       kernel_zero
//       for k_in in K_inner_tile chunks:
//         kernel_compute(K_inner_tile)
//       kernel_store
//
// K is packed in full (no outer K block) because splitting K across pack
// calls would require repacking on every (jr, ir) visit -- that would erase
// the win.
// =============================================================================

namespace SMEKernels1x4SymZAInOut {

    // =========================================================================
    // PACK A  (M x K) -> single-panel layout (GS = SVL)
    // Each m-step (SVL rows) gets its own contiguous K_curr * SVL block.
    // Within a block per k-step: SVL contiguous floats = one column of A.
    //
    // Transpose strategy: ZA-native, 2-tile rolling pipeline.
    //
    // For each pair of 16x16 source blocks:
    //   1. load 16 rows of block 0, write into horizontal slices of tile 0
    //   2. load 16 rows of block 1, write into horizontal slices of tile 1
    //   3. read 16 vertical slices of tile 0 (= transpose), store
    //   4. read 16 vertical slices of tile 1 (= transpose), store
    //
    // The point of using two tiles: between any tile-N write and the next
    // tile-N read, ~32 unrelated instructions sit in between -- enough to
    // hide the MOVA write-to-read latency that bottlenecked the single-tile
    // version. Same total MOVA count, just better scheduling.
    //
    // M-tail: per-row predicates zero inactive rows on load; tile rows beyond
    // rows_here are written as zero -> vertical reads return zero in those
    // lanes. Behaviorally identical to the butterfly path.
    //
    // K odd-block tail: if K_full / SVL is odd, the last full-SVL block falls
    // out of the pair loop and is handled by the single-tile path below.
    //
    // ZA ownership: __arm_out("za") declares that we clobber ZA. Caller is
    // __arm_new("za") and only invokes pack_A when no live accumulator state
    // is held (start of each m-strip, before kernel_zero).
    // =========================================================================
    __attribute__((noinline))
    void pack_A_streaming(const float* A, float* packed_A,
                          size_t M_curr, size_t K_curr,
                          size_t curr_row, size_t curr_col, size_t K) __arm_out("za") __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const size_t K_full = (K_curr / SVL) * SVL;
        const size_t K_pair = (K_full / (2 * SVL)) * (2 * SVL);
        const size_t GS = SVL;

        const svbool_t pg = svptrue_b32();
        const svbool_t pfalse = svpfalse_b();
        const svfloat32_t und = svundef_f32();

        for (size_t m = 0; m < M_curr; m += SVL) {
            const float* row_base = A + (m + curr_row) * K + curr_col;
            size_t m_idx = m / SVL;
            float* block_base = packed_A + m_idx * SVL * K_curr;
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

            // -----------------------------------------------------------------
            // 2-tile pipelined main loop: 2 blocks per iteration (k += 2*SVL).
            // -----------------------------------------------------------------
            for (; k < K_pair; k += 2 * SVL) {
                // ---- Block 0 at K=k -> tile 0 ----
                svfloat32_t a0  = svld1_f32(p0,  row_base + k);
                svfloat32_t a1  = svld1_f32(p1,  row_base + k +  1*K);
                svfloat32_t a2  = svld1_f32(p2,  row_base + k +  2*K);
                svfloat32_t a3  = svld1_f32(p3,  row_base + k +  3*K);
                svfloat32_t a4  = svld1_f32(p4,  row_base + k +  4*K);
                svfloat32_t a5  = svld1_f32(p5,  row_base + k +  5*K);
                svfloat32_t a6  = svld1_f32(p6,  row_base + k +  6*K);
                svfloat32_t a7  = svld1_f32(p7,  row_base + k +  7*K);
                svfloat32_t a8  = svld1_f32(p8,  row_base + k +  8*K);
                svfloat32_t a9  = svld1_f32(p9,  row_base + k +  9*K);
                svfloat32_t a10 = svld1_f32(p10, row_base + k + 10*K);
                svfloat32_t a11 = svld1_f32(p11, row_base + k + 11*K);
                svfloat32_t a12 = svld1_f32(p12, row_base + k + 12*K);
                svfloat32_t a13 = svld1_f32(p13, row_base + k + 13*K);
                svfloat32_t a14 = svld1_f32(p14, row_base + k + 14*K);
                svfloat32_t a15 = svld1_f32(p15, row_base + k + 15*K);

                svwrite_hor_za32_f32_m(0,  0, pg, a0);
                svwrite_hor_za32_f32_m(0,  1, pg, a1);
                svwrite_hor_za32_f32_m(0,  2, pg, a2);
                svwrite_hor_za32_f32_m(0,  3, pg, a3);
                svwrite_hor_za32_f32_m(0,  4, pg, a4);
                svwrite_hor_za32_f32_m(0,  5, pg, a5);
                svwrite_hor_za32_f32_m(0,  6, pg, a6);
                svwrite_hor_za32_f32_m(0,  7, pg, a7);
                svwrite_hor_za32_f32_m(0,  8, pg, a8);
                svwrite_hor_za32_f32_m(0,  9, pg, a9);
                svwrite_hor_za32_f32_m(0, 10, pg, a10);
                svwrite_hor_za32_f32_m(0, 11, pg, a11);
                svwrite_hor_za32_f32_m(0, 12, pg, a12);
                svwrite_hor_za32_f32_m(0, 13, pg, a13);
                svwrite_hor_za32_f32_m(0, 14, pg, a14);
                svwrite_hor_za32_f32_m(0, 15, pg, a15);

                // ---- Block 1 at K=k+SVL -> tile 1 ----
                svfloat32_t b0  = svld1_f32(p0,  row_base + k + SVL);
                svfloat32_t b1  = svld1_f32(p1,  row_base + k + SVL +  1*K);
                svfloat32_t b2  = svld1_f32(p2,  row_base + k + SVL +  2*K);
                svfloat32_t b3  = svld1_f32(p3,  row_base + k + SVL +  3*K);
                svfloat32_t b4  = svld1_f32(p4,  row_base + k + SVL +  4*K);
                svfloat32_t b5  = svld1_f32(p5,  row_base + k + SVL +  5*K);
                svfloat32_t b6  = svld1_f32(p6,  row_base + k + SVL +  6*K);
                svfloat32_t b7  = svld1_f32(p7,  row_base + k + SVL +  7*K);
                svfloat32_t b8  = svld1_f32(p8,  row_base + k + SVL +  8*K);
                svfloat32_t b9  = svld1_f32(p9,  row_base + k + SVL +  9*K);
                svfloat32_t b10 = svld1_f32(p10, row_base + k + SVL + 10*K);
                svfloat32_t b11 = svld1_f32(p11, row_base + k + SVL + 11*K);
                svfloat32_t b12 = svld1_f32(p12, row_base + k + SVL + 12*K);
                svfloat32_t b13 = svld1_f32(p13, row_base + k + SVL + 13*K);
                svfloat32_t b14 = svld1_f32(p14, row_base + k + SVL + 14*K);
                svfloat32_t b15 = svld1_f32(p15, row_base + k + SVL + 15*K);

                svwrite_hor_za32_f32_m(1,  0, pg, b0);
                svwrite_hor_za32_f32_m(1,  1, pg, b1);
                svwrite_hor_za32_f32_m(1,  2, pg, b2);
                svwrite_hor_za32_f32_m(1,  3, pg, b3);
                svwrite_hor_za32_f32_m(1,  4, pg, b4);
                svwrite_hor_za32_f32_m(1,  5, pg, b5);
                svwrite_hor_za32_f32_m(1,  6, pg, b6);
                svwrite_hor_za32_f32_m(1,  7, pg, b7);
                svwrite_hor_za32_f32_m(1,  8, pg, b8);
                svwrite_hor_za32_f32_m(1,  9, pg, b9);
                svwrite_hor_za32_f32_m(1, 10, pg, b10);
                svwrite_hor_za32_f32_m(1, 11, pg, b11);
                svwrite_hor_za32_f32_m(1, 12, pg, b12);
                svwrite_hor_za32_f32_m(1, 13, pg, b13);
                svwrite_hor_za32_f32_m(1, 14, pg, b14);
                svwrite_hor_za32_f32_m(1, 15, pg, b15);

                // ---- Read tile 0 vertical slices -> packed_A ----
                float* outA = block_base + k * GS;
                svst1_f32(pg, outA +  0*GS, svread_ver_za32_f32_m(und, pg, 0,  0));
                svst1_f32(pg, outA +  1*GS, svread_ver_za32_f32_m(und, pg, 0,  1));
                svst1_f32(pg, outA +  2*GS, svread_ver_za32_f32_m(und, pg, 0,  2));
                svst1_f32(pg, outA +  3*GS, svread_ver_za32_f32_m(und, pg, 0,  3));
                svst1_f32(pg, outA +  4*GS, svread_ver_za32_f32_m(und, pg, 0,  4));
                svst1_f32(pg, outA +  5*GS, svread_ver_za32_f32_m(und, pg, 0,  5));
                svst1_f32(pg, outA +  6*GS, svread_ver_za32_f32_m(und, pg, 0,  6));
                svst1_f32(pg, outA +  7*GS, svread_ver_za32_f32_m(und, pg, 0,  7));
                svst1_f32(pg, outA +  8*GS, svread_ver_za32_f32_m(und, pg, 0,  8));
                svst1_f32(pg, outA +  9*GS, svread_ver_za32_f32_m(und, pg, 0,  9));
                svst1_f32(pg, outA + 10*GS, svread_ver_za32_f32_m(und, pg, 0, 10));
                svst1_f32(pg, outA + 11*GS, svread_ver_za32_f32_m(und, pg, 0, 11));
                svst1_f32(pg, outA + 12*GS, svread_ver_za32_f32_m(und, pg, 0, 12));
                svst1_f32(pg, outA + 13*GS, svread_ver_za32_f32_m(und, pg, 0, 13));
                svst1_f32(pg, outA + 14*GS, svread_ver_za32_f32_m(und, pg, 0, 14));
                svst1_f32(pg, outA + 15*GS, svread_ver_za32_f32_m(und, pg, 0, 15));

                // ---- Read tile 1 vertical slices -> packed_A ----
                float* outB = block_base + (k + SVL) * GS;
                svst1_f32(pg, outB +  0*GS, svread_ver_za32_f32_m(und, pg, 1,  0));
                svst1_f32(pg, outB +  1*GS, svread_ver_za32_f32_m(und, pg, 1,  1));
                svst1_f32(pg, outB +  2*GS, svread_ver_za32_f32_m(und, pg, 1,  2));
                svst1_f32(pg, outB +  3*GS, svread_ver_za32_f32_m(und, pg, 1,  3));
                svst1_f32(pg, outB +  4*GS, svread_ver_za32_f32_m(und, pg, 1,  4));
                svst1_f32(pg, outB +  5*GS, svread_ver_za32_f32_m(und, pg, 1,  5));
                svst1_f32(pg, outB +  6*GS, svread_ver_za32_f32_m(und, pg, 1,  6));
                svst1_f32(pg, outB +  7*GS, svread_ver_za32_f32_m(und, pg, 1,  7));
                svst1_f32(pg, outB +  8*GS, svread_ver_za32_f32_m(und, pg, 1,  8));
                svst1_f32(pg, outB +  9*GS, svread_ver_za32_f32_m(und, pg, 1,  9));
                svst1_f32(pg, outB + 10*GS, svread_ver_za32_f32_m(und, pg, 1, 10));
                svst1_f32(pg, outB + 11*GS, svread_ver_za32_f32_m(und, pg, 1, 11));
                svst1_f32(pg, outB + 12*GS, svread_ver_za32_f32_m(und, pg, 1, 12));
                svst1_f32(pg, outB + 13*GS, svread_ver_za32_f32_m(und, pg, 1, 13));
                svst1_f32(pg, outB + 14*GS, svread_ver_za32_f32_m(und, pg, 1, 14));
                svst1_f32(pg, outB + 15*GS, svread_ver_za32_f32_m(und, pg, 1, 15));
            }

            // -----------------------------------------------------------------
            // Single-tile leftover: handles the final block if K_full/SVL is odd.
            // -----------------------------------------------------------------
            for (; k < K_full; k += SVL) {
                svfloat32_t r0  = svld1_f32(p0,  row_base + k);
                svfloat32_t r1  = svld1_f32(p1,  row_base + k +  1*K);
                svfloat32_t r2  = svld1_f32(p2,  row_base + k +  2*K);
                svfloat32_t r3  = svld1_f32(p3,  row_base + k +  3*K);
                svfloat32_t r4  = svld1_f32(p4,  row_base + k +  4*K);
                svfloat32_t r5  = svld1_f32(p5,  row_base + k +  5*K);
                svfloat32_t r6  = svld1_f32(p6,  row_base + k +  6*K);
                svfloat32_t r7  = svld1_f32(p7,  row_base + k +  7*K);
                svfloat32_t r8  = svld1_f32(p8,  row_base + k +  8*K);
                svfloat32_t r9  = svld1_f32(p9,  row_base + k +  9*K);
                svfloat32_t r10 = svld1_f32(p10, row_base + k + 10*K);
                svfloat32_t r11 = svld1_f32(p11, row_base + k + 11*K);
                svfloat32_t r12 = svld1_f32(p12, row_base + k + 12*K);
                svfloat32_t r13 = svld1_f32(p13, row_base + k + 13*K);
                svfloat32_t r14 = svld1_f32(p14, row_base + k + 14*K);
                svfloat32_t r15 = svld1_f32(p15, row_base + k + 15*K);

                svwrite_hor_za32_f32_m(0,  0, pg, r0);
                svwrite_hor_za32_f32_m(0,  1, pg, r1);
                svwrite_hor_za32_f32_m(0,  2, pg, r2);
                svwrite_hor_za32_f32_m(0,  3, pg, r3);
                svwrite_hor_za32_f32_m(0,  4, pg, r4);
                svwrite_hor_za32_f32_m(0,  5, pg, r5);
                svwrite_hor_za32_f32_m(0,  6, pg, r6);
                svwrite_hor_za32_f32_m(0,  7, pg, r7);
                svwrite_hor_za32_f32_m(0,  8, pg, r8);
                svwrite_hor_za32_f32_m(0,  9, pg, r9);
                svwrite_hor_za32_f32_m(0, 10, pg, r10);
                svwrite_hor_za32_f32_m(0, 11, pg, r11);
                svwrite_hor_za32_f32_m(0, 12, pg, r12);
                svwrite_hor_za32_f32_m(0, 13, pg, r13);
                svwrite_hor_za32_f32_m(0, 14, pg, r14);
                svwrite_hor_za32_f32_m(0, 15, pg, r15);

                float* out = block_base + k * GS;
                svst1_f32(pg, out +  0*GS, svread_ver_za32_f32_m(und, pg, 0,  0));
                svst1_f32(pg, out +  1*GS, svread_ver_za32_f32_m(und, pg, 0,  1));
                svst1_f32(pg, out +  2*GS, svread_ver_za32_f32_m(und, pg, 0,  2));
                svst1_f32(pg, out +  3*GS, svread_ver_za32_f32_m(und, pg, 0,  3));
                svst1_f32(pg, out +  4*GS, svread_ver_za32_f32_m(und, pg, 0,  4));
                svst1_f32(pg, out +  5*GS, svread_ver_za32_f32_m(und, pg, 0,  5));
                svst1_f32(pg, out +  6*GS, svread_ver_za32_f32_m(und, pg, 0,  6));
                svst1_f32(pg, out +  7*GS, svread_ver_za32_f32_m(und, pg, 0,  7));
                svst1_f32(pg, out +  8*GS, svread_ver_za32_f32_m(und, pg, 0,  8));
                svst1_f32(pg, out +  9*GS, svread_ver_za32_f32_m(und, pg, 0,  9));
                svst1_f32(pg, out + 10*GS, svread_ver_za32_f32_m(und, pg, 0, 10));
                svst1_f32(pg, out + 11*GS, svread_ver_za32_f32_m(und, pg, 0, 11));
                svst1_f32(pg, out + 12*GS, svread_ver_za32_f32_m(und, pg, 0, 12));
                svst1_f32(pg, out + 13*GS, svread_ver_za32_f32_m(und, pg, 0, 13));
                svst1_f32(pg, out + 14*GS, svread_ver_za32_f32_m(und, pg, 0, 14));
                svst1_f32(pg, out + 15*GS, svread_ver_za32_f32_m(und, pg, 0, 15));
            }

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
    // PACK B  (K x N) -> 4-panel interleaved layout (GS = 4*SVL)
    // Each jr-group (4*SVL cols) gets its own contiguous K_curr * 4*SVL block.
    // Layout per k-step: [panel0_SVL | panel1_SVL | panel2_SVL | panel3_SVL]
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
    // KERNEL ZERO -- svzero_za only. Called once per output tile.
    // =========================================================================
    __attribute__((noinline))
    void kernel_zero() __arm_out("za") __arm_streaming {
        svzero_za();
    }

    // =========================================================================
    // KERNEL COMPUTE -- pure FMOPA loop, ZA passes through __arm_inout.
    // Body is the 1x4-sym micro-kernel without svzero_za and without the
    // STORE_ZA_TILE writeback. May be called many times in sequence on the
    // same ZA scope; FMOPA accumulates onto whatever is already in ZA.
    //
    // Schedule per 4 k-steps (unchanged from 1x4-sym):
    //   1 A x4 load + 4 B x4 loads + 16 svmopa, software-pipelined.
    // =========================================================================
    __attribute__((noinline))
    void kernel_compute(const float* RESTRICT packed_A, const float* RESTRICT packed_B,
                        size_t K_curr) __arm_inout("za") __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t pg = svptrue_b32();
        const svcount_t pg4 = svptrue_c32();
        const size_t GS_B = 4 * SVL;

        // Software prefetch the NEXT call's chunks. With K_inner_tile ~ 40,
        // each call is short enough that the HW prefetcher barely warms up
        // before the call ends and pointers jump to the next chunk. Pre-
        // issuing a handful of T0 prefetches gives the prefetcher a head
        // start on the new streams. Out-of-bounds prefetches on the last
        // call are benign (treated as hints).
        {
            const float* next_pA = packed_A + K_curr * SVL;
            const float* next_pB = packed_B + K_curr * GS_B;
            __builtin_prefetch(next_pA,             0, 3);
            __builtin_prefetch(next_pB,             0, 3);
            __builtin_prefetch(next_pB + GS_B,      0, 3);
        }

        const float* pA = packed_A;
        const float* pB = packed_B;

        const size_t K_main = (K_curr / 4) * 4;

        if (K_main >= 4) {
            svfloat32x4_t a_x4 = svld1_f32_x4(pg4, pA); pA += 4 * SVL;
            svfloat32_t a0 = svget4_f32(a_x4, 0);
            svfloat32_t a1 = svget4_f32(a_x4, 1);
            svfloat32_t a2 = svget4_f32(a_x4, 2);
            svfloat32_t a3 = svget4_f32(a_x4, 3);

            svfloat32x4_t b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;

            for (size_t k = 0; k < K_main - 4; k += 4) {
                svfloat32_t b0 = svget4_f32(b_x4, 0);
                svfloat32_t b1 = svget4_f32(b_x4, 1);
                svfloat32_t b2 = svget4_f32(b_x4, 2);
                svfloat32_t b3 = svget4_f32(b_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a0, b0);
                svmopa_za32_f32_m(1, pg, pg, a0, b1);
                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
                svmopa_za32_f32_m(2, pg, pg, a0, b2);
                svmopa_za32_f32_m(3, pg, pg, a0, b3);

                b0 = svget4_f32(b_x4, 0);
                b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2);
                b3 = svget4_f32(b_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a1, b0);
                svmopa_za32_f32_m(1, pg, pg, a1, b1);
                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
                svmopa_za32_f32_m(2, pg, pg, a1, b2);
                svmopa_za32_f32_m(3, pg, pg, a1, b3);

                b0 = svget4_f32(b_x4, 0);
                b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2);
                b3 = svget4_f32(b_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a2, b0);
                svmopa_za32_f32_m(1, pg, pg, a2, b1);
                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
                svmopa_za32_f32_m(2, pg, pg, a2, b2);
                svmopa_za32_f32_m(3, pg, pg, a2, b3);

                b0 = svget4_f32(b_x4, 0);
                b1 = svget4_f32(b_x4, 1);
                b2 = svget4_f32(b_x4, 2);
                b3 = svget4_f32(b_x4, 3);

                svmopa_za32_f32_m(0, pg, pg, a3, b0);
                svmopa_za32_f32_m(1, pg, pg, a3, b1);
                a_x4 = svld1_f32_x4(pg4, pA); pA += 4 * SVL;
                svmopa_za32_f32_m(2, pg, pg, a3, b2);
                svmopa_za32_f32_m(3, pg, pg, a3, b3);

                a0 = svget4_f32(a_x4, 0);
                a1 = svget4_f32(a_x4, 1);
                a2 = svget4_f32(a_x4, 2);
                a3 = svget4_f32(a_x4, 3);
                b_x4 = svld1_f32_x4(pg4, pB); pB += GS_B;
            }

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
    // KERNEL STORE -- ZA tiles 0..3 -> C with += semantics.
    // ZA_i lands at C[0..SVL, i*SVL..(i+1)*SVL]. The += keeps the path
    // compatible with the edge-tile scratch buffer (pre-zeroed) and with any
    // future outer-K blocking that revisits the same C tile.
    // =========================================================================
    __attribute__((noinline))
    void kernel_store(float* RESTRICT C, size_t wide_of_C) __arm_in("za") __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t pg = svptrue_b32();
        svfloat32_t inactive = svundef_f32();

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
    // MAIN DRIVER -- ZA-resident accumulation across K_inner_tile chunks.
    //
    // For each (m, n) strip:
    //   1. pack A and B for the FULL K once
    //   2. for each output tile (jr, ir):
    //        kernel_zero
    //        for k_in in K_inner_tile chunks: kernel_compute
    //        kernel_store
    //
    // K_inner_tile sweep on M4 @ 2048^3 (1 thread):
    //    K_inner_tile=2048 (1 call/tile) :  992 GFLOPS, IPC 0.74
    //    K_inner_tile= 256 (8 calls/tile): 1131 GFLOPS, IPC 0.79
    //    K_inner_tile=  64 (32 calls/tile):1194 GFLOPS, IPC 1.40   <- best
    // Smaller chunks issue ~67% more total instructions but run 20% faster.
    // The function-call boundaries appear to drain a same-tile FMOPA
    // dependency chain that builds up in long uninterrupted bursts.
    // =========================================================================
    struct FreeDeleter { void operator()(void* p) { std::free(p); } };
    using AlignedBuffer = std::unique_ptr<float[], FreeDeleter>;

    __arm_locally_streaming __arm_new("za")
    __attribute__((noinline))
    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N) {
        const size_t SVL = static_cast<size_t>(svcntsw());

        constexpr size_t M_tile = 1024;
        constexpr size_t N_tile = 128;
        // Smaller is faster on M4 (see header block). 64 was the empirical
        // sweet spot -- short enough to cap the same-tile FMOPA dependency
        // chain, long enough that prologue/epilogue isn't most of the work.
        constexpr size_t K_inner_tile = 40;

        const size_t M_step = 1 * SVL;          // 16
        const size_t N_step = 4 * SVL;          // 64
        const size_t GS_A   = SVL;              // pack_A k-step group stride
        const size_t GS_B   = 4 * SVL;          // pack_B k-step group stride

        AlignedBuffer packed_A(static_cast<float*>(
            std::aligned_alloc(64, M_tile * K * sizeof(float))));
        AlignedBuffer packed_B(static_cast<float*>(
            std::aligned_alloc(64, K * N_tile * sizeof(float))));
        AlignedBuffer C_scratch(static_cast<float*>(
            std::aligned_alloc(64, M_step * N_step * sizeof(float))));

        for (size_t m = 0; m < M; m += M_tile) {
            size_t mc = std::min(M_tile, M - m);

            // Pack A for the full K of this M-strip ONCE; reused across all n.
            // (A's data doesn't depend on n, so this hoist is what matches the
            // 1x4-sym baseline's structure.)
            pack_A_streaming(A, packed_A.get(), mc, K, m, 0, K);

            for (size_t n = 0; n < N; n += N_tile) {
                size_t nc = std::min(N_tile, N - n);

                // pack_B depends on n, repacked per n-iteration.
                pack_B_streaming(B, packed_B.get(), nc, K, 0, N, n);

                for (size_t jr = 0; jr < nc; jr += N_step) {
                    size_t n_rem = nc - jr;
                    for (size_t ir = 0; ir < mc; ir += M_step) {
                        size_t m_rem = mc - ir;
                        bool m_tail = m_rem < M_step;
                        bool n_tail = n_rem < N_step;

                        const float* pA_tile = packed_A.get() + ir * K;
                        const float* pB_tile = packed_B.get() + jr * K;

                        if (!m_tail && !n_tail) {
                            kernel_zero();
                            for (size_t k_in = 0; k_in < K; k_in += K_inner_tile) {
                                size_t kic = std::min(K_inner_tile, K - k_in);
                                kernel_compute(pA_tile + k_in * GS_A,
                                               pB_tile + k_in * GS_B,
                                               kic);
                            }
                            kernel_store(C + (m + ir) * N + (n + jr), N);
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

                            kernel_zero();
                            for (size_t k_in = 0; k_in < K; k_in += K_inner_tile) {
                                size_t kic = std::min(K_inner_tile, K - k_in);
                                kernel_compute(pA_tile + k_in * GS_A,
                                               pB_tile + k_in * GS_B,
                                               kic);
                            }
                            kernel_store(C_scratch.get(), N_step);

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

} // namespace SMEKernels1x4SymZAInOut
