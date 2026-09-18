#include "sme-1x4-bstream-reg3.hpp"

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <memory>
#include <algorithm>

#define RESTRICT __restrict__

// =============================================================================
// SME 1x4, B streamed from memory, prefetch depth 3  (EXPERIMENT)
//
// B IS NEVER PACKED. Measured 2026-09-10 at 256^3: routing B through a staging
// buffer in memory cost -51% (double buffered) to -73% (single buffered),
// while dropping the packed B panel entirely cost nothing at all (1461.6 vs
// 1469.6 GFLOP/s). The micro-kernel sits close to the load/store issue limit --
// 5 memory instructions per 16 svmopa -- so any scheme that adds memory traffic
// on the B path loses more than the locality it buys. B is therefore read
// straight from memory at stride N and fed to the svmopa from registers.
//
// DEPTH 3: three B quads are kept in flight, so each load is issued 12 svmopa
// before its use instead of immediately before it. Costs 4 more live Z
// registers (20 of 32) and adds no memory instructions at all -- the whole
// point, after the staging-buffer attempt lost on exactly that axis.
//
// A is packed once, in full, before any loop runs.
// =============================================================================

namespace SMEKernels1x4BStreamReg3 {

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

    struct FreeDeleter { void operator()(void* p) { std::free(p); } };
    using AlignedBuffer = std::unique_ptr<float[], FreeDeleter>;

    __arm_locally_streaming __arm_new("za")
    __attribute__((noinline))
    void run_multiplication(const float* A, const float* B, float* C,
                            size_t M, size_t K, size_t N) {
        const size_t SVL    = static_cast<size_t>(svcntsw());
        const size_t M_step = 1 * SVL;       // rows per ZA tile column block
        const size_t N_step = 4 * SVL;       // columns per tile: 4 ZA tiles wide
        const size_t GRP    = 4;             // k-steps per A x4 load

        const svbool_t  pg  = svptrue_b32();
        const svcount_t pg4 = svptrue_c32();

        // M is rounded up to M_step because pack_A consumes a whole SVL-row
        // block even when the last one is partial.
        const size_t M_padded = ((M + M_step - 1) / M_step) * M_step;
        AlignedBuffer packed_A(static_cast<float*>(
            std::aligned_alloc(64, M_padded * K * sizeof(float))));
        pack_A_streaming(A, packed_A.get(), M, K, 0, 0, K);

        const size_t K_main = (K / GRP) * GRP;

        for (size_t n = 0; n < N; n += N_step) {
            for (size_t ir = 0; ir < M; ir += M_step) {
                // A panel ir is a contiguous K * M_step block at ir * K.
                // bsrc walks B down column block n at stride N.
                const float* pA   = packed_A.get() + ir * K;
                const float* bsrc = B + n;

                svzero_za();
                    if (K_main >= GRP) {
                        // PROLOGUE (pipeline fill): the first three k-steps of B.
                        // Nothing to hide these behind; they are the only exposed
                        // B loads in the whole tile.
                        svfloat32x4_t c0 = svld1_f32_x4(pg4, bsrc + 0 * N);
                        svfloat32x4_t c1 = svld1_f32_x4(pg4, bsrc + 1 * N);
                        svfloat32x4_t c2 = svld1_f32_x4(pg4, bsrc + 2 * N);

                        // STEADY: each group loads the k-steps the NEXT group will
                        // consume, so every load is issued 12 svmopa ahead of its
                        // use and rides the ZA pipeline. Four B quads plus one A
                        // quad stay live -- 20 of 32 Z registers, no spill.
                        // The last group has no successor to load for, so it is
                        // peeled off below rather than guarded by a branch here.
                        for (size_t k = 0; k + GRP < K_main; k += GRP) {
                        svfloat32x4_t a_x4 = svld1_f32_x4(pg4, pA); pA += GRP * SVL;
                        svfloat32_t a0 = svget4_f32(a_x4, 0);
                        svfloat32_t a1 = svget4_f32(a_x4, 1);
                        svfloat32_t a2 = svget4_f32(a_x4, 2);
                        svfloat32_t a3 = svget4_f32(a_x4, 3);

                            // PREFETCH: k+3, used 12 svmopa below.
                            svfloat32x4_t c3 = svld1_f32_x4(pg4, bsrc + 3 * N);
                            svmopa_za32_f32_m(0, pg, pg, a0, svget4_f32(c0, 0));
                            svmopa_za32_f32_m(1, pg, pg, a0, svget4_f32(c0, 1));
                            svmopa_za32_f32_m(2, pg, pg, a0, svget4_f32(c0, 2));
                            svmopa_za32_f32_m(3, pg, pg, a0, svget4_f32(c0, 3));

                            // PREFETCH: k+4, the next group's first k-step.
                            c0 = svld1_f32_x4(pg4, bsrc + 4 * N);
                            svmopa_za32_f32_m(0, pg, pg, a1, svget4_f32(c1, 0));
                            svmopa_za32_f32_m(1, pg, pg, a1, svget4_f32(c1, 1));
                            svmopa_za32_f32_m(2, pg, pg, a1, svget4_f32(c1, 2));
                            svmopa_za32_f32_m(3, pg, pg, a1, svget4_f32(c1, 3));

                            // PREFETCH: k+5.
                            c1 = svld1_f32_x4(pg4, bsrc + 5 * N);
                            svmopa_za32_f32_m(0, pg, pg, a2, svget4_f32(c2, 0));
                            svmopa_za32_f32_m(1, pg, pg, a2, svget4_f32(c2, 1));
                            svmopa_za32_f32_m(2, pg, pg, a2, svget4_f32(c2, 2));
                            svmopa_za32_f32_m(3, pg, pg, a2, svget4_f32(c2, 3));

                            // PREFETCH: k+6.
                            c2 = svld1_f32_x4(pg4, bsrc + 6 * N);
                            svmopa_za32_f32_m(0, pg, pg, a3, svget4_f32(c3, 0));
                            svmopa_za32_f32_m(1, pg, pg, a3, svget4_f32(c3, 1));
                            svmopa_za32_f32_m(2, pg, pg, a3, svget4_f32(c3, 2));
                            svmopa_za32_f32_m(3, pg, pg, a3, svget4_f32(c3, 3));

                            bsrc += GRP * N;
                        }

                        // EPILOGUE: the last full group. c0,c1,c2 are already in
                        // registers; only k+3 is still missing, and there is no
                        // successor group to load for.
                        svfloat32x4_t a_x4 = svld1_f32_x4(pg4, pA); pA += GRP * SVL;
                        svfloat32_t a0 = svget4_f32(a_x4, 0);
                        svfloat32_t a1 = svget4_f32(a_x4, 1);
                        svfloat32_t a2 = svget4_f32(a_x4, 2);
                        svfloat32_t a3 = svget4_f32(a_x4, 3);
                        svfloat32x4_t c3 = svld1_f32_x4(pg4, bsrc + 3 * N);
                        svmopa_za32_f32_m(0, pg, pg, a0, svget4_f32(c0, 0));
                        svmopa_za32_f32_m(1, pg, pg, a0, svget4_f32(c0, 1));
                        svmopa_za32_f32_m(2, pg, pg, a0, svget4_f32(c0, 2));
                        svmopa_za32_f32_m(3, pg, pg, a0, svget4_f32(c0, 3));
                        svmopa_za32_f32_m(0, pg, pg, a1, svget4_f32(c1, 0));
                        svmopa_za32_f32_m(1, pg, pg, a1, svget4_f32(c1, 1));
                        svmopa_za32_f32_m(2, pg, pg, a1, svget4_f32(c1, 2));
                        svmopa_za32_f32_m(3, pg, pg, a1, svget4_f32(c1, 3));
                        svmopa_za32_f32_m(0, pg, pg, a2, svget4_f32(c2, 0));
                        svmopa_za32_f32_m(1, pg, pg, a2, svget4_f32(c2, 1));
                        svmopa_za32_f32_m(2, pg, pg, a2, svget4_f32(c2, 2));
                        svmopa_za32_f32_m(3, pg, pg, a2, svget4_f32(c2, 3));
                        svmopa_za32_f32_m(0, pg, pg, a3, svget4_f32(c3, 0));
                        svmopa_za32_f32_m(1, pg, pg, a3, svget4_f32(c3, 1));
                        svmopa_za32_f32_m(2, pg, pg, a3, svget4_f32(c3, 2));
                        svmopa_za32_f32_m(3, pg, pg, a3, svget4_f32(c3, 3));
                        bsrc += GRP * N;
                    }
                    // K TAIL: the last K % GRP k-steps, read one at a time.
                    for (size_t k = K_main; k < K; k++) {
                        svfloat32_t   a = svld1_f32(pg, pA); pA += SVL;
                        svfloat32x4_t b = svld1_f32_x4(pg4, bsrc); bsrc += N;
                        svmopa_za32_f32_m(0, pg, pg, a, svget4_f32(b, 0));
                        svmopa_za32_f32_m(1, pg, pg, a, svget4_f32(b, 1));
                        svmopa_za32_f32_m(2, pg, pg, a, svget4_f32(b, 2));
                        svmopa_za32_f32_m(3, pg, pg, a, svget4_f32(b, 3));
                    }
                store_za(C + ir * N + n, N);
            }
        }
    }

} // namespace SMEKernels1x4BStreamReg3
