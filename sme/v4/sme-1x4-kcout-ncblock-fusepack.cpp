#include "sme-1x4-kcout-ncblock-fusepack.hpp"
#include "sme/support/lab_hooks.hpp"

#include <arm_sme.h>
#include <arm_sve.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <algorithm>

#define RESTRICT __restrict__

#ifndef MAXMULSK_FUSEPACK_FILL
#define MAXMULSK_FUSEPACK_FILL 0
#endif

// 0 = control build: packing is still hoisted, but no B row goes into an A
// chunk slot, so fuse_pack packs the A block and then the B block back to back.
#ifndef MAXMULSK_FUSEPACK_FUSE
#define MAXMULSK_FUSEPACK_FUSE 1
#endif

// B traversal inside fuse_pack. 1 (default since 2026-09-26): row order -- all
// 64-column panels of row k, then row k+1, so each B row is read as one
// contiguous Nc-float run. 0: column order, as the original pack_B (one panel
// down all k), kept to reproduce the earlier measurements. The packed layout is
// the same either way.
#ifndef MAXMULSK_FUSEPACK_B_ROWMAJOR
#define MAXMULSK_FUSEPACK_B_ROWMAJOR 1
#endif

// Software prefetch distances in the fused pack (prfm pldl2keep; 0 = off).
// PFA: A chunks ahead on the same A row; PFB: B rows ahead. Chosen from the
// Lab prefetch search of 2026-09-26: PFA = 1 helped or was neutral on every
// geometry, PFB > 0 helped far-strided B but cost 30-40% on contiguous B.
#ifndef MAXMULSK_FUSEPACK_PFA
#define MAXMULSK_FUSEPACK_PFA 1
#endif
#ifndef MAXMULSK_FUSEPACK_PFB
#define MAXMULSK_FUSEPACK_PFB 0
#endif

// Byte offset added to the start of packed B inside its allocation (a multiple
// of 64). Large aligned_alloc blocks start on 64 KiB / 1 MiB boundaries, so
// packed A and packed B otherwise begin at the same cache-set offset. 0 keeps
// the allocation exactly as before. Experiment of 2026-09-27.
#ifndef MAXMULSK_FUSEPACK_SKEW_B
#define MAXMULSK_FUSEPACK_SKEW_B 0
#endif
static_assert(MAXMULSK_FUSEPACK_SKEW_B % 64 == 0, "packed B skew must keep 64-byte alignment");

// =============================================================================
// Copy of sme-1x4-kcout-ncblock-apack4za.cpp. Everything from here down to the
// end of store_za_add is carried over verbatim (macro prefix NB_ -> FK_, the
// helpers made static), except pack_B_streaming, which is dropped: B is packed
// by fuse_pack below, with the same full-width 64-column row copies. Only the
// driver below store_za_add is new. See the header for the loop structure.
// =============================================================================

// Output group g of the vertical drain is four consecutive vertical slices of
// one ZA tile: tile g/4, first column 4*(g%4). Both must be compile-time
// constants -- the tile operand of svread_ver_za32 is an immediate.
#define FK_T(g) ((g) / 4)
#define FK_C(g) (4 * ((g) % 4))
#define FK_GROUP(g)                                                            \
    svcreate4_f32(svread_ver_za32_f32_m(zu, pg, FK_T(g), FK_C(g) + 0),         \
                  svread_ver_za32_f32_m(zu, pg, FK_T(g), FK_C(g) + 1),         \
                  svread_ver_za32_f32_m(zu, pg, FK_T(g), FK_C(g) + 2),         \
                  svread_ver_za32_f32_m(zu, pg, FK_T(g), FK_C(g) + 3))

// One steady step: store the group a Z set already holds, then refill that same
// Z set with the group two ahead. Store operand is consumed before the refill,
// so the two x4 store streams never wait on each other.
#define FK_DRAIN(g)                                                            \
    svst1_f32_x4(p4, out + (g) * 64, s0);                                      \
    s0 = FK_GROUP((g) + 2);                                                    \
    svst1_f32_x4(p4, out + ((g) + 1) * 64, s1);                                \
    s1 = FK_GROUP((g) + 3);

#define FK_PACK_A_KP_CHUNK(SRC, STRIDE)                                           \
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
        MAXMULSK_HOOK_BEGIN(pack_a_read_transform);                            \
        svfloat32x4_t q0 = svld1_f32_x4(p4, (SRC) + 0 * (STRIDE));             \
        svfloat32x4_t q1 = svld1_f32_x4(p4, (SRC) + 1 * (STRIDE));             \
        for (uint32_t r = 0; r < 14; r += 2) {                         \
            svwrite_hor_za32_f32_m(0, r, pg, svget4_f32(q0, 0));       \
            svwrite_hor_za32_f32_m(1, r, pg, svget4_f32(q0, 1));       \
            svwrite_hor_za32_f32_m(2, r, pg, svget4_f32(q0, 2));       \
            svwrite_hor_za32_f32_m(3, r, pg, svget4_f32(q0, 3));       \
            /* every component of q0 is consumed; refill it */         \
            q0 = svld1_f32_x4(p4, (SRC) + (r + 2) * (STRIDE));                 \
            svwrite_hor_za32_f32_m(0, r + 1, pg, svget4_f32(q1, 0));   \
            svwrite_hor_za32_f32_m(1, r + 1, pg, svget4_f32(q1, 1));   \
            svwrite_hor_za32_f32_m(2, r + 1, pg, svget4_f32(q1, 2));   \
            svwrite_hor_za32_f32_m(3, r + 1, pg, svget4_f32(q1, 3));   \
            q1 = svld1_f32_x4(p4, (SRC) + (r + 3) * (STRIDE));                 \
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
        MAXMULSK_HOOK_END(pack_a_read_transform);                              \
        MAXMULSK_HOOK_BEGIN(pack_a_transform_store);                           \
        float* const out = pA + kp * 16;                         \
        svfloat32x4_t s0 = FK_GROUP(0);                                \
        svfloat32x4_t s1 = FK_GROUP(1);                                \
        /* Static unroll: the tile operand of svread_ver_za32 must */  \
        /* be an immediate, so g cannot be a runtime variable. */      \
        FK_DRAIN(0)  FK_DRAIN(2)  FK_DRAIN(4)  FK_DRAIN(6)             \
        FK_DRAIN(8)  FK_DRAIN(10) FK_DRAIN(12)                         \
        /* DRAIN: s0 holds group 14 (ZA3 vertical 8..11), */           \
        /* s1 holds group 15 (ZA3 vertical 12..15). No further */      \
        /* extraction -- the two stores are back to back. */           \
        svst1_f32_x4(p4, out + 14 * 64, s0);                           \
        svst1_f32_x4(p4, out + 15 * 64, s1);                           \
        MAXMULSK_HOOK_END(pack_a_transform_store);                             \
    } while (0)

namespace MAXMULSK_FUSEPACK_NS {

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
    static void micro_kernel_1x4(float* RESTRICT packed_A, float* RESTRICT packed_B,
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

    // Edge tiles only: drain the full 16x64 accumulator into a dense scratch,
    // from which the driver copies just the valid rows and columns. Same body as
    // store_za with a fixed 64-float row stride; the full-tile path does not go
    // through here and is left byte-identical.
    __attribute__((noinline))
    static void store_za_tile(float* RESTRICT scratch) __arm_in("za") __arm_streaming {
        const svbool_t pg = svptrue_b32();
        const svcount_t pn = svptrue_c32();
        const size_t SVL = static_cast<size_t>(svcntsw());
        svfloat32_t inactive = svundef_f32();
        for (size_t i = 0; i < SVL; i++) {
            svfloat32x4_t row = svcreate4_f32(
                svread_hor_za32_f32_m(inactive, pg, 0, (uint32_t)i),
                svread_hor_za32_f32_m(inactive, pg, 1, (uint32_t)i),
                svread_hor_za32_f32_m(inactive, pg, 2, (uint32_t)i),
                svread_hor_za32_f32_m(inactive, pg, 3, (uint32_t)i));
            svst1_f32_x4(pn, scratch + i * 4 * SVL, row);
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

    int fill_slots_enabled() { return MAXMULSK_FUSEPACK_FILL; }

    static Support fk_classify(size_t M, size_t K, size_t N, const Blocking& b) {
        if (M == 0 || K == 0 || N == 0)  return Support::Unsupported;
        if (b.Mc == 0 || b.Mc % 16 != 0) return Support::BadBlocking;
        if (b.Nc == 0 || b.Nc % 64 != 0) return Support::BadBlocking;
        if (b.Kc == 0 || b.Kc % 64 != 0) return Support::BadBlocking;
        return Support::Native;
    }

// One B row of the fused pack, software-pipelined by one: store the row loaded
// on the previous step, then load the next. Invariants the macro relies on:
//   * bp holds a row that has been loaded and not yet stored; its destination
//     is bd_prev. So every step stores exactly one row and loads exactly one.
//   * bleft counts rows not yet loaded; callers never step with bleft == 0.
//   * Packed B panel p starts at packed_B + p*Kc*64 (the ALLOCATED Kc, as in
//     the original driver) and row k of it is at +k*64, so within a panel the
//     destination advances by 64 and at k == kcl it jumps to the next panel
//     slot while the source moves 64 columns right.
// Like the original driver, every panel is read 64 columns wide, including the
// last one of a ragged N.
#define FK_BSTEP()                                                             \
    do {                                                                       \
        svst1_f32_x4(p4, bd_prev, bp);                                         \
        bp = svld1_f32_x4(p4, bs);                                             \
        bd_prev = bd; bleft--;                                                 \
        FK_BADVANCE();                                                         \
    } while (0)

// Moves the B cursor (bs source, bd destination) to the next 64-float piece.
// Row order: panel index bi runs 0..npanels-1 across row k (source +64,
// destination +Kc*64 to the next panel slot), then wraps to row k+1 (source
// +N from the row start, destination +64 from the row's panel-0 slot).
// Column order: the original walk, k = 0..kcl-1 down one panel, then the next.
#if MAXMULSK_FUSEPACK_B_ROWMAJOR
#define FK_BADVANCE()                                                          \
    do {                                                                       \
        if (++bi == npanels) { bi = 0; bsrc_row += N; bs = bsrc_row; bd_row += 64; bd = bd_row; } \
        else { bs += 64; bd += Kc * 64; }                                      \
        if (MAXMULSK_FUSEPACK_PFB) __asm__ volatile("prfm pldl2keep, [%0]" :: "r"(bs + MAXMULSK_FUSEPACK_PFB * N)); \
    } while (0)
#else
#define FK_BADVANCE()                                                          \
    do {                                                                       \
        if (++bk == kcl) {                                                     \
            bk = 0; bsrc_panel += 64; bs = bsrc_panel;                         \
            bd_panel += Kc * 64; bd = bd_panel;                                \
        } else { bs += N; bd += 64; }                                          \
        if (MAXMULSK_FUSEPACK_PFB) __asm__ volatile("prfm pldl2keep, [%0]" :: "r"(bs + MAXMULSK_FUSEPACK_PFB * N)); \
    } while (0)
#endif

// Prefetch of the A rows' next chunk(s), issued next to the FILL loads.
#define FK_PFA_ROW(ROWPTR)                                                     \
    do { if (MAXMULSK_FUSEPACK_PFA) __asm__ volatile("prfm pldl2keep, [%0]" :: "r"((ROWPTR) + MAXMULSK_FUSEPACK_PFA * 64)); } while (0)

#define FK_SLOT(i)                                                             \
    do { for (uint32_t q_ = quota[i]; q_ && bleft; q_--) FK_BSTEP(); } while (0)

#if MAXMULSK_FUSEPACK_FILL
#define FK_FILL_SLOT(i) FK_SLOT(i)
#else
#define FK_FILL_SLOT(i) ((void)0)
#endif

// FK_PACK_A_KP_CHUNK with B slots: 0-6 after each FILL step, 7 after the FILL
// tail, 8-14 after each DRAIN step, 15 after the final stores. The A work is
// the same instruction sequence as the original macro. ZA is scratch only:
// every slice read in the DRAIN was written in this chunk's FILL, so nothing is
// live across chunks and no zeroing is needed.
#define FK_PACK_A_KP_CHUNK_FUSED(SRC, STRIDE)                                  \
    do {                                                                       \
        svfloat32x4_t q0 = svld1_f32_x4(p4, (SRC) + 0 * (STRIDE));             \
        svfloat32x4_t q1 = svld1_f32_x4(p4, (SRC) + 1 * (STRIDE));             \
        for (uint32_t r = 0; r < 14; r += 2) {                                 \
            svwrite_hor_za32_f32_m(0, r, pg, svget4_f32(q0, 0));               \
            svwrite_hor_za32_f32_m(1, r, pg, svget4_f32(q0, 1));               \
            svwrite_hor_za32_f32_m(2, r, pg, svget4_f32(q0, 2));               \
            svwrite_hor_za32_f32_m(3, r, pg, svget4_f32(q0, 3));               \
            q0 = svld1_f32_x4(p4, (SRC) + (r + 2) * (STRIDE));                 \
            svwrite_hor_za32_f32_m(0, r + 1, pg, svget4_f32(q1, 0));           \
            svwrite_hor_za32_f32_m(1, r + 1, pg, svget4_f32(q1, 1));           \
            svwrite_hor_za32_f32_m(2, r + 1, pg, svget4_f32(q1, 2));           \
            svwrite_hor_za32_f32_m(3, r + 1, pg, svget4_f32(q1, 3));           \
            q1 = svld1_f32_x4(p4, (SRC) + (r + 3) * (STRIDE));                 \
            FK_PFA_ROW((SRC) + r * (STRIDE));                                  \
            FK_PFA_ROW((SRC) + (r + 1) * (STRIDE));                            \
            FK_FILL_SLOT(r / 2);                                               \
        }                                                                      \
        FK_PFA_ROW((SRC) + 14 * (STRIDE));                                     \
        FK_PFA_ROW((SRC) + 15 * (STRIDE));                                     \
        svwrite_hor_za32_f32_m(0, 14, pg, svget4_f32(q0, 0));                  \
        svwrite_hor_za32_f32_m(1, 14, pg, svget4_f32(q0, 1));                  \
        svwrite_hor_za32_f32_m(2, 14, pg, svget4_f32(q0, 2));                  \
        svwrite_hor_za32_f32_m(3, 14, pg, svget4_f32(q0, 3));                  \
        svwrite_hor_za32_f32_m(0, 15, pg, svget4_f32(q1, 0));                  \
        svwrite_hor_za32_f32_m(1, 15, pg, svget4_f32(q1, 1));                  \
        svwrite_hor_za32_f32_m(2, 15, pg, svget4_f32(q1, 2));                  \
        svwrite_hor_za32_f32_m(3, 15, pg, svget4_f32(q1, 3));                  \
        FK_FILL_SLOT(7);                                                       \
        float* const out = pA + kp * 16;                                       \
        svfloat32x4_t s0 = FK_GROUP(0);                                        \
        svfloat32x4_t s1 = FK_GROUP(1);                                        \
        FK_DRAIN(0)  FK_SLOT(8);  FK_DRAIN(2)  FK_SLOT(9);                     \
        FK_DRAIN(4)  FK_SLOT(10); FK_DRAIN(6)  FK_SLOT(11);                    \
        FK_DRAIN(8)  FK_SLOT(12); FK_DRAIN(10) FK_SLOT(13);                    \
        FK_DRAIN(12) FK_SLOT(14);                                              \
        svst1_f32_x4(p4, out + 14 * 64, s0);                                   \
        svst1_f32_x4(p4, out + 15 * 64, s1);                                   \
        FK_SLOT(15);                                                           \
    } while (0)

// Edge A chunk (rows < 16 or kr < 64), verbatim from the original driver: a
// dense zero-filled 16x64 staging block packed with the same macro at stride 64.
// Zeroing uses predicated SVE stores, never memset, which would lower to
// __arm_sc_memset inside streaming code.
#define FK_PACK_A_EDGE_CHUNK()                                                 \
    do {                                                                       \
        for (size_t r = 0; r < 16; r++) {                                      \
            const size_t have = (r < rows) ? kr : 0;                           \
            const float* srow = arow_base + (r < rows ? r : 0) * K + kp;       \
            float* drow = stage + r * 64;                                      \
            for (size_t j = 0; j < 64; j += SVL) {                             \
                const svbool_t pv = svwhilelt_b32_u64(j, have);                \
                svst1_f32(pg, drow + j, svld1_f32(pv, srow + j));              \
            }                                                                  \
        }                                                                      \
        FK_PACK_A_KP_CHUNK(stage, 64);                                         \
    } while (0)

    // Packs the Mc block starting at row mm into packed_A, one 16-row slot per
    // (ir/16) at stride Kc_pad*16 -- the slot layout of the original driver.
    // ZA is used as transpose scratch only (see the chunk macros).
    __attribute__((always_inline)) static inline
    void pack_A_block(const float* A, size_t K, size_t mm, size_t mcl,
                      size_t kk, size_t kcl, size_t Kc_pad,
                      float* packed_A, float* stage) __arm_streaming __arm_inout("za") {
        const svbool_t  pg = svptrue_b32();
        const svcount_t p4 = svptrue_c32();
        const svfloat32_t zu = svundef_f32();
        const size_t SVL = static_cast<size_t>(svcntsw());
        for (size_t ir = 0; ir < mcl; ir += 16) {
            float* const pA = packed_A + (ir / 16) * (Kc_pad * 16);
            const size_t rows = std::min<size_t>(16, mcl - ir);
            const float* const arow_base = A + (mm + ir) * K + kk;
            for (size_t kp = 0; kp < kcl; kp += 64) {
                const size_t kr = std::min<size_t>(64, kcl - kp);
                if (rows == 16 && kr == 64) {
                    const float* const arow = arow_base + kp;
                    FK_PACK_A_KP_CHUNK(arow, K);
                } else {
                    FK_PACK_A_EDGE_CHUNK();
                }
            }
        }
    }

    // Packs the whole (kcl x ncl) B block at columns nn into packed_B AND the Mc
    // block at row 0 into packed_A, with B rows spread over the slots of every
    // FULL A chunk. Rows left over when the full chunks run out (or all of B,
    // when there is no full chunk) are copied after the A work.
    __attribute__((always_inline)) static inline
    void fuse_pack(const float* A, const float* B, size_t K, size_t N,
                   size_t mcl, size_t nn, size_t ncl, size_t kk, size_t kcl,
                   size_t Kc, size_t Kc_pad,
                   float* packed_A, float* packed_B, float* stage) __arm_streaming __arm_inout("za") {
        const svbool_t  pg = svptrue_b32();
        const svcount_t p4 = svptrue_c32();
        const svfloat32_t zu = svundef_f32();
        const size_t SVL = static_cast<size_t>(svcntsw());

        const size_t npanels = (ncl + 63) / 64;
        const size_t full_chunks = (mcl / 16) * (kcl / 64);
        const size_t brows = npanels * kcl;

        uint32_t quota[16] = {0};
        if (full_chunks && MAXMULSK_FUSEPACK_FUSE) {
            const uint32_t R = static_cast<uint32_t>((brows + full_chunks - 1) / full_chunks);
            const uint32_t first = MAXMULSK_FUSEPACK_FILL ? 0 : 8;
            const uint32_t nslots = 16 - first;
            for (uint32_t i = 0; i < nslots; i++)
                quota[first + i] = R / nslots + (i < R % nslots ? 1 : 0);
        }

        const float* bsrc_panel = B + kk * N + nn;
        const float* bsrc_row = bsrc_panel;
        const float* bs = bsrc_panel;
        float* bd_panel = packed_B;
        float* bd_row = packed_B;
        float* bd = bd_panel;
        float* bd_prev = bd;
        size_t bk = 0, bi = 0;
        size_t bleft = brows;
        (void)bk; (void)bi; (void)bsrc_row; (void)bd_row; (void)bd_panel;
        // Prime the pipeline: load the first piece without storing anything.
        svfloat32x4_t bp = svld1_f32_x4(p4, bs);
        bd_prev = bd; bleft--;
        FK_BADVANCE();

        for (size_t ir = 0; ir < mcl; ir += 16) {
            float* const pA = packed_A + (ir / 16) * (Kc_pad * 16);
            const size_t rows = std::min<size_t>(16, mcl - ir);
            const float* const arow_base = A + ir * K + kk;
            for (size_t kp = 0; kp < kcl; kp += 64) {
                const size_t kr = std::min<size_t>(64, kcl - kp);
                if (rows == 16 && kr == 64) {
                    const float* const arow = arow_base + kp;
                    FK_PACK_A_KP_CHUNK_FUSED(arow, K);
                } else {
                    FK_PACK_A_EDGE_CHUNK();
                }
            }
        }

        while (bleft) FK_BSTEP();
        svst1_f32_x4(p4, bd_prev, bp);
    }

    // Computes the Mc block at row mm against the packed Nc block, in the same
    // tile order as the original driver (64-column panel outer, 16-row panel
    // inner) with the same ZA zeroing and the same full/edge writeback, so C is
    // bit-identical. packed_A / packed_B must already hold this block.
    __attribute__((always_inline)) static inline
    void compute_block(float* C, size_t N, size_t mm, size_t mcl,
                       size_t nn, size_t ncl, size_t kcl, bool first_k,
                       size_t Kc, size_t Kc_pad,
                       float* packed_A, float* packed_B, float* scratch) __arm_streaming __arm_inout("za") {
        const size_t SVL = static_cast<size_t>(svcntsw());
        for (size_t nr = 0; nr < ncl; nr += 64) {
            float* const pB = packed_B + (nr / 64) * (Kc * 64);
            for (size_t ir = 0; ir < mcl; ir += 16) {
                float* const pA = packed_A + (ir / 16) * (Kc_pad * 16);
                const size_t rows = std::min<size_t>(16, mcl - ir);

                MAXMULSK_HOOK_EVENT(za_init);
                MAXMULSK_HOOK_BEGIN(za_init);
                svzero_za();
                MAXMULSK_HOOK_END(za_init);
                MAXMULSK_HOOK_EVENT(compute);
                MAXMULSK_HOOK_BEGIN(compute);
                micro_kernel_1x4(pA, pB, kcl);
                MAXMULSK_HOOK_END(compute);

                float* const dstC = C + (mm + ir) * N + (nn + nr);
                const size_t cols = std::min<size_t>(64, ncl - nr);
                MAXMULSK_HOOK_EVENT(writeback);
                MAXMULSK_HOOK_BEGIN(writeback);
                if (rows == 16 && cols == 64) {
                    if (first_k) store_za(dstC, N);
                    else         store_za_add(dstC, N);
                } else {
                    store_za_tile(scratch);
                    for (size_t r = 0; r < rows; r++) {
                        const float* sr = scratch + r * 64;
                        float* cr = dstC + r * N;
                        for (size_t j = 0; j < cols; j += SVL) {
                            const svbool_t pv = svwhilelt_b32_u64(j, cols);
                            svfloat32_t v = svld1_f32(pv, sr + j);
                            if (!first_k) v = svadd_f32_m(pv, v, svld1_f32(pv, cr + j));
                            svst1_f32(pv, cr + j, v);
                        }
                    }
                }
                MAXMULSK_HOOK_END(writeback);
            }
        }
    }

    // One streaming region for the whole GEMM. The mm = 0 round is peeled out of
    // the Mc loop so the fused pack needs no branch: it runs exactly once per
    // (kk, nn), where both a fresh B block and a fresh A block are needed.
    __arm_locally_streaming __arm_new("za")
    __attribute__((noinline))
    static bool run_streaming(const float* A, const float* B, float* C,
                              size_t M, size_t K, size_t N,
                              float* packed_A, float* packed_B,
                              float* stage, float* scratch,
                              size_t Kc, size_t Nc, size_t Mc) {
        if (static_cast<size_t>(svcntsw()) != 16) return false;
        const size_t Kc_pad = ((Kc + 63) / 64) * 64;

        for (size_t kk = 0; kk < K; kk += Kc) {
            const size_t kcl = std::min(Kc, K - kk);
            const bool first_k = (kk == 0);
            for (size_t nn = 0; nn < N; nn += Nc) {
                const size_t ncl = std::min(Nc, N - nn);

                const size_t mcl0 = std::min(Mc, M);
                // Lab scope pack_b here covers the WHOLE fused pack (B block plus
                // the mm = 0 A block); pack_a covers only the A-only packs below.
                MAXMULSK_HOOK_EVENT(pack_b);
                MAXMULSK_HOOK_BEGIN(pack_b);
                fuse_pack(A, B, K, N, mcl0, nn, ncl, kk, kcl, Kc, Kc_pad, packed_A, packed_B, stage);
                MAXMULSK_HOOK_END(pack_b);
                compute_block(C, N, 0, mcl0, nn, ncl, kcl, first_k, Kc, Kc_pad, packed_A, packed_B, scratch);

                for (size_t mm = Mc; mm < M; mm += Mc) {
                    const size_t mcl = std::min(Mc, M - mm);
                    MAXMULSK_HOOK_EVENT(pack_a);
                    MAXMULSK_HOOK_BEGIN(pack_a);
                    pack_A_block(A, K, mm, mcl, kk, kcl, Kc_pad, packed_A, stage);
                    MAXMULSK_HOOK_END(pack_a);
                    compute_block(C, N, mm, mcl, nn, ncl, kcl, first_k, Kc, Kc_pad, packed_A, packed_B, scratch);
                }
            }
        }
        return true;
    }

    // Workspace layout shared by both entry points: packed A (Mc x Kc_pad), packed
    // B (Nc x Kc, starting MAXMULSK_FUSEPACK_SKEW_B bytes into its region), the
    // 16x64 staging block and the 16x64 C scratch -- each region rounded up to 64
    // bytes so every pointer handed to the streaming code is 64-byte aligned.
    struct Layout { size_t Kc, Nc, Mc, Kc_pad, a_bytes, b_bytes, tile_bytes, total; };
    static Layout layout(size_t M, size_t K, size_t N, const Blocking& b) {
        Layout l;
        l.Kc = std::min(b.Kc, K);
        l.Nc = std::min(b.Nc, ((N + 63) / 64) * 64);
        l.Mc = std::min(b.Mc, ((M + 15) / 16) * 16);
        l.Kc_pad = ((l.Kc + 63) / 64) * 64;
        const auto r64 = [](size_t x) { return (x + 63) / 64 * 64; };
        l.a_bytes = r64(l.Mc * l.Kc_pad * sizeof(float));
        l.b_bytes = r64(l.Nc * l.Kc * sizeof(float) + MAXMULSK_FUSEPACK_SKEW_B);
        l.tile_bytes = 16 * 64 * sizeof(float);
        l.total = l.a_bytes + l.b_bytes + 2 * l.tile_bytes;
        return l;
    }

    size_t workspace_bytes(size_t M, size_t K, size_t N, const Blocking& b) {
        return fk_classify(M, K, N, b) == Support::Native ? layout(M, K, N, b).total : 0;
    }

    // Invariant: ws is 64-byte aligned and holds at least workspace_bytes() bytes;
    // nothing in it needs to be initialised (every packed slot and the staging
    // block are written before they are read, as in the allocating path).
    Support run_multiplication_ws(const float* A, const float* B, float* C,
                                  size_t M, size_t K, size_t N, const Blocking& b,
                                  void* ws, size_t ws_bytes) {
        const Support s = fk_classify(M, K, N, b);
        if (s != Support::Native) return s;
        const Layout l = layout(M, K, N, b);
        if (!ws || ws_bytes < l.total || reinterpret_cast<uintptr_t>(ws) % 64 != 0) return Support::AllocationFailed;
        char* p = static_cast<char*>(ws);
        float* packed_A = reinterpret_cast<float*>(p);
        float* packed_B = reinterpret_cast<float*>(p + l.a_bytes + MAXMULSK_FUSEPACK_SKEW_B);
        float* stage    = reinterpret_cast<float*>(p + l.a_bytes + l.b_bytes);
        float* scratch  = reinterpret_cast<float*>(p + l.a_bytes + l.b_bytes + l.tile_bytes);
        return run_streaming(A, B, C, M, K, N, packed_A, packed_B, stage, scratch, l.Kc, l.Nc, l.Mc)
                   ? Support::Native : Support::UnsupportedVectorLength;
    }

    // Allocating entry point: one allocation per call, freed on return.
    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N, const Blocking& b) {
        const Support s = fk_classify(M, K, N, b);
        if (s != Support::Native) return s;
        const size_t bytes = layout(M, K, N, b).total;
        std::unique_ptr<char[], FreeDeleter> ws(static_cast<char*>(std::aligned_alloc(64, bytes)));
        if (!ws) return Support::AllocationFailed;
        return run_multiplication_ws(A, B, C, M, K, N, b, ws.get(), bytes);
    }

} // namespace MAXMULSK_FUSEPACK_NS
