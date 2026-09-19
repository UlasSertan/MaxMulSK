#include "sme-1x4-splitunit.hpp"
#include "sme/support/lab_hooks.hpp"

#include <arm_neon.h>
#include <arm_sme.h>
#include <arm_sve.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sys/qos.h>

#define RESTRICT __restrict__

// The ZA packing macro, the micro-kernel and both writeback bodies are the
// ncblock-apack4za ones, carried over unchanged (v5 changes who feeds them,
// not what they do).
#define NB_T(g) ((g) / 4)
#define NB_C(g) (4 * ((g) % 4))
#define NB_GROUP(g)                                                            \
    svcreate4_f32(svread_ver_za32_f32_m(zu, pg, NB_T(g), NB_C(g) + 0),         \
                  svread_ver_za32_f32_m(zu, pg, NB_T(g), NB_C(g) + 1),         \
                  svread_ver_za32_f32_m(zu, pg, NB_T(g), NB_C(g) + 2),         \
                  svread_ver_za32_f32_m(zu, pg, NB_T(g), NB_C(g) + 3))

// One steady step: store the group a Z set already holds, then refill that same
// Z set with the group two ahead. Store operand is consumed before the refill,
// so the two x4 store streams never wait on each other.
#define NB_DRAIN(g)                                                            \
    svst1_f32_x4(p4, out + (g) * 64, s0);                                      \
    s0 = NB_GROUP((g) + 2);                                                    \
    svst1_f32_x4(p4, out + ((g) + 1) * 64, s1);                                \
    s1 = NB_GROUP((g) + 3);

#define NB_PACK_A_KP_CHUNK(SRC, STRIDE)                                           \
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
        svfloat32x4_t s0 = NB_GROUP(0);                                \
        svfloat32x4_t s1 = NB_GROUP(1);                                \
        /* Static unroll: the tile operand of svread_ver_za32 must */  \
        /* be an immediate, so g cannot be a runtime variable. */      \
        NB_DRAIN(0)  NB_DRAIN(2)  NB_DRAIN(4)  NB_DRAIN(6)             \
        NB_DRAIN(8)  NB_DRAIN(10) NB_DRAIN(12)                         \
        /* DRAIN: s0 holds group 14 (ZA3 vertical 8..11), */           \
        /* s1 holds group 15 (ZA3 vertical 12..15). No further */      \
        /* extraction -- the two stores are back to back. */           \
        svst1_f32_x4(p4, out + 14 * 64, s0);                           \
        svst1_f32_x4(p4, out + 15 * 64, s1);                           \
        MAXMULSK_HOOK_END(pack_a_transform_store);                             \
    } while (0)

namespace SMEKernels1x4SplitUnit {

    // =========================================================================
    // NEON side (normal mode). No SME/SVE intrinsic may appear here.
    // =========================================================================

    // A rows m..m+15, columns kk..kk+kcl -> dst[r*kcl + k] (row-major panel,
    // 16 x kcl contiguous). Pure copy; the transpose happens later in ZA.
    __attribute__((noinline))
    static void prepack_A_panel(const float* RESTRICT A, size_t K, size_t m, size_t kk, size_t kcl, float* RESTRICT dst) {
        for (size_t r = 0; r < 16; r++) {
            const float* s = A + (m + r) * K + kk;
            float* d = dst + r * kcl;
            for (size_t k = 0; k < kcl; k += 32) {
                float32x4x4_t v0 = vld1q_f32_x4(s + k), v1 = vld1q_f32_x4(s + k + 16);
                vst1q_f32_x4(d + k, v0); vst1q_f32_x4(d + k + 16, v1);
            }
        }
    }
    // B rows kk..kk+kcl, columns n..n+63 -> dst[k*64 + j]: same layout the
    // streaming pack_B produced; one x4 pair per k row.
    __attribute__((noinline))
    static void pack_B_panel(const float* RESTRICT B, size_t N, size_t kk, size_t n, size_t kcl, float* RESTRICT dst) {
        for (size_t k = 0; k < kcl; k++) {
            const float* s = B + (kk + k) * N + n;
            float32x4x4_t v0 = vld1q_f32_x4(s), v1 = vld1q_f32_x4(s + 16), v2 = vld1q_f32_x4(s + 32), v3 = vld1q_f32_x4(s + 48);
            float* d = dst + k * 64;
            vst1q_f32_x4(d, v0); vst1q_f32_x4(d + 16, v1); vst1q_f32_x4(d + 32, v2); vst1q_f32_x4(d + 48, v3);
        }
    }
    void probe_prepack_A(const float* A, size_t K, size_t m, size_t kk, size_t kcl, float* dst) { prepack_A_panel(A, K, m, kk, kcl, dst); }
    void probe_pack_B(const float* B, size_t N, size_t kk, size_t n, size_t kcl, float* dst) { pack_B_panel(B, N, kk, n, kcl, dst); }

    // Streaming-mode B panel copy, the ncblock one, for L2-resident B (the SME
    // thread's own load path is faster than a NEON core's from L2).
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
    // Worker pool: persistent threads spinning on a job generation counter.
    // A job is a list of panel tasks; workers claim tasks with one atomic
    // fetch_add, in the order the SME thread will need them, and publish each
    // finished panel with a release store on its flag.
    // =========================================================================
    struct Task { uint8_t kind; uint32_t idx; };   // kind 0 = prepack A panel idx, 1 = pack B panel idx
    struct Job {
        const float* A; const float* B; size_t K, N, kk, kcl;
        float* prepack_A; float* packed_B;
        std::vector<Task> tasks;
        std::atomic<uint32_t> next{0};
        std::atomic<uint32_t>* a_ready; std::atomic<uint32_t>* b_ready;
        std::atomic<uint32_t> done{0};
    };
    struct Pool {
        std::vector<std::thread> threads;
        std::atomic<Job*> job{nullptr};
        std::atomic<uint64_t> generation{0};
        std::atomic<bool> quit{false};
        explicit Pool(size_t n) {
            for (size_t i = 0; i < n; i++) threads.emplace_back([this] { loop(); });
        }
        ~Pool() { quit.store(true); generation.fetch_add(1); for (auto& t : threads) t.join(); }
        void loop() {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
            uint64_t seen = 0;
            for (;;) {
                while (generation.load(std::memory_order_acquire) == seen) { __asm__ volatile("yield"); }
                seen = generation.load(std::memory_order_acquire);
                if (quit.load()) return;
                Job* j = job.load(std::memory_order_acquire);
                if (!j) continue;
                for (;;) {
                    const uint32_t t = j->next.fetch_add(1, std::memory_order_relaxed);
                    if (t >= j->tasks.size()) break;
                    const Task& tk = j->tasks[t];
                    if (tk.kind == 0) { prepack_A_panel(j->A, j->K, 16 * tk.idx, j->kk, j->kcl, j->prepack_A + (size_t)tk.idx * 16 * j->kcl); j->a_ready[tk.idx].store(1, std::memory_order_release); }
                    else              { pack_B_panel(j->B, j->N, j->kk, 64 * tk.idx, j->kcl, j->packed_B + (size_t)tk.idx * j->kcl * 64);   j->b_ready[tk.idx].store(1, std::memory_order_release); }
                }
                j->done.fetch_add(1, std::memory_order_release);
            }
        }
    };
    static std::unique_ptr<Pool> g_pool; static size_t g_pool_n = 0;
    static Pool& pool(size_t n) { if (!g_pool || g_pool_n != n) { g_pool.reset(); g_pool = std::make_unique<Pool>(n); g_pool_n = n; } return *g_pool; }

    // =========================================================================
    // SME side (streaming). Carried over from ncblock-apack4za.
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
    // ldb = floats between consecutive k rows of the B operand: 64 for a packed
    // panel, N for reading B straight from the source (direct-B mode). Only the
    // pointer increment changes; the load/MOPA schedule is the v4 one.
    void micro_kernel_1x4(float* RESTRICT packed_A, const float* RESTRICT packed_B,
                          size_t K_curr, size_t ldb) __arm_inout("za") __arm_streaming {
        const size_t SVL = static_cast<size_t>(svcntsw());
        const svbool_t pg = svptrue_b32();
        const svcount_t pg4 = svptrue_c32();
        const size_t GS_B = ldb; // B stride per k-step

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
    static inline void spin_until(std::atomic<uint32_t>* f) __arm_streaming_compatible {
        while (f->load(std::memory_order_acquire) == 0) { __asm__ volatile("yield"); }
    }

    __arm_locally_streaming __arm_new("za")
    __attribute__((noinline))
    static bool run_slab(const float* RESTRICT A, const float* RESTRICT B, size_t K,
                         const float* RESTRICT prepack_A, float* RESTRICT packed_A, float* RESTRICT packed_B, float* RESTRICT C,
                         size_t M, size_t N, size_t kk, size_t kcl, bool first_k, bool a_by_workers, bool b_by_workers, bool order_mn, bool direct_b,
                         std::atomic<uint32_t>* a_ready, std::atomic<uint32_t>* b_ready) {
        if (static_cast<size_t>(svcntsw()) != 16) return false;
        const svbool_t  pg = svptrue_b32();
        const svcount_t p4 = svptrue_c32();
        const svfloat32_t zu = svundef_f32();
        const size_t na = M / 16, nb = N / 64;

        // B not owned by workers: the SME thread copies every panel itself, up front
        // (L2-resident by the l2_bytes rule; one pass at the streaming copy rate).
        if (!b_by_workers && !direct_b) {
            for (size_t b = 0; b < nb; b++) {
                MAXMULSK_HOOK_EVENT(pack_b); MAXMULSK_HOOK_BEGIN(pack_b);
                pack_B_streaming(B, packed_B + b * (kcl * 64), 64, kcl, kk, N, 64 * b);
                MAXMULSK_HOOK_END(pack_b);
            }
        }
        // Lambdas here were measured 4x slower (2026-09-19; not inlined, captures
        // spilled), so the two orders are written out with macros.
#define V5_TRANSPOSE_A(a)                                                                     \
        do {                                                                                  \
            if (a_by_workers) { MAXMULSK_HOOK_BEGIN(wait_a); spin_until(&a_ready[(a)]); MAXMULSK_HOOK_END(wait_a); } \
            const float* const src_ = a_by_workers ? prepack_A + (a) * 16 * kcl : A + (16 * (a)) * K + kk; \
            const size_t stride_ = a_by_workers ? kcl : K;                                    \
            float* const pA = packed_A + (a) * (kcl * 16);                                    \
            for (size_t kp = 0; kp < kcl; kp += 64) {                                         \
                MAXMULSK_HOOK_EVENT(pack_a); MAXMULSK_HOOK_BEGIN(pack_a);                     \
                const float* const arow = src_ + kp;                                          \
                NB_PACK_A_KP_CHUNK(arow, stride_);                                            \
                MAXMULSK_HOOK_END(pack_a);                                                    \
            }                                                                                 \
        } while (0)
#define V5_TILE(a, b)                                                                         \
        do {                                                                                  \
            float* const pA_ = packed_A + (a) * (kcl * 16);                                   \
            const float* const pB_ = direct_b ? B + kk * N + 64 * (b) : packed_B + (b) * (kcl * 64); \
            const size_t ldb_ = direct_b ? N : 64;                                            \
            MAXMULSK_HOOK_EVENT(za_init); MAXMULSK_HOOK_BEGIN(za_init);                       \
            svzero_za();                                                                      \
            MAXMULSK_HOOK_END(za_init);                                                       \
            MAXMULSK_HOOK_EVENT(compute); MAXMULSK_HOOK_BEGIN(compute);                       \
            micro_kernel_1x4(pA_, pB_, kcl, ldb_);                                            \
            MAXMULSK_HOOK_END(compute);                                                       \
            float* const dstC = C + (16 * (a)) * N + 64 * (b);                                \
            MAXMULSK_HOOK_EVENT(writeback); MAXMULSK_HOOK_BEGIN(writeback);                   \
            if (first_k) store_za(dstC, N); else store_za_add(dstC, N);                       \
            MAXMULSK_HOOK_END(writeback);                                                     \
        } while (0)
        if (order_mn) {
            // M -> N: whole packed_B fits L2. Each A panel is transposed as soon as
            // it is available and used against every B panel: compute starts after
            // ONE A panel; each B panel is waited for at its first use.
            for (size_t a = 0; a < na; a++) {
                V5_TRANSPOSE_A(a);
                for (size_t b = 0; b < nb; b++) {
                    if (b_by_workers && a == 0) { MAXMULSK_HOOK_BEGIN(wait_b); spin_until(&b_ready[b]); MAXMULSK_HOOK_END(wait_b); }
                    V5_TILE(a, b);
                }
            }
        } else {
            // N -> M: packed_B does not fit L2. All of A first (small when this order
            // is chosen), then every B panel is consumed once, by all A panels, right
            // after it arrives -- B never goes round L2 twice.
            for (size_t a = 0; a < na; a++) V5_TRANSPOSE_A(a);
            for (size_t b = 0; b < nb; b++) {
                if (b_by_workers) { MAXMULSK_HOOK_BEGIN(wait_b); spin_until(&b_ready[b]); MAXMULSK_HOOK_END(wait_b); }
                for (size_t a = 0; a < na; a++) V5_TILE(a, b);
            }
        }
#undef V5_TRANSPOSE_A
#undef V5_TILE
        (void)zu; (void)p4; (void)pg;
        return true;
    }

    Support classify(size_t M, size_t K, size_t N, const Params& p) {
        if (M == 0 || K == 0 || N == 0)  return Support::Unsupported;
        if (M % 16 != 0)                 return Support::UnsupportedMTail;
        if (N % 64 != 0)                 return Support::UnsupportedNTail;
        if (K % 64 != 0)                 return Support::UnsupportedKTail;
        if (p.workers == 0 || p.workers > 8 || p.Kc % 64 != 0) return Support::BadParams;
        return Support::Native;
    }
    static bool by_workers(size_t bytes, const Params& p) { return p.l2_bytes == 0 || bytes > p.l2_bytes; }
    void counts(size_t M, size_t K, size_t N, const Params& p, Counts* out) {
        const size_t Kc = std::min(p.Kc, K);
        out->kc_slices = (K + Kc - 1) / Kc; out->a_panels = out->kc_slices * (M / 16); out->b_panels = out->kc_slices * (N / 64);
        out->microkernel_calls = out->kc_slices * (M / 16) * (N / 64);
        out->a_by_workers = by_workers(M * K * sizeof(float), p); out->b_by_workers = !p.direct_b && by_workers(K * N * sizeof(float), p);
    }

    struct FreeDeleter { void operator()(void* q) { std::free(q); } };
    using AlignedBuffer = std::unique_ptr<float[], FreeDeleter>;

    // Buffers and flags kept across calls (grown on demand). Declared here so a
    // reader sees that `allocation` is not paid per call — unlike v4c and
    // MpGEMM, which allocate inside every call. Reported in the adapter detail.
    struct Context {
        AlignedBuffer prepack_A, packed_A, packed_B; size_t cap_A = 0, cap_B = 0, cap_prepack = 0;
        std::vector<std::atomic<uint32_t>> a_ready, b_ready;
        std::vector<Task> tasks;
    };
    static Context g_ctx;
    static bool grow(AlignedBuffer& buf, size_t& cap, size_t floats) {
        if (cap >= floats) return true;
        buf.reset(static_cast<float*>(std::aligned_alloc(64, floats * sizeof(float)))); cap = buf ? floats : 0; return (bool)buf;
    }
    void shutdown_workers() { g_pool.reset(); g_pool_n = 0; g_ctx = Context{}; }

    Support run_multiplication(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, const Params& p) {
        const Support s = classify(M, K, N, p);
        if (s != Support::Native) return s;
        const size_t Kc = std::min(p.Kc, K);
        const size_t na = M / 16, nb = N / 64;
        const bool a_w = by_workers(M * K * sizeof(float), p), b_w = !p.direct_b && by_workers(K * N * sizeof(float), p);
        MAXMULSK_HOOK_EVENT(allocation); MAXMULSK_HOOK_BEGIN(allocation);
        if (!grow(g_ctx.packed_A, g_ctx.cap_A, M * Kc) || (!p.direct_b && !grow(g_ctx.packed_B, g_ctx.cap_B, N * Kc))) return Support::AllocationFailed;
        if (a_w && !grow(g_ctx.prepack_A, g_ctx.cap_prepack, M * Kc)) return Support::AllocationFailed;
        if (g_ctx.a_ready.size() < na) g_ctx.a_ready = std::vector<std::atomic<uint32_t>>(na);
        if (g_ctx.b_ready.size() < nb) g_ctx.b_ready = std::vector<std::atomic<uint32_t>>(nb);
        MAXMULSK_HOOK_END(allocation);
        const bool use_pool = a_w || b_w;
        Pool* pl = use_pool ? &pool(p.workers) : nullptr;

        for (size_t kk = 0; kk < K; kk += Kc) {
            const size_t kcl = std::min(Kc, K - kk);
            const bool order_mn = N * kcl * sizeof(float) <= (6u << 20);   // packed_B slab within a 6 MiB L2 budget
            Job job;
            if (use_pool) {
                for (size_t i = 0; i < na; i++) g_ctx.a_ready[i].store(0, std::memory_order_relaxed);
                for (size_t i = 0; i < nb; i++) g_ctx.b_ready[i].store(0, std::memory_order_relaxed);
                job.A = A; job.B = B; job.K = K; job.N = N; job.kk = kk; job.kcl = kcl;
                job.prepack_A = g_ctx.prepack_A.get(); job.packed_B = g_ctx.packed_B.get(); job.a_ready = g_ctx.a_ready.data(); job.b_ready = g_ctx.b_ready.data();
                // task order = the SME thread's consumption order: M->N needs every B
                // panel for the first A panel (B first, then A); N->M needs all of A
                // before the first B panel (A first, then B).
                g_ctx.tasks.clear();
                if (order_mn) {
                    if (b_w) for (uint32_t b = 0; b < nb; b++) g_ctx.tasks.push_back({1, b});
                    if (a_w) for (uint32_t a = 0; a < na; a++) g_ctx.tasks.push_back({0, a});
                } else {
                    if (a_w) for (uint32_t a = 0; a < na; a++) g_ctx.tasks.push_back({0, a});
                    if (b_w) for (uint32_t b = 0; b < nb; b++) g_ctx.tasks.push_back({1, b});
                }
                job.tasks.swap(g_ctx.tasks);
                pl->job.store(&job, std::memory_order_release);
                pl->generation.fetch_add(1, std::memory_order_release);
            }
            const bool ok = run_slab(A, B, K, g_ctx.prepack_A.get(), g_ctx.packed_A.get(), g_ctx.packed_B.get(), C,
                                     M, N, kk, kcl, kk == 0, a_w, b_w, order_mn, p.direct_b, g_ctx.a_ready.data(), g_ctx.b_ready.data());
            if (use_pool) {
                while (job.done.load(std::memory_order_acquire) < p.workers) { __asm__ volatile("yield"); }
                pl->job.store(nullptr, std::memory_order_release);
                job.tasks.swap(g_ctx.tasks);
            }
            if (!ok) return Support::UnsupportedVectorLength;
        }
        return Support::Native;
    }

} // namespace SMEKernels1x4SplitUnit
