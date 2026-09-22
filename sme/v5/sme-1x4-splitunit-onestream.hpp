// =============================================================================
// v5 — split-unit GEMM (recon-stage implementation, 2026-09-19).
//
// The SME unit is one per cluster and does not see L1D; the NEON load/store
// path is per core (measured: MaxMulSK-Lab docs/device/m4-base.md). So the
// DRAM/L2 movement is done by NEON worker threads in normal mode and ONE
// thread stays in streaming mode for the ZA transpose of A and the mopa loop:
//
//   workers (NEON):  prepack_A  A rows -> contiguous 16 x kcl panels (row-major)
//                    pack_B     B      -> kcl x 64 panels (k-major, the layout
//                                         the micro-kernel reads)
//   SME thread:      pack_A     prepack panel -> ZA -> packed_A (k-major),
//                    compute    1x4 micro-kernel, writeback ZA -> C
//
// Ownership: every packed panel has a ready flag; the SME thread waits on it
// (acquire) before touching the panel; workers set it (release) after their
// stores. Workers never enter streaming mode (NEON is SIGILL there); the SME
// thread never runs NEON. Per Kc slab the pipeline is: submit all panel tasks,
// consume, join; the next slab starts when every worker is idle.
//
// SEMANTICS: C = A*B (first Kc slab overwrites, later slabs accumulate).
// First step supports full tiles only: M%16==0, N%64==0, K%64==0.
// Changes the comparison class: multi-thread packing + single SME compute.
// =============================================================================
#pragma once
// ONE-STREAM copy of sme-1x4-splitunit (2026-09-22): the whole K loop runs in ONE
// streaming region (the original left streaming mode between Kc slabs to join the
// workers). Workers run one slab ahead into ping-pong buffers.
#include <cstddef>

namespace SMEKernels1x4SplitUnitOneStream {

    struct Params {
        size_t workers  = 2;         // NEON worker threads (persistent, spinning)
        size_t Kc       = 2048;      // K slab; one slab at 256^3
        // An operand whose bytes fit under this is assumed L2-resident and is
        // packed by the SME thread itself (its load path is ~5x a NEON core's
        // from L2); larger operands go to the workers. 0 = workers for everything.
        size_t l2_bytes = 4u << 20;
        // direct-B: no B packing at all; the micro-kernel reads B from the source
        // with row stride N (each k-step is still one contiguous 256 B x4 load).
        bool   direct_b = false;
        // H2: with direct_b, workers can pull the NEXT B panel's rows into the
        // shared L2 without writing anything: 2 = NEON loads (asm sink),
        // 3 = prfm pldl2keep per 128 B line. 0/1 = no prefetch.
        int    b_prefetch = 0;
        // After a worker writes a packed panel: 0 nothing, 1 dc cvac every line
        // (clean to L2), 2 dc civac (clean + drop from the writer's L1).
        int    flush = 0;
        // Who produces packed_A (k-major micro-panels):
        //   0  workers copy rows (prepack), the SME thread transposes in ZA (default)
        //   1  workers transpose with NEON 4x4 zip/trn blocks -> packed_A directly
        //   2  workers gather columns with lane loads (one lane load per element)
        // With 1/2 the SME thread never leaves compute for A.
        int    a_mode = 0;
        // With a_mode 1/2: the SME thread transposes the first `a_split` panels
        // itself (no wait at the start), workers do the rest -> pipeline.
        size_t a_split = 0;
        // Worker thread QoS: 0 = USER_INTERACTIVE (scheduler prefers P-cores),
        // 1 = BACKGROUND (steered to E-cores). Placement experiment only.
        int    worker_qos = 0;
    };

    enum class Support {
        Native, UnsupportedMTail, UnsupportedNTail, UnsupportedKTail,
        UnsupportedVectorLength, BadParams, AllocationFailed, Unsupported,
    };

    struct Counts {
        size_t kc_slices, a_panels, b_panels, microkernel_calls;
        bool   a_by_workers, b_by_workers;
    };

    Support classify(size_t M, size_t K, size_t N, const Params& p);
    void    counts(size_t M, size_t K, size_t N, const Params& p, Counts* out);

    // TEST ONLY: NEON packers, callable from normal mode, for layout checks.
    void probe_prepack_A(const float* A, size_t K, size_t m, size_t kk, size_t kcl, float* dst);
    void probe_pack_B(const float* B, size_t N, size_t kk, size_t n, size_t kcl, float* dst);
    void probe_pack_A_neon(int mode, const float* A, size_t K, size_t m, size_t kk, size_t kcl, float* dst);  // mode 1 transpose, 2 gather
    // TEST ONLY: `tiles` back-to-back micro-kernel calls on one packed A panel and
    // one B panel (ldb = 64 packed / N direct), ZA zeroed per tile, no writeback.
    void probe_microkernel(const float* pA, const float* pB, size_t kcl, size_t ldb, size_t tiles);
    // TEST ONLY: streaming-mode row copy of one 16 x kcl A panel (the SME thread's own prepack).
    void probe_prepack_A_streaming(const float* A, size_t K, size_t m, size_t kk, size_t kcl, float* dst);

    Support run_multiplication(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N, const Params& p);

    // Persistent worker pool and the packed buffers/flags are kept across calls
    // (grown on demand); `allocation` in the hooks is therefore ~0 after the
    // first call. Call to release everything.
    void shutdown_workers();

} // namespace SMEKernels1x4SplitUnitOneStream
