// bench/maxmulsk_sme_adapter.hpp
// Benchmark-only adapter over the MaxMulSK SME kernels.
//
// Two things the benchmark needs that the shipped API does not provide:
//
//  1. A uniform dispatch over the six SME variants, so the same driver loop can
//     time all of them.
//
//  2. A *kernel-only* path — packing hoisted out of the timed region. The
//     shipped `run_multiplication` is BLIS-style: it interleaves packing with
//     compute inside the cache-blocking loop nest, reusing one small
//     `M_tile x K_tile` A buffer and one `K_tile x N_tile` B buffer. There is no
//     entry point that takes pre-packed operands.
//
//     Rather than modify the kernels, `Prepacked` replays the driver's exact
//     loop nest twice: once calling only the shipped `pack_A_streaming` /
//     `pack_B_streaming` into one large buffer holding every block the driver
//     would visit, and once calling only the shipped micro-kernel, reading each
//     block back from that buffer. Loop order, tile sizes, micro-kernel input
//     layout, edge-tile handling and C accumulation order are all identical to
//     the shipped driver — the only difference is where the packed bytes live.
//
//     LIMITATION, and it is not a small one: the shipped driver reads its
//     packed panels out of a small, hot, repeatedly-reused buffer, whereas the
//     kernel-only path reads them out of a large cold one. The kernel-only
//     numbers therefore do NOT decompose the end-to-end time; they measure the
//     micro-kernel plus the memory traffic of streaming full-size pre-packed
//     operands. This is the same footing KleidiAI's kernel-only path is on
//     (its pre-packed operands are also full-size and cold), which is why the
//     comparison is meaningful even though neither is a pure ALU measurement.
//
// GEMM SEMANTICS: every MaxMulSK micro-kernel *accumulates* (`C += A*B`); it
// never overwrites C. Producing `C = A*B` therefore requires C to be zeroed
// first, and that zeroing is the caller's cost. The benchmark times it. See
// `zero_cost_note()`.

#pragma once

#include <cstddef>
#include <memory>

namespace MaxMulSK {

enum class Kernel {
    Sme4x1,          // n->k->m, M_tile 64,   K_tile 2048, N_tile 1024, 4SVL x 1SVL
    Sme2x2,          // n->k->m, M_tile 64,   K_tile 1024, N_tile 512,  2SVL x 2SVL
    Sme1x4,          // m->k->n, M_tile 1024, K_tile 2048, N_tile 64,   1SVL x 4SVL
    Sme1x4Sym,       // m->k->n, M_tile 1024, K_tile 2048, N_tile 64,   1SVL x 4SVL
    Sme4x1ZAPack,    // n->k->m, M_tile 128,  K_tile 1024, N_tile 512,  4SVL x 1SVL
    Sme1x4SymZAIO,   // m->n,    no K blocking; ZA-resident K accumulation
    // Experimental. Loop order m->n->K with K innermost; A and B packed over the
    // full K; ZA zeroed once per C tile, accumulated across all of K, stored
    // once. Acc issues one micro-kernel call per K tile, AccKc splits each K
    // tile into Kc sub-chunks - same arithmetic, different invocation
    // granularity.
    Sme1x4Acc,
    Sme1x4AccKc,
    // Outer-Kc variants: the whole M x N sweep runs once per Kc slice of K, so
    // the packed buffers scale with Kc instead of K and stay inside L2 on
    // large-K shapes. Each C tile is visited K/Kc times - first panel
    // overwrites, the rest accumulate. KcOut repacks per tile like Acc;
    // FastKcOut packs each region once per panel and reuses it.
    // End-to-end only, see has_kernel_only().
    Sme1x4AccKcOut,
    Sme1x4AccFastKcOut,
};

inline constexpr Kernel kAllKernels[] = {
    Kernel::Sme4x1, Kernel::Sme2x2, Kernel::Sme1x4,
    Kernel::Sme1x4Sym, Kernel::Sme4x1ZAPack, Kernel::Sme1x4SymZAIO,
    Kernel::Sme1x4Acc, Kernel::Sme1x4AccKc,
    Kernel::Sme1x4AccKcOut, Kernel::Sme1x4AccFastKcOut,
};

const char* name(Kernel k);   // e.g. "SMEKernels4x1::run_multiplication"
const char* label(Kernel k);  // e.g. "SME 4x1"

// Streaming vector length in 32-bit words (16 on an M4, SVL = 512 bits).
size_t svl_words();

// End-to-end: the shipped driver, unmodified.
//
// Most kernels accumulate (`C += A*B`), so the caller must zero C first and that
// memset is part of the cost of using them. The Acc variants overwrite
// (`C = A*B`) and need no such memset. overwrites_C() reports which, so the
// benchmark can time each kernel for the work it actually requires instead of
// charging every kernel the same fixed toll.
bool overwrites_C(Kernel k);

void run(Kernel k, const float* A, const float* B, float* C,
         size_t M, size_t K, size_t N);

// Kernel-only support. False for Sme1x4SymZAIO: it does not block over K (it
// packs the full K up front) and splits the micro-kernel into zero/compute/
// store around ZA-resident accumulation, so it does not share the skeleton the
// adapter replays. It is benchmarked end-to-end only.
//
// Also false for the two outer-Kc variants, for a different reason: their whole
// mechanism IS the packing schedule. Hoisting packing out of the timed region
// would erase the thing being measured, and would require materialising the
// full-K packed buffers those kernels exist to avoid. A prepacked number for
// them would be misleading rather than merely absent.
bool has_kernel_only(Kernel k);

// Experiment 1: the true hot-microkernel ceiling.
//
// `Prepacked` above is a pre-packed *full GEMM*: it still walks the whole
// output and streams full-size packed operands, so it carries macro-loop and
// cache-traversal effects. `HotMicro` removes those. It builds the packed A/B
// panels for ONE output tile (M_step x N_step) over a chosen Kc, small enough
// to stay resident, and invokes only the shipped micro-kernel, repeatedly.
//
// Panel sizes are uniform across the variants: A is M_step*Kc floats, B is
// N_step*Kc floats, and every MaxMulSK micro-kernel produces a 1024-element
// tile (4SVL x 1SVL, 2SVL x 2SVL or 1SVL x 4SVL at SVL = 16).
class HotMicro {
public:
    HotMicro();
    ~HotMicro();
    HotMicro(const HotMicro&)            = delete;
    HotMicro& operator=(const HotMicro&) = delete;

    // Untimed. A_small is M_step x Kc row-major, B_small is Kc x N_step
    // row-major; both are packed with the kernel's own shipped pack routines so
    // the panel layout is exactly what the micro-kernel expects.
    void prepare(Kernel k, size_t Kc, const float* A_small, const float* B_small);

    size_t m_step() const;
    size_t n_step() const;
    size_t panel_bytes() const;   // packed A + packed B + the output tile

    // Timed. `reps` back-to-back micro-kernel invocations with streaming mode
    // entered ONCE around the whole batch — this is what the shipped driver
    // does, and it is the number that answers "how well does the kernel feed
    // the pipeline".
    void run_batch(size_t reps) const;

    // Timed. Same work, but streaming mode and the ZA scope are entered and
    // left per invocation. KleidiAI's kai_run_matmul_* does smstart/smstop on
    // every call because that is where its API boundary sits, so this variant
    // exists to measure that confound rather than let it silently favour us.
    void run_batch_isolated_streaming(size_t reps) const;

    // Only meaningful for the Acc variants, whose micro-kernel neither zeroes
    // ZA nor writes it back: zero once, run `reps` accumulating calls, store
    // once. run_batch() above keeps the zero+compute+store per invocation so it
    // stays comparable with the other kernels; this one shows the FMOPA rate
    // with those costs amortised, which is the whole point of the variant.
    // No-op for kernels whose micro-kernel owns its own ZA lifetime.
    void run_batch_amortized(size_t reps) const;
    bool supports_amortized() const;

    void zero_tile();
    const float* tile() const;    // M_step x N_step, row-major, stride n_step()

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Prepacked {
public:
    Prepacked();
    ~Prepacked();
    Prepacked(const Prepacked&)            = delete;
    Prepacked& operator=(const Prepacked&) = delete;

    // Untimed. Allocates, then runs the shipped packing routines over every
    // block the driver would visit. Must be called before compute().
    void prepare(Kernel k, const float* A, const float* B,
                 size_t M, size_t K, size_t N);

    // Timed. Replays the driver's loop nest with packing removed.
    // Accumulates into C.
    void compute(float* C) const;

    // Total pre-packed bytes held (A blocks + B blocks).
    size_t packed_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace MaxMulSK
