// bench/kleidiai_gemm.hpp
// Thin C++ wrapper over Arm KleidiAI's fp32 GEMM micro-kernels, for benchmark
// comparison against our own NEON/SME kernels.
//
// KleidiAI micro-kernels do not take row-major A/B directly: they consume
// operands already packed into a kernel-specific layout. Accelerate, OpenBLAS
// and our own kernels all pack internally as part of the measured call, so the
// packing step is split out here to let the benchmark report both
//   pack + matmul  (comparable to every other library measured), and
//   matmul only    (the micro-kernel in isolation).
//
// KleidiAI also ships no cache blocking above the micro-kernel — that is the
// calling runtime's job — so matmul() drives the kernels through an outer
// N-block loop. Without it the large sizes measure the missing loop rather than
// the kernel: a bare 4096^3 call runs at ~890 GFLOPS, the same kernel N-blocked
// runs at ~1700. See docs/BENCHMARKS.md §0.5.
//
// When the build is configured without KleidiAI (-DMAXMULSK_WITH_KLEIDIAI=OFF,
// or the dependency could not be fetched), this still compiles: available()
// returns false and the benchmark skips the KleidiAI columns, the same way it
// degrades when OpenBLAS is missing.

#pragma once

#include <cstddef>
#include <memory>

namespace KleidiAI {

enum class Kernel {
    // kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa
    // SME2 FMOPA, 2VL x 2VL output tile. Both operands packed.
    Sme2Mopa2VL,
    // kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa
    // SME2 FMOPA, 8VS x 8VS output tile with 16VSx4VS / 8VSx4VS / 4VSx… edge
    // variants. Both operands packed.
    Sme2Mopa8VS,
    // kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla
    // NEON FMLA, 6x8 output tile. Only B is packed; A is read row-major.
    Neon6x8Mla,
};

// Human-readable kernel name (the full KleidiAI symbol).
const char* name(Kernel k);

// Short column label for benchmark tables ("2VL", "8VS", "6x8").
const char* label(Kernel k);

// True if this binary was built against KleidiAI. When false, Gemm is a no-op
// and its results must not be reported.
bool available();

// Experiment 2: KleidiAI's own pack + compute driven by a MaxMulSK-style
// panel-blocked caller. Only the orchestration differs — the micro-kernel and
// the packing routines are KleidiAI's own, unmodified.
//
//   for N-block: for K-panel: pack B panel
//                             for M-block: pack A panel, consume immediately
//
// The one thing this cannot mirror: the fp32 SME2 kernels overwrite C and
// expose no beta=1 / accumulate flag, so when K is blocked the caller must
// compute into a scratch tile and add. That add is a caller-side cost forced by
// the API, not a kernel deficiency. Kc >= K takes a direct-write fast path with
// no scratch and no add.
struct PanelConfig {
    size_t Mc = 0, Kc = 0, Nc = 0;   // 0 means "whole dimension"
};

// Timings for one panel-blocked run, in milliseconds. Filled only by the
// instrumented entry point; the plain one leaves them zero.
struct PanelBreakdown {
    double pack_a_ms = 0, pack_b_ms = 0, compute_ms = 0, accumulate_ms = 0, other_ms = 0;
};

// One packed-operand GEMM, reused across benchmark iterations.
//
// reshape() owns every allocation, so pack() and matmul() can be timed without
// measuring malloc. Call order is reshape -> pack -> matmul.
class Gemm {
public:
    explicit Gemm(Kernel k);
    ~Gemm();

    Gemm(const Gemm&)            = delete;
    Gemm& operator=(const Gemm&) = delete;

    // Size the packed-operand buffers for an M x K by K x N product.
    // Untimed: allocates.
    void reshape(size_t M, size_t N, size_t K);

    // Pack the operands this kernel needs. Both A and B for the SME2 kernels,
    // B only for the NEON kernel. Timed.
    void pack(const float* A, const float* B);

    // C = A * B, row-major, from the packed operands. `A` is used only by
    // kernels that take an unpacked left-hand side. Timed.
    void matmul(const float* A, float* C);

    // End-to-end: allocate the packed-operand buffers, pack, and compute, all
    // in one call. Exists so the end-to-end benchmark is symmetric with
    // MaxMulSK, whose run_multiplication allocates its packing buffers on every
    // call and offers no way to hoist them. reshape() must have been called for
    // this shape first (it fixes the buffer sizes and the N-block size); it
    // allocates nothing that this call reuses.
    void run_end_to_end(const float* A, const float* B, float* C);

    // Outer N-block width, in columns. KleidiAI ships no cache blocking above
    // the micro-kernel — the calling runtime supplies it — so this is a tuning
    // parameter of the *caller*, not of the kernel. reshape() picks a default
    // from a packed-B-panel size budget; set_n_block() overrides it, and the
    // benchmark autotunes over a candidate set so that KleidiAI is measured at
    // its best block size, the same way MaxMulSK is measured at its best of six
    // kernel variants. 0 restores the heuristic. Call after reshape().
    void   set_n_block(size_t columns);
    size_t n_block() const;

    // Experiment 2. Panel-blocked end-to-end: allocates its (small) panel
    // buffers, then packs and consumes one panel at a time. reshape() must have
    // been called for this shape. Supported for the two fp32 SME2 kernels.
    void run_panel_blocked(const float* A, const float* B, float* C,
                           const PanelConfig& cfg);

    // Same, with per-stage timers. Slower than run_panel_blocked because of the
    // clock calls, so it is used for the breakdown only, never for the headline
    // timing.
    PanelBreakdown run_panel_blocked_instrumented(const float* A, const float* B,
                                                  float* C, const PanelConfig& cfg);

    bool supports_panel_blocking() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Shared body of the two panel-blocked entry points; `bd` non-null enables
    // the per-stage timers.
    void panel_blocked_impl(const float* A, const float* B, float* C,
                            const PanelConfig& cfg, PanelBreakdown* bd);
};

// Experiment 1: the hot micro-kernel ceiling. Invokes the fp32 SME2 matmul on a
// single m_step x n_step output tile over a chosen Kc, with the packed panels
// small enough to stay resident. Note that KleidiAI's entry point enters and
// leaves streaming mode on every call — that is where its API boundary sits and
// it cannot be hoisted without editing the library.
class HotTile {
public:
    HotTile();
    ~HotTile();
    HotTile(const HotTile&)            = delete;
    HotTile& operator=(const HotTile&) = delete;

    // Untimed. A_small is m_step x Kc row-major, B_small is Kc x n_step.
    void prepare(Kernel k, size_t Kc, const float* A_small, const float* B_small);

    size_t m_step() const;
    size_t n_step() const;
    size_t panel_bytes() const;

    void run_batch(size_t reps) const;   // timed
    const float* tile() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace KleidiAI
