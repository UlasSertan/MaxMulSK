#pragma once

#include <cstddef>

namespace SMETest {

    // Kernel selector — picks which SME variant to exercise.
    enum class Kernel { K4x1, K2x2, K1x4, K1x4Sym, K1x4SymZAInOut, K4x1ZAPack };

    // Full per-kernel suite: pack correctness → GEMM correctness → benchmark.
    void run(Kernel k);

    // Cross-kernel side-by-side: same sizes, all four kernels, GFLOPS table.
    void run_comparison();

    // 4x1-only: time pack_A / pack_B / micro-kernel separately.
    void run_timing_breakdown();

    // Single-kernel, single-size, N-iteration driver for Instruments PMU profiling.
    void profile(Kernel k, std::size_t M, std::size_t K, std::size_t N, int iters);

} // namespace SMETest
