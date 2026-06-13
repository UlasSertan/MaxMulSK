#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <omp.h>

#include "common/utils.hpp"
#include "neon/neon-8x12.hpp"
#include "neon/test_neon.hpp"
#include "sme/sme-4x1.hpp"
#include "sme/test_sme.hpp"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// =============================================================================
// HELPERS
// =============================================================================
namespace {

    // Run NEON kernel N times, return average ms (includes one warmup run)
    double bench_neon(const float* A, const float* B, float* C,
                      size_t M, size_t N, size_t K, int iters) {
        GEMM::package(A, B, C, M, N, K); // warmup
        auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i)
            GEMM::package(A, B, C, M, N, K);
        return Ms(Clock::now() - t0).count() / iters;
    }

    // Pick iteration count so each size runs for a stable duration
    int iters_for(size_t N) {
        if (N <=   32) return 50000;
        if (N <=  128) return 5000;
        if (N <=  256) return 500;
        if (N <=  512) return 50;
        return 10;
    }

    void print_row(const std::string& label, size_t M, size_t N, size_t K,
                   double ms, bool correct, bool show_correct) {
        double gflops = Utils::compute_gflops(M, N, K, ms);
        std::cout << std::left  << std::setw(28) << label
                  << std::right << std::setw(8)  << std::fixed << std::setprecision(2) << ms     << " ms"
                  << std::setw(10) << std::setprecision(1) << gflops << " GFLOPS";
        if (show_correct)
            std::cout << "  " << (correct ? "PASS" : "FAIL");
        std::cout << "\n";
    }

} // namespace

// =============================================================================
// BENCHMARK NAMESPACE
// =============================================================================
namespace Benchmark {

    // -------------------------------------------------------------------------
    // 1. Large correctness test — 1067x1067x1067 (forces all three tails:
    //    M%8=3, N%12=11, K%4=3)
    // -------------------------------------------------------------------------
    void neon_correctness_large() {
        constexpr size_t M = 1067, N = 1067, K = 1067;

        std::cout << "\n========================================\n";
        std::cout << "  NEON Correctness — " << M << "x" << N << "x" << K << "\n";
        std::cout << "  (M%8=" << M%8 << "  N%12=" << N%12 << "  K%4=" << K%4 << ")\n";
        std::cout << "========================================\n";

        std::vector<float> A(M * K), B(K * N);
        std::vector<float> C_ref(M * N, 0.0f), C_neon(M * N, 0.0f);
        Utils::fill_random(A);
        Utils::fill_random(B);

        // Scalar reference
        auto t0 = Clock::now();
        Utils::multiply_scalar(A.data(), B.data(), C_ref.data(), M, N, K);
        double dt_scalar = Ms(Clock::now() - t0).count();

        // NEON
        auto t1 = Clock::now();
        GEMM::package(A.data(), B.data(), C_neon.data(), M, N, K);
        double dt_neon = Ms(Clock::now() - t1).count();

        bool ok = Utils::check_correctness(C_ref.data(), C_neon.data(), M * N, "NEON");

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Scalar : " << dt_scalar << " ms\n";
        std::cout << "  NEON   : " << dt_neon   << " ms  "
                  << Utils::compute_gflops(M, N, K, dt_neon) << " GFLOPS\n";
        std::cout << "  Result : " << (ok ? "PASS" : "FAIL") << "\n";
    }

    // -------------------------------------------------------------------------
    // 2. GFLOPS sweep — aligned sizes + edge-case sizes
    // -------------------------------------------------------------------------
    void neon_speed_sweep() {
        std::cout << "\n========================================\n";
        std::cout << "  NEON Speed Sweep\n";
        std::cout << "========================================\n";

        struct Case {
            size_t M, N, K;
            const char* label;
        };

        // Aligned: all tails are zero, measures pure NEON throughput
        // Edge:    deliberately chosen to force each tail combination
        const Case cases[] = {
            // --- Aligned ---
            {   4,   4,   4, "4^3         [tiny / scalar]"    },
            {  16,  16,  16, "16^3        [aligned]"          },
            {  64,  64,  64, "64^3        [aligned]"          },
            { 128, 128, 128, "128^3       [aligned]"          },
            { 256, 256, 256, "256^3       [aligned]"          },
            { 512, 512, 512, "512^3       [aligned]"          },
            {1024,1024,1024, "1024^3      [aligned]"          },
            {1024,1020,1024, "1024x1020x1024 [pre-edge baseline]"},
            // --- Edge: M tail only (M%8 != 0) ---
            {  65,  64,  64, "65x64x64    [M tail]"           },
            { 513, 512, 512, "513x512x512 [M tail]"           },
            // --- Edge: N tail only (N%12 != 0) ---
            {  64,  65,  64, "64x65x64    [N tail]"           },
            { 512, 509, 512, "512x509x512 [N tail]"           },
            // --- Edge: K tail only (K%4 != 0) ---
            {  64,  64,  65, "64x64x65    [K tail]"           },
            { 512, 512, 513, "512x512x513 [K tail]"           },
            // --- Edge: all tails ---
            {  67,  67,  67, "67^3        [all tails]"        },
            { 513, 509, 513, "513x509x513 [all tails]"        },
        };

        // Header
        std::cout << std::left  << std::setw(28) << "  Size"
                  << std::right << std::setw(10) << "Time"
                  << std::setw(14) << "GFLOPS" << "\n";
        std::cout << "  " << std::string(52, '-') << "\n";

        for (auto& c : cases) {
            const int iters = iters_for(std::max({c.M, c.N, c.K}));

            std::vector<float> A(c.M * c.K), B(c.K * c.N), Cout(c.M * c.N);
            Utils::fill_random(A);
            Utils::fill_random(B);

            double ms = bench_neon(A.data(), B.data(), Cout.data(), c.M, c.N, c.K, iters);
            print_row(std::string("  ") + c.label, c.M, c.N, c.K, ms, true, false);
        }
    }

    // -------------------------------------------------------------------------
    // 3. Single run: Scalar vs NEON at a given size
    // -------------------------------------------------------------------------
    void run_single(const float* A, const float* B, float* C_neon,
                    size_t M, size_t N, size_t K) {
        std::cout << "\n========================================\n";
        std::cout << "  Single Run — " << M << "x" << N << "x" << K << "\n";
        std::cout << "========================================\n";

        std::vector<float> C_ref(M * N, 0.0f);
        auto t0 = Clock::now();
        Utils::multiply_scalar(A, B, C_ref.data(), M, N, K);
        double dt_scalar = Ms(Clock::now() - t0).count();

        std::fill(C_neon, C_neon + M * N, 0.0f);
        auto t1 = Clock::now();
        GEMM::package(A, B, C_neon, M, N, K);
        double dt_neon = Ms(Clock::now() - t1).count();

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Scalar : " << dt_scalar << " ms | "
                  << Utils::compute_gflops(M, N, K, dt_scalar) << " GFLOPS\n";
        std::cout << "  NEON   : " << dt_neon << " ms | "
                  << Utils::compute_gflops(M, N, K, dt_neon) << " GFLOPS"
                  << " | " << dt_scalar / dt_neon << "x\n";

        bool ok = Utils::check_correctness(C_ref.data(), C_neon, M * N, "NEON");
        std::cout << "  Correctness: " << (ok ? "OK" : "FAIL") << "\n";
    }

    void run_stress(int iterations, const float* A, const float* B, float* C,
                    size_t M, size_t N, size_t K) {
        GEMM::package(A, B, C, M, N, K); // warmup

        auto start = Clock::now();
        for (int i = 0; i < iterations; ++i)
            GEMM::package(A, B, C, M, N, K);
        double avg_ms = Ms(Clock::now() - start).count() / iterations;
        std::cout << "  [NEON x" << iterations << "]  avg="
                  << std::fixed << std::setprecision(2) << avg_ms << " ms  "
                  << std::setprecision(1) << Utils::compute_gflops(M, N, K, avg_ms) << " GFLOPS\n";
    }

} // namespace Benchmark

// =============================================================================
// MAIN
// =============================================================================
int main() {
    omp_set_num_threads(1);
    std::cout << "  Threads: " << omp_get_max_threads() << "\n";

    std::cout << "========================================\n";
    std::cout << "  GEMM — Scalar / NEON / SME\n";
    std::cout << "  Single thread\n";
    std::cout << "========================================\n";

    // ==========================================================
    // FULL SWEEP — enable everything we want fresh numbers for.
    // For Instruments / power profiling, set FULL_SWEEP to 0 and
    // uncomment exactly one profile() call below.
    // ==========================================================
#define FULL_SWEEP 0
#if FULL_SWEEP

    // --- NEON ---
    NEONTest::run();
    Benchmark::neon_correctness_large();
    Benchmark::neon_speed_sweep();
    {
        constexpr size_t M = 1024, N = 1024, K = 1024;
        std::vector<float> A(M * K), B(K * N), C_neon(M * N);
        Utils::fill_random(A);
        Utils::fill_random(B);
        Benchmark::run_single(A.data(), B.data(), C_neon.data(), M, N, K);
        std::cout << "\n========================================\n";
        std::cout << "  Stress Test (50 iterations, 1024^3)\n";
        std::cout << "========================================\n";
        Benchmark::run_stress(50, A.data(), B.data(), C_neon.data(), M, N, K);
    }

    // --- SME per-kernel suites ---
    SMETest::run(SMETest::Kernel::K4x1);
    SMETest::run(SMETest::Kernel::K2x2);
    SMETest::run(SMETest::Kernel::K1x4);
    SMETest::run(SMETest::Kernel::K1x4Sym);
    SMETest::run(SMETest::Kernel::K1x4SymZAInOut);
    // SMETest::run(SMETest::Kernel::K4x1ZAPack); // disabled: heap-corrupt crash at 32^3 (TODO §2)

    // --- Cross-kernel comparison + 4x1 phase breakdown ---
    SMETest::run_comparison();
    SMETest::run_timing_breakdown();

#endif  // FULL_SWEEP

    // ==========================================================
    // PROFILING DRIVERS — uncomment exactly ONE for scripts/profile.sh /
    // scripts/profile_power.sh runs (set FULL_SWEEP to 0 above).
    // ==========================================================
    {
        [[maybe_unused]] constexpr std::size_t M = 2048, K = 2048, N = 2048;
        [[maybe_unused]] constexpr int iters = 100;

        // SMETest::profile(SMETest::Kernel::K4x1,           M, K, N, iters);
        // SMETest::profile(SMETest::Kernel::K2x2,           M, K, N, iters);
        // SMETest::profile(SMETest::Kernel::K1x4,           M, K, N, iters);
        // SMETest::profile(SMETest::Kernel::K1x4Sym,        M, K, N, iters);
        SMETest::run(SMETest::Kernel::K1x4SymZAInOut); // TEMP: butterfly K_inner=40 baseline
        // SMETest::profile(SMETest::Kernel::K1x4SymZAInOut, M, K, N, iters);
        // SMETest::profile(SMETest::Kernel::K4x1ZAPack,     M, K, N, iters);
    }

    return 0;
}
