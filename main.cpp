#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <omp.h>

#include "common/utils.hpp"
#include "neon/GEMMKernels.hpp"
#include "neon/test_neon.hpp"
#include "sme/SME-GEMMKernels.hpp"
#include "sme/test_sme.hpp"

// =============================================================================
// BENCHMARK
// =============================================================================
namespace Benchmark {

    void run_single(const float* A, const float* B,
                    float* C_neon, float* C_sme,
                    size_t M, size_t N, size_t K) {
        std::cout << "\n--- Single Run (correctness + timing) ---\n";

        // Scalar reference
        std::vector<float> C_ref(M * N, 0.0f);
        auto t0 = std::chrono::high_resolution_clock::now();
        Utils::multiply_scalar(A, B, C_ref.data(), M, N, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt_scalar = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // NEON
        std::fill(C_neon, C_neon + M * N, 0.0f);
        auto t2 = std::chrono::high_resolution_clock::now();
        GEMM::package(A, B, C_neon, M, N, K);
        auto t3 = std::chrono::high_resolution_clock::now();
        double dt_neon = std::chrono::duration<double, std::milli>(t3 - t2).count();

        // SME
        std::fill(C_sme, C_sme + M * N, 0.0f);
        auto t4 = std::chrono::high_resolution_clock::now();
        SMEKernels::run_multiplication(A, B, C_sme, M, K, N);
        auto t5 = std::chrono::high_resolution_clock::now();
        double dt_sme = std::chrono::duration<double, std::milli>(t5 - t4).count();

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Scalar : " << dt_scalar << " ms | "
                  << Utils::compute_gflops(M, N, K, dt_scalar) << " GFLOPS\n";
        std::cout << "  NEON   : " << dt_neon   << " ms | "
                  << Utils::compute_gflops(M, N, K, dt_neon)   << " GFLOPS"
                  << " | " << dt_scalar / dt_neon  << "x\n";
        std::cout << "  SME    : " << dt_sme    << " ms | "
                  << Utils::compute_gflops(M, N, K, dt_sme)    << " GFLOPS"
                  << " | " << dt_scalar / dt_sme   << "x\n";

        bool neon_ok = Utils::check_correctness(C_ref.data(), C_neon, M * N, "NEON");
        bool sme_ok  = Utils::check_correctness(C_ref.data(), C_sme,  M * N, "SME");
        std::cout << "  Correctness: NEON=" << (neon_ok ? "OK" : "FAIL")
                  << "  SME=" << (sme_ok ? "OK" : "FAIL") << "\n";
    }

    void run_stress(int iterations, const float* A, const float* B, float* C,
                    size_t M, size_t N, size_t K, const std::string& label) {
        // Warmup
        if (label == "NEON") GEMM::package(A, B, C, M, N, K);
        else                  SMEKernels::run_multiplication(A, B, C, M, K, N);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            if (label == "NEON") GEMM::package(A, B, C, M, N, K);
            else                  SMEKernels::run_multiplication(A, B, C, M, K, N);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        std::cout << "  [" << label << " x" << iterations << "]  avg=" << avg_ms << " ms"
                  << "  " << Utils::compute_gflops(M, N, K, avg_ms) << " GFLOPS\n";
    }

} // namespace Benchmark

// =============================================================================
// MAIN
// =============================================================================
int main() {
    omp_set_num_threads(1);

    const size_t M = 1024;
    const size_t N = 1020; // multiple of 12 (NEON tile width)
    const size_t K = 1024;

    std::cout << "========================================\n";
    std::cout << "  GEMM Benchmark — Scalar / NEON / SME\n";
    std::cout << "========================================\n";
    std::cout << "M=" << M << "  N=" << N << "  K=" << K << "  threads=1\n";

    std::vector<float> A(M * K), B(K * N);
    std::vector<float> C_neon(M * N), C_sme(M * N);
    Utils::fill_random(A);
    Utils::fill_random(B);

    // --- Unit tests ---
    NEONTest::run();
    SMETest::run();

    // --- Benchmark ---
    Benchmark::run_single(A.data(), B.data(), C_neon.data(), C_sme.data(), M, N, K);

    std::cout << "\n--- Stress Tests (50 iterations) ---\n";
    Benchmark::run_stress(50, A.data(), B.data(), C_neon.data(), M, N, K, "NEON");
    Benchmark::run_stress(50, A.data(), B.data(), C_sme.data(),  M, N, K, "SME");

    return 0;
}
