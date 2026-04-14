#include "test_sme.hpp"
#include "SME-GEMMKernels.hpp"
#include "SME-GEMMKernels2x2.hpp"
#include "../common/utils.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <arm_sve.h>

namespace SMETest {

    __arm_locally_streaming
    static void run_packing() {
        std::cout << "\n--- SME Packing ---\n";

        const size_t SVL = static_cast<size_t>(svcntw());
        const size_t M = 20, K = 35, N = 41;

        // Deterministic input for easy manual verification
        std::vector<float> A(M * K), B(K * N);
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < K; j++)
                A[i * K + j] = static_cast<float>(i * 1000 + j);
        for (size_t i = 0; i < K; i++)
            for (size_t j = 0; j < N; j++)
                B[i * N + j] = static_cast<float>(i * 1000 + j);

        const size_t M_curr = (M / SVL) * SVL;
        const size_t K_curr = K;
        const size_t N_curr = N;

        // pack_B: expected layout — packed_B[n * K_curr + k] == B[k * N + n]
        std::vector<float> packed_B(K_curr * ((N_curr + SVL - 1) / SVL) * SVL, -999.0f);
        SMEKernels::pack_B_streaming(B.data(), packed_B.data(), N_curr, K_curr, 0, N, 0);

        bool pack_B_ok = true;
        for (size_t k = 0; k < K_curr && pack_B_ok; k++) {
            for (size_t n = 0; n < N_curr && pack_B_ok; n++) {
                size_t panel = (n / SVL) * SVL;
                size_t lane  = n % SVL;
                float expected = B[k * N + n];
                float got      = packed_B[panel * K_curr + k * SVL + lane];
                if (std::abs(expected - got) > 1e-5f) {
                    std::cout << "  pack_B FAIL: B[" << k << "," << n << "]="
                              << expected << " got=" << got << "\n";
                    pack_B_ok = false;
                }
            }
        }
        std::cout << "  pack_B : " << (pack_B_ok ? "PASS" : "FAIL") << "\n";

        // pack_A: expected layout — packed_A[panel * K_curr * SVL + k * SVL + row] == A[m * K + k]
        std::vector<float> packed_A(M_curr * K_curr * 2, -999.0f);
        SMEKernels::pack_A_streaming(A.data(), packed_A.data(), M_curr, K_curr, 0, 0, K);

        bool pack_A_ok = true;
        for (size_t m = 0; m < M_curr && pack_A_ok; m++) {
            size_t panel = m / SVL;
            size_t lane  = m % SVL;
            for (size_t k = 0; k < K_curr && pack_A_ok; k++) {
                float expected = A[m * K + k];
                float got      = packed_A[panel * K_curr * SVL + k * SVL + lane];
                if (std::abs(expected - got) > 1e-5f) {
                    std::cout << "  pack_A FAIL: A[" << m << "," << k << "]="
                              << expected << " got=" << got << "\n";
                    pack_A_ok = false;
                }
            }
        }
        std::cout << "  pack_A : " << (pack_A_ok ? "PASS" : "FAIL") << "\n";
    }

    static void run_correctness() {
        std::cout << "\n--- SME Correctness ---\n";

        struct TestCase { size_t M, K, N; const char* name; };
        TestCase cases[] = {
            { 16,  16,  16, "Small  ( 16x16x16 )"},
            { 64,  64,  64, "Medium ( 64x64x64 )"},
            {128, 128, 128, "Large  (128x128x128)"},
            { 20,  35,  41, "Non-aligned (20x35x41)"},
        };

        for (auto& tc : cases) {
            std::vector<float> A(tc.M * tc.K);
            std::vector<float> B(tc.K * tc.N);
            std::vector<float> C_ref(tc.M * tc.N, 0.0f);
            std::vector<float> C_sme(tc.M * tc.N, 0.0f);

            Utils::fill_random(A);
            Utils::fill_random(B);
            Utils::multiply_scalar(A.data(), B.data(), C_ref.data(), tc.M, tc.N, tc.K);
            SMEKernels::run_multiplication(A.data(), B.data(), C_sme.data(), tc.M, tc.K, tc.N);

            bool ok = Utils::check_correctness(C_ref.data(), C_sme.data(), tc.M * tc.N, "SME");
            std::cout << "  " << tc.name << " : " << (ok ? "PASS" : "FAIL") << "\n";
        }
    }

    static void run_benchmark() {
            std::cout << "\n--- Performance Benchmark (GFLOPS) ---\n";

            // Performans ölçümü için matris boyutlarının donanımı zorlayacak kadar
            // büyük olması (örneğin önbelleğe sığmaması) daha doğru sonuç verir.
            struct TestCase { size_t M, N, K; const char* name; };
            TestCase cases[] = {
                { 256,  256,  256, "Small  ( 256x256x256 )"},
                { 512,  512,  512, "Medium ( 512x512x512 )"},
                {1024, 1024, 1024, "Large  (1024x1024x1024)"},
                {2048, 2048, 2048, "Huge   (2048x2048x2048)"}
            };

            const int num_iterations = 10; // Daha tutarlı bir ortalama elde etmek için

            for (auto& tc : cases) {
                std::vector<float> A(tc.M * tc.K);
                std::vector<float> B(tc.K * tc.N);
                std::vector<float> C(tc.M * tc.N, 0.0f);

                Utils::fill_random(A);
                Utils::fill_random(B);

                // Isınma (Warm-up) turu: Önbelleği doldurmak ve işlemci frekansını
                // maksimum seviyeye (turbo boost vb.) çekmek için bir kez boşa çalıştırıyoruz.
                SMEKernels::run_multiplication(A.data(), B.data(), C.data(), tc.M, tc.K, tc.N);

                // Zaman ölçümünü başlat
                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < num_iterations; i++) {
                    // NEON testi için burayı GEMM::package(A.data(), B.data(), C.data(), tc.M, tc.N, tc.K);
                    // olarak değiştirmelisin.
                    SMEKernels::run_multiplication(A.data(), B.data(), C.data(), tc.M, tc.K, tc.N);
                }

                // Zaman ölçümünü bitir
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = end - start;
                double seconds = diff.count();

                // GFLOPS Hesabı
                // Toplam FLOPs = 2 * M * N * K * iterasyon_sayısı
                double total_flops = 2.0 * tc.M * tc.N * tc.K * num_iterations;
                double gflops = (total_flops / 1e9) / seconds;
                double ms_per_iter = (seconds / num_iterations) * 1000.0;

                std::cout << "  " << tc.name << " : "
                          << gflops << " GFLOPS ("
                          << ms_per_iter << " ms/iter)\n";
            }
        }

    // =================================================================
    // 2x2 KERNEL TESTS
    // =================================================================

    __arm_locally_streaming
    static void run_packing_2x2() {
        std::cout << "\n--- SME 2x2 Packing ---\n";

        const size_t SVL = static_cast<size_t>(svcntw());
        const size_t M = 20, K = 35, N = 41;

        std::vector<float> A(M * K), B(K * N);
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < K; j++)
                A[i * K + j] = static_cast<float>(i * 1000 + j);
        for (size_t i = 0; i < K; i++)
            for (size_t j = 0; j < N; j++)
                B[i * N + j] = static_cast<float>(i * 1000 + j);

        const size_t M_curr = (M / SVL) * SVL;
        const size_t K_curr = K;
        const size_t N_curr = N;

        // pack_A (same layout as 4x1)
        std::vector<float> packed_A(M_curr * K_curr * 2, -999.0f);
        SMEKernels2x2::pack_A_streaming(A.data(), packed_A.data(), M_curr, K_curr, 0, 0, K);

        bool pack_A_ok = true;
        for (size_t m = 0; m < M_curr && pack_A_ok; m++) {
            size_t panel = m / SVL;
            size_t lane  = m % SVL;
            for (size_t k = 0; k < K_curr && pack_A_ok; k++) {
                float expected = A[m * K + k];
                float got      = packed_A[panel * K_curr * SVL + k * SVL + lane];
                if (std::abs(expected - got) > 1e-5f) {
                    std::cout << "  pack_A FAIL: A[" << m << "," << k << "]="
                              << expected << " got=" << got << "\n";
                    pack_A_ok = false;
                }
            }
        }
        std::cout << "  pack_A : " << (pack_A_ok ? "PASS" : "FAIL") << "\n";

        // pack_B: interleaved 2*SVL layout
        // For each 2*SVL panel, layout is [b0_k | b1_k] per k-step
        const size_t panel_width = 2 * SVL;
        const size_t num_panels = (N_curr + panel_width - 1) / panel_width;
        std::vector<float> packed_B(num_panels * K_curr * panel_width, -999.0f);
        SMEKernels2x2::pack_B_streaming(B.data(), packed_B.data(), N_curr, K_curr, 0, N, 0);

        bool pack_B_ok = true;
        for (size_t k = 0; k < K_curr && pack_B_ok; k++) {
            for (size_t n = 0; n < N_curr && pack_B_ok; n++) {
                size_t panel_idx = n / panel_width;
                size_t within    = n % panel_width;
                // Each k-step is panel_width floats; panel starts at panel_idx * K_curr * panel_width
                float expected = B[k * N + n];
                float got      = packed_B[panel_idx * K_curr * panel_width + k * panel_width + within];
                if (std::abs(expected - got) > 1e-5f) {
                    std::cout << "  pack_B FAIL: B[" << k << "," << n << "]="
                              << expected << " got=" << got << "\n";
                    pack_B_ok = false;
                }
            }
        }
        std::cout << "  pack_B : " << (pack_B_ok ? "PASS" : "FAIL") << "\n";
    }

    static void run_correctness_2x2() {
        std::cout << "\n--- SME 2x2 Correctness ---\n";

        struct TestCase { size_t M, K, N; const char* name; };
        TestCase cases[] = {
            { 16,  16,  16, "Small  ( 16x16x16 )"},
            { 64,  64,  64, "Medium ( 64x64x64 )"},
            {128, 128, 128, "Large  (128x128x128)"},
            { 20,  35,  41, "Non-aligned (20x35x41)"},
            { 32,  32,  32, "Exact 2*SVL (32x32x32)"},
            {256, 256, 256, "256x256x256"},
            {512, 512, 512, "512x512x512"},
        };

        for (auto& tc : cases) {
            std::vector<float> A(tc.M * tc.K);
            std::vector<float> B(tc.K * tc.N);
            std::vector<float> C_ref(tc.M * tc.N, 0.0f);
            std::vector<float> C_sme(tc.M * tc.N, 0.0f);

            Utils::fill_random(A);
            Utils::fill_random(B);
            Utils::multiply_scalar(A.data(), B.data(), C_ref.data(), tc.M, tc.N, tc.K);
            SMEKernels2x2::run_multiplication(A.data(), B.data(), C_sme.data(), tc.M, tc.K, tc.N);

            bool ok = Utils::check_correctness(C_ref.data(), C_sme.data(), tc.M * tc.N, "SME-2x2");
            std::cout << "  " << tc.name << " : " << (ok ? "PASS" : "FAIL") << "\n";
        }
    }

    static void run_benchmark_2x2() {
        std::cout << "\n--- SME 2x2 Performance Benchmark (GFLOPS) ---\n";

        struct TestCase { size_t M, N, K; const char* name; };
        TestCase cases[] = {
            { 256,  256,  256, "Small  ( 256x256x256 )"},
            { 512,  512,  512, "Medium ( 512x512x512 )"},
            {1024, 1024, 1024, "Large  (1024x1024x1024)"},
            {2048, 2048, 2048, "Huge   (2048x2048x2048)"}
        };

        const int num_iterations = 10;

        for (auto& tc : cases) {
            std::vector<float> A(tc.M * tc.K);
            std::vector<float> B(tc.K * tc.N);
            std::vector<float> C(tc.M * tc.N, 0.0f);

            Utils::fill_random(A);
            Utils::fill_random(B);

            // Warmup
            SMEKernels2x2::run_multiplication(A.data(), B.data(), C.data(), tc.M, tc.K, tc.N);

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_iterations; i++)
                SMEKernels2x2::run_multiplication(A.data(), B.data(), C.data(), tc.M, tc.K, tc.N);
            auto end = std::chrono::high_resolution_clock::now();

            double seconds = std::chrono::duration<double>(end - start).count();
            double total_flops = 2.0 * tc.M * tc.N * tc.K * num_iterations;
            double gflops = (total_flops / 1e9) / seconds;
            double ms_per_iter = (seconds / num_iterations) * 1000.0;

            std::cout << "  " << tc.name << " : "
                      << gflops << " GFLOPS ("
                      << ms_per_iter << " ms/iter)\n";
        }
    }

    // =================================================================
    // SIDE-BY-SIDE COMPARISON: 4x1 vs 2x2
    // =================================================================

    static void run_comparison_impl() {
        std::cout << "\n========================================\n";
        std::cout << "  4x1 vs 2x2 Side-by-Side Comparison\n";
        std::cout << "========================================\n";

        struct TestCase { size_t M, N, K; const char* name; };
        TestCase cases[] = {
            { 256,  256,  256, " 256^3"},
            { 512,  512,  512, " 512^3"},
            {1024, 1024, 1024, "1024^3"},
            {2048, 2048, 2048, "2048^3"},
        };

        const int num_iterations = 10;

        std::cout << std::left << std::setw(12) << "  Size"
                  << std::right << std::setw(14) << "4x1 GFLOPS"
                  << std::setw(14) << "2x2 GFLOPS"
                  << std::setw(10) << "Winner" << "\n";
        std::cout << "  " << std::string(48, '-') << "\n";

        for (auto& tc : cases) {
            std::vector<float> A(tc.M * tc.K);
            std::vector<float> B(tc.K * tc.N);
            std::vector<float> C(tc.M * tc.N, 0.0f);

            Utils::fill_random(A);
            Utils::fill_random(B);

            // --- 4x1 ---
            SMEKernels::run_multiplication(A.data(), B.data(), C.data(), tc.M, tc.K, tc.N);
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_iterations; i++)
                SMEKernels::run_multiplication(A.data(), B.data(), C.data(), tc.M, tc.K, tc.N);
            auto t1 = std::chrono::high_resolution_clock::now();
            double sec_4x1 = std::chrono::duration<double>(t1 - t0).count();
            double gflops_4x1 = (2.0 * tc.M * tc.N * tc.K * num_iterations / 1e9) / sec_4x1;

            // --- 2x2 ---
            std::fill(C.begin(), C.end(), 0.0f);
            SMEKernels2x2::run_multiplication(A.data(), B.data(), C.data(), tc.M, tc.K, tc.N);
            auto t2 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_iterations; i++)
                SMEKernels2x2::run_multiplication(A.data(), B.data(), C.data(), tc.M, tc.K, tc.N);
            auto t3 = std::chrono::high_resolution_clock::now();
            double sec_2x2 = std::chrono::duration<double>(t3 - t2).count();
            double gflops_2x2 = (2.0 * tc.M * tc.N * tc.K * num_iterations / 1e9) / sec_2x2;

            const char* winner = (gflops_2x2 > gflops_4x1) ? "2x2" : "4x1";

            std::cout << std::left << std::setw(12) << (std::string("  ") + tc.name)
                      << std::right << std::fixed << std::setprecision(1)
                      << std::setw(14) << gflops_4x1
                      << std::setw(14) << gflops_2x2
                      << std::setw(10) << winner << "\n";
        }
    }

    // =================================================================
    // ENTRY POINTS
    // =================================================================

    void run() {
        std::cout << "========== SME 4x1 Tests ==========\n";
        run_packing();
        run_correctness();
        run_benchmark();
        std::cout << "====================================\n";
    }

    void run_2x2() {
        std::cout << "========== SME 2x2 Tests ==========\n";
        run_packing_2x2();
        run_correctness_2x2();
        run_benchmark_2x2();
        std::cout << "====================================\n";
    }

    void run_comparison() {
        run_comparison_impl();
    }

} // namespace SMETest
