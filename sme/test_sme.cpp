#include "test_sme.hpp"
#include "SME-GEMMKernels.hpp"
#include "../common/utils.hpp"

#include <iostream>
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

    void run() {
        std::cout << "========== SME Tests ==========\n";
        run_packing();
        run_correctness();
        run_benchmark();
        std::cout << "===============================\n";
    }

} // namespace SMETest
