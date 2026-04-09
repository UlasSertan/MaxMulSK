#include "test_neon.hpp"
#include "GEMMKernels.hpp"
#include "../common/utils.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>

namespace NEONTest {

    struct TestCase {
        size_t M, N, K;
        const char* name;
    };

    static void run_correctness() {
        std::cout << "\n--- NEON Correctness ---\n";

        // N must be a multiple of 12 (kernel tile width constraint)
        TestCase cases[] = {
            { 64,  60,  64, "Small  ( 64x60x64  )"},
            {128, 120, 128, "Medium (128x120x128)"},
            {256, 240, 256, "Large  (256x240x256)"},
        };

        for (auto& tc : cases) {
            std::vector<float> A(tc.M * tc.K);
            std::vector<float> B(tc.K * tc.N);
            std::vector<float> C_ref(tc.M * tc.N, 0.0f);
            std::vector<float> C_neon(tc.M * tc.N, 0.0f);

            Utils::fill_random(A);
            Utils::fill_random(B);
            Utils::multiply_scalar(A.data(), B.data(), C_ref.data(), tc.M, tc.N, tc.K);
            GEMM::package(A.data(), B.data(), C_neon.data(), tc.M, tc.N, tc.K);

            bool ok = Utils::check_correctness(C_ref.data(), C_neon.data(), tc.M * tc.N, "NEON");
            std::cout << "  " << tc.name << " : " << (ok ? "PASS" : "FAIL") << "\n";
        }
    }

    static void run_packing() {
        std::cout << "\n--- NEON Packing ---\n";

        // Verify transpose_8x4: an 8x4 block of A transposed into packed layout
        // Expected: packed[k*8 + row] == A[row*K + k]
        const size_t K = 16;
        std::vector<float> src(8 * K);
        std::vector<float> dst(32, -1.0f); // 8 rows * 4 cols

        for (size_t r = 0; r < 8; r++)
            for (size_t c = 0; c < K; c++)
                src[r * K + c] = static_cast<float>(r * 100 + c);

        GEMM::transpose_8x4(src.data(), dst.data(), K);

        bool ok = true;
        for (size_t k = 0; k < 4 && ok; k++) {
            for (size_t r = 0; r < 8 && ok; r++) {
                float expected = src[r * K + k];
                float got      = dst[k * 8 + r];
                if (expected != got) {
                    std::cout << "  transpose_8x4 FAIL at k=" << k << " r=" << r
                              << " expected=" << expected << " got=" << got << "\n";
                    ok = false;
                }
            }
        }
        std::cout << "  transpose_8x4 : " << (ok ? "PASS" : "FAIL") << "\n";
    }

    void run() {
        std::cout << "========== NEON Tests ==========\n";
        run_packing();
        run_correctness();
        std::cout << "================================\n";
    }

} // namespace NEONTest
