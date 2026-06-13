#include "test_neon.hpp"
#include "neon-8x12.hpp"
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

        TestCase cases[] = {
            // Fully aligned
            { 64,  60,  64, "Aligned     ( 64x60x64  )"},
            {128, 120, 128, "Aligned     (128x120x128)"},
            {256, 240, 256, "Aligned     (256x240x256)"},
            // M tail (M % 8 != 0)
            { 67,  60,  64, "M tail      ( 67x60x64  )"},
            // N tail (N % 12 != 0)
            { 64,  65,  64, "N tail      ( 64x65x64  )"},
            // K tail (K % 4 != 0)
            { 64,  60,  65, "K tail      ( 64x60x65  )"},
            // All tails at once
            { 67,  65,  65, "All tails   ( 67x65x65  )"},
            // Smaller than one kernel tile
            {  4,   4,   4, "Tiny        (  4x4x4   )"},
            // Non-uniform non-aligned
            {100, 100, 100, "Non-aligned (100x100x100)"},
            // N > Nc_cache (1020): forces J>1 in the j loop, exposes the
            // N tail accumulation bug if the j==0 guard is missing
            {128, 1100, 128, "N > Nc_cache (128x1100x128)"},
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
