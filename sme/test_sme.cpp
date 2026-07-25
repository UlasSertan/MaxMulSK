#include "test_sme.hpp"
#include "sme-4x1.hpp"
#include "sme-2x2.hpp"
#include "sme-1x4.hpp"
#include "sme-1x4-sym.hpp"
#include "sme-1x4-sym-zainout.hpp"
#include "sme-4x1-zapack.hpp"
#include "../common/utils.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <string>
#include <arm_sve.h>

namespace SMETest {

    using Clock = std::chrono::high_resolution_clock;

    // =========================================================================
    // SHARED TEST CASES
    // =========================================================================

    struct CorrectnessCase { std::size_t M, K, N; const char* name; };
    static const CorrectnessCase kCorrectnessCases[] = {
        {  16,  16,  16, "16x16x16        [small]"           },
        {  32,  32,  32, "32x32x32        [exact 2*SVL]"     },
        {  64,  64,  64, "64x64x64        [aligned]"         },
        {  67,  67,  67, "67x67x67        [all tails]"       },
        { 128, 128, 128, "128x128x128     [aligned]"         },
        {  20,  35,  41, "20x35x41        [non-aligned]"     },
        { 256, 256, 256, "256x256x256     [medium]"          },
        { 512, 512, 512, "512x512x512     [large]"           },
    };

    struct BenchCase { std::size_t M, K, N; const char* name; };
    static const BenchCase kBenchCases[] = {
        {  256,  256,  256, " 256^3" },
        {  512,  512,  512, " 512^3" },
        { 1024, 1024, 1024, "1024^3" },
        { 2048, 2048, 2048, "2048^3" },
    };

    static const BenchCase kComparisonCases[] = {
        {  256,  256,  256, " 256^3" },
        {  512,  512,  512, " 512^3" },
        { 1024, 1024, 1024, "1024^3" },
        { 2048, 2048, 2048, "2048^3" },
        { 4096, 4096, 4096, "4096^3" },
    };

    // Iteration policy:
    //   run(k)            — flat 10, matches historical numbers in docs/BENCHMARKS.md
    //   run_comparison()  — bumped to 50 for stable kernel ordering
    //   profile()         — caller-controlled (main.cpp picks)
    static constexpr int kBenchIters      = 10;
    static constexpr int kComparisonIters = 50;

    // =========================================================================
    // KERNEL DISPATCH
    // =========================================================================

    static const char* kernel_name(Kernel k) {
        switch (k) {
            case Kernel::K4x1:        return "4x1";
            case Kernel::K2x2:        return "2x2";
            case Kernel::K1x4:        return "1x4";
            case Kernel::K1x4Sym:        return "1x4-sym";
            case Kernel::K1x4SymZAInOut: return "1x4ZAIO";
            case Kernel::K4x1ZAPack:     return "4x1-ZAPack";
        }
        return "?";
    }

    static void run_kernel(Kernel k,
                           const float* A, const float* B, float* C,
                           std::size_t M, std::size_t K, std::size_t N) {
        switch (k) {
            case Kernel::K4x1:        SMEKernels4x1::run_multiplication(A, B, C, M, K, N);       return;
            case Kernel::K2x2:        SMEKernels2x2::run_multiplication(A, B, C, M, K, N);       return;
            case Kernel::K1x4:        SMEKernels1x4::run_multiplication(A, B, C, M, K, N);       return;
            case Kernel::K1x4Sym:        SMEKernels1x4Sym::run_multiplication(A, B, C, M, K, N);        return;
            case Kernel::K1x4SymZAInOut: SMEKernels1x4SymZAInOut::run_multiplication(A, B, C, M, K, N); return;
            case Kernel::K4x1ZAPack:     SMEKernels4x1ZAPack::run_multiplication(A, B, C, M, K, N);     return;
        }
    }

    // All kernels now have a scratch-buffer fallback for edge tiles
    // (BUG-4x1-SMALL-M / BUG-ZAPACK-WRONG fixed), so any M/K/N is legal.
    static bool kernel_can_handle(Kernel /*k*/, std::size_t /*M*/, std::size_t /*K*/, std::size_t /*N*/) {
        return true;
    }

    // =========================================================================
    // PACK CORRECTNESS — one helper per layout family.
    // Bodies are the previous run_packing_* logic minus the banner.
    // The layout-verification index math is what differs between kernels.
    // =========================================================================

    // 4x1 layout:
    //   pack_A: 4-panel interleaved, GS = 4*SVL
    //   pack_B: SVL-wide panels
    __arm_locally_streaming
    static void check_pack_4x1() {
        const std::size_t SVL = static_cast<std::size_t>(svcntw());
        const std::size_t M = 20, K = 35, N = 41;

        std::vector<float> A(M * K), B(K * N);
        for (std::size_t i = 0; i < M; i++)
            for (std::size_t j = 0; j < K; j++)
                A[i * K + j] = static_cast<float>(i * 1000 + j);
        for (std::size_t i = 0; i < K; i++)
            for (std::size_t j = 0; j < N; j++)
                B[i * N + j] = static_cast<float>(i * 1000 + j);

        const std::size_t M_curr = (M / SVL) * SVL;
        const std::size_t K_curr = K;
        const std::size_t N_curr = N;

        std::vector<float> packed_B(K_curr * ((N_curr + SVL - 1) / SVL) * SVL, -999.0f);
        SMEKernels4x1::pack_B_streaming(B.data(), packed_B.data(), N_curr, K_curr, 0, N, 0);

        bool pack_B_ok = true;
        for (std::size_t k = 0; k < K_curr && pack_B_ok; k++) {
            for (std::size_t n = 0; n < N_curr && pack_B_ok; n++) {
                std::size_t panel = (n / SVL) * SVL;
                std::size_t lane  = n % SVL;
                if (std::abs(B[k * N + n] - packed_B[panel * K_curr + k * SVL + lane]) > 1e-5f)
                    pack_B_ok = false;
            }
        }
        std::cout << "  pack_B : " << (pack_B_ok ? "PASS" : "FAIL") << "\n";

        const std::size_t GS = 4 * SVL;
        const std::size_t groups_4x1 = (M_curr + GS - 1) / GS;
        std::vector<float> packed_A(groups_4x1 * GS * K_curr, -999.0f);
        SMEKernels4x1::pack_A_streaming(A.data(), packed_A.data(), M_curr, K_curr, 0, 0, K);

        bool pack_A_ok = true;
        for (std::size_t m = 0; m < M_curr && pack_A_ok; m++) {
            std::size_t panel = m / SVL;
            std::size_t lane  = m % SVL;
            for (std::size_t k = 0; k < K_curr && pack_A_ok; k++) {
                if (std::abs(A[m * K + k] - packed_A[k * GS + panel * SVL + lane]) > 1e-5f)
                    pack_A_ok = false;
            }
        }
        std::cout << "  pack_A : " << (pack_A_ok ? "PASS" : "FAIL") << "\n";
    }

    // Same layout as 4x1; pack_A uses ZA-based transpose internally,
    // hence __arm_new("za") on the wrapper.
    __arm_locally_streaming __arm_new("za")
    static void check_pack_4x1ZAPack() {
        const std::size_t SVL = static_cast<std::size_t>(svcntw());
        const std::size_t M = 20, K = 35, N = 41;

        std::vector<float> A(M * K), B(K * N);
        for (std::size_t i = 0; i < M; i++)
            for (std::size_t j = 0; j < K; j++)
                A[i * K + j] = static_cast<float>(i * 1000 + j);
        for (std::size_t i = 0; i < K; i++)
            for (std::size_t j = 0; j < N; j++)
                B[i * N + j] = static_cast<float>(i * 1000 + j);

        const std::size_t M_curr = (M / SVL) * SVL;
        const std::size_t K_curr = K;
        const std::size_t N_curr = N;

        std::vector<float> packed_B(K_curr * ((N_curr + SVL - 1) / SVL) * SVL, -999.0f);
        SMEKernels4x1ZAPack::pack_B_streaming(B.data(), packed_B.data(), N_curr, K_curr, 0, N, 0);

        bool pack_B_ok = true;
        for (std::size_t k = 0; k < K_curr && pack_B_ok; k++) {
            for (std::size_t n = 0; n < N_curr && pack_B_ok; n++) {
                std::size_t panel = (n / SVL) * SVL;
                std::size_t lane  = n % SVL;
                if (std::abs(B[k * N + n] - packed_B[panel * K_curr + k * SVL + lane]) > 1e-5f)
                    pack_B_ok = false;
            }
        }
        std::cout << "  pack_B : " << (pack_B_ok ? "PASS" : "FAIL") << "\n";

        const std::size_t GS = 4 * SVL;
        const std::size_t groups_zap = (M_curr + GS - 1) / GS;
        std::vector<float> packed_A(groups_zap * GS * K_curr, -999.0f);
        SMEKernels4x1ZAPack::pack_A_streaming(A.data(), packed_A.data(), M_curr, K_curr, 0, 0, K);

        bool pack_A_ok = true;
        for (std::size_t m = 0; m < M_curr && pack_A_ok; m++) {
            std::size_t panel = m / SVL;
            std::size_t lane  = m % SVL;
            for (std::size_t k = 0; k < K_curr && pack_A_ok; k++) {
                if (std::abs(A[m * K + k] - packed_A[k * GS + panel * SVL + lane]) > 1e-5f)
                    pack_A_ok = false;
            }
        }
        std::cout << "  pack_A : " << (pack_A_ok ? "PASS" : "FAIL") << "\n";
    }

    // 2x2 layout:
    //   pack_A: 2-panel interleaved per group, GS = 2*SVL
    //   pack_B: 2*SVL-wide interleaved panels
    __arm_locally_streaming __arm_new("za")
    static void check_pack_2x2() {
        const std::size_t SVL = static_cast<std::size_t>(svcntw());
        const std::size_t M = 20, K = 35, N = 41;

        std::vector<float> A(M * K), B(K * N);
        for (std::size_t i = 0; i < M; i++)
            for (std::size_t j = 0; j < K; j++)
                A[i * K + j] = static_cast<float>(i * 1000 + j);
        for (std::size_t i = 0; i < K; i++)
            for (std::size_t j = 0; j < N; j++)
                B[i * N + j] = static_cast<float>(i * 1000 + j);

        const std::size_t M_curr = (M / SVL) * SVL;
        const std::size_t K_curr = K;
        const std::size_t N_curr = N;

        std::vector<float> packed_A(M_curr * K_curr * 2, -999.0f);
        SMEKernels2x2::pack_A_streaming(A.data(), packed_A.data(), M_curr, K_curr, 0, 0, K);

        bool pack_A_ok = true;
        const std::size_t GS_2x2 = 2 * SVL;
        for (std::size_t m = 0; m < M_curr && pack_A_ok; m++) {
            std::size_t panel_global = m / SVL;
            std::size_t p_local      = panel_global % 2;
            std::size_t group        = panel_global / 2;
            std::size_t group_offset = group * GS_2x2 * K_curr;
            std::size_t lane         = m % SVL;
            for (std::size_t k = 0; k < K_curr && pack_A_ok; k++) {
                if (std::abs(A[m * K + k] - packed_A[group_offset + k * GS_2x2 + p_local * SVL + lane]) > 1e-5f)
                    pack_A_ok = false;
            }
        }
        std::cout << "  pack_A : " << (pack_A_ok ? "PASS" : "FAIL") << "\n";

        const std::size_t panel_width = 2 * SVL;
        const std::size_t num_panels  = (N_curr + panel_width - 1) / panel_width;
        std::vector<float> packed_B(num_panels * K_curr * panel_width, -999.0f);
        SMEKernels2x2::pack_B_streaming(B.data(), packed_B.data(), N_curr, K_curr, 0, N, 0);

        bool pack_B_ok = true;
        for (std::size_t k = 0; k < K_curr && pack_B_ok; k++) {
            for (std::size_t n = 0; n < N_curr && pack_B_ok; n++) {
                std::size_t panel_idx = n / panel_width;
                std::size_t within    = n % panel_width;
                if (std::abs(B[k * N + n] - packed_B[panel_idx * K_curr * panel_width + k * panel_width + within]) > 1e-5f)
                    pack_B_ok = false;
            }
        }
        std::cout << "  pack_B : " << (pack_B_ok ? "PASS" : "FAIL") << "\n";
    }

    // 1x4 layout:
    //   pack_A: single SVL-wide panel per m-step (GS = SVL)
    //   pack_B: 4-panel interleaved per group (GS = 4*SVL)
    __arm_locally_streaming
    static void check_pack_1x4() {
        const std::size_t SVL = static_cast<std::size_t>(svcntw());
        const std::size_t M = 32, K = 35, N = 128;  // M, N must be multiples of SVL / 4*SVL

        std::vector<float> A(M * K), B(K * N);
        for (std::size_t i = 0; i < M; i++)
            for (std::size_t j = 0; j < K; j++)
                A[i * K + j] = static_cast<float>(i * 1000 + j);
        for (std::size_t i = 0; i < K; i++)
            for (std::size_t j = 0; j < N; j++)
                B[i * N + j] = static_cast<float>(i * 1000 + j);

        const std::size_t M_curr = M;
        const std::size_t K_curr = K;
        const std::size_t N_curr = N;

        std::vector<float> packed_A(M_curr * K_curr + SVL, -999.0f);
        SMEKernels1x4::pack_A_streaming(A.data(), packed_A.data(), M_curr, K_curr, 0, 0, K);

        bool pack_A_ok = true;
        for (std::size_t m = 0; m < M_curr && pack_A_ok; m++) {
            std::size_t m_idx      = m / SVL;
            std::size_t lane       = m % SVL;
            std::size_t block_base = m_idx * SVL * K_curr;
            for (std::size_t k = 0; k < K_curr && pack_A_ok; k++) {
                if (std::abs(A[m * K + k] - packed_A[block_base + k * SVL + lane]) > 1e-5f)
                    pack_A_ok = false;
            }
        }
        std::cout << "  pack_A : " << (pack_A_ok ? "PASS" : "FAIL") << "\n";

        const std::size_t GS_B       = 4 * SVL;
        const std::size_t num_groups = (N_curr + GS_B - 1) / GS_B;
        std::vector<float> packed_B(num_groups * K_curr * GS_B, -999.0f);
        SMEKernels1x4::pack_B_streaming(B.data(), packed_B.data(), N_curr, K_curr, 0, N, 0);

        bool pack_B_ok = true;
        for (std::size_t k = 0; k < K_curr && pack_B_ok; k++) {
            for (std::size_t n = 0; n < N_curr && pack_B_ok; n++) {
                std::size_t g      = n / GS_B;
                std::size_t within = n % GS_B;
                if (std::abs(B[k * N + n] - packed_B[g * K_curr * GS_B + k * GS_B + within]) > 1e-5f)
                    pack_B_ok = false;
            }
        }
        std::cout << "  pack_B : " << (pack_B_ok ? "PASS" : "FAIL") << "\n";
    }

    // 1x4-sym layout: identical to 1x4 (pack_A: SVL panel, GS=SVL;
    //                                    pack_B: 4-panel interleaved, GS=4*SVL).
    // Only the micro-kernel load schedule differs.
    __arm_locally_streaming
    static void check_pack_1x4_sym() {
        const std::size_t SVL = static_cast<std::size_t>(svcntw());
        const std::size_t M = 32, K = 35, N = 128;

        std::vector<float> A(M * K), B(K * N);
        for (std::size_t i = 0; i < M; i++)
            for (std::size_t j = 0; j < K; j++)
                A[i * K + j] = static_cast<float>(i * 1000 + j);
        for (std::size_t i = 0; i < K; i++)
            for (std::size_t j = 0; j < N; j++)
                B[i * N + j] = static_cast<float>(i * 1000 + j);

        const std::size_t M_curr = M;
        const std::size_t K_curr = K;
        const std::size_t N_curr = N;

        std::vector<float> packed_A(M_curr * K_curr + SVL, -999.0f);
        SMEKernels1x4Sym::pack_A_streaming(A.data(), packed_A.data(), M_curr, K_curr, 0, 0, K);

        bool pack_A_ok = true;
        for (std::size_t m = 0; m < M_curr && pack_A_ok; m++) {
            std::size_t m_idx      = m / SVL;
            std::size_t lane       = m % SVL;
            std::size_t block_base = m_idx * SVL * K_curr;
            for (std::size_t k = 0; k < K_curr && pack_A_ok; k++) {
                if (std::abs(A[m * K + k] - packed_A[block_base + k * SVL + lane]) > 1e-5f)
                    pack_A_ok = false;
            }
        }
        std::cout << "  pack_A : " << (pack_A_ok ? "PASS" : "FAIL") << "\n";

        const std::size_t GS_B       = 4 * SVL;
        const std::size_t num_groups = (N_curr + GS_B - 1) / GS_B;
        std::vector<float> packed_B(num_groups * K_curr * GS_B, -999.0f);
        SMEKernels1x4Sym::pack_B_streaming(B.data(), packed_B.data(), N_curr, K_curr, 0, N, 0);

        bool pack_B_ok = true;
        for (std::size_t k = 0; k < K_curr && pack_B_ok; k++) {
            for (std::size_t n = 0; n < N_curr && pack_B_ok; n++) {
                std::size_t g      = n / GS_B;
                std::size_t within = n % GS_B;
                if (std::abs(B[k * N + n] - packed_B[g * K_curr * GS_B + k * GS_B + within]) > 1e-5f)
                    pack_B_ok = false;
            }
        }
        std::cout << "  pack_B : " << (pack_B_ok ? "PASS" : "FAIL") << "\n";
    }

    // 1x4-symZAInOut layout: pack_A uses ZA tile 0 as transpose scratch, so the
    // wrapper must own a ZA scope (__arm_new("za")) to legally call into a
    // shared-ZA function. Same pattern as check_pack_4x1ZAPack above.
    __arm_locally_streaming __arm_new("za")
    static void check_pack_1x4_zainout() {
        const std::size_t SVL = static_cast<std::size_t>(svcntw());
        const std::size_t M = 32, K = 35, N = 128;

        std::vector<float> A(M * K), B(K * N);
        for (std::size_t i = 0; i < M; i++)
            for (std::size_t j = 0; j < K; j++)
                A[i * K + j] = static_cast<float>(i * 1000 + j);
        for (std::size_t i = 0; i < K; i++)
            for (std::size_t j = 0; j < N; j++)
                B[i * N + j] = static_cast<float>(i * 1000 + j);

        const std::size_t M_curr = M;
        const std::size_t K_curr = K;
        const std::size_t N_curr = N;

        std::vector<float> packed_A(M_curr * K_curr + SVL, -999.0f);
        SMEKernels1x4SymZAInOut::pack_A_streaming(A.data(), packed_A.data(), M_curr, K_curr, 0, 0, K);

        bool pack_A_ok = true;
        for (std::size_t m = 0; m < M_curr && pack_A_ok; m++) {
            std::size_t m_idx      = m / SVL;
            std::size_t lane       = m % SVL;
            std::size_t block_base = m_idx * SVL * K_curr;
            for (std::size_t k = 0; k < K_curr && pack_A_ok; k++) {
                if (std::abs(A[m * K + k] - packed_A[block_base + k * SVL + lane]) > 1e-5f)
                    pack_A_ok = false;
            }
        }
        std::cout << "  pack_A : " << (pack_A_ok ? "PASS" : "FAIL") << "\n";

        const std::size_t GS_B       = 4 * SVL;
        const std::size_t num_groups = (N_curr + GS_B - 1) / GS_B;
        std::vector<float> packed_B(num_groups * K_curr * GS_B, -999.0f);
        SMEKernels1x4SymZAInOut::pack_B_streaming(B.data(), packed_B.data(), N_curr, K_curr, 0, N, 0);

        bool pack_B_ok = true;
        for (std::size_t k = 0; k < K_curr && pack_B_ok; k++) {
            for (std::size_t n = 0; n < N_curr && pack_B_ok; n++) {
                std::size_t g      = n / GS_B;
                std::size_t within = n % GS_B;
                if (std::abs(B[k * N + n] - packed_B[g * K_curr * GS_B + k * GS_B + within]) > 1e-5f)
                    pack_B_ok = false;
            }
        }
        std::cout << "  pack_B : " << (pack_B_ok ? "PASS" : "FAIL") << "\n";
    }

    // =========================================================================
    // PHASES
    // =========================================================================

    static void phase_pack_correctness(Kernel k) {
        std::cout << "\n--- Phase 1: Pack correctness ---\n";
        switch (k) {
            case Kernel::K4x1:           check_pack_4x1();         break;
            case Kernel::K4x1ZAPack:     check_pack_4x1ZAPack();   break;
            case Kernel::K2x2:           check_pack_2x2();         break;
            case Kernel::K1x4:           check_pack_1x4();         break;
            case Kernel::K1x4Sym:        check_pack_1x4_sym();     break;
            case Kernel::K1x4SymZAInOut: check_pack_1x4_zainout(); break;
        }
    }

    static void phase_gemm_correctness(Kernel k) {
        std::cout << "\n--- Phase 2: GEMM correctness vs scalar ---\n";

        for (auto& tc : kCorrectnessCases) {
            if (!kernel_can_handle(k, tc.M, tc.K, tc.N)) {
                std::cout << "  " << std::left << std::setw(36) << tc.name
                          << " : SKIP (writeback would overflow C)\n";
                continue;
            }

            std::vector<float> A(tc.M * tc.K), B(tc.K * tc.N);
            std::vector<float> C_ref(tc.M * tc.N, 0.0f), C_sme(tc.M * tc.N, 0.0f);

            Utils::fill_random(A);
            Utils::fill_random(B);
            Utils::multiply_scalar(A.data(), B.data(), C_ref.data(), tc.M, tc.N, tc.K);
            run_kernel(k, A.data(), B.data(), C_sme.data(), tc.M, tc.K, tc.N);

            bool ok = Utils::check_correctness(C_ref.data(), C_sme.data(), tc.M * tc.N, kernel_name(k));
            std::cout << "  " << std::left << std::setw(36) << tc.name
                      << " : " << (ok ? "PASS" : "FAIL") << "\n";
        }
    }

    static void phase_benchmark(Kernel k, int iters) {
        std::cout << "\n--- Phase 3: Benchmark (iters=" << iters << ") ---\n";

        for (auto& tc : kBenchCases) {
            std::vector<float> A(tc.M * tc.K), B(tc.K * tc.N), C(tc.M * tc.N, 0.0f);
            Utils::fill_random(A);
            Utils::fill_random(B);

            run_kernel(k, A.data(), B.data(), C.data(), tc.M, tc.K, tc.N); // warmup

            auto t0 = Clock::now();
            for (int i = 0; i < iters; i++)
                run_kernel(k, A.data(), B.data(), C.data(), tc.M, tc.K, tc.N);
            double sec = std::chrono::duration<double>(Clock::now() - t0).count();

            double total_flops = 2.0 * tc.M * tc.N * tc.K * iters;
            double gflops      = (total_flops / 1e9) / sec;
            double ms_per_iter = (sec / iters) * 1000.0;

            std::cout << "  " << std::left << std::setw(8) << tc.name
                      << " : " << std::right << std::fixed << std::setprecision(1)
                      << std::setw(7) << gflops << " GFLOPS"
                      << "   (" << std::setprecision(2) << ms_per_iter << " ms/iter)\n";
        }
    }

    // =========================================================================
    // PUBLIC: per-kernel suite
    // =========================================================================

    void run(Kernel k) {
        std::cout << "\n========================================\n";
        std::cout << "  SME " << kernel_name(k) << " — full suite\n";
        std::cout << "========================================\n";

        phase_pack_correctness(k);
        phase_gemm_correctness(k);
        phase_benchmark(k, kBenchIters);

        std::cout << "========================================\n";
    }

    // =========================================================================
    // PUBLIC: cross-kernel comparison
    // =========================================================================

    // Interleaved comparison: each outer iteration runs every kernel once, so
    // cache state averages out across kernels instead of giving later-in-the-
    // -array kernels a warmer L2/L3. Iteration counts are sized so each kernel
    // gets ~stable wall time; small sizes get more iters to outrun timer noise.
    void run_comparison() {
        const Kernel all[] = {
            Kernel::K4x1, Kernel::K2x2, Kernel::K1x4, Kernel::K1x4Sym,
            Kernel::K1x4SymZAInOut, Kernel::K4x1ZAPack
        };
        constexpr int N_K = sizeof(all) / sizeof(all[0]);

        auto iters_for_size = [](std::size_t n) -> int {
            if (n <=  256) return 400;
            if (n <=  512) return 200;
            if (n <= 1024) return 100;
            if (n <= 2048) return 50;
            return 20; // 4096^3
        };

        std::cout << "\n=====================================================\n";
        std::cout << "  Side-by-side: 4x1 vs 2x2 vs 1x4 vs 1x4-sym vs 1x4ZAIO vs 4x1-ZAPack   (interleaved)\n";
        std::cout << "=====================================================\n";

        std::cout << std::left << std::setw(10) << "  Size"
                  << std::right
                  << std::setw(8)  << "iters"
                  << std::setw(12) << "4x1"
                  << std::setw(12) << "2x2"
                  << std::setw(12) << "1x4"
                  << std::setw(12) << "1x4-sym"
                  << std::setw(12) << "1x4ZAIO"
                  << std::setw(12) << "4x1-ZAP"
                  << std::setw(12) << "Winner" << "\n";
        std::cout << "  " << std::string(102, '-') << "\n";

        for (auto& tc : kComparisonCases) {
            const int iters = iters_for_size(std::max({tc.M, tc.K, tc.N}));

            std::vector<float> A(tc.M * tc.K), B(tc.K * tc.N), C(tc.M * tc.N, 0.0f);
            Utils::fill_random(A);
            Utils::fill_random(B);

            // Warmup: run each kernel once so all four have paid the page-fault
            // / icache-fill / data-touch cost before timing starts.
            for (int i = 0; i < N_K; i++)
                run_kernel(all[i], A.data(), B.data(), C.data(), tc.M, tc.K, tc.N);

            // Interleaved timing: outer loop is iterations, inner loop is
            // kernels. Each kernel sees the same average cache state.
            // C is left to accumulate — every kernel pays the same C r/w cost.
            double sec_total[N_K] = { 0 };
            for (int it = 0; it < iters; it++) {
                for (int i = 0; i < N_K; i++) {
                    auto t0 = Clock::now();
                    run_kernel(all[i], A.data(), B.data(), C.data(), tc.M, tc.K, tc.N);
                    sec_total[i] += std::chrono::duration<double>(Clock::now() - t0).count();
                }
            }

            double gflops[N_K] = { 0 };
            for (int i = 0; i < N_K; i++)
                gflops[i] = (2.0 * tc.M * tc.N * tc.K * iters / 1e9) / sec_total[i];

            int best = 0;
            for (int i = 1; i < N_K; i++) if (gflops[i] > gflops[best]) best = i;

            std::cout << std::left << std::setw(10) << (std::string("  ") + tc.name)
                      << std::right
                      << std::setw(8) << iters
                      << std::fixed << std::setprecision(1);
            for (int i = 0; i < N_K; i++) std::cout << std::setw(12) << gflops[i];
            std::cout << std::setw(12) << kernel_name(all[best]) << "\n";
        }
    }

    // =========================================================================
    // PUBLIC: 4x1 timing breakdown (pack_A vs pack_B vs micro-kernel)
    // =========================================================================

    __arm_locally_streaming __arm_new("za")
    __attribute__((noinline))
    static void timing_inner_4x1(
        const float* A, const float* B, float* C,
        float* packed_A, float* packed_B,
        std::size_t M, std::size_t K, std::size_t N,
        double& t_packA, double& t_packB, double& t_kernel)
    {
        const std::size_t SVL = static_cast<std::size_t>(svcntw());
        constexpr std::size_t M_tile = 64;
        constexpr std::size_t K_tile = 2048;
        constexpr std::size_t N_tile = 1024;
        const std::size_t M_step = 4 * SVL;
        const std::size_t N_step = 1 * SVL;

        for (std::size_t n = 0; n < N; n += N_tile) {
            std::size_t nc = std::min(N_tile, N - n);
            for (std::size_t k = 0; k < K; k += K_tile) {
                std::size_t kc = std::min(K_tile, K - k);

                auto t0 = Clock::now();
                SMEKernels4x1::pack_B_streaming(B, packed_B, nc, kc, k, N, n);
                auto t1 = Clock::now();
                t_packB += std::chrono::duration<double>(t1 - t0).count();

                for (std::size_t m = 0; m < M; m += M_tile) {
                    std::size_t mc = std::min(M_tile, M - m);

                    auto t2 = Clock::now();
                    SMEKernels4x1::pack_A_streaming(A, packed_A, mc, kc, m, k, K);
                    auto t3 = Clock::now();
                    t_packA += std::chrono::duration<double>(t3 - t2).count();

                    auto t4 = Clock::now();
                    for (std::size_t jr = 0; jr < nc; jr += N_step) {
                        for (std::size_t ir = 0; ir < mc; ir += M_step) {
                            SMEKernels4x1::micro_kernel_4x1(
                                packed_A + ir * kc,
                                packed_B + jr * kc,
                                C + (m + ir) * N + (n + jr),
                                kc, N);
                        }
                    }
                    auto t5 = Clock::now();
                    t_kernel += std::chrono::duration<double>(t5 - t4).count();
                }
            }
        }
    }

    void run_timing_breakdown() {
        std::cout << "\n========================================\n";
        std::cout << "  Timing Breakdown: pack_A / pack_B / kernel  (4x1 only)\n";
        std::cout << "========================================\n";

        constexpr std::size_t M_tile = 64;
        constexpr std::size_t K_tile = 2048;
        constexpr std::size_t N_tile = 1024;

        auto* packed_A = static_cast<float*>(std::aligned_alloc(64, M_tile * K_tile * sizeof(float)));
        auto* packed_B = static_cast<float*>(std::aligned_alloc(64, K_tile * N_tile * sizeof(float)));

        std::cout << "  Size      pack_A(ms)  pack_B(ms)  kernel(ms)  total(ms)   pack%\n";
        std::cout << "  ----------------------------------------------------------------\n";

        const int iters = 5;

        for (auto& tc : kBenchCases) {
            std::vector<float> A(tc.M * tc.K), B(tc.K * tc.N), C(tc.M * tc.N, 0.0f);
            Utils::fill_random(A);
            Utils::fill_random(B);

            SMEKernels4x1::run_multiplication(A.data(), B.data(), C.data(), tc.M, tc.K, tc.N); // warmup

            double t_packA = 0, t_packB = 0, t_kernel = 0;
            for (int iter = 0; iter < iters; iter++) {
                std::fill(C.begin(), C.end(), 0.0f);
                timing_inner_4x1(A.data(), B.data(), C.data(),
                                 packed_A, packed_B,
                                 tc.M, tc.K, tc.N,
                                 t_packA, t_packB, t_kernel);
            }

            double total    = t_packA + t_packB + t_kernel;
            double pack_pct = (t_packA + t_packB) / total * 100.0;
            std::cout << std::fixed << std::setprecision(2)
                      << "  " << std::left << std::setw(8) << tc.name
                      << std::right
                      << std::setw(10) << (t_packA  / iters * 1000.0)
                      << std::setw(12) << (t_packB  / iters * 1000.0)
                      << std::setw(12) << (t_kernel / iters * 1000.0)
                      << std::setw(12) << (total    / iters * 1000.0)
                      << std::setw(8)  << pack_pct << "%"
                      << "\n";
        }

        std::free(packed_A);
        std::free(packed_B);
    }

    // =========================================================================
    // PUBLIC: Instruments profiling driver
    // =========================================================================

    void profile(Kernel k, std::size_t M, std::size_t K, std::size_t N, int iters) {
        std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);
        Utils::fill_random(A);
        Utils::fill_random(B);

        run_kernel(k, A.data(), B.data(), C.data(), M, K, N); // warmup

        auto t0 = Clock::now();
        for (int i = 0; i < iters; i++)
            run_kernel(k, A.data(), B.data(), C.data(), M, K, N);
        double sec = std::chrono::duration<double>(Clock::now() - t0).count();

        double gflops      = (2.0 * M * N * K * iters / 1e9) / sec;
        double ms_per_iter = (sec / iters) * 1000.0;
        std::cout << "  [" << kernel_name(k) << "] " << M << "x" << K << "x" << N
                  << " iters=" << iters
                  << " : " << std::fixed << std::setprecision(1) << gflops << " GFLOPS ("
                  << std::setprecision(2) << ms_per_iter << " ms/iter)\n";
    }

} // namespace SMETest
