// bench/bench_compare.cpp
// Single-threaded GEMM comparison: our NEON vs Apple Accelerate vs OpenBLAS
//
// Build via CMake target `bench_compare`.
// Run via bench/run_bench.sh (sets VECLIB_MAXIMUM_THREADS=1 and
// OPENBLAS_NUM_THREADS=1 for a fair single-thread comparison).

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstdlib>

// Apple Accelerate (AMX-backed on M4) — linked at compile time
#include <Accelerate/Accelerate.h>

// dlopen/dlsym: load OpenBLAS at runtime to avoid cblas_sgemm symbol collision
// with Accelerate (both export the same name).
#include <dlfcn.h>

// Our NEON kernel
#include "../neon/neon-8x12.hpp"

// Our SME Kernel
#include "../sme/sme-4x1.hpp"       // 4x1 version
#include "../sme/sme-2x2.hpp"       // 2x2 version
#include "../sme/sme-4x1-zapack.hpp" // 4x1 with ZA-based pack_A

#include <omp.h>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// =============================================================================
// Helpers
// =============================================================================

static void fill_random(std::vector<float>& v) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& x : v) x = dis(gen);
}

static double gflops(size_t M, size_t N, size_t K, double ms) {
    return (2.0 * double(M) * double(N) * double(K)) / (ms * 1e-3) / 1e9;
}

static float max_diff(const float* a, const float* b, size_t n) {
    float d = 0.0f;
    for (size_t i = 0; i < n; ++i)
        d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

template<typename Fn>
static double bench(Fn&& fn, int iters) {
    fn(); // warmup
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) fn();
    return Ms(Clock::now() - t0).count() / iters;
}

static int iters_for(size_t maxdim) {
    if (maxdim <=   64) return 10000;
    if (maxdim <=  128) return  2000;
    if (maxdim <=  256) return   500;
    if (maxdim <=  512) return    50;
    if (maxdim <= 1024) return    20;
    if (maxdim <= 2048) return     5;
    return 3;
}

// =============================================================================
// BLAS wrappers
// =============================================================================

// Accelerate cblas_sgemm — linked at compile time, resolves to Accelerate
// (AMX-backed on M4).
static void accel_sgemm(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

// OpenBLAS cblas_sgemm — loaded at runtime via dlopen so its symbol doesn't
// collide with Accelerate's cblas_sgemm which is linked at compile time.
using cblas_sgemm_fn = void (*)(int, int, int, int, int, int,
                                float, const float*, int,
                                const float*, int, float, float*, int);
static cblas_sgemm_fn g_openblas_sgemm = nullptr;

static bool load_openblas() {
    void* handle = dlopen("/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
                          RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "  [warn] Could not load OpenBLAS: " << dlerror() << "\n";
        return false;
    }
    g_openblas_sgemm = reinterpret_cast<cblas_sgemm_fn>(
        dlsym(handle, "cblas_sgemm"));
    if (!g_openblas_sgemm) {
        std::cerr << "  [warn] cblas_sgemm not found in OpenBLAS: " << dlerror() << "\n";
        return false;
    }
    return true;
}

// CblasRowMajor=101, CblasNoTrans=111
static void openblas_sgemm(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    g_openblas_sgemm(101, 111, 111, M, N, K,
                     1.0f, A, K, B, N, 0.0f, C, N);
}

// =============================================================================
// Main
// =============================================================================
int main() {
    omp_set_num_threads(1);
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS",   "1", 1);

    const bool have_openblas = load_openblas();

    std::cout << "===========================================================================\n";
    std::cout << "  Single-thread GEMM: NEON (ours) vs Accelerate (AMX) vs OpenBLAS (NEON)\n";
    std::cout << "  float32, row-major C = A*B\n";
    if (!have_openblas)
        std::cout << "  [OpenBLAS unavailable — OBlas column will show 0]\n";
    std::cout << "===========================================================================\n\n";

    struct Case { size_t M, N, K; const char* tag; };
    const Case cases[] = {
        {    8,    8,    8, "tiny"            },
        {   16,   16,   16, "tiny"            },
        {   32,   32,   32, "small"           },
        {   64,   64,   64, "small"           },
        {  128,  128,  128, "L2"              },
        {  256,  256,  256, "L2"              },
        {  512,  512,  512, "L3"              },
        { 1024, 1024, 1024, "L3"              },
        { 2048, 2048, 2048, "mem-bound"       },
        { 4096, 4096, 4096, "mem-bound"       },
        { 4095, 4095, 4095, "mem-bound"       },
        { 1024, 1020, 1024, "N=85x12 aligned" },
        { 2048,   64,  512, "tall-skinny"     },
        {  512,   64, 2048, "wide-flat"       },
        {   65,   65,   65, "all tails +1"    },
        {  513,  509,  513, "all tails mixed" },
        { 1025, 1021, 1025, "all tails large" },
        {  128, 1100,  128, "N > Nc_cache"    },
        {  512, 2048,  512, "N >> Nc_cache"   },
    };

    constexpr int W_SIZE = 22;
    constexpr int W_TAG  = 18;
    constexpr int W_NUM  = 10;

    auto hdr = [&](const char* s) {
        std::cout << std::right << std::setw(W_NUM) << s;
    };

    std::cout << std::left << std::setw(W_SIZE) << "Size (MxKxN)"
              << std::setw(W_TAG) << "Tag";
    hdr("NEON GF"); hdr("SME 4x1"); hdr("4x1 ZP"); hdr("SME 2x2");
    hdr("Accel GF"); hdr("OBlas GF"); hdr("vs Accel"); hdr("vs OBlas");
    std::cout << std::right << std::setw(9) << "MaxDiff\n";
    std::cout << std::string(W_SIZE + W_TAG + W_NUM * 8 + 9, '-') << "\n";

    for (const auto& c : cases) {
        const int iters = iters_for(std::max({c.M, c.N, c.K}));

        std::vector<float> A(c.M * c.K), B(c.K * c.N);
        std::vector<float> C_neon(c.M * c.N), C_accel(c.M * c.N), C_oblas(c.M * c.N);
        fill_random(A);
        fill_random(B);

        // All SME kernels have scratch-buffer edge-tile fallbacks now — no
        // C padding needed (BUG-4x1-SMALL-M / BUG-ZAPACK-WRONG fixed).
        const size_t sme_C_floats = c.M * c.N;

        // SME 4x1 measurement
        std::vector<float> C_sme41(sme_C_floats, 0.0f);
        double ms_sme41 = bench([&]{
            SMEKernels4x1::run_multiplication(A.data(), B.data(), C_sme41.data(), c.M, c.K, c.N);
        }, iters);

        // SME 4x1 ZAPack measurement (pack_A uses ZA horizontal-write / vertical-read transpose)
        std::vector<float> C_sme41zp(sme_C_floats, 0.0f);
        double ms_sme41zp = bench([&]{
            SMEKernels4x1ZAPack::run_multiplication(A.data(), B.data(), C_sme41zp.data(), c.M, c.K, c.N);
        }, iters);

        // SME 2x2 measurement (has scratch-buffer fallback, doesn't need padding,
        // but use the same buffer size for consistency)
        std::vector<float> C_sme22(sme_C_floats, 0.0f);
        double ms_sme22 = bench([&]{
            SMEKernels2x2::run_multiplication(A.data(), B.data(), C_sme22.data(), c.M, c.K, c.N);
        }, iters);

        double gf_sme41   = gflops(c.M, c.N, c.K, ms_sme41);
        double gf_sme41zp = gflops(c.M, c.N, c.K, ms_sme41zp);
        double gf_sme22   = gflops(c.M, c.N, c.K, ms_sme22);
        double ms_neon  = bench([&]{ GEMM::package(A.data(), B.data(), C_neon.data(),  c.M, c.N, c.K); }, iters);
        double ms_accel = bench([&]{ accel_sgemm(A.data(), B.data(), C_accel.data(), (int)c.M, (int)c.N, (int)c.K); }, iters);
        double ms_oblas = have_openblas
            ? bench([&]{ openblas_sgemm(A.data(), B.data(), C_oblas.data(), (int)c.M, (int)c.N, (int)c.K); }, iters)
            : 0.0;

        double gf_neon  = gflops(c.M, c.N, c.K, ms_neon);
        double gf_accel = gflops(c.M, c.N, c.K, ms_accel);
        double gf_oblas = gflops(c.M, c.N, c.K, ms_oblas);

        // Correctness vs Accelerate (ground truth)
        float diff_accel = max_diff(C_neon.data(), C_accel.data(), c.M * c.N);
        float diff_oblas = max_diff(C_neon.data(), C_oblas.data(), c.M * c.N);
        float diff       = std::max(diff_accel, diff_oblas);

        std::string label = std::to_string(c.M) + "x" + std::to_string(c.K) + "x" + std::to_string(c.N);

        std::cout << std::left  << std::fixed
                  << std::setw(W_SIZE) << label
                  << std::setw(W_TAG)  << c.tag
                  << std::right
                  << std::setw(W_NUM) << std::setprecision(1) << gf_neon
                  << std::setw(W_NUM) << std::setprecision(1) << gf_sme41
                  << std::setw(W_NUM) << std::setprecision(1) << gf_sme41zp
                  << std::setw(W_NUM) << std::setprecision(1) << gf_sme22
                  << std::setw(W_NUM) << std::setprecision(1) << gf_accel
                  << std::setw(W_NUM) << std::setprecision(1) << gf_oblas
                  << std::setw(W_NUM) << std::setprecision(3) << (gf_sme22 / gf_accel) // SME 2x2 vs AMX ratio
                  << std::setw(W_NUM) << std::setprecision(3) << (gf_sme22 / gf_oblas) // SME 2x2 vs OpenBLAS
                  << std::setw(9)     << std::setprecision(5) << diff
                  << "\n";
    }

    std::cout << "\n  NEON GF  = our kernel   |  Accel GF = Accelerate cblas (AMX)\n";
    std::cout << "  OBlas GF = OpenBLAS cblas (NEON, no AMX)\n";
    std::cout << "  vs Accel / vs OBlas: ratio > 1.0 means our kernel is faster\n";
    std::cout << "  MaxDiff: max(|NEON-Accel|, |NEON-OBlas|)\n";

    return 0;
}
