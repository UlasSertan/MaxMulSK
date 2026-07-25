// bench/bench_profile.cpp
// Single-library, single-size, N-iteration driver for Instruments PMU /
// powermetrics profiling. Mirrors the output format of SMETest::profile()
// in sme/test_sme.cpp so scripts/profile.sh and scripts/profile_power.sh can reuse the
// same parsers (looks for "GFLOPS" / "iters=" / "Threads:" lines).
//
// Usage:
//   bench_profile <accel|oblas> <M> <K> <N> <iters>
//
// Env: forces single-thread for both Accelerate (vecLib) and OpenBLAS.

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>
#include <random>
#include <functional>

#include <Accelerate/Accelerate.h>
#include <dlfcn.h>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static void fill_random(std::vector<float>& v) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& x : v) x = dis(gen);
}

static void accel_sgemm(const float* A, const float* B, float* C,
                        int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

using cblas_sgemm_fn = void (*)(int, int, int, int, int, int,
                                float, const float*, int,
                                const float*, int, float, float*, int);
static cblas_sgemm_fn g_openblas_sgemm = nullptr;

// Try $OPENBLAS_DYLIB first, then the dynamic loader's own search path, then
// the usual Homebrew prefixes (Apple Silicon, then Intel).
static void* dlopen_openblas() {
    const char* env = std::getenv("OPENBLAS_DYLIB");
    const char* candidates[] = {
        env,
        "libopenblas.dylib",
        "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
        "/usr/local/opt/openblas/lib/libopenblas.dylib",
    };
    for (const char* path : candidates) {
        if (!path) continue;
        if (void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL)) return h;
    }
    return nullptr;
}

static bool load_openblas() {
    void* handle = dlopen_openblas();
    if (!handle) {
        std::cerr << "  [error] could not load libopenblas.dylib: " << dlerror() << "\n"
                  << "          install with 'brew install openblas', or set "
                     "OPENBLAS_DYLIB=/path/to/libopenblas.dylib\n";
        return false;
    }
    g_openblas_sgemm = reinterpret_cast<cblas_sgemm_fn>(
        dlsym(handle, "cblas_sgemm"));
    if (!g_openblas_sgemm) {
        std::cerr << "  [error] dlsym cblas_sgemm: " << dlerror() << "\n";
        return false;
    }
    return true;
}

static void openblas_sgemm(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    g_openblas_sgemm(101, 111, 111, M, N, K,
                     1.0f, A, K, B, N, 0.0f, C, N);
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: " << argv[0] << " <accel|oblas> M K N iters\n";
        return 1;
    }
    const std::string lib = argv[1];
    const int M = std::atoi(argv[2]);
    const int K = std::atoi(argv[3]);
    const int N = std::atoi(argv[4]);
    const int iters = std::atoi(argv[5]);

    omp_set_num_threads(1);
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
    setenv("OPENBLAS_NUM_THREADS",   "1", 1);
    std::cout << "  Threads: " << omp_get_max_threads() << "\n";

    std::vector<float> A(static_cast<size_t>(M) * K),
                       B(static_cast<size_t>(K) * N),
                       C(static_cast<size_t>(M) * N);
    fill_random(A); fill_random(B);

    std::function<void()> fn;
    std::string label;
    if (lib == "accel") {
        label = "accel";
        fn = [&]{ accel_sgemm(A.data(), B.data(), C.data(), M, N, K); };
    } else if (lib == "oblas") {
        if (!load_openblas()) return 1;
        label = "oblas";
        fn = [&]{ openblas_sgemm(A.data(), B.data(), C.data(), M, N, K); };
    } else {
        std::cerr << "unknown library: " << lib << "  (expected accel|oblas)\n";
        return 1;
    }

    fn(); // warmup
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) fn();
    double sec = Ms(Clock::now() - t0).count() / 1000.0;

    double gflops      = (2.0 * double(M) * double(N) * double(K) * iters / 1e9) / sec;
    double ms_per_iter = sec * 1000.0 / iters;
    std::cout << "  [" << label << "] " << M << "x" << K << "x" << N
              << " iters=" << iters
              << " : " << std::fixed << std::setprecision(1) << gflops << " GFLOPS ("
              << std::setprecision(2) << ms_per_iter << " ms/iter)\n";
    return 0;
}
