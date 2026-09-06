// bench/energy_bench.cpp
// One implementation, one shape, run in a tight loop for a fixed wall time.
//
// This binary does not measure power itself. bench/energy_bench.sh wraps each
// invocation in its own powermetrics session, so every sample that session
// collects belongs to this workload and no timestamp alignment is needed. All
// this has to do is keep the machine busy with exactly one implementation for a
// known number of FLOPs, then report that count.
//
//   energy_bench <impl> <M> <K> <N> <seconds>
//
// Prints one machine-readable line:
//   RESULT impl=<..> M=.. K=.. N=.. iters=.. wall_s=.. gflop=.. gflops=..
//
// Setup, allocation, packing-buffer warmup and correctness are all outside the
// measured window; the window is the same end-to-end GEMM the benchmarks time
// (unpacked A and B in, finished C out, packing included).

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <functional>
#include <string>
#include <vector>

#include <omp.h>

#include "kleidiai_gemm.hpp"
#include "maxmulsk_sme_adapter.hpp"

using Clock = std::chrono::steady_clock;

namespace {

void fill_random(std::vector<float>& v, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& x : v) x = dis(gen);
}

// Keeps the optimiser honest and gives a cheap sanity signal per run.
volatile double g_sink = 0.0;
void consume(const std::vector<float>& C) {
    double acc = 0.0;
    for (size_t i = 0; i < C.size(); i += 4096) acc += double(C[i]);
    g_sink = g_sink + acc;
}

struct MMEntry { const char* name; MaxMulSK::Kernel k; };
const MMEntry kMM[] = {
    {"sme-4x1",     MaxMulSK::Kernel::Sme4x1},
    {"sme-1x4sym",  MaxMulSK::Kernel::Sme1x4Sym},
    {"sme-1x4acc",  MaxMulSK::Kernel::Sme1x4Acc},
    {"sme-1x4acckc",MaxMulSK::Kernel::Sme1x4AccKc},
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
            "usage: %s <impl> <M> <K> <N> <seconds>\n"
            "  impl: sme-4x1 | sme-1x4sym | sme-1x4acc | sme-1x4acckc |\n"
            "        kleidiai-fullpack | kleidiai-panel | accelerate\n", argv[0]);
        return 2;
    }
    const std::string impl = argv[1];
    const size_t M = std::strtoull(argv[2], nullptr, 10);
    const size_t K = std::strtoull(argv[3], nullptr, 10);
    const size_t N = std::strtoull(argv[4], nullptr, 10);
    const double seconds = std::strtod(argv[5], nullptr);

    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
    omp_set_num_threads(1);

    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);
    fill_random(A, 42);
    fill_random(B, 43);
    const size_t c_bytes = M * N * sizeof(float);

    // Pick the work function. Everything it needs is built before timing.
    std::function<void()> run;
    KleidiAI::Gemm kai(KleidiAI::Kernel::Sme2Mopa2VL);
    KleidiAI::PanelConfig panel{};

    bool matched = false;
    for (const auto& e : kMM) {
        if (impl != e.name) continue;
        matched = true;
        const bool needs_zero = !MaxMulSK::overwrites_C(e.k);
        const MaxMulSK::Kernel k = e.k;
        run = [&, k, needs_zero] {
            if (needs_zero) std::memset(C.data(), 0, c_bytes);
            MaxMulSK::run(k, A.data(), B.data(), C.data(), M, K, N);
        };
        break;
    }
    if (!matched && impl == "kleidiai-fullpack") {
        matched = true;
        kai.reshape(M, N, K);
        run = [&] { kai.run_end_to_end(A.data(), B.data(), C.data()); };
    }
    if (!matched && impl == "kleidiai-panel") {
        matched = true;
        kai.reshape(M, N, K);
        // Small untimed sweep, same spirit as the main benchmark: the blocking
        // is the caller's choice, so measuring a badly-blocked KleidiAI would
        // report our configuration as its energy.
        double best_ms = 0.0;
        for (size_t mc : {size_t(64), size_t(256), size_t(0)})
            for (size_t kc : {size_t(2048), size_t(0)})
                for (size_t nc : {size_t(512), size_t(0)}) {
                    if (nc && N / nc > 8) continue;
                    const KleidiAI::PanelConfig cfg{mc, kc, nc};
                    kai.run_panel_blocked(A.data(), B.data(), C.data(), cfg);
                    const auto t0 = Clock::now();
                    kai.run_panel_blocked(A.data(), B.data(), C.data(), cfg);
                    const double ms =
                        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                    if (best_ms == 0.0 || ms < best_ms) { best_ms = ms; panel = cfg; }
                }
        run = [&] { kai.run_panel_blocked(A.data(), B.data(), C.data(), panel); };
    }
    if (!matched && impl == "accelerate") {
        matched = true;
        run = [&] {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)M, (int)N, (int)K, 1.0f, A.data(), (int)K,
                        B.data(), (int)N, 0.0f, C.data(), (int)N);
        };
    }
    if (!matched) { std::fprintf(stderr, "unknown impl: %s\n", impl.c_str()); return 2; }

    for (int i = 0; i < 3; ++i) run();   // warm caches and first-touch pages

    // ---- measured window ----
    size_t iters = 0;
    const auto t0 = Clock::now();
    double elapsed = 0.0;
    do {
        run();
        ++iters;
        elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
    } while (elapsed < seconds);
    // ---- end ----

    consume(C);
    const double flop = 2.0 * double(M) * double(N) * double(K) * double(iters);
    std::printf("RESULT impl=%s M=%zu K=%zu N=%zu iters=%zu wall_s=%.4f gflop=%.4f gflops=%.2f\n",
                impl.c_str(), M, K, N, iters, elapsed, flop / 1e9,
                flop / elapsed / 1e9);
    if (g_sink == 12345.6789) std::printf("");
    return 0;
}
