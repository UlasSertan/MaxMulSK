// bench/bench_vs_openblas.cpp
// MaxMulSK vs OpenBLAS, single-thread FP32 GEMM, head to head and nothing else.
//
// MaxMulSK is represented by SMEKernels1x4AccKcOut, which is the current
// flagship: ZA-resident K accumulation, overwriting row-major store, ZA-transpose
// pack_A, an outer Kc panel, and the shape-dependent blocking table in
// sme/gemm_tuning.hpp. Whichever blocking the table picks for a shape is what
// gets timed; the table's choice is printed so the number can be reproduced.
//
// OpenBLAS is loaded with dlopen rather than linked, because Accelerate is
// linked into this process and both export cblas_sgemm. Its version and the
// kernel family it selected are read out of the library itself and printed --
// "the version brew says is installed" is not evidence about what actually ran.
//
// Measurement rules, same as the rest of bench/:
//   - FLOP counts in uint64_t, times in double, never signed 32-bit;
//   - GFLOP/s from the MEDIAN, and three separate timing blocks per shape with
//     the spread reported, since between-block drift on this machine is larger
//     than within-block sample noise;
//   - nothing allocated or initialised inside a timed region;
//   - C prefilled with a sentinel: both sides OVERWRITE C (OpenBLAS with
//     beta = 0), so a tile that never gets written shows up instead of passing
//     silently;
//   - a checksum of C is printed so the calls cannot be optimised away.
#include "../sme/sme-1x4-acc-kcout.hpp"
#include "../sme/gemm_tuning.hpp"

#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// CBLAS takes int dimensions; everything else here stays size_t / uint64_t.
using cblas_sgemm_fn = void (*)(int, int, int, int, int, int,
                                float, const float*, int,
                                const float*, int, float, float*, int);
using get_config_fn   = char* (*)();
using get_corename_fn = char* (*)();
using set_threads_fn  = void  (*)(int);

cblas_sgemm_fn g_sgemm = nullptr;
std::string    g_config, g_corename;

bool load_openblas() {
    const char* env = std::getenv("OPENBLAS_DYLIB");
    const char* candidates[] = {
        env,
        "libopenblas.dylib",
        "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
        "/usr/local/opt/openblas/lib/libopenblas.dylib",
    };
    void* h = nullptr;
    for (const char* p : candidates) {
        if (!p) continue;
        if ((h = dlopen(p, RTLD_NOW | RTLD_LOCAL))) break;
    }
    if (!h) {
        std::fprintf(stderr,
            "error: could not load OpenBLAS (%s)\n"
            "       brew install openblas, or set OPENBLAS_DYLIB=/path/to/libopenblas.dylib\n",
            dlerror());
        return false;
    }
    g_sgemm = reinterpret_cast<cblas_sgemm_fn>(dlsym(h, "cblas_sgemm"));
    if (!g_sgemm) {
        std::fprintf(stderr, "error: cblas_sgemm not found in OpenBLAS\n");
        return false;
    }
    // Pin to one thread through the library's own entry point, not just the
    // environment variable: OPENBLAS_NUM_THREADS is read at load time and a
    // caller who forgets it would otherwise get a multi-threaded number here.
    if (auto st = reinterpret_cast<set_threads_fn>(dlsym(h, "openblas_set_num_threads")))
        st(1);
    if (auto gc = reinterpret_cast<get_config_fn>(dlsym(h, "openblas_get_config")))
        if (const char* s = gc()) g_config = s;
    if (auto gn = reinterpret_cast<get_corename_fn>(dlsym(h, "openblas_get_corename")))
        if (const char* s = gn()) g_corename = s;
    return true;
}

// CblasRowMajor = 101, CblasNoTrans = 111. beta = 0 so C is overwritten, which
// matches SMEKernels1x4AccKcOut's C = A*B semantics.
void openblas_gemm(const float* A, const float* B, float* C,
                   size_t M, size_t K, size_t N) {
    g_sgemm(101, 111, 111, (int)M, (int)N, (int)K,
            1.0f, A, (int)K, B, (int)N, 0.0f, C, (int)N);
}

struct Shape { size_t M, K, N; const char* group; };

const Shape kShapes[] = {
    { 256,   256,  256, "square"},
    { 512,   512,  512, "square"},
    {1024,  1024, 1024, "square"},
    {2048,  2048, 2048, "square"},
    {4096,  4096, 4096, "square"},
    {  64,  8192,  512, "llm"},
    {  64, 16384,  512, "llm"},
    {  64, 32768,  512, "llm"},
    { 128,  8192,  512, "llm"},
    { 128, 16384,  512, "llm"},
    { 128, 32768,  512, "llm"},
};

constexpr float kSentinel = 12345.0f;

struct Timing { double gflops, spread_pct; };

template <typename F>
Timing time_it(F&& run, uint64_t flops, double budget_s) {
    auto t0 = Clock::now();
    run();
    const double one = std::chrono::duration<double>(Clock::now() - t0).count();
    // The upper clamp has to be generous: at 256^3 a single call is ~24 us, so
    // a cap of 400 gave ~10 ms blocks and 35-60% spread between them, while the
    // parameter sweep -- same kernel, same shape, 4000 reps -- saw 0.0%. Too few
    // reps is not a small error here, it is the difference between 1019 and
    // 1415 GFLOP/s for the same code.
    const size_t reps = std::clamp<size_t>((size_t)(budget_s / std::fmax(one, 1e-6)), 3, 20000);

    std::vector<double> blocks;
    for (int b = 0; b < 3; b++) {
        std::vector<double> t;
        t.reserve(reps);
        for (size_t r = 0; r < reps; r++) {
            auto a = Clock::now();
            run();
            t.push_back(std::chrono::duration<double, std::milli>(Clock::now() - a).count());
        }
        blocks.push_back(median(t));
    }
    const double m = median(blocks);
    const double lo = *std::min_element(blocks.begin(), blocks.end());
    const double hi = *std::max_element(blocks.begin(), blocks.end());
    return { (double)flops / (m * 1e6), 100.0 * (hi - lo) / m };
}

}  // namespace

// A shape is skipped if it appears in MAXMULSK_SKIP as "MxKxN", comma
// separated. This exists because a BLAS under test can crash: OpenBLAS
// develop's ARMV9SME target dies with SIGILL at 4096^3 on this machine, and
// without a way to step over it the shapes after it never get measured.
bool skipped(size_t M, size_t K, size_t N) {
    const char* list = std::getenv("MAXMULSK_SKIP");
    if (!list || !*list) return false;
    char want[48];
    std::snprintf(want, sizeof want, "%zux%zux%zu", M, K, N);
    return std::strstr(list, want) != nullptr;
}

int main(int argc, char** argv) {
    const char* csv_path = (argc > 1) ? argv[1] : nullptr;

    // Line buffered: if the library under test dies by signal, everything
    // measured up to that point must still reach the terminal.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    if (!load_openblas()) return 1;

    std::printf("=====================================================================\n");
    std::printf("  MaxMulSK vs OpenBLAS - single-thread FP32 GEMM, Apple M4\n");
    std::printf("=====================================================================\n\n");
    std::printf("  MaxMulSK   : SMEKernels1x4AccKcOut (blocking from sme/gemm_tuning.hpp)\n");
    std::printf("  OpenBLAS   : %s\n", g_config.empty() ? "(version string unavailable)" : g_config.c_str());
    std::printf("  OB core    : %s\n", g_corename.empty() ? "(unknown)" : g_corename.c_str());
    std::printf("  Threads    : 1 (openblas_set_num_threads(1), plus OPENBLAS_NUM_THREADS)\n");
    std::printf("  Semantics  : both compute C = A*B (OpenBLAS beta = 0)\n\n");

    std::mt19937 rng(20260909);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::FILE* csv = nullptr;
    if (csv_path) {
        csv = std::fopen(csv_path, "w");
        if (csv) std::fprintf(csv, "group,M,K,N,implementation,gflops,spread_pct,"
                                   "blocking,max_abs_error,rel_frobenius\n");
    }

    // Global warm-up before any timing. Without it the FIRST shape in the list
    // absorbs the process's page faults and the CPU's frequency ramp, and pays
    // for both: on one run 256^3 reported 941 GFLOP/s at 60.8% block spread
    // against 1391 at 0.0% when it was not first. Both implementations are
    // exercised, on a shape that is not in the results.
    {
        const size_t w = 768;
        std::vector<float> A(w * w), B(w * w), C(w * w);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        for (int i = 0; i < 3; i++) {
            SMEKernels1x4AccKcOut::run_multiplication(A.data(), B.data(), C.data(), w, w, w);
            openblas_gemm(A.data(), B.data(), C.data(), w, w, w);
        }
    }

    std::printf("%-8s %-18s %11s %8s   %11s %8s   %9s  %s\n",
                "group", "M x K x N", "MaxMulSK", "spread", "OpenBLAS", "spread",
                "ratio", "max|diff|");
    std::printf("%s\n", std::string(96, '-').c_str());

    double sink = 0.0;
    for (const Shape& s : kShapes) {
        if (skipped(s.M, s.K, s.N)) {
            char sb[32];
            std::snprintf(sb, sizeof sb, "%zux%zux%zu", s.M, s.K, s.N);
            std::printf("%-8s %-18s %11s %8s   %11s %8s   %9s  (MAXMULSK_SKIP)\n",
                        s.group, sb, "-", "-", "-", "-", "-");
            continue;
        }
        const uint64_t flops = 2ull * s.M * s.N * s.K;

        std::vector<float> A(s.M * s.K), B(s.K * s.N);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        std::vector<float> C_ours(s.M * s.N, kSentinel);
        std::vector<float> C_ob  (s.M * s.N, kSentinel);

        auto run_ours = [&] {
            SMEKernels1x4AccKcOut::run_multiplication(A.data(), B.data(), C_ours.data(),
                                                      s.M, s.K, s.N);
        };
        auto run_ob = [&] { openblas_gemm(A.data(), B.data(), C_ob.data(), s.M, s.K, s.N); };

        run_ours();
        run_ob();

        double max_abs = 0.0, num = 0.0, den = 0.0;
        size_t unwritten = 0;
        for (size_t i = 0; i < C_ours.size(); i++) {
            if (C_ours[i] == kSentinel || C_ob[i] == kSentinel) unwritten++;
            const double d = (double)C_ours[i] - (double)C_ob[i];
            max_abs = std::fmax(max_abs, std::fabs(d));
            num += d * d;
            den += (double)C_ob[i] * (double)C_ob[i];
        }
        const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;

        const double budget = 1.0;
        const Timing to = time_it(run_ours, flops, budget);
        const Timing tb = time_it(run_ob,   flops, budget);
        sink += (double)C_ours[0] + (double)C_ob[s.M * s.N - 1];

        const auto blk = MaxMulSK::tuning::select(s.M, s.K, s.N);
        char blkbuf[48];
        std::snprintf(blkbuf, sizeof blkbuf, "%zu/%zu/%zu/%s",
                      blk.M_tile, blk.N_tile, blk.Kc, blk.ir_outer ? "swap" : "-");
        char shapebuf[32];
        std::snprintf(shapebuf, sizeof shapebuf, "%zux%zux%zu", s.M, s.K, s.N);

        std::printf("%-8s %-18s %11.1f %7.1f%%   %11.1f %7.1f%%   %8.2fx  %.2e%s\n",
                    s.group, shapebuf, to.gflops, to.spread_pct,
                    tb.gflops, tb.spread_pct, to.gflops / tb.gflops, max_abs,
                    unwritten ? "  UNWRITTEN!" : "");

        if (csv) {
            std::fprintf(csv, "%s,%zu,%zu,%zu,MaxMulSK,%.2f,%.2f,%s,%.6e,%.6e\n",
                         s.group, s.M, s.K, s.N, to.gflops, to.spread_pct, blkbuf, max_abs, rel);
            std::fprintf(csv, "%s,%zu,%zu,%zu,OpenBLAS,%.2f,%.2f,,,\n",
                         s.group, s.M, s.K, s.N, tb.gflops, tb.spread_pct);
            std::fflush(csv);   // survive a crash in the library under test
        }
    }

    std::printf("\n  ratio > 1 means MaxMulSK is ahead. GFLOP/s from medians of three\n");
    std::printf("  timing blocks; spread is (max-min)/median across those blocks.\n");
    std::printf("  This machine has no thread pinning: treat differences under ~5%% as noise.\n");
    std::printf("\n  checksum %.6e\n", sink);
    if (csv) { std::fclose(csv); std::printf("  -> %s\n", csv_path); }
    return 0;
}
