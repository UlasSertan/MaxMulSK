// bench/bench_headline.cpp
// The headline comparison: MaxMulSK vs Apple Accelerate vs OpenBLAS, single
// thread, FP32, end to end. Three end-to-end GEMM implementations measured in
// ONE run, because putting numbers from different runs on one chart is exactly
// the mistake this repo tries not to make.
//
// KleidiAI is deliberately absent. It ships micro-kernels and packing, not a
// blocked end-to-end GEMM; comparing it here would mean lending it our own
// blocking, which makes it a component study rather than a baseline. It belongs
// in the micro-kernel and prepacked experiments (docs/BENCHMARKS.md 0.6-0.9).
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
#include "../sme/v3/sme-1x4-acc-kcout.hpp"
#include "../sme/support/gemm_tuning.hpp"

#include <Accelerate/Accelerate.h>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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

cblas_sgemm_fn g_sgemm = nullptr;   // OpenBLAS, resolved through dlopen
std::string    g_config, g_corename;

// Accelerate is called directly. Its cblas_sgemm is linked; OpenBLAS is opened
// RTLD_LOCAL so its identically-named symbol never enters the global namespace
// and cannot answer this call instead.
void accelerate_gemm(const float* A, const float* B, float* C,
                     size_t M, size_t K, size_t N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K, 1.0f, A, (int)K, B, (int)N, 0.0f, C, (int)N);
}

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

// All implementations for one shape are timed together, and the ORDER ROTATES
// between blocks. Timing them one after another instead would hand a permanent
// advantage to whoever runs later: the first implementation pulls A and B into
// cache and the next one inherits them warm. With three implementations and
// three blocks, rotation puts each of them first exactly once, so that effect
// averages out instead of accumulating on one name.
std::vector<Timing> time_all(const std::vector<std::function<void()>>& runs,
                             uint64_t flops, double budget_s) {
    const size_t n = runs.size();

    // Calibrate on the slowest implementation so every one of them gets at
    // least the intended block length.
    double slowest = 0.0;
    for (const auto& r : runs) {
        auto t0 = Clock::now();
        r();
        slowest = std::fmax(slowest, std::chrono::duration<double>(Clock::now() - t0).count());
    }
    // The upper clamp has to be generous: at 256^3 a single call is ~24 us, so
    // a cap of 400 gave ~10 ms blocks and 35-60% spread between them, while the
    // parameter sweep -- same kernel, same shape, 4000 reps -- saw 0.0%. Too few
    // reps is not a small error here, it is the difference between 1019 and
    // 1415 GFLOP/s for the same code.
    const size_t reps = std::clamp<size_t>((size_t)(budget_s / std::fmax(slowest, 1e-6)), 3, 20000);

    std::vector<std::vector<double>> blocks(n);
    for (size_t b = 0; b < n; b++) {
        for (size_t i = 0; i < n; i++) {
            const size_t idx = (i + b) % n;          // rotate who goes first
            std::vector<double> t;
            t.reserve(reps);
            for (size_t r = 0; r < reps; r++) {
                auto a = Clock::now();
                runs[idx]();
                t.push_back(std::chrono::duration<double, std::milli>(Clock::now() - a).count());
            }
            blocks[idx].push_back(median(t));
        }
    }

    std::vector<Timing> out;
    for (size_t i = 0; i < n; i++) {
        const double m = median(blocks[i]);
        const double lo = *std::min_element(blocks[i].begin(), blocks[i].end());
        const double hi = *std::max_element(blocks[i].begin(), blocks[i].end());
        out.push_back({ (double)flops / (m * 1e6), 100.0 * (hi - lo) / m });
    }
    return out;
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
    std::printf("  MaxMulSK   : SMEKernels1x4AccKcOut (blocking from sme/support/gemm_tuning.hpp)\n");
    std::printf("  Accelerate : linked cblas_sgemm (runs on SME - measured, docs/BENCHMARKS.md 0.17)\n");
    std::printf("  OpenBLAS   : %s\n", g_config.empty() ? "(version string unavailable)" : g_config.c_str());
    std::printf("  OB core    : %s\n", g_corename.empty() ? "(unknown)" : g_corename.c_str());
    std::printf("  Threads    : 1 (openblas_set_num_threads(1), VECLIB_MAXIMUM_THREADS, OPENBLAS_NUM_THREADS)\n");
    std::printf("  Semantics  : all three compute C = A*B (beta = 0)\n\n");

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
            accelerate_gemm(A.data(), B.data(), C.data(), w, w, w);
            openblas_gemm(A.data(), B.data(), C.data(), w, w, w);
        }
    }

    std::printf("%-8s %-16s %10s %7s %10s %7s %10s %7s   %8s %8s\n",
                "group", "M x K x N", "MaxMulSK", "sp", "Accelerate", "sp",
                "OpenBLAS", "sp", "vs Accel", "vs OB");
    std::printf("%s\n", std::string(104, '-').c_str());

    double sink = 0.0;
    for (const Shape& s : kShapes) {
        if (skipped(s.M, s.K, s.N)) {
            char sb[32];
            std::snprintf(sb, sizeof sb, "%zux%zux%zu", s.M, s.K, s.N);
            std::printf("%-8s %-16s %10s %7s %10s %7s %10s %7s   %8s %8s\n",
                        s.group, sb, "-", "-", "-", "-", "-", "-", "skipped", "");
            continue;
        }
        const uint64_t flops = 2ull * s.M * s.N * s.K;

        std::vector<float> A(s.M * s.K), B(s.K * s.N);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        std::vector<float> C_ours(s.M * s.N, kSentinel);
        std::vector<float> C_acc (s.M * s.N, kSentinel);
        std::vector<float> C_ob  (s.M * s.N, kSentinel);

        auto run_ours = [&] {
            SMEKernels1x4AccKcOut::run_multiplication(A.data(), B.data(), C_ours.data(),
                                                      s.M, s.K, s.N);
        };
        auto run_acc = [&] { accelerate_gemm(A.data(), B.data(), C_acc.data(), s.M, s.K, s.N); };
        auto run_ob  = [&] { openblas_gemm (A.data(), B.data(), C_ob.data(),  s.M, s.K, s.N); };

        run_ours();
        run_acc();
        run_ob();

        // Correctness is checked against Accelerate, which is the most mature of
        // the three. C is prefilled with a sentinel because all three OVERWRITE
        // C, so a tile nobody wrote shows up instead of passing silently.
        double max_abs = 0.0, num = 0.0, den = 0.0;
        size_t unwritten = 0;
        for (size_t i = 0; i < C_ours.size(); i++) {
            if (C_ours[i] == kSentinel || C_acc[i] == kSentinel || C_ob[i] == kSentinel) unwritten++;
            const double d = (double)C_ours[i] - (double)C_acc[i];
            max_abs = std::fmax(max_abs, std::fabs(d));
            num += d * d;
            den += (double)C_acc[i] * (double)C_acc[i];
        }
        const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;

        const std::vector<Timing> t = time_all({run_ours, run_acc, run_ob}, flops, 1.0);
        const Timing to = t[0], ta = t[1], tb = t[2];
        sink += (double)C_ours[0] + (double)C_acc[0] + (double)C_ob[s.M * s.N - 1];

        const auto blk = MaxMulSK::tuning::select(s.M, s.K, s.N);
        char blkbuf[48];
        std::snprintf(blkbuf, sizeof blkbuf, "%zu/%zu/%zu/%s",
                      blk.M_tile, blk.N_tile, blk.Kc, blk.ir_outer ? "swap" : "-");
        char shapebuf[32];
        std::snprintf(shapebuf, sizeof shapebuf, "%zux%zux%zu", s.M, s.K, s.N);

        std::printf("%-8s %-16s %10.1f %6.1f%% %10.1f %6.1f%% %10.1f %6.1f%%   %7.2fx %7.2fx%s\n",
                    s.group, shapebuf, to.gflops, to.spread_pct,
                    ta.gflops, ta.spread_pct, tb.gflops, tb.spread_pct,
                    to.gflops / ta.gflops, to.gflops / tb.gflops,
                    unwritten ? "  UNWRITTEN!" : "");

        if (csv) {
            std::fprintf(csv, "%s,%zu,%zu,%zu,MaxMulSK,%.2f,%.2f,%s,%.6e,%.6e\n",
                         s.group, s.M, s.K, s.N, to.gflops, to.spread_pct, blkbuf, max_abs, rel);
            std::fprintf(csv, "%s,%zu,%zu,%zu,Accelerate,%.2f,%.2f,,,\n",
                         s.group, s.M, s.K, s.N, ta.gflops, ta.spread_pct);
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
