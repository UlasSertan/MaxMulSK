// bench/benchmark_maxmul_vs_kleidiai.cpp
//
// Fair single-thread FP32 GEMM comparison: MaxMulSK's SME kernels vs Arm
// KleidiAI's FP32 SME2 micro-kernels, on Apple M4.
//
// Two modes, kept strictly separate:
//
//   end_to_end   from unpacked row-major A and B to finished C, including all
//                packing and the packed-buffer allocations each implementation
//                needs. This is "normal library usage".
//
//   kernel_only  all packing done before the timed region; the timed region
//                contains only the reusable compute over pre-packed operands.
//
// Everything else is held identical: same A/B (fixed seed), same row-major
// layout, same GEMM semantics (C = A*B, alpha=1, beta=0), same warmups, same
// adaptive repetition policy, same timer, same thread count.
//
// Build:  cmake --build <dir> --target benchmark_maxmul_vs_kleidiai
// Run:    ./benchmark_maxmul_vs_kleidiai [output.csv]

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <pthread.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <omp.h>

#include "kleidiai_gemm.hpp"
#include "ggml_gemm_baseline.hpp"
#include "maxmulsk_sme_adapter.hpp"

#ifndef BENCH_CXX_FLAGS
#define BENCH_CXX_FLAGS "(unknown)"
#endif
#ifndef BENCH_KLEIDIAI_VERSION
#define BENCH_KLEIDIAI_VERSION "(unknown)"
#endif
#ifndef BENCH_GIT_REV
#define BENCH_GIT_REV "(unknown)"
#endif
#ifndef BENCH_LLAMA_REV
#define BENCH_LLAMA_REV "(unknown)"
#endif
#ifndef BENCH_GGML_KAI_VER
#define BENCH_GGML_KAI_VER "(unknown)"
#endif

using Clock = std::chrono::steady_clock;  // monotonic; mach_absolute_time on macOS
static_assert(Clock::is_steady, "need a monotonic clock");

// Consumed after every timed region so nothing can be optimised away.
static volatile double g_sink = 0.0;

// =============================================================================
// Shapes
// =============================================================================

struct Shape {
    size_t M, K, N;
    const char* category;  // "square" | "large_k"
};

static const Shape kShapes[] = {
    {  256,   256,  256, "square"  },
    {  512,   512,  512, "square"  },
    { 1024,  1024, 1024, "square"  },
    { 2048,  2048, 2048, "square"  },
    { 4096,  4096, 4096, "square"  },

    {   64,  8192,  512, "large_k" },
    {   64, 16384,  512, "large_k" },
    {   64, 32768,  512, "large_k" },
    {  128,  8192,  512, "large_k" },
    {  128, 16384,  512, "large_k" },
    {  128, 32768,  512, "large_k" },
};

// KleidiAI FP32 SME/SME2 GEMM paths. The NEON kernel in kleidiai_gemm.hpp is
// deliberately excluded: the comparison asked for is the SME/SME2 path.
static const KleidiAI::Kernel kKaiKernels[] = {
    KleidiAI::Kernel::Sme2Mopa2VL,
    KleidiAI::Kernel::Sme2Mopa8VS,
};

// =============================================================================
// Timing
// =============================================================================

struct Stats {
    double min_ms = 0, mean_ms = 0, median_ms = 0, stddev_ms = 0;
    size_t reps = 0;
};

static Stats summarize(std::vector<double> ms) {
    Stats s;
    s.reps = ms.size();
    if (ms.empty()) return s;
    std::sort(ms.begin(), ms.end());
    s.min_ms = ms.front();
    s.median_ms = (ms.size() % 2) ? ms[ms.size() / 2]
                                  : 0.5 * (ms[ms.size() / 2 - 1] + ms[ms.size() / 2]);
    s.mean_ms = std::accumulate(ms.begin(), ms.end(), 0.0) / double(ms.size());
    double acc = 0.0;
    for (double v : ms) acc += (v - s.mean_ms) * (v - s.mean_ms);
    s.stddev_ms = (ms.size() > 1) ? std::sqrt(acc / double(ms.size() - 1)) : 0.0;
    return s;
}

// Three warmups (the last one timed, only to size the sample count), then a
// batch of timed samples. The first, cold invocation is never a reported value.
template <class F>
static Stats measure(F&& f, double target_s = 0.35,
                     size_t min_reps = 7, size_t max_reps = 151) {
    f();
    f();
    const auto w0 = Clock::now();
    f();
    const auto w1 = Clock::now();

    const double est_s = std::chrono::duration<double>(w1 - w0).count();
    size_t reps = min_reps;
    if (est_s > 0.0) reps = static_cast<size_t>(std::ceil(target_s / est_s));
    reps = std::clamp(reps, min_reps, max_reps);

    std::vector<double> ms;
    ms.reserve(reps);
    for (size_t i = 0; i < reps; ++i) {
        const auto t0 = Clock::now();
        f();
        const auto t1 = Clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return summarize(std::move(ms));
}

// 64-bit throughout: 2*M*N*K exceeds int32 for several of these shapes.
static double gflops_of(size_t M, size_t N, size_t K, double median_ms) {
    if (median_ms <= 0.0) return 0.0;
    const double flops = 2.0 * static_cast<double>(M)
                             * static_cast<double>(N)
                             * static_cast<double>(K);
    return flops / (median_ms * 1e-3) / 1e9;
}

// =============================================================================
// Correctness
// =============================================================================

struct Errors {
    double max_abs = 0.0;   // vs the FP32 reference SGEMM
    double rel_l2  = 0.0;   // ||X - ref||_F / ||ref||_F
    double max_abs_fp64 = 0.0;  // vs FP64 accumulation, on sampled entries
    bool   ok = false;
};

// Loose enough for FP32 accumulation-order differences at K = 32768, tight
// enough that a real indexing or blocking bug cannot hide under it.
static constexpr double kRelL2Tolerance = 1e-4;

struct Fp64Sample { size_t i, j; double value; };

static std::vector<Fp64Sample> fp64_samples(const std::vector<float>& A,
                                            const std::vector<float>& B,
                                            size_t M, size_t K, size_t N,
                                            size_t count) {
    std::vector<Fp64Sample> out;
    std::mt19937_64 rng(1234567);
    for (size_t s = 0; s < count; ++s) {
        const size_t i = rng() % M;
        const size_t j = rng() % N;
        double acc = 0.0;  // FP64 accumulation of an FP32 product
        for (size_t k = 0; k < K; ++k)
            acc += double(A[i * K + k]) * double(B[k * N + j]);
        out.push_back({i, j, acc});
    }
    return out;
}

static Errors check(const std::vector<float>& X, const std::vector<float>& ref,
                    size_t M, size_t N,
                    const std::vector<Fp64Sample>& samples) {
    Errors e;
    double num = 0.0, den = 0.0;
    for (size_t idx = 0; idx < M * N; ++idx) {
        const double d = double(X[idx]) - double(ref[idx]);
        e.max_abs = std::max(e.max_abs, std::abs(d));
        num += d * d;
        den += double(ref[idx]) * double(ref[idx]);
    }
    e.rel_l2 = (den > 0.0) ? std::sqrt(num / den) : 0.0;
    for (const auto& s : samples)
        e.max_abs_fp64 = std::max(e.max_abs_fp64,
                                  std::abs(double(X[s.i * N + s.j]) - s.value));
    e.ok = std::isfinite(e.rel_l2) && e.rel_l2 <= kRelL2Tolerance;
    return e;
}

// =============================================================================
// Results
// =============================================================================

struct Row {
    std::string experiment;   // hot_microkernel | prepacked_full_gemm |
                              // end_to_end | panel_blocked | ggml_cpu
    std::string category, mode, impl;
    std::string variant;      // geometry / config detail; may be empty
    size_t M = 0, N = 0, K = 0;
    size_t Kc = 0, reps_inner = 0;   // hot_microkernel only
    Stats t;
    double gflops = 0.0;
    double ns_per_invocation = 0.0;  // hot_microkernel only
    double packing_ms = -1.0;        // -1 = N/A
    double compute_ms = -1.0;
    Errors e;
    bool is_maxmulsk = false, is_kleidiai = false;
};

static std::vector<Row> g_rows;

static const Row* best_of(const std::string& category, const std::string& mode,
                          size_t M, size_t N, size_t K, bool maxmulsk) {
    const Row* best = nullptr;
    for (const auto& r : g_rows) {
        if (r.category != category || r.mode != mode) continue;
        if (r.M != M || r.N != N || r.K != K) continue;
        if (maxmulsk ? !r.is_maxmulsk : !r.is_kleidiai) continue;
        if (!r.e.ok) continue;
        if (!best || r.gflops > best->gflops) best = &r;
    }
    return best;
}

static const Row* find_row(const std::string& mode, size_t M, size_t N, size_t K,
                           const std::string& impl) {
    for (const auto& r : g_rows)
        if (r.mode == mode && r.M == M && r.N == N && r.K == K && r.impl == impl)
            return &r;
    return nullptr;
}

// =============================================================================
// Environment
// =============================================================================

static std::string sysctl_str(const char* key) {
    size_t len = 0;
    if (sysctlbyname(key, nullptr, &len, nullptr, 0) != 0 || len == 0) return "(n/a)";
    std::string v(len, '\0');
    if (sysctlbyname(key, v.data(), &len, nullptr, 0) != 0) return "(n/a)";
    if (!v.empty() && v.back() == '\0') v.pop_back();
    return v;
}

static uint64_t sysctl_u64(const char* key) {
    uint64_t v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(key, &v, &len, nullptr, 0) != 0) return 0;
    return v;
}

static void print_environment() {
    std::printf("=====================================================================\n");
    std::printf("  MaxMulSK vs Arm KleidiAI — single-thread FP32 GEMM, Apple M4\n");
    std::printf("=====================================================================\n\n");
    std::printf("  Hardware\n");
    std::printf("    CPU                 : %s\n", sysctl_str("machdep.cpu.brand_string").c_str());
    std::printf("    P-cores / E-cores   : %llu / %llu\n",
                (unsigned long long)sysctl_u64("hw.perflevel0.physicalcpu"),
                (unsigned long long)sysctl_u64("hw.perflevel1.physicalcpu"));
    std::printf("    P-core L1d / L2     : %llu KiB / %llu MiB\n",
                (unsigned long long)(sysctl_u64("hw.perflevel0.l1dcachesize") / 1024),
                (unsigned long long)(sysctl_u64("hw.perflevel0.l2cachesize") / (1024 * 1024)));
    std::printf("    RAM                 : %.1f GiB\n",
                double(sysctl_u64("hw.memsize")) / (1024.0 * 1024.0 * 1024.0));
    std::printf("    SME / SME2          : %s / %s\n",
                sysctl_u64("hw.optional.arm.FEAT_SME") ? "yes" : "no",
                sysctl_u64("hw.optional.arm.FEAT_SME2") ? "yes" : "no");
    std::printf("    Streaming VL        : %zu x 32-bit words (%zu bits)\n",
                MaxMulSK::svl_words(), MaxMulSK::svl_words() * 32);
    std::printf("    OS                  : %s %s\n",
                sysctl_str("kern.ostype").c_str(), sysctl_str("kern.osrelease").c_str());

    std::printf("\n  Build\n");
    std::printf("    Compiler            : %s\n", __VERSION__);
    std::printf("    Benchmark flags     : %s\n", BENCH_CXX_FLAGS);
    std::printf("    SME sources built at: -march=armv8.7-a+sme+sme2 (MaxMulSK kernels + adapter)\n");
    std::printf("    KleidiAI            : %s (built from source; SME asm at\n"
                "                          -march=armv8.2-a+sve+sve2, KleidiAI's own flags)\n",
                BENCH_KLEIDIAI_VERSION);
    std::printf("    MaxMulSK revision   : %s\n", BENCH_GIT_REV);
    std::printf("    KleidiAI present    : %s\n", KleidiAI::available() ? "yes" : "NO");
    std::printf("    llama.cpp / ggml    : %s\n", BENCH_LLAMA_REV);
    std::printf("    ggml build          : %s\n", GgmlCpu::build_description().c_str());
    std::printf("    ggml FP32 mul_mat   : ggml_compute_forward_mul_mat -> generic chunked\n");
    std::printf("                          path -> ggml_compute_forward_mul_mat_one_chunk ->\n");
    std::printf("                          ggml_vec_dot_f32 (one NEON dot product per output\n");
    std::printf("                          element). NOT SME, NOT an external library.\n");
    std::printf("                          llamafile/tinyBLAS is NOT used, despite\n");
    std::printf("                          GGML_LLAMAFILE=ON and sgemm.cpp being linked:\n");
    std::printf("                          ggml-cpu.c does\n");
    std::printf("                            #if defined(__ARM_FEATURE_SVE) ||\n");
    std::printf("                                defined(__ARM_FEATURE_MATMUL_INT8)\n");
    std::printf("                            #undef GGML_USE_LLAMAFILE\n");
    std::printf("                          and ggml builds its CPU variant with +i8mm, which\n");
    std::printf("                          defines __ARM_FEATURE_MATMUL_INT8. Verified two\n");
    std::printf("                          ways: an instrumented build shows the #if block is\n");
    std::printf("                          compiled out, and a GGML_LLAMAFILE=OFF build gives\n");
    std::printf("                          identical timings (42.6 vs 42.8 GFLOP/s at 1024^3).\n");
    std::printf("                          This is upstream behaviour on Apple Silicon, not a\n");
    std::printf("                          misconfiguration of this benchmark.\n");

    std::printf("\n  Threading\n");
    std::printf("    OpenMP threads      : %d (MaxMulSK SME drivers are single-thread anyway)\n",
                omp_get_max_threads());
    std::printf("    VECLIB_MAXIMUM_THREADS = 1 (reference SGEMM only)\n");
    std::printf("    ggml KleidiAI build : %s\n",
                GgmlCpu::kleidiai_build() ? "GGML_CPU_KLEIDIAI=ON" : "OFF (llama-kleidiai row absent)");
    std::printf("    ggml KleidiAI ver   : %s\n", BENCH_GGML_KAI_VER);
    std::printf("    KleidiAI dispatch   : verified, not inferred --\n");
    std::printf("                          (a) static: kai_run_matmul_clamp_f32_f32p2vlx1_\n");
    std::printf("                              f32p2vlx1biasf32_sme2_mopa is defined in\n");
    std::printf("                              libkleidiai.a and referenced by libggml-cpu.a\n");
    std::printf("                          (b) runtime: src0 allocated in the CPU_KLEIDIAI\n");
    std::printf("                              extra buffer type (required by supports_op)\n");
    std::printf("                          (c) an instrumented build resolved the executed\n");
    std::printf("                              run_kernel_ex via dladdr to\n");
    std::printf("                              kernel_run_fn10<kai_run_matmul_clamp_f32_\n");
    std::printf("                              f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>, with\n");
    std::printf("                              m_step=n_step=mr=nr=32, kr=sr=1. That build is\n");
    std::printf("                              separate; no such logging exists here.\n");
    std::printf("    ggml n_threads      : 1 (set via ggml_backend_cpu_set_n_threads;\n");
    std::printf("                          verified below by CPU-time/wall-time ratio)\n");
    std::printf("    QoS                 : USER_INTERACTIVE requested (biases to P-cores).\n");
    std::printf("                          macOS exposes no thread-to-core pinning API, so\n"
                "                          core residency cannot be pinned, only biased.\n");

    std::printf("\n  Methodology\n");
    std::printf("    Timer               : std::chrono::steady_clock (monotonic)\n");
    std::printf("    Warmups             : 3 per measurement, none reported\n");
    std::printf("    Samples             : adaptive, >= 7, targeting ~0.35 s of timed work\n");
    std::printf("    Headline metric     : GFLOP/s from the MEDIAN time; FLOPs = 2*M*N*K in double\n");
    std::printf("    Inputs              : U(-1,1), mt19937 seed 42, identical for every impl\n");
    std::printf("    Reference           : Accelerate cblas_sgemm, plus FP64 dot products on\n"
                "                          32 sampled output entries per shape\n");
    std::printf("    Tolerance           : relative L2 <= %.0e\n", kRelL2Tolerance);
    std::printf("\n");
}

// =============================================================================
// KleidiAI N-block autotuning
//
// KleidiAI ships no cache blocking above its micro-kernel; the calling runtime
// provides it, so the block width is a property of the caller, not the library.
// Leaving it at one fixed heuristic would measure my choice of block size as if
// it were KleidiAI's performance. MaxMulSK is reported as the best of its six
// kernel variants, so the symmetric treatment is to report KleidiAI at its best
// block width. This picks it per (shape, kernel) before any timed region.
// =============================================================================

static size_t autotune_n_block(KleidiAI::Gemm& g, const float* A, const float* B,
                               float* C, size_t M, size_t N, size_t K) {
    static const size_t kCandidates[] = {32, 64, 128, 256, 512, 1024, 2048};
    size_t best_nb = 0;
    double best_ms = 0.0;

    auto try_nb = [&](size_t nb) {
        g.set_n_block(nb);
        const size_t actual = g.n_block();
        g.pack(A, B);
        g.matmul(A, C);                       // warmup
        std::vector<double> ms;
        for (int i = 0; i < 5; ++i) {
            const auto t0 = Clock::now();
            g.matmul(A, C);
            const auto t1 = Clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        if (best_nb == 0 || med < best_ms) { best_ms = med; best_nb = actual; }
    };

    g.set_n_block(0);                          // the built-in heuristic
    try_nb(0);
    for (size_t nb : kCandidates)
        if (nb <= N) try_nb(nb);
    try_nb(N);                                 // one monolithic call

    g.set_n_block(best_nb);
    (void)M; (void)K;
    return best_nb;
}

// =============================================================================
// Benchmark
// =============================================================================

// Chosen N-block width per (kernel, shape), reported alongside the results.
static std::map<std::string, size_t> g_kai_nblock;
static std::map<std::string, KleidiAI::PanelBreakdown> g_panel_breakdown;
static std::map<std::string, double> g_ggml_transpose_ms;
// Measured, not assumed. Filled once, from a large shape.
static double g_ggml_cpu_wall_ratio = -1.0;
static double g_ggml_kai_wall_ratio = -1.0;
// One-off src0 repack into the KleidiAI packed layout: llama.cpp pays this at
// model load, not per token, so it is reported rather than timed per call.
static std::map<std::string, double> g_kai_weight_pack_ms;

static std::string shape_key(size_t M, size_t N, size_t K) {
    return std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K);
}

static void fill_random(std::vector<float>& v, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& x : v) x = dis(gen);
}

static void consume(const std::vector<float>& C) {
    double acc = 0.0;
    for (size_t i = 0; i < C.size(); i += 4096) acc += double(C[i]);
    g_sink = g_sink + acc;
}

static void run_shape(const Shape& sh) {
    const size_t M = sh.M, K = sh.K, N = sh.N;
    std::printf("  [%s] %zu x %zu x %zu (MxKxN) ...", sh.category, M, K, N);
    std::fflush(stdout);

    std::vector<float> A(M * K), B(K * N), C(M * N), C_ref(M * N);
    fill_random(A, 42);
    fill_random(B, 43);

    // Reference (FP32) and FP64 spot checks. Neither is inside a timed region.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K, 1.0f, A.data(), (int)K,
                B.data(), (int)N, 0.0f, C_ref.data(), (int)N);
    const auto samples = fp64_samples(A, B, M, K, N, 32);

    const size_t c_bytes = M * N * sizeof(float);
    auto zero_C = [&] { std::memset(C.data(), 0, c_bytes); };

    auto add_row_ex = [&](const char* experiment, const char* mode,
                          const std::string& impl, const std::string& variant,
                          const Stats& t, const Errors& e, bool mm, bool kai) -> size_t {
        Row r;
        r.experiment = experiment;
        r.category = sh.category; r.mode = mode; r.impl = impl; r.variant = variant;
        r.M = M; r.N = N; r.K = K;
        r.t = t; r.gflops = gflops_of(M, N, K, t.median_ms);
        r.e = e; r.is_maxmulsk = mm; r.is_kleidiai = kai;
        g_rows.push_back(std::move(r));
        return g_rows.size() - 1;
    };
    auto add_row = [&](const char* mode, const std::string& impl, const Stats& t,
                       const Errors& e, bool mm, bool kai) {
        const char* exp = (std::string(mode) == "kernel_only") ? "prepacked_full_gemm"
                        : (std::string(mode) == "zero_only")   ? "zero_only"
                                                               : "end_to_end";
        add_row_ex(exp, mode, impl, "", t, e, mm, kai);
    };

    // --- diagnostic: the memset that MaxMulSK's C += A*B semantics require ---
    {
        const Stats t = measure(zero_C, 0.1, 7, 51);
        Errors e; e.ok = true;
        add_row("zero_only", "memset C", t, e, false, false);
    }

    // ------------------------------------------------------------------
    // end_to_end
    // ------------------------------------------------------------------
    for (MaxMulSK::Kernel k : MaxMulSK::kAllKernels) {
        // Time each kernel for the work it actually needs. The accumulating
        // kernels require C zeroed first to produce C = A*B, and that memset is
        // a real cost of using them, so it is inside the timed region. The Acc
        // variants overwrite C and need no memset, so they are not charged one.
        const bool needs_zero = !MaxMulSK::overwrites_C(k);

        // Correctness: prefill C with a sentinel rather than zero. For an
        // accumulating kernel that is then zeroed anyway; for an overwriting one
        // it proves the claim, since any forgotten accumulate would leave the
        // sentinel behind and fail the check.
        std::fill(C.begin(), C.end(), 12345.0f);
        if (needs_zero) zero_C();
        MaxMulSK::run(k, A.data(), B.data(), C.data(), M, K, N);
        const Errors e = check(C, C_ref, M, N, samples);

        const Stats t = measure([&] {
            if (needs_zero) zero_C();
            MaxMulSK::run(k, A.data(), B.data(), C.data(), M, K, N);
        });
        consume(C);
        add_row_ex("end_to_end", "end_to_end", MaxMulSK::label(k),
                   needs_zero ? "C+=A*B, memset timed" : "C=A*B, no memset",
                   t, e, true, false);
    }

    if (KleidiAI::available()) {
        for (KleidiAI::Kernel k : kKaiKernels) {
            KleidiAI::Gemm g(k);
            g.reshape(M, N, K);
            g_kai_nblock[std::string(KleidiAI::label(k)) + "@" + shape_key(M, N, K)] =
                autotune_n_block(g, A.data(), B.data(), C.data(), M, N, K);
            g.run_end_to_end(A.data(), B.data(), C.data());
            const Errors e = check(C, C_ref, M, N, samples);
            const Stats t = measure([&] { g.run_end_to_end(A.data(), B.data(), C.data()); });
            consume(C);
            add_row("end_to_end", std::string("KAI ") + KleidiAI::label(k), t, e, false, true);
        }
    }

    {  // third-party baseline, not part of the required comparison
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)M, (int)N, (int)K, 1.0f, A.data(), (int)K,
                    B.data(), (int)N, 0.0f, C.data(), (int)N);
        const Errors e = check(C, C_ref, M, N, samples);
        const Stats t = measure([&] {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)M, (int)N, (int)K, 1.0f, A.data(), (int)K,
                        B.data(), (int)N, 0.0f, C.data(), (int)N);
        });
        consume(C);
        add_row("end_to_end", "Accelerate", t, e, false, false);
    }

    // ------------------------------------------------------------------
    // kernel_only
    // ------------------------------------------------------------------
    for (MaxMulSK::Kernel k : MaxMulSK::kAllKernels) {
        if (!MaxMulSK::has_kernel_only(k)) continue;
        MaxMulSK::Prepacked pp;
        pp.prepare(k, A.data(), B.data(), M, K, N);  // untimed
        const bool needs_zero = !MaxMulSK::overwrites_C(k);

        std::fill(C.begin(), C.end(), 12345.0f);
        if (needs_zero) zero_C();
        pp.compute(C.data());
        const Errors e = check(C, C_ref, M, N, samples);

        const Stats t = measure([&] {
            if (needs_zero) zero_C();
            pp.compute(C.data());
        });
        consume(C);
        add_row_ex("prepacked_full_gemm", "kernel_only", MaxMulSK::label(k),
                   needs_zero ? "C+=A*B, memset timed" : "C=A*B, no memset",
                   t, e, true, false);
    }

    if (KleidiAI::available()) {
        for (KleidiAI::Kernel k : kKaiKernels) {
            KleidiAI::Gemm g(k);
            g.reshape(M, N, K);
            g.set_n_block(g_kai_nblock[std::string(KleidiAI::label(k)) + "@" +
                                       shape_key(M, N, K)]);
            g.pack(A.data(), B.data());  // untimed
            g.matmul(A.data(), C.data());
            const Errors e = check(C, C_ref, M, N, samples);
            const Stats t = measure([&] { g.matmul(A.data(), C.data()); });
            consume(C);
            add_row("kernel_only", std::string("KAI ") + KleidiAI::label(k), t, e, false, true);
        }
    }

    // ------------------------------------------------------------------
    // Experiment 2: KleidiAI's own kernel + packing, driven by a
    // MaxMulSK-style panel-blocked caller. Mc/Kc/Nc autotuned per shape for
    // the same reason the N-block width is: the blocking is the caller's
    // choice, and a bad one would be reported as KleidiAI's performance.
    // ------------------------------------------------------------------
    if (KleidiAI::available()) {
        for (KleidiAI::Kernel k : kKaiKernels) {
            KleidiAI::Gemm g(k);
            g.reshape(M, N, K);
            if (!g.supports_panel_blocking()) continue;

            static const size_t kMc[] = {64, 256, 0};        // 0 = whole M
            static const size_t kKc[] = {2048, 0};           // 0 = whole K
            static const size_t kNc[] = {128, 512, 0};       // 0 = whole N

            KleidiAI::PanelConfig best{};
            double best_ms = 0.0;
            for (size_t mc : kMc)
                for (size_t kc : kKc)
                    for (size_t nc : kNc) {
                        // A is re-packed once per N block, so total A packing
                        // traffic scales with N/Nc. Candidates below N/Nc = 8
                        // are pathological (they dominated autotune runtime and
                        // never won); skip them.
                        if (nc && N / nc > 8) continue;
                        const KleidiAI::PanelConfig cfg{mc, kc, nc};
                        g.run_panel_blocked(A.data(), B.data(), C.data(), cfg);  // warm
                        std::vector<double> ms;
                        for (int i = 0; i < 2; ++i) {
                            const auto t0 = Clock::now();
                            g.run_panel_blocked(A.data(), B.data(), C.data(), cfg);
                            ms.push_back(std::chrono::duration<double, std::milli>(
                                             Clock::now() - t0).count());
                        }
                        const double v = std::min(ms[0], ms[1]);
                        if (best_ms == 0.0 || v < best_ms) { best_ms = v; best = cfg; }
                    }

            g.run_panel_blocked(A.data(), B.data(), C.data(), best);
            const Errors e = check(C, C_ref, M, N, samples);
            const Stats t = measure([&] {
                g.run_panel_blocked(A.data(), B.data(), C.data(), best);
            });
            consume(C);

            char variant[96];
            std::snprintf(variant, sizeof(variant), "Mc=%zu Kc=%zu Nc=%zu",
                          best.Mc ? best.Mc : M, best.Kc ? best.Kc : K,
                          best.Nc ? best.Nc : N);
            const size_t ri = add_row_ex("panel_blocked", "end_to_end",
                                         std::string("KAIpanel ") + KleidiAI::label(k),
                                         variant, t, e, false, true);

            // One instrumented run for the stage breakdown. Its total is a
            // little higher than the headline median because of the clock
            // calls, so only the proportions are meaningful.
            const KleidiAI::PanelBreakdown bd =
                g.run_panel_blocked_instrumented(A.data(), B.data(), C.data(), best);
            g_rows[ri].packing_ms = bd.pack_a_ms + bd.pack_b_ms;
            g_rows[ri].compute_ms = bd.compute_ms;
            g_panel_breakdown[std::string(KleidiAI::label(k)) + "@" + shape_key(M, N, K)] = bd;
        }
    }

    // ------------------------------------------------------------------
    // Experiment 3: llama.cpp / ggml CPU FP32 matmul, two dispatch paths.
    //
    //   llama-native-f32    src0 in the default CPU buffer type -> ggml's own
    //                       generic path (ggml_vec_dot_f32; llamafile/tinyBLAS
    //                       is #undef'd upstream on i8mm targets)
    //   llama-kleidiai-f32  src0 in the "CPU_KLEIDIAI" extra buffer type, which
    //                       is what KleidiAI's supports_op() requires and how
    //                       llama.cpp allocates model weights -> ggml's KleidiAI
    //                       integration -> fp32 SME2 micro-kernel
    //
    // Both go through the ordinary ggml graph/backend flow; neither calls a
    // KleidiAI entry point directly.
    // ------------------------------------------------------------------
    for (int kai_mode = 0; kai_mode < 2; ++kai_mode) {
        if (kai_mode == 1 && !GgmlCpu::kleidiai_build()) continue;
        const char* impl = kai_mode ? "llama-kleidiai" : "llama-native";

        GgmlCpu::Gemm gg;
        if (!gg.prepare(M, K, N, A.data(), B.data(), /*n_threads=*/1, kai_mode != 0)) {
            std::printf(" [%s prepare failed]", impl);
            continue;
        }
        if (kai_mode == 1 && !gg.using_kleidiai_buft()) {
            std::printf(" [CPU_KLEIDIAI buffer type unavailable]");
            continue;
        }
        gg.run();
        std::vector<float> C_ggml(M * N);
        gg.read_result(C_ggml.data());
        const Errors e = check(C_ggml, C_ref, M, N, samples);

        const Stats t = measure([&] { gg.run(); }, 0.35, 5, 101);
        consume(C_ggml);
        add_row_ex("ggml_cpu", "end_to_end", impl,
                   kai_mode ? "graph exec, CPU_KLEIDIAI buft" : "graph exec, default buft",
                   t, e, false, false);

        const Stats t2 = measure([&] { gg.run_with_setup(); }, 0.35, 5, 101);
        add_row_ex("ggml_cpu", "end_to_end", std::string(impl) + "+setup",
                   "graph rebuild+exec", t2, e, false, false);

        if (kai_mode == 0) {
            g_ggml_transpose_ms[shape_key(M, N, K)] = gg.b_transpose_ms();
            if (g_ggml_cpu_wall_ratio < 0.0 && M >= 1024)
                g_ggml_cpu_wall_ratio = gg.measure_cpu_to_wall_ratio(5);
        } else {
            g_kai_weight_pack_ms[shape_key(M, N, K)] = gg.weight_pack_ms();
            if (g_ggml_kai_wall_ratio < 0.0 && M >= 1024)
                g_ggml_kai_wall_ratio = gg.measure_cpu_to_wall_ratio(5);
        }
    }

    std::printf(" done\n");
    std::fflush(stdout);
}

// =============================================================================
// Experiment 1 — Hot Microkernel Ceiling
//
// The prepacked_full_gemm results are NOT this: they still traverse the whole
// output and stream full-size packed operands. Here the packed panels for a
// SINGLE output tile are built up front and kept small enough to stay resident,
// and the timed region contains only repeated micro-kernel invocations.
//
// Every MaxMulSK micro-kernel produces a 1024-element tile (4SVLx1SVL,
// 2SVLx2SVL or 1SVLx4SVL at SVL=16) and reads M_step*Kc + N_step*Kc packed
// floats, so all variants move the same bytes per invocation. KleidiAI's fp32
// SME2 entry points are invoked on their own m_step x n_step tile.
//
// ASYMMETRY THAT CANNOT BE REMOVED: MaxMulSK's micro-kernels are __arm_streaming
// leaf functions, so a batch runs with streaming mode entered once, exactly as
// the shipped driver does. KleidiAI's kai_run_matmul_* does smstart/smstop
// inside every call because that is where its API boundary is. To keep this
// visible rather than silently in our favour, MaxMulSK is reported twice: with
// streaming entered once per batch, and with it entered per invocation.
// =============================================================================

static const size_t kHotKc[] = {128, 256, 512, 1024, 2048};

// Time `reps_inner` invocations per timed sample; returns ns per invocation.
template <class F>
static Stats measure_batched(F&& batch, size_t /*reps_inner*/) {
    return measure(batch, 0.30, 9, 201);
}

static void run_hot_microkernel() {
    std::printf("  Experiment 1: hot micro-kernel ceiling ...");
    std::fflush(stdout);

    for (size_t Kc : kHotKc) {
        // Master panels; each kernel takes the leading m_step rows / n_step
        // columns, so every variant sees the same numbers.
        constexpr size_t kMaxStep = 64;
        std::vector<float> A_master(kMaxStep * Kc), B_master(Kc * kMaxStep);
        fill_random(A_master, 7);
        fill_random(B_master, 11);

        auto slice_B = [&](size_t n_step) {
            std::vector<float> Bs(Kc * n_step);
            for (size_t k = 0; k < Kc; ++k)
                for (size_t j = 0; j < n_step; ++j)
                    Bs[k * n_step + j] = B_master[k * kMaxStep + j];
            return Bs;
        };

        auto record = [&](const char* impl, const char* variant,
                          size_t m_step, size_t n_step, size_t reps_inner,
                          const Stats& t, const Errors& e, bool mm, bool kai) {
            Row r;
            r.experiment = "hot_microkernel";
            r.category = "hot"; r.mode = "hot_microkernel";
            r.impl = impl; r.variant = variant;
            r.M = m_step; r.N = n_step; r.K = Kc; r.Kc = Kc;
            r.reps_inner = reps_inner;
            r.t = t;
            r.ns_per_invocation = t.median_ms * 1e6 / double(reps_inner);
            const double flops = 2.0 * double(m_step) * double(n_step) * double(Kc);
            r.gflops = flops / (r.ns_per_invocation * 1e-9) / 1e9;
            r.e = e; r.is_maxmulsk = mm; r.is_kleidiai = kai;
            g_rows.push_back(std::move(r));
        };

        // ---- MaxMulSK ----
        for (MaxMulSK::Kernel k : MaxMulSK::kAllKernels) {
            if (!MaxMulSK::has_kernel_only(k)) continue;   // ZAIO has no isolable tile
            MaxMulSK::HotMicro hm;
            {
                MaxMulSK::HotMicro probe;
                probe.prepare(k, Kc, A_master.data(), B_master.data());
                const std::vector<float> Bs = slice_B(probe.n_step());
                hm.prepare(k, Kc, A_master.data(), Bs.data());
            }
            const size_t ms_ = hm.m_step(), ns_ = hm.n_step();

            // Correctness: one invocation from a zeroed tile must equal
            // A_small (ms_ x Kc) * B_small (Kc x ns_).
            const std::vector<float> Bs = slice_B(ns_);
            std::vector<float> ref(ms_ * ns_);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)ms_, (int)ns_, (int)Kc, 1.0f,
                        A_master.data(), (int)Kc, Bs.data(), (int)ns_,
                        0.0f, ref.data(), (int)ns_);
            hm.zero_tile();
            hm.run_batch(1);
            std::vector<float> got(hm.tile(), hm.tile() + ms_ * ns_);
            const Errors e = check(got, ref, ms_, ns_, {});

            // Size the inner batch so one timed sample is ~2 ms.
            const auto c0 = Clock::now();
            hm.run_batch(64);
            const double per_call_s =
                std::chrono::duration<double>(Clock::now() - c0).count() / 64.0;
            size_t reps_inner = 64;
            if (per_call_s > 0) reps_inner = (size_t)std::ceil(0.002 / per_call_s);
            reps_inner = std::clamp<size_t>(reps_inner, 64, 200000);

            char geom[64];
            std::snprintf(geom, sizeof(geom), "%zux%zu tile", ms_, ns_);
            record(MaxMulSK::label(k), geom, ms_, ns_, reps_inner,
                   measure_batched([&] { hm.run_batch(reps_inner); }, reps_inner),
                   e, true, false);

            char geom2[80];
            std::snprintf(geom2, sizeof(geom2), "%zux%zu, smstart/call", ms_, ns_);
            record((std::string(MaxMulSK::label(k)) + " [iso]").c_str(), geom2,
                   ms_, ns_, reps_inner,
                   measure_batched([&] { hm.run_batch_isolated_streaming(reps_inner); },
                                   reps_inner),
                   e, true, false);

            // The Acc variants do not own ZA inside the micro-kernel, so they
            // can also be measured with the zero and the ZA->C store amortised
            // across the batch instead of paid per invocation. That is the
            // number the variant exists to move; the row above stays directly
            // comparable with every other kernel.
            if (hm.supports_amortized()) {
                char geom3[80];
                std::snprintf(geom3, sizeof(geom3), "%zux%zu, zero+store amort", ms_, ns_);
                record((std::string(MaxMulSK::label(k)) + " [amort]").c_str(), geom3,
                       ms_, ns_, reps_inner,
                       measure_batched([&] { hm.run_batch_amortized(reps_inner); },
                                       reps_inner),
                       e, true, false);
            }
        }

        // ---- KleidiAI ----
        if (KleidiAI::available()) {
            for (KleidiAI::Kernel k : kKaiKernels) {
                KleidiAI::HotTile ht;
                {
                    KleidiAI::HotTile probe;
                    probe.prepare(k, Kc, A_master.data(), B_master.data());
                    const std::vector<float> Bs = slice_B(probe.n_step());
                    ht.prepare(k, Kc, A_master.data(), Bs.data());
                }
                const size_t ms_ = ht.m_step(), ns_ = ht.n_step();
                const std::vector<float> Bs = slice_B(ns_);
                std::vector<float> ref(ms_ * ns_);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            (int)ms_, (int)ns_, (int)Kc, 1.0f,
                            A_master.data(), (int)Kc, Bs.data(), (int)ns_,
                            0.0f, ref.data(), (int)ns_);
                ht.run_batch(1);
                std::vector<float> got(ht.tile(), ht.tile() + ms_ * ns_);
                const Errors e = check(got, ref, ms_, ns_, {});

                const auto c0 = Clock::now();
                ht.run_batch(64);
                const double per_call_s =
                    std::chrono::duration<double>(Clock::now() - c0).count() / 64.0;
                size_t reps_inner = 64;
                if (per_call_s > 0) reps_inner = (size_t)std::ceil(0.002 / per_call_s);
                reps_inner = std::clamp<size_t>(reps_inner, 64, 200000);

                char geom[64];
                std::snprintf(geom, sizeof(geom), "%zux%zu tile", ms_, ns_);
                record((std::string("KAI ") + KleidiAI::label(k)).c_str(), geom,
                       ms_, ns_, reps_inner,
                       measure_batched([&] { ht.run_batch(reps_inner); }, reps_inner),
                       e, false, true);
            }
        }
    }
    std::printf(" done\n");
    std::fflush(stdout);
}

// =============================================================================
// Output
// =============================================================================

static std::string shape_label(size_t M, size_t K, size_t N) {
    return std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);
}

static void headline_table(const char* category, const char* mode) {
    std::printf("%-20s %14s %12s %14s %12s %10s %10s  %s\n",
                "M x K x N",
                "MaxMulSK ms", "MaxMulSK GF", "KleidiAI ms", "KleidiAI GF",
                "ratio", "diff %", "correctness");
    std::printf("%s\n", std::string(120, '-').c_str());

    for (const Shape& sh : kShapes) {
        if (std::string(sh.category) != category) continue;
        const Row* mm = best_of(category, mode, sh.M, sh.N, sh.K, true);
        const Row* ka = best_of(category, mode, sh.M, sh.N, sh.K, false);
        if (!mm || !ka) {
            std::printf("%-20s  (missing result)\n", shape_label(sh.M, sh.K, sh.N).c_str());
            continue;
        }
        const double ratio = mm->gflops / ka->gflops;
        const double diff  = (ratio - 1.0) * 100.0;
        std::printf("%-20s %14.3f %12.1f %14.3f %12.1f %10.3f %+9.1f%%  %s\n",
                    shape_label(sh.M, sh.K, sh.N).c_str(),
                    mm->t.median_ms, mm->gflops, ka->t.median_ms, ka->gflops,
                    ratio, diff, (mm->e.ok && ka->e.ok) ? "PASS" : "FAIL");
    }
    std::printf("\n  Best MaxMulSK / best KleidiAI variant per shape:\n");
    for (const Shape& sh : kShapes) {
        if (std::string(sh.category) != category) continue;
        const Row* mm = best_of(category, mode, sh.M, sh.N, sh.K, true);
        const Row* ka = best_of(category, mode, sh.M, sh.N, sh.K, false);
        std::printf("    %-20s %-14s vs %-14s\n", shape_label(sh.M, sh.K, sh.N).c_str(),
                    mm ? mm->impl.c_str() : "-", ka ? ka->impl.c_str() : "-");
    }
    std::printf("\n");
}

static void detail_table(const char* category, const char* mode) {
    std::printf("%-20s %-14s %10s %10s %10s %10s %8s %10s\n",
                "M x K x N", "implementation",
                "median ms", "mean ms", "min ms", "stddev", "reps", "GFLOP/s");
    std::printf("%s\n", std::string(106, '-').c_str());
    for (const Shape& sh : kShapes) {
        if (std::string(sh.category) != category) continue;
        for (const auto& r : g_rows) {
            if (r.category != category || r.mode != mode) continue;
            if (r.M != sh.M || r.N != sh.N || r.K != sh.K) continue;
            std::printf("%-20s %-14s %10.3f %10.3f %10.3f %10.3f %8zu %10.1f\n",
                        shape_label(r.M, r.K, r.N).c_str(), r.impl.c_str(),
                        r.t.median_ms, r.t.mean_ms, r.t.min_ms, r.t.stddev_ms,
                        r.t.reps, r.gflops);
        }
    }
    std::printf("\n");
}

static void correctness_table() {
    std::printf("%-20s %-12s %-14s %14s %14s %16s  %s\n",
                "M x K x N", "mode", "implementation",
                "max abs err", "rel L2 err", "max abs vs FP64", "status");
    std::printf("%s\n", std::string(112, '-').c_str());
    for (const auto& r : g_rows) {
        if (r.mode == "zero_only") continue;
        std::printf("%-20s %-12s %-14s %14.3e %14.3e %16.3e  %s\n",
                    shape_label(r.M, r.K, r.N).c_str(), r.mode.c_str(), r.impl.c_str(),
                    r.e.max_abs, r.e.rel_l2, r.e.max_abs_fp64, r.e.ok ? "PASS" : "FAIL");
    }
    std::printf("\n");
}

static const Row* best_hot(size_t Kc, bool maxmulsk, bool allow_isolated) {
    const Row* best = nullptr;
    for (const auto& r : g_rows) {
        if (r.experiment != "hot_microkernel" || r.Kc != Kc) continue;
        if (maxmulsk ? !r.is_maxmulsk : !r.is_kleidiai) continue;
        if (!allow_isolated && r.impl.find("[iso]") != std::string::npos) continue;
        if (!r.e.ok) continue;
        if (!best || r.gflops > best->gflops) best = &r;
    }
    return best;
}

static void table_A() {
    std::printf("=====================================================================\n");
    std::printf("  Table A - Hot Microkernel (inner-kernel quality)\n");
    std::printf("=====================================================================\n\n");
    std::printf("  Packed panels for ONE output tile, resident; timed region is the\n");
    std::printf("  micro-kernel only. MaxMulSK enters streaming mode once per batch\n");
    std::printf("  (as its driver does); KleidiAI's entry point does smstart/smstop per\n");
    std::printf("  call. The [iso] rows re-measure MaxMulSK with per-call streaming so\n");
    std::printf("  that difference is visible rather than assumed away.\n\n");

    std::printf("%6s %14s %14s %8s  %-16s %-16s\n",
                "Kc", "MaxMulSK GF", "KleidiAI GF", "ratio",
                "MaxMul geometry", "KAI geometry");
    std::printf("%s\n", std::string(82, '-').c_str());
    for (size_t Kc : kHotKc) {
        const Row* mm = best_hot(Kc, true, false);
        const Row* ka = best_hot(Kc, false, false);
        if (!mm || !ka) continue;
        std::printf("%6zu %14.1f %14.1f %8.3f  %-16s %-16s\n",
                    Kc, mm->gflops, ka->gflops, mm->gflops / ka->gflops,
                    (mm->impl + " " + mm->variant).substr(0, 16).c_str(),
                    (ka->impl + " " + ka->variant).substr(0, 16).c_str());
    }

    std::printf("\n  Per-variant detail\n\n");
    std::printf("%6s %-18s %-22s %10s %12s %14s %10s\n",
                "Kc", "implementation", "geometry", "reps/sample",
                "ns/invoke", "GFLOP/s", "correct");
    std::printf("%s\n", std::string(98, '-').c_str());
    for (size_t Kc : kHotKc)
        for (const auto& r : g_rows) {
            if (r.experiment != "hot_microkernel" || r.Kc != Kc) continue;
            std::printf("%6zu %-18s %-22s %10zu %12.1f %14.1f %10s\n",
                        r.Kc, r.impl.c_str(), r.variant.c_str(), r.reps_inner,
                        r.ns_per_invocation, r.gflops, r.e.ok ? "PASS" : "FAIL");
        }
    std::printf("\n");
}

static const Row* find_exp(const char* experiment, size_t M, size_t N, size_t K,
                           bool best_by_gflops, const char* impl_prefix) {
    const Row* best = nullptr;
    for (const auto& r : g_rows) {
        if (r.experiment != experiment) continue;
        if (r.M != M || r.N != N || r.K != K) continue;
        if (impl_prefix && r.impl.rfind(impl_prefix, 0) != 0) continue;
        if (!r.e.ok) continue;
        if (!best || (best_by_gflops && r.gflops > best->gflops)) best = &r;
    }
    return best;
}

static void table_B() {
    std::printf("=====================================================================\n");
    std::printf("  Table B - Controlled End-to-End Flow (dataflow / packing quality)\n");
    std::printf("=====================================================================\n\n");
    std::printf("  Same micro-kernels as Table A, now inside three different whole-GEMM\n");
    std::printf("  organizations. 'KAI panel' is KleidiAI's own kernel and packing\n");
    std::printf("  routines driven by a MaxMulSK-style panel-blocked caller.\n\n");

    std::printf("%-20s %12s %12s %12s %11s %11s\n",
                "M x K x N", "MaxMulSK", "KAI orig", "KAI panel",
                "MM/orig", "MM/panel");
    std::printf("%s\n", std::string(84, '-').c_str());
    for (const Shape& sh : kShapes) {
        const Row* mm = best_of(sh.category, "end_to_end", sh.M, sh.N, sh.K, true);
        const Row* ko = find_exp("end_to_end", sh.M, sh.N, sh.K, true, "KAI ");
        const Row* kp = find_exp("panel_blocked", sh.M, sh.N, sh.K, true, "KAIpanel ");
        if (!mm || !ko || !kp) continue;
        std::printf("%-20s %12.1f %12.1f %12.1f %11.3f %11.3f\n",
                    shape_label(sh.M, sh.K, sh.N).c_str(),
                    mm->gflops, ko->gflops, kp->gflops,
                    mm->gflops / ko->gflops, mm->gflops / kp->gflops);
    }

    std::printf("\n  Panel-blocked configuration chosen, and its stage breakdown.\n");
    std::printf("  Percentages are of the instrumented run, whose total is slightly\n");
    std::printf("  above the headline median because of the clock calls.\n\n");
    std::printf("%-20s %-12s %-22s %9s %9s %9s %9s\n",
                "M x K x N", "kernel", "config", "packA %", "packB %", "compute %", "accum %");
    std::printf("%s\n", std::string(96, '-').c_str());
    for (const Shape& sh : kShapes) {
        for (const char* lbl : {"2VL", "8VS"}) {
            auto it = g_panel_breakdown.find(std::string(lbl) + "@" +
                                             shape_key(sh.M, sh.N, sh.K));
            if (it == g_panel_breakdown.end()) continue;
            const auto& b = it->second;
            const double tot = b.pack_a_ms + b.pack_b_ms + b.compute_ms +
                               b.accumulate_ms + b.other_ms;
            if (tot <= 0) continue;
            const Row* kp = nullptr;
            for (const auto& r : g_rows)
                if (r.experiment == "panel_blocked" && r.M == sh.M && r.N == sh.N &&
                    r.K == sh.K && r.impl == std::string("KAIpanel ") + lbl) kp = &r;
            std::printf("%-20s %-12s %-22s %8.1f%% %8.1f%% %8.1f%% %8.1f%%\n",
                        shape_label(sh.M, sh.K, sh.N).c_str(), lbl,
                        kp ? kp->variant.c_str() : "-",
                        100.0 * b.pack_a_ms / tot, 100.0 * b.pack_b_ms / tot,
                        100.0 * b.compute_ms / tot, 100.0 * b.accumulate_ms / tot);
        }
    }
    std::printf("\n");
}

static const Row* find_exact(const char* experiment, size_t M, size_t N, size_t K,
                             const char* impl) {
    for (const auto& r : g_rows)
        if (r.experiment == experiment && r.M == M && r.N == N && r.K == K &&
            r.impl == impl && r.e.ok)
            return &r;
    return nullptr;
}

static void table_C() {
    std::printf("=====================================================================\n");
    std::printf("  Table C - Runtime-Style CPU Comparison (vs a real inference runtime)\n");
    std::printf("=====================================================================\n\n");
    std::printf("%-20s %-16s %11s %11s %10s %10s\n",
                "M x K x N", "implementation", "median ms", "GFLOP/s",
                "vs MaxMul", "correct");
    std::printf("%s\n", std::string(84, '-').c_str());
    for (const Shape& sh : kShapes) {
        const Row* mm = best_of(sh.category, "end_to_end", sh.M, sh.N, sh.K, true);
        if (!mm) continue;
        struct Entry { const char* name; const Row* r; };
        const Row* ko = find_exp("end_to_end",   sh.M, sh.N, sh.K, true, "KAI ");
        const Row* kp = find_exp("panel_blocked",sh.M, sh.N, sh.K, true, "KAIpanel ");
        const Row* gn = find_exact("ggml_cpu", sh.M, sh.N, sh.K, "llama-native");
        const Row* gk = find_exact("ggml_cpu", sh.M, sh.N, sh.K, "llama-kleidiai");
        const Entry es[] = {
            {"MaxMulSK",        mm},
            {"KleidiAI direct", ko},
            {"KleidiAI panel",  kp},
            {"llama-native-f32",gn},
            {"llama-kleidiai",  gk},
        };
        for (const auto& en : es) {
            if (!en.r) continue;
            std::printf("%-20s %-16s %11.3f %11.1f %10.3f %10s\n",
                        shape_label(sh.M, sh.K, sh.N).c_str(), en.name,
                        en.r->t.median_ms, en.r->gflops,
                        en.r->gflops / mm->gflops, en.r->e.ok ? "PASS" : "FAIL");
        }
        std::printf("\n");
    }
    std::printf("  ggml note: the one-off B transpose into ggml's [K, N] weight layout\n");
    std::printf("  is NOT in the timed region (llama.cpp stores weights that way).\n");
    std::printf("  Measured transpose cost, for reference:\n\n");
    std::printf("%-20s %14s\n", "M x K x N", "transpose ms");
    std::printf("%s\n", std::string(36, '-').c_str());
    for (const Shape& sh : kShapes) {
        auto it = g_ggml_transpose_ms.find(shape_key(sh.M, sh.N, sh.K));
        if (it != g_ggml_transpose_ms.end())
            std::printf("%-20s %14.3f\n", shape_label(sh.M, sh.K, sh.N).c_str(), it->second);
    }
    std::printf("\n");
}

static void table_D() {
    std::printf("=====================================================================\n");
    std::printf("  Table D - Same KleidiAI primitive, two callers\n");
    std::printf("=====================================================================\n\n");
    std::printf("  Both columns execute the identical micro-kernel,\n");
    std::printf("  kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa\n");
    std::printf("  (m_step=n_step=mr=nr=32, kr=sr=1). The difference is entirely the\n");
    std::printf("  surrounding orchestration.\n\n");
    std::printf("  Two structural differences to keep in mind when reading the ratio:\n");
    std::printf("   1. llama.cpp maps ggml's src0 to KleidiAI's RHS, so the kernel sees\n");
    std::printf("      M and N swapped relative to the direct rows. On the skinny shapes\n");
    std::printf("      (M=64, N=512) that is a materially different problem for it.\n");
    std::printf("   2. llama.cpp pre-packs src0 once into the CPU_KLEIDIAI buffer (a\n");
    std::printf("      model-load cost, excluded here and reported separately); the\n");
    std::printf("      direct end-to-end rows pack both operands inside the timed call.\n");
    std::printf("      So this is not a like-for-like efficiency ratio - it is a\n");
    std::printf("      comparison of two whole flows around one primitive.\n\n");

    std::printf("%-20s %13s %13s %13s %11s %13s\n",
                "M x K x N", "KAI direct", "KAI panel", "llama+KAI",
                "llama/best", "wt pack ms");
    std::printf("%s\n", std::string(92, '-').c_str());
    for (const Shape& sh : kShapes) {
        const Row* ko = find_exp("end_to_end",    sh.M, sh.N, sh.K, true, "KAI ");
        const Row* kp = find_exp("panel_blocked", sh.M, sh.N, sh.K, true, "KAIpanel ");
        const Row* gk = find_exact("ggml_cpu", sh.M, sh.N, sh.K, "llama-kleidiai");
        if (!ko || !kp || !gk) continue;
        const double best = std::max(ko->gflops, kp->gflops);
        auto it = g_kai_weight_pack_ms.find(shape_key(sh.M, sh.N, sh.K));
        std::printf("%-20s %13.1f %13.1f %13.1f %11.3f %13.3f\n",
                    shape_label(sh.M, sh.K, sh.N).c_str(),
                    ko->gflops, kp->gflops, gk->gflops, gk->gflops / best,
                    it != g_kai_weight_pack_ms.end() ? it->second : 0.0);
    }
    std::printf("\n  llama.cpp KleidiAI vs its own native path (the acceleration ggml\n");
    std::printf("  actually gains by enabling GGML_CPU_KLEIDIAI):\n\n");
    std::printf("%-20s %14s %14s %10s\n", "M x K x N", "native GF", "KleidiAI GF", "speedup");
    std::printf("%s\n", std::string(62, '-').c_str());
    for (const Shape& sh : kShapes) {
        const Row* gn = find_exact("ggml_cpu", sh.M, sh.N, sh.K, "llama-native");
        const Row* gk = find_exact("ggml_cpu", sh.M, sh.N, sh.K, "llama-kleidiai");
        if (!gn || !gk) continue;
        std::printf("%-20s %14.1f %14.1f %10.2fx\n",
                    shape_label(sh.M, sh.K, sh.N).c_str(),
                    gn->gflops, gk->gflops, gk->gflops / gn->gflops);
    }
    std::printf("\n");
}

static void write_csv(const char* path) {
    std::FILE* f = std::fopen(path, "w");
    if (!f) { std::printf("  [warn] could not open %s for writing\n", path); return; }
    std::fprintf(f, "experiment,category,mode,M,N,K,implementation,variant,"
                    "median_ms,mean_ms,min_ms,stddev_ms,reps,ns_per_invocation,"
                    "gflops,packing_ms,compute_ms,max_abs_error,relative_error,"
                    "max_abs_error_vs_fp64,status\n");
    // Free-text fields can contain commas (e.g. "64x16, smstart/call"), so they
    // are quoted rather than emitted raw.
    auto q = [](const std::string& v) { return "\"" + v + "\""; };
    auto num_or_na = [](char* buf, size_t n, double v) {
        if (v < 0) std::snprintf(buf, n, "NA");
        else       std::snprintf(buf, n, "%.6f", v);
        return buf;
    };
    for (const auto& r : g_rows) {
        char pk[32], cp[32], ns[32];
        num_or_na(pk, sizeof(pk), r.packing_ms);
        num_or_na(cp, sizeof(cp), r.compute_ms);
        if (r.ns_per_invocation > 0) std::snprintf(ns, sizeof(ns), "%.3f", r.ns_per_invocation);
        else                          std::snprintf(ns, sizeof(ns), "NA");
        std::fprintf(f, "%s,%s,%s,%zu,%zu,%zu,%s,%s,"
                        "%.6f,%.6f,%.6f,%.6f,%zu,%s,%.3f,%s,%s,%.6e,%.6e,%.6e,%s\n",
                     r.experiment.c_str(), r.category.c_str(), r.mode.c_str(),
                     r.M, r.N, r.K, q(r.impl).c_str(), q(r.variant).c_str(),
                     r.t.median_ms, r.t.mean_ms, r.t.min_ms, r.t.stddev_ms, r.t.reps,
                     ns, r.gflops, pk, cp,
                     r.e.max_abs, r.e.rel_l2, r.e.max_abs_fp64,
                     r.e.ok ? "PASS" : "FAIL");
    }
    std::fclose(f);
    std::printf("  CSV written to %s\n\n", path);
}

// =============================================================================
// Packing analysis
// =============================================================================

static void packing_analysis() {
    std::printf("=====================================================================\n");
    std::printf("  Packing analysis\n");
    std::printf("=====================================================================\n\n");

    std::printf("  Per-implementation packing + allocation cost, derived as\n");
    std::printf("  (end_to_end median) - (kernel_only median). Both modes compute the\n");
    std::printf("  same C; the difference is packing, the packed-buffer allocation, and\n");
    std::printf("  for MaxMulSK the fact that end_to_end reads packed panels from a small\n");
    std::printf("  hot buffer while kernel_only streams them from a large cold one.\n\n");

    std::printf("%-20s %-14s %12s %12s %12s %10s\n",
                "M x K x N", "implementation", "e2e ms", "kernel ms", "delta ms", "pack %");
    std::printf("%s\n", std::string(86, '-').c_str());
    for (const Shape& sh : kShapes) {
        for (const auto& r : g_rows) {
            if (r.mode != "end_to_end") continue;
            if (r.M != sh.M || r.N != sh.N || r.K != sh.K) continue;
            const Row* ko = find_row("kernel_only", r.M, r.N, r.K, r.impl);
            if (!ko) continue;
            const double d = r.t.median_ms - ko->t.median_ms;
            std::printf("%-20s %-14s %12.3f %12.3f %12.3f %9.1f%%\n",
                        shape_label(r.M, r.K, r.N).c_str(), r.impl.c_str(),
                        r.t.median_ms, ko->t.median_ms, d,
                        r.t.median_ms > 0 ? 100.0 * d / r.t.median_ms : 0.0);
        }
    }

    std::printf("\n  Cost of the memset that MaxMulSK's C += A*B semantics require.\n");
    std::printf("  Charged to the accumulating kernels only. SME 1x4Acc and\n");
    std::printf("  SME 1x4AccKc overwrite C (their ZA->C store does not read C\n");
    std::printf("  back), need no memset, and are not charged one:\n\n");
    std::printf("%-20s %12s %14s\n", "M x K x N", "memset ms", "share of e2e");
    std::printf("%s\n", std::string(48, '-').c_str());
    for (const Shape& sh : kShapes) {
        const Row* z = find_row("zero_only", sh.M, sh.N, sh.K, "memset C");
        const Row* mm = best_of(sh.category, "end_to_end", sh.M, sh.N, sh.K, true);
        if (!z || !mm) continue;
        std::printf("%-20s %12.4f %13.2f%%\n", shape_label(sh.M, sh.K, sh.N).c_str(),
                    z->t.median_ms, 100.0 * z->t.median_ms / mm->t.median_ms);
    }

    std::printf("\n  KleidiAI outer N-block width chosen by autotuning (columns).\n");
    std::printf("  KleidiAI has no cache blocking of its own; this is the caller's\n");
    std::printf("  parameter and is tuned per shape so KleidiAI is measured at its best,\n");
    std::printf("  mirroring MaxMulSK being reported as the best of its six variants.\n\n");
    std::printf("%-20s %12s %12s\n", "M x K x N", "KAI 2VL", "KAI 8VS");
    std::printf("%s\n", std::string(46, '-').c_str());
    for (const Shape& sh : kShapes) {
        const std::string key = shape_key(sh.M, sh.N, sh.K);
        std::printf("%-20s", shape_label(sh.M, sh.K, sh.N).c_str());
        for (const char* lbl : {"2VL", "8VS"}) {
            auto it = g_kai_nblock.find(std::string(lbl) + "@" + key);
            if (it != g_kai_nblock.end()) std::printf(" %12zu", it->second);
            else                          std::printf(" %12s", "-");
        }
        std::printf("\n");
    }

    std::printf("\n  Ratio trend (MaxMulSK best / KleidiAI best, >1 means MaxMulSK ahead):\n\n");
    std::printf("%-20s %14s %14s\n", "M x K x N", "end_to_end", "kernel_only");
    std::printf("%s\n", std::string(50, '-').c_str());
    for (const Shape& sh : kShapes) {
        const Row* e_mm = best_of(sh.category, "end_to_end", sh.M, sh.N, sh.K, true);
        const Row* e_ka = best_of(sh.category, "end_to_end", sh.M, sh.N, sh.K, false);
        const Row* k_mm = best_of(sh.category, "kernel_only", sh.M, sh.N, sh.K, true);
        const Row* k_ka = best_of(sh.category, "kernel_only", sh.M, sh.N, sh.K, false);
        std::printf("%-20s", shape_label(sh.M, sh.K, sh.N).c_str());
        if (e_mm && e_ka) std::printf(" %14.3f", e_mm->gflops / e_ka->gflops);
        else              std::printf(" %14s", "-");
        if (k_mm && k_ka) std::printf(" %14.3f", k_mm->gflops / k_ka->gflops);
        else              std::printf(" %14s", "-");
        std::printf("\n");
    }
    std::printf("\n");
}

static void compact_summary() {
    std::printf("=====================================================================\n");
    std::printf("  Compact summary (copy/paste)\n");
    std::printf("=====================================================================\n\n");
    std::printf("| Shape (MxKxN) | mode | MaxMulSK GF | KleidiAI GF | ratio | winner |\n");
    std::printf("|---|---|---:|---:|---:|---|\n");
    for (const Shape& sh : kShapes) {
        for (const char* mode : {"end_to_end", "kernel_only"}) {
            const Row* mm = best_of(sh.category, mode, sh.M, sh.N, sh.K, true);
            const Row* ka = best_of(sh.category, mode, sh.M, sh.N, sh.K, false);
            if (!mm || !ka) continue;
            const double ratio = mm->gflops / ka->gflops;
            std::printf("| %s | %s | %.1f | %.1f | %.2f | %s |\n",
                        shape_label(sh.M, sh.K, sh.N).c_str(), mode,
                        mm->gflops, ka->gflops, ratio,
                        ratio >= 1.0 ? "MaxMulSK" : "KleidiAI");
        }
    }
    std::printf("\n");
}

// =============================================================================

int main(int argc, char** argv) {
    const char* csv_path = (argc > 1) ? argv[1] : "bench_maxmul_vs_kleidiai.csv";

    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
    omp_set_num_threads(1);
    // macOS has no thread-affinity API. USER_INTERACTIVE QoS is the available
    // lever: it biases the scheduler strongly toward the P-core cluster.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    GgmlCpu::start_log_capture();
    print_environment();

    if (!KleidiAI::available()) {
        std::printf("  FATAL: built without KleidiAI; nothing to compare against.\n"
                    "         Reconfigure with -DMAXMULSK_WITH_KLEIDIAI=ON.\n");
        return 1;
    }

    std::printf("  Running (this takes several minutes) ...\n");
    run_hot_microkernel();
    for (const Shape& sh : kShapes) run_shape(sh);
    std::printf("\n");
    {
        const std::string lg = GgmlCpu::ggml_log_capture();
        if (!lg.empty()) {
            std::printf("  ggml's own KleidiAI init diagnostics (captured via ggml_log_set):\n");
            size_t pos = 0;
            while (pos < lg.size()) {
                const size_t nl = lg.find('\n', pos);
                const std::string line = lg.substr(pos, nl == std::string::npos ? nl : nl - pos);
                if (!line.empty()) std::printf("    %s\n", line.c_str());
                if (nl == std::string::npos) break;
                pos = nl + 1;
            }
            std::printf("\n");
        }
    }
    if (g_ggml_kai_wall_ratio > 0.0) {
        std::printf("  llama-kleidiai thread check: CPU time / wall time = %.2f\n\n",
                    g_ggml_kai_wall_ratio);
    }
    if (g_ggml_cpu_wall_ratio > 0.0) {
        std::printf("  ggml thread check: CPU time / wall time = %.2f over a large shape\n",
                    g_ggml_cpu_wall_ratio);
        std::printf("  (~1.0 confirms a single worker actually executed the matmul;\n");
        std::printf("   a 4-thread run would land near 4.0.)\n\n");
    }

    std::printf("=====================================================================\n");
    std::printf("  Square GEMM Benchmark\n");
    std::printf("=====================================================================\n\n");
    std::printf("## End-to-End GEMM: Packing + Compute\n\n");
    headline_table("square", "end_to_end");
    std::printf("## Kernel-Only GEMM: Packing Excluded\n\n");
    headline_table("square", "kernel_only");

    std::printf("=====================================================================\n");
    std::printf("  LLM-Like Rectangular / Large-K GEMM Benchmark\n");
    std::printf("=====================================================================\n\n");
    std::printf("## End-to-End GEMM: Packing + Compute\n\n");
    headline_table("large_k", "end_to_end");
    std::printf("## Kernel-Only GEMM: Packing Excluded\n\n");
    headline_table("large_k", "kernel_only");

    std::printf("=====================================================================\n");
    std::printf("  Per-variant detail\n");
    std::printf("=====================================================================\n\n");
    std::printf("-- square, end_to_end --\n\n");     detail_table("square", "end_to_end");
    std::printf("-- square, kernel_only --\n\n");    detail_table("square", "kernel_only");
    std::printf("-- large_k, end_to_end --\n\n");    detail_table("large_k", "end_to_end");
    std::printf("-- large_k, kernel_only --\n\n");   detail_table("large_k", "kernel_only");

    std::printf("=====================================================================\n");
    std::printf("  Correctness\n");
    std::printf("=====================================================================\n\n");
    correctness_table();

    table_A();
    table_B();
    table_C();
    table_D();
    packing_analysis();
    compact_summary();
    write_csv(csv_path);

    // Keep the sink observable.
    if (g_sink == 12345.6789) std::printf("");
    return 0;
}
