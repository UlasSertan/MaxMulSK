// =============================================================================
// MaxMulSK (v3 kernels) vs MpGEMM, single-threaded FP32, row-major.
//
// MpGEMM: https://github.com/mpgemm/MPGEMM  -- "Demystifying ARM SME to
// Optimize General Matrix Multiplications", IPDPS '26. Same hardware, same ISA,
// same precision, so GFLOP/s is directly comparable here.
//
// Methodology matches bench_headline: reps calibrated on the slowest entrant,
// call order rotated per shape so no implementation always runs first into a
// cold cache, median of the per-rep timings, line-buffered output.
// =============================================================================
#include "sme/v3/sme-1x4-acc-kcout.hpp"
#include "sme/v3/sme-2x2-acc-kcout.hpp"
#include "sme/v3/sme-4x1-acc-kcout.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <functional>
#include <Accelerate/Accelerate.h>

extern "C" void row_sgemm(int m, int n, int k, float* XA, float* XB, float* XC);

// MpGEMM's hand-written .S kernels clobber the callee-saved SIMD registers
// d8-d15 without restoring them; verified 2026-09-10 by holding known values in
// d8/d9/d14 across row_sgemm and reading back zeros. That is an AAPCS64
// violation and it destroys any floating-point value the caller had live across
// the call -- in this harness it silently turned every timing into inf and then
// into 0. The empty asm below names those registers as clobbered, which forces
// this wrapper to spill and reload them around the call, so the damage cannot
// escape into the timing code. Nothing about MpGEMM's own work is changed.
__attribute__((noinline))
static void mpgemm_sgemm(size_t M, size_t K, size_t N,
                         const float* A, const float* B, float* C) {
    row_sgemm((int)M, (int)N, (int)K, const_cast<float*>(A), const_cast<float*>(B), C);
    __asm__ volatile("" ::: "d8","d9","d10","d11","d12","d13","d14","d15","memory");
}

using Clock = std::chrono::steady_clock;

struct Shape { size_t M, K, N; const char* tag; };

// Our own set, then the four MpGEMM-family shapes: large N with short K, plus
// one long-K reduction case.
static const Shape kShapes[] = {
    {  256,   256,   256, "square"   },
    {  512,   512,   512, "square"   },
    { 1024,  1024,  1024, "square"   },
    { 2048,  2048,  2048, "square"   },
    { 4096,  4096,  4096, "square"   },
    {   64,  8192,   512, "llm"      },
    {   64, 16384,   512, "llm"      },
    {   64, 32768,   512, "llm"      },
    {  128,  8192,   512, "llm"      },
    {  128, 16384,   512, "llm"      },
    {  128, 32768,   512, "llm"      },
    // TABLE III of the MpGEMM paper: GEMM shapes from DeepSeek (IDs 1-18) and
    // LLaMA (IDs 19-24). The paper lists them as (M, N, K); they are written
    // here in this file's (M, K, N) order.
    {    64,  7168,  2112, "deepseek-id1" },
    {    64,  1536, 24576, "deepseek-id2" },
    {    64,   512, 32768, "deepseek-id3" },
    {    64, 16384,  7168, "deepseek-id4" },
    {    64,  7168,  4096, "deepseek-id5" },
    {    64,  2048,  7168, "deepseek-id6" },
    {   128,  7168,  2112, "deepseek-id7" },
    {   128,  1536, 24576, "deepseek-id8" },
    {   128,   512, 32768, "deepseek-id9" },
    {   128, 16384,  7168, "deepseek-id10" },
    {   128,  7168,  4096, "deepseek-id11" },
    {   128,  2048,  7168, "deepseek-id12" },
    {  4096,  7168,  2112, "deepseek-id13" },
    {  4096,  1536, 24576, "deepseek-id14" },
    {  4096,   512, 32768, "deepseek-id15" },
    {  4096, 16384,  7168, "deepseek-id16" },
    {  4096,  7168,  4096, "deepseek-id17" },
    {  4096,  2048,  7168, "deepseek-id18" },
    {  4096,  4096,   256, "llama-id19" },
    { 11008,  4096,   256, "llama-id20" },
    {  4096, 11008,   256, "llama-id21" },
    {  5120,  5120,   256, "llama-id22" },
    { 13824,  5120,   256, "llama-id23" },
    {  5120, 13824,   256, "llama-id24" },
};

struct Impl { const char* name; std::function<void(const float*, const float*, float*, size_t, size_t, size_t)> run; };

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    // Accelerate multi-threads by default; this comparison is single-thread.
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);

    std::vector<Impl> impls = {
        {"MaxMulSK-1x4", [](const float* A, const float* B, float* C, size_t M, size_t K, size_t N){
            SMEKernels1x4AccKcOut::run_multiplication(A, B, C, M, K, N); }},
        {"MaxMulSK-2x2", [](const float* A, const float* B, float* C, size_t M, size_t K, size_t N){
            SMEKernels2x2AccKcOut::run_multiplication(A, B, C, M, K, N); }},
        {"MaxMulSK-4x1", [](const float* A, const float* B, float* C, size_t M, size_t K, size_t N){
            SMEKernels4x1AccKcOut::run_multiplication(A, B, C, M, K, N); }},
        // MpGEMM's kernels take pc and accumulate across K panels, so C must be
        // zeroed by the caller. That memset is charged to MpGEMM's time, which
        // is the honest accounting: our kernels overwrite C and pay no such cost.
        {"Accelerate", [](const float* A, const float* B, float* C, size_t M, size_t K, size_t N){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K,
                        1.0f, A, (int)K, B, (int)N, 0.0f, C, (int)N); }},
        {"MpGEMM-sgemm", [](const float* A, const float* B, float* C, size_t M, size_t K, size_t N){
            // row_sgemm overwrites C (verified: two back-to-back calls leave
            // C unchanged), so it needs no pre-zeroing and is not charged one.
            mpgemm_sgemm(M, K, N, A, B, C); }},
    };

    const char* skip = std::getenv("MAXMULSK_SKIP");
    if (skip) impls.erase(std::remove_if(impls.begin(), impls.end(),
        [&](const Impl& i){ return std::string(i.name).find(skip) != std::string::npos; }), impls.end());

    const int shape_lo = (argc > 1) ? std::atoi(argv[1]) : 0;
    const int shape_hi = (argc > 2) ? std::atoi(argv[2]) : (int)(sizeof(kShapes)/sizeof(kShapes[0]));

    std::printf("shape,tag,M,K,N,impl,reps,ms,GFLOPs,checksum\n");

    std::mt19937 rng(20260910);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // One global warm-up so the first shape does not absorb page-fault and
    // frequency-ramp cost that the others never see.
    {
        std::vector<float> a(256*256), b(256*256), c(256*256);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);
        for (auto& im : impls) for (int i = 0; i < 20; i++) im.run(a.data(), b.data(), c.data(), 256, 256, 256);
    }

    for (int si = shape_lo; si < shape_hi; si++) {
        const Shape& s = kShapes[si];
        std::vector<float> A(s.M * s.K), B(s.K * s.N), C(s.M * s.N);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);

        const double flops = 2.0 * (double)s.M * (double)s.K * (double)s.N;

        // Calibrate on the SLOWEST implementation so every entrant gets the same
        // rep count; calibrating per-implementation would give the slow one
        // fewer reps and a noisier median.
        double worst = 0.0;
        for (auto& im : impls) {
            im.run(A.data(), B.data(), C.data(), s.M, s.K, s.N);      // warm this shape
            auto t0 = Clock::now();
            im.run(A.data(), B.data(), C.data(), s.M, s.K, s.N);
            worst = std::max(worst, (double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count() * 1e-9);
        }
        size_t reps = (size_t)std::clamp(0.35 / std::max(worst, 1e-9), 5.0, 20000.0);

        // Cyclic Latin square on the call order: for shape si the order starts
        // at impls[si % n]. Over any n consecutive shapes each implementation
        // occupies each position exactly once, so none of them systematically
        // runs first into a cold cache or last into a warm one.
        for (size_t r = 0; r < impls.size(); r++) {
            Impl& im = impls[(si + r) % impls.size()];
            std::vector<double> t; t.reserve(reps);
            for (size_t i = 0; i < reps; i++) {
                auto t0 = Clock::now();
                im.run(A.data(), B.data(), C.data(), s.M, s.K, s.N);
                t.push_back((double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count() * 1e-6);
            }
            double ms = median(t);
            double sum = 0.0;
            for (size_t i = 0; i < C.size(); i += 977) sum += C[i];
            std::printf("%d,%s,%zu,%zu,%zu,%s,%zu,%.4f,%.1f,%.6f\n",
                        si, s.tag, s.M, s.K, s.N, im.name, reps, ms, flops / (ms * 1e6), sum);
        }
    }
    return 0;
}
