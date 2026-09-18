// =============================================================================
// R0 (sme/v3 acc-kcout, default blocking) vs F2 (sme/v4 mc16-apack4za-bonce)
// vs F2-ZaPres (F2 with pack_B_streaming declared __arm_preserves("za"))
// at the F2 target shape only: M=11008, K=4096, N=256, FP32, single thread.
//
// Both entrants allocate their packed buffers inside their own call, so the
// allocation scope is identical and neither is charged a cost the other avoids.
// Order is rotated per run, reps are calibrated on the slower entrant so both
// get the same count, and every run is reported so the spread is visible.
// =============================================================================
#include "sme/v3/sme-1x4-acc-kcout.hpp"
#include "sme/v4/sme-1x4-kcout-mc16-apack4za-bonce.hpp"
#include "sme/v4/sme-1x4-kcout-mc16-apack4za-bonce-zapres.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <Accelerate/Accelerate.h>

extern "C" void row_sgemm(int m, int n, int k, float* XA, float* XB, float* XC);

// MpGEMM's hand-written .S kernels clobber the callee-saved SIMD registers
// d8-d15 without restoring them (verified 2026-09-10 by holding known values in
// d8/d9/d14 across row_sgemm and reading back zeros). Naming them as clobbered
// forces this wrapper to spill and reload them around the call, so the damage
// cannot reach the timing code. MpGEMM's own work is unchanged.
__attribute__((noinline))
static void mpgemm_sgemm(size_t M_, size_t K_, size_t N_,
                         const float* A, const float* B, float* C) {
    row_sgemm((int)M_, (int)N_, (int)K_, const_cast<float*>(A), const_cast<float*>(B), C);
    __asm__ volatile("" ::: "d8","d9","d10","d11","d12","d13","d14","d15","memory");
}

using Clock = std::chrono::steady_clock;
static const size_t M = 11008, K = 4096, N = 256;

static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n/2] : 0.5 * (v[n/2-1] + v[n/2]);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);  // this comparison is single-thread
    const int runs = (argc > 1) ? std::atoi(argv[1]) : 9;

    std::vector<float> A(M*K), B(K*N), C(M*N);
    std::mt19937 rng(913); std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& v : A) v = d(rng);
    for (auto& v : B) v = d(rng);

    struct Impl { const char* name; void (*run)(const float*, const float*, float*); };
    static const float *gA, *gB; static float* gC;
    gA = A.data(); gB = B.data(); gC = C.data();
    Impl impls[] = {
        {"R0",        [](const float* a, const float* b, float* c){
            SMEKernels1x4AccKcOut::run_multiplication(a, b, c, M, K, N); }},
        {"F2",        [](const float* a, const float* b, float* c){
            SMEKernels1x4KcOutMc16Apack4ZaBonce::run_multiplication_f2(a, b, c, M, K, N); }},
        {"F2-ZaPres", [](const float* a, const float* b, float* c){
            SMEKernels1x4KcOutMc16Apack4ZaBonceZaPres::run_multiplication_f2(a, b, c, M, K, N); }},
        {"Accelerate", [](const float* a, const float* b, float* c){
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K,
                        1.0f, a, (int)K, b, (int)N, 0.0f, c, (int)N); }},
        // row_sgemm overwrites C, so it is not charged a pre-zeroing pass.
        {"MpGEMM",     [](const float* a, const float* b, float* c){
            mpgemm_sgemm(M, K, N, a, b, c); }},
    };
    const int NI = (int)(sizeof(impls)/sizeof(impls[0]));

    // Global warm-up so the first timed entrant does not absorb page-fault and
    // frequency-ramp cost the others never see.
    for (int i = 0; i < 3; i++) for (auto& im : impls) im.run(gA, gB, gC);

    const double flops = 2.0 * M * K * N;
    double slowest = 0;
    for (auto& im : impls) {
        auto t1 = Clock::now(); im.run(gA, gB, gC);
        slowest = std::max(slowest, (double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t1).count()*1e-9);
    }
    const size_t reps = (size_t)std::clamp(0.35 / slowest, 3.0, 200.0);

    std::printf("run,impl,reps,ms,GFLOPs,checksum\n");
    std::vector<std::vector<double>> g(NI);
    for (int r = 0; r < runs; r++) {
        // Cyclic rotation: run r starts at impls[r % NI], so over any NI runs
        // each entrant occupies each slot exactly once.
        for (int slot = 0; slot < NI; slot++) {
            const int idx = (r + slot) % NI;
            std::vector<double> t; t.reserve(reps);
            for (size_t i = 0; i < reps; i++) {
                auto a = Clock::now();
                impls[idx].run(gA, gB, gC);
                t.push_back((double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-a).count()*1e-6);
            }
            double ms = med(t), gf = flops / (ms * 1e6);
            double s = 0; for (size_t i = 0; i < C.size(); i += 1013) s += C[i];
            g[idx].push_back(gf);
            std::printf("%d,%s,%zu,%.3f,%.1f,%.6f\n", r, impls[idx].name, reps, ms, gf, s);
        }
    }
    std::printf("\n");
    for (int i = 0; i < NI; i++) {
        auto mm = std::minmax_element(g[i].begin(), g[i].end());
        std::printf("%-10s medyan %.1f  min %.1f  max %.1f  yayilma %.2f%%\n",
                    impls[i].name, med(g[i]), *mm.first, *mm.second,
                    100.0*(*mm.second-*mm.first)/med(g[i]));
    }
    for (int i = 1; i < NI; i++)
        std::printf("%s/R0 = %.3fx\n", impls[i].name, med(g[i])/med(g[0]));
    std::printf("F2-ZaPres/F2 = %.4fx\n", med(g[2])/med(g[1]));
    return 0;
}
